"""Train enhanced uncertainty-guided local diffusion from a fitted Stage-3 model.

The script loads only the successful deterministic dual-domain heads from an
existing uncertainty-guided Stage-3 checkpoint. The old diffusion branches are
not reused because their normalization no longer matches the new remaining-
error scales.

Default schedule:
- 20 epochs: low-rate deterministic region-specialization fine-tuning;
- 120 epochs: frozen-deterministic diffusion with oracle-to-predicted mask curriculum;
- 40 epochs: low-rate joint fine-tuning;
- 12 deterministic DDIM steps at validation and inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import build_loaders
from losses import SAMLoss
from models.stage3_uncertainty_guided_diffusion_v2 import (
    UncertaintyGuidedDualDomainDiffusionRefinerV2,
)
from train_stage2_coefficients import (
    first_spectral_difference,
    second_spectral_difference,
)
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import (
    evaluate,
    parse_uncertainty_guided_args,
    spatial_correlation,
)
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def parse_v2_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage3_initial_checkpoint",
        type=str,
        default=(
            "./checkpoints_stage3_extended/"
            "stage3_uncertainty_guided_diffusion/PaviaU/"
            "uncertainty_guided_best_psnr.pth"
        ),
    )
    parser.add_argument("--v2_specialization_epochs", type=int, default=20)
    parser.add_argument("--v2_diffusion_epochs", type=int, default=120)
    parser.add_argument("--v2_joint_epochs", type=int, default=40)
    parser.add_argument("--v2_specialization_lr", type=float, default=2e-6)
    parser.add_argument("--v2_diffusion_lr", type=float, default=5e-5)
    parser.add_argument("--v2_joint_det_lr", type=float, default=1e-6)
    parser.add_argument("--v2_joint_diff_lr", type=float, default=1e-5)
    parser.add_argument("--v2_direct_x0_weight", type=float, default=0.7)
    parser.add_argument("--v2_oracle_curriculum_fraction", type=float, default=0.6)
    parser.add_argument(
        "--v2_det_high_uncertainty_suppression",
        type=float,
        default=0.7,
    )
    parser.add_argument("--v2_scale_oracle_weight", type=float, default=0.5)
    parser.add_argument("--v2_scale_estimation_batches", type=int, default=0)
    parser.add_argument("--v2_grad_clip", type=float, default=1.0)

    parser.add_argument("--v2_lambda_det_nll", type=float, default=1.0)
    parser.add_argument("--v2_lambda_det_x0", type=float, default=0.2)
    parser.add_argument("--v2_lambda_uncertainty", type=float, default=0.1)
    parser.add_argument("--v2_lambda_mask", type=float, default=0.1)
    parser.add_argument("--v2_lambda_det_reconstruction", type=float, default=0.15)
    parser.add_argument("--v2_lambda_noise", type=float, default=1.0)
    parser.add_argument("--v2_lambda_direct_x0", type=float, default=1.0)
    parser.add_argument("--v2_lambda_hybrid_x0", type=float, default=0.5)
    parser.add_argument("--v2_lambda_final_l1", type=float, default=0.5)
    parser.add_argument("--v2_lambda_final_sam", type=float, default=0.2)
    parser.add_argument("--v2_lambda_final_sgrad1", type=float, default=0.05)
    parser.add_argument("--v2_lambda_final_sgrad2", type=float, default=0.02)
    parser.add_argument("--v2_lambda_orthogonality", type=float, default=0.05)
    parser.add_argument("--v2_joint_det_weight", type=float, default=0.25)

    specific, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_uncertainty_guided_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    integer_options = (
        cfg.v2_specialization_epochs,
        cfg.v2_diffusion_epochs,
        cfg.v2_joint_epochs,
        cfg.v2_scale_estimation_batches,
    )
    if any(value < 0 for value in integer_options):
        raise ValueError("V2 epoch and scale-estimation counts must be non-negative")
    if not 0.0 <= cfg.v2_direct_x0_weight <= 1.0:
        raise ValueError("v2_direct_x0_weight must be in [0, 1]")
    if not 0.0 < cfg.v2_oracle_curriculum_fraction <= 1.0:
        raise ValueError("v2_oracle_curriculum_fraction must be in (0, 1]")
    if not 0.0 <= cfg.v2_det_high_uncertainty_suppression < 1.0:
        raise ValueError("deterministic suppression must be in [0, 1)")
    if not 0.0 <= cfg.v2_scale_oracle_weight <= 1.0:
        raise ValueError("v2_scale_oracle_weight must be in [0, 1]")
    cfg.epochs = (
        cfg.v2_specialization_epochs
        + cfg.v2_diffusion_epochs
        + cfg.v2_joint_epochs
    )
    cfg.stage3_inference_steps = 12
    cfg.stage3_initial_noise = "zero"
    return cfg


def build_v2_model(cfg, stage2, device: torch.device):
    return UncertaintyGuidedDualDomainDiffusionRefinerV2(
        stage2_model=stage2,
        deterministic_hidden_channels=cfg.stage3_det_hidden_channels,
        deterministic_blocks=cfg.stage3_det_blocks,
        diffusion_hidden_channels=cfg.stage3_diff_hidden_channels,
        diffusion_blocks=cfg.stage3_diff_blocks,
        time_channels=cfg.stage3_time_channels,
        diffusion_timesteps=cfg.stage3_diffusion_timesteps,
        beta_start=cfg.stage3_beta_start,
        beta_end=cfg.stage3_beta_end,
        max_normalized_residual=cfg.stage3_max_normalized_residual,
        clean_clip=cfg.stage3_clean_clip,
        log_variance_min=cfg.stage3_log_variance_min,
        log_variance_max=cfg.stage3_log_variance_max,
        basis_mask_threshold=cfg.stage3_basis_mask_threshold,
        orthogonal_mask_threshold=cfg.stage3_orthogonal_mask_threshold,
        mask_temperature=cfg.stage3_mask_temperature,
        mask_spread_floor=cfg.stage3_mask_spread_floor,
        msi_residual_gain=cfg.stage3_msi_residual_gain,
        residual_scale_floor=cfg.stage3_residual_scale_floor,
        direct_x0_weight=cfg.v2_direct_x0_weight,
    ).to(device)


def _torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_deterministic_initialization(
    model: UncertaintyGuidedDualDomainDiffusionRefinerV2,
    path: str,
    device: torch.device,
) -> int:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage-3 initialization checkpoint not found: {path}")
    state = _torch_load(path, device)
    source = state["model"] if "model" in state else state
    allowed_prefixes = (
        "coefficient_deterministic.",
        "orthogonal_deterministic.",
    )
    allowed_exact = {
        "coefficient_residual_scale",
        "orthogonal_residual_scale",
    }
    filtered = {
        key: value
        for key, value in source.items()
        if key.startswith(allowed_prefixes) or key in allowed_exact
    }
    if not any(key.startswith("coefficient_deterministic.") for key in filtered):
        raise RuntimeError("No deterministic Stage-3 parameters found in checkpoint")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected deterministic initialization keys: {unexpected}")
    print(
        f"Loaded deterministic Stage-3 initialization from {path}; "
        f"new diffusion parameters remain zero-initialized ({len(missing)} missing keys)."
    )
    return int(state.get("epoch", 0))


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1e-6)


def weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 0.25,
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(
        prediction,
        target,
        beta=beta,
        reduction="none",
    )
    return weighted_mean(elementwise, weight)


def reconstruction_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sam_loss: SAMLoss,
    l1_weight: float,
    sam_weight: float,
    sgrad1_weight: float,
    sgrad2_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    parts = {
        "l1": F.l1_loss(prediction, target),
        "sam": sam_loss(prediction, target),
        "sgrad1": F.l1_loss(
            first_spectral_difference(prediction),
            first_spectral_difference(target),
        ),
        "sgrad2": F.l1_loss(
            second_spectral_difference(prediction),
            second_spectral_difference(target),
        ),
    }
    total = (
        l1_weight * parts["l1"]
        + sam_weight * parts["sam"]
        + sgrad1_weight * parts["sgrad1"]
        + sgrad2_weight * parts["sgrad2"]
    )
    return total, parts


@torch.no_grad()
def estimate_diffusion_scales(
    model: UncertaintyGuidedDualDomainDiffusionRefinerV2,
    loader,
    device: torch.device,
    oracle_weight: float = 0.5,
    max_batches: int = 0,
) -> Dict[str, torch.Tensor]:
    model.eval()
    coefficient_square_sum = torch.zeros(
        model.basis_rank,
        device=device,
        dtype=torch.float64,
    )
    coefficient_weight_sum = torch.zeros((), device=device, dtype=torch.float64)
    orthogonal_square_sum = torch.zeros((), device=device, dtype=torch.float64)
    orthogonal_weight_sum = torch.zeros((), device=device, dtype=torch.float64)

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        stage2_outputs = model.stage2_forward(batch["lr_hsi"], batch["hr_msi"])
        deterministic = model.deterministic_forward_from_stage2(
            stage2_outputs,
            batch["hr_msi"],
        )
        residual = batch["gt"] - stage2_outputs["reconstructed_hsi"]
        target_coefficient, _, target_orthogonal = model.decompose_residual(
            residual,
            stage2_outputs["basis"],
        )
        remaining_coefficient = (
            target_coefficient
            - deterministic["deterministic_coefficient_residual"]
        )
        remaining_orthogonal = model.project_orthogonal(
            target_orthogonal
            - deterministic["deterministic_orthogonal_residual"],
            stage2_outputs["basis"],
        )
        base_coefficient_scale = model.coefficient_residual_scale.view(1, -1, 1, 1)
        base_orthogonal_scale = model.orthogonal_residual_scale.view(1, 1, 1, 1)
        oracle_basis = model.error_to_oracle_mask(
            remaining_coefficient / base_coefficient_scale,
            model.basis_mask_threshold,
        )
        oracle_orthogonal = model.error_to_oracle_mask(
            remaining_orthogonal / base_orthogonal_scale,
            model.orthogonal_mask_threshold,
        )
        basis_weight = (
            oracle_weight * oracle_basis
            + (1.0 - oracle_weight) * deterministic["basis_mask"]
        ).double()
        orthogonal_weight = (
            oracle_weight * oracle_orthogonal
            + (1.0 - oracle_weight) * deterministic["orthogonal_mask"]
        ).double()
        coefficient_square_sum += (
            remaining_coefficient.double().square() * basis_weight
        ).sum(dim=(0, 2, 3))
        coefficient_weight_sum += basis_weight.sum()
        orthogonal_square_sum += (
            remaining_orthogonal.double().square() * orthogonal_weight
        ).sum()
        orthogonal_weight_sum += orthogonal_weight.sum() * model.n_bands

    if coefficient_weight_sum.item() <= 0 or orthogonal_weight_sum.item() <= 0:
        raise RuntimeError("No valid samples for diffusion-scale estimation")
    coefficient_scale = torch.sqrt(
        coefficient_square_sum / coefficient_weight_sum
    ).float()
    orthogonal_scale = torch.sqrt(
        orthogonal_square_sum / orthogonal_weight_sum
    ).float()
    model.set_diffusion_scales(coefficient_scale, orthogonal_scale)
    return {
        "coefficient": model.coefficient_diffusion_scale.detach().clone(),
        "orthogonal": model.orthogonal_diffusion_scale.detach().clone(),
    }


def oracle_mix_for_epoch(epoch: int, cfg, phase: str) -> float:
    if phase != "diffusion" or cfg.v2_diffusion_epochs <= 0:
        return 0.0
    local_epoch = epoch - cfg.v2_specialization_epochs
    curriculum_epochs = max(
        int(round(
            cfg.v2_diffusion_epochs * cfg.v2_oracle_curriculum_fraction
        )),
        1,
    )
    if local_epoch >= curriculum_epochs:
        return 0.0
    if curriculum_epochs == 1:
        return 1.0
    return max(1.0 - local_epoch / float(curriculum_epochs - 1), 0.0)


def set_requires_grad(parameters: Iterable[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def configure_phase(model, optimizer, epoch: int, cfg):
    specialization_end = cfg.v2_specialization_epochs
    diffusion_end = specialization_end + cfg.v2_diffusion_epochs
    if epoch < specialization_end:
        phase = "specialization"
        det_enabled, diff_enabled = True, False
        det_lr, diff_lr = cfg.v2_specialization_lr, 0.0
    elif epoch < diffusion_end:
        phase = "diffusion"
        det_enabled, diff_enabled = False, True
        det_lr, diff_lr = 0.0, cfg.v2_diffusion_lr
    else:
        phase = "joint"
        det_enabled, diff_enabled = True, True
        det_lr, diff_lr = cfg.v2_joint_det_lr, cfg.v2_joint_diff_lr
    set_requires_grad(model.deterministic_parameters(), det_enabled)
    set_requires_grad(model.diffusion_parameters(), diff_enabled)
    for group in optimizer.param_groups:
        if group["group_name"] == "deterministic":
            group["lr"] = det_lr
        else:
            group["lr"] = diff_lr
    return phase, diff_enabled, det_lr, diff_lr


def compute_losses(model, outputs, gt, sam_loss, cfg, phase: str):
    suppression = cfg.v2_det_high_uncertainty_suppression
    basis_det_weight = 1.0 - suppression * outputs["basis_oracle_mask"]
    orthogonal_det_weight = 1.0 - suppression * outputs[
        "orthogonal_oracle_mask"
    ]
    basis_error = (
        outputs["deterministic_coefficient_normalized"]
        - outputs["target_coefficient_normalized"]
    )
    orthogonal_error = (
        outputs["deterministic_orthogonal_normalized"]
        - outputs["target_orthogonal_normalized"]
    )
    basis_log_variance = outputs["deterministic_coefficient_log_variance"]
    orthogonal_log_variance = outputs[
        "deterministic_orthogonal_log_variance"
    ]
    basis_nll = weighted_mean(
        torch.exp(-basis_log_variance) * basis_error.square()
        + basis_log_variance,
        basis_det_weight,
    )
    orthogonal_nll = weighted_mean(
        torch.exp(-orthogonal_log_variance) * orthogonal_error.square()
        + orthogonal_log_variance,
        orthogonal_det_weight,
    )
    basis_x0 = weighted_smooth_l1(
        outputs["deterministic_coefficient_normalized"],
        outputs["target_coefficient_normalized"],
        basis_det_weight,
    )
    orthogonal_x0 = weighted_smooth_l1(
        outputs["deterministic_orthogonal_normalized"],
        outputs["target_orthogonal_normalized"],
        orthogonal_det_weight,
    )
    uncertainty_loss = (
        F.smooth_l1_loss(
            torch.exp(0.5 * basis_log_variance),
            basis_error.detach().abs(),
            beta=0.25,
        )
        + F.smooth_l1_loss(
            torch.exp(0.5 * orthogonal_log_variance),
            orthogonal_error.detach().abs(),
            beta=0.25,
        )
    )
    mask_loss = (
        F.smooth_l1_loss(
            outputs["basis_mask"],
            outputs["basis_oracle_mask"],
            beta=0.1,
        )
        + F.smooth_l1_loss(
            outputs["orthogonal_mask"],
            outputs["orthogonal_oracle_mask"],
            beta=0.1,
        )
    )
    det_reconstruction, det_parts = reconstruction_objective(
        outputs["deterministic_hsi"],
        gt,
        sam_loss,
        0.5,
        0.2,
        0.05,
        0.02,
    )
    det_objective = (
        cfg.v2_lambda_det_nll * (basis_nll + orthogonal_nll)
        + cfg.v2_lambda_det_x0 * (basis_x0 + orthogonal_x0)
        + cfg.v2_lambda_uncertainty * uncertainty_loss
        + cfg.v2_lambda_mask * mask_loss
        + cfg.v2_lambda_det_reconstruction * det_reconstruction
    )

    zero = gt.new_zeros(())
    diffusion_noise = zero
    diffusion_direct_x0 = zero
    diffusion_hybrid_x0 = zero
    if phase != "specialization":
        basis_mask = outputs["basis_train_mask"]
        orthogonal_mask = outputs["orthogonal_train_mask"]
        target_basis_noise = outputs["coefficient_noise"] * basis_mask
        target_orthogonal_noise = outputs["orthogonal_noise"] * orthogonal_mask
        diffusion_noise = (
            weighted_mean(
                (
                    outputs["predicted_coefficient_noise"]
                    - target_basis_noise
                ).square(),
                basis_mask,
            )
            + weighted_mean(
                (
                    outputs["predicted_orthogonal_noise"]
                    - target_orthogonal_noise
                ).square(),
                orthogonal_mask,
            )
        )
        target_basis_x0 = outputs["remaining_coefficient_normalized"] * basis_mask
        target_orthogonal_x0 = outputs["remaining_orthogonal_normalized"] * orthogonal_mask
        diffusion_direct_x0 = (
            weighted_smooth_l1(
                outputs["direct_coefficient_x0"],
                target_basis_x0,
                basis_mask,
            )
            + weighted_smooth_l1(
                outputs["direct_orthogonal_x0"],
                target_orthogonal_x0,
                orthogonal_mask,
            )
        )
        diffusion_hybrid_x0 = (
            weighted_smooth_l1(
                outputs["predicted_coefficient_clean"],
                target_basis_x0,
                basis_mask,
            )
            + weighted_smooth_l1(
                outputs["predicted_orthogonal_clean"],
                target_orthogonal_x0,
                orthogonal_mask,
            )
        )

    final_reconstruction, final_parts = reconstruction_objective(
        outputs["refined_hsi"],
        gt,
        sam_loss,
        cfg.v2_lambda_final_l1,
        cfg.v2_lambda_final_sam,
        cfg.v2_lambda_final_sgrad1,
        cfg.v2_lambda_final_sgrad2,
    )
    orthogonality = model.project_to_basis_coefficients(
        outputs["deterministic_orthogonal_residual"]
        + outputs["diffusion_orthogonal_residual"],
        outputs["basis"],
    ).abs().mean()
    diffusion_objective = (
        cfg.v2_lambda_noise * diffusion_noise
        + cfg.v2_lambda_direct_x0 * diffusion_direct_x0
        + cfg.v2_lambda_hybrid_x0 * diffusion_hybrid_x0
        + final_reconstruction
        + cfg.v2_lambda_orthogonality * orthogonality
    )
    if phase == "specialization":
        total = det_objective
    elif phase == "diffusion":
        total = diffusion_objective
    else:
        total = (
            cfg.v2_joint_det_weight * det_objective
            + diffusion_objective
        )
    return {
        "total": total,
        "det_objective": det_objective,
        "diffusion_objective": diffusion_objective,
        "basis_nll": basis_nll,
        "orthogonal_nll": orthogonal_nll,
        "basis_x0": basis_x0,
        "orthogonal_x0": orthogonal_x0,
        "uncertainty": uncertainty_loss,
        "mask": mask_loss,
        "det_l1": det_parts["l1"],
        "det_sam": det_parts["sam"],
        "diffusion_noise": diffusion_noise,
        "diffusion_direct_x0": diffusion_direct_x0,
        "diffusion_hybrid_x0": diffusion_hybrid_x0,
        "final_l1": final_parts["l1"],
        "final_sam": final_parts["sam"],
        "orthogonality": orthogonality,
        "basis_train_mask": outputs["basis_train_mask"].mean().detach(),
        "orthogonal_train_mask": outputs["orthogonal_train_mask"].mean().detach(),
        "diff_basis_abs": outputs["diffusion_parallel_residual"].abs().mean().detach(),
        "diff_orthogonal_abs": outputs["diffusion_orthogonal_residual"].abs().mean().detach(),
    }


LOSS_NAMES = [
    "total",
    "det_objective",
    "diffusion_objective",
    "basis_nll",
    "orthogonal_nll",
    "basis_x0",
    "orthogonal_x0",
    "uncertainty",
    "mask",
    "det_l1",
    "det_sam",
    "diffusion_noise",
    "diffusion_direct_x0",
    "diffusion_hybrid_x0",
    "final_l1",
    "final_sam",
    "orthogonality",
    "basis_train_mask",
    "orthogonal_train_mask",
    "diff_basis_abs",
    "diff_orthogonal_abs",
]


def train_one_epoch(
    model,
    loader,
    optimizer,
    sam_loss,
    cfg,
    device,
    phase: str,
    run_diffusion: bool,
    oracle_mix: float,
):
    model.train()
    meters = {name: AverageMeter() for name in LOSS_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_forward(
            batch["lr_hsi"],
            batch["hr_msi"],
            batch["gt"],
            run_diffusion=run_diffusion,
            oracle_mask_mix=oracle_mix,
        )
        losses = compute_losses(model, outputs, batch["gt"], sam_loss, cfg, phase)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite V2 loss in phase {phase}")
        losses["total"].backward()
        if cfg.v2_grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                cfg.v2_grad_clip,
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite V2 gradient in phase {phase}")
        optimizer.step()
        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def export_outputs(model, loader, output_dir: str, cfg, device):
    ensure_dir(output_dir)
    batch = move_to_device(next(iter(loader)), device)
    outputs = model.sample(
        batch["lr_hsi"],
        batch["hr_msi"],
        inference_steps=12,
        initial_noise="zero",
    )
    np.savez_compressed(
        os.path.join(output_dir, "stage3_uncertainty_guided_v2_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        stage2_hsi=outputs["stage2_hsi"].detach().cpu().numpy(),
        deterministic_hsi=outputs["stage3_deterministic_hsi"].detach().cpu().numpy(),
        final_hsi=outputs["refined_hsi"].detach().cpu().numpy(),
        basis_mask=outputs["basis_mask"].detach().cpu().numpy(),
        orthogonal_mask=outputs["orthogonal_mask"].detach().cpu().numpy(),
        diffusion_parallel=outputs[
            "stage3_diffusion_parallel_residual"
        ].detach().cpu().numpy(),
        diffusion_orthogonal=outputs[
            "stage3_diffusion_orthogonal_residual"
        ].detach().cpu().numpy(),
        coefficient_diffusion_scale=model.coefficient_diffusion_scale.detach().cpu().numpy(),
        orthogonal_diffusion_scale=model.orthogonal_diffusion_scale.detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_v2_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_v2_model(cfg, stage2, device)

    deterministic_parameters = list(model.deterministic_parameters())
    diffusion_parameters = list(model.diffusion_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": deterministic_parameters,
                "lr": cfg.v2_specialization_lr,
                "group_name": "deterministic",
            },
            {
                "params": diffusion_parameters,
                "lr": 0.0,
                "group_name": "diffusion",
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage3_uncertainty_guided_diffusion_v2",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage3_uncertainty_guided_diffusion_v2",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage3_uncertainty_guided_diffusion_v2",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "v2_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "v2_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "v2_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "v2_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=True,
            map_location=str(device),
        )
        source_epoch = -1
    else:
        source_epoch = load_deterministic_initialization(
            model,
            cfg.stage3_initial_checkpoint,
            device,
        )
        start_epoch = 0
        scales = estimate_diffusion_scales(
            model,
            train_loader,
            device,
            oracle_weight=cfg.v2_scale_oracle_weight,
            max_batches=cfg.v2_scale_estimation_batches,
        )
        write_log(
            log_path,
            "Initial diffusion scales | coefficient min/median/max="
            f"({scales['coefficient'].min().item():.6e}, "
            f"{scales['coefficient'].median().item():.6e}, "
            f"{scales['coefficient'].max().item():.6e}), "
            f"orthogonal={scales['orthogonal'].item():.6e}.",
        )

    initial = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"V2 start | source_epoch={source_epoch} | "
        f"det={initial['deterministic_psnr']:.4f} dB/"
        f"{initial['deterministic_sam']:.4f} deg | "
        f"final={initial['final_psnr']:.4f} dB/"
        f"{initial['final_sam']:.4f} deg | DDIM=12.",
    )

    metric_fields = [
        "stage2_psnr",
        "stage2_sam",
        "deterministic_psnr",
        "deterministic_sam",
        "final_psnr",
        "final_sam",
        "psnr_gain_over_stage2",
        "sam_gain_over_stage2",
        "diffusion_psnr_gain_over_deterministic",
        "diffusion_sam_gain_over_deterministic",
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        [
            "epoch",
            "phase",
            "oracle_mix",
            "det_lr",
            "diff_lr",
            *metric_fields,
            *[f"train_{name}" for name in LOSS_NAMES],
        ],
    )
    best_selection = initial["selection"]
    best_psnr = initial["final_psnr"]
    best_sam = initial["final_sam"]
    initial_extra = {
        "stage": "stage3_uncertainty_guided_diffusion_v2",
        "source_checkpoint": cfg.stage3_initial_checkpoint,
        "source_epoch": source_epoch,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, initial_extra)

    sam_loss = SAMLoss()
    scale_reestimated = start_epoch > cfg.v2_specialization_epochs
    for epoch in range(start_epoch, cfg.epochs):
        if (
            epoch == cfg.v2_specialization_epochs
            and not scale_reestimated
        ):
            scales = estimate_diffusion_scales(
                model,
                train_loader,
                device,
                oracle_weight=cfg.v2_scale_oracle_weight,
                max_batches=cfg.v2_scale_estimation_batches,
            )
            scale_reestimated = True
            write_log(
                log_path,
                "Post-specialization diffusion scales | coefficient min/median/max="
                f"({scales['coefficient'].min().item():.6e}, "
                f"{scales['coefficient'].median().item():.6e}, "
                f"{scales['coefficient'].max().item():.6e}), "
                f"orthogonal={scales['orthogonal'].item():.6e}.",
            )

        phase, run_diffusion, det_lr, diff_lr = configure_phase(
            model,
            optimizer,
            epoch,
            cfg,
        )
        oracle_mix = oracle_mix_for_epoch(epoch, cfg, phase)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            phase,
            run_diffusion,
            oracle_mix,
        )

        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={"phase": phase, "oracle_mix": oracle_mix},
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | phase={phase} | "
            f"det={val['deterministic_psnr']:.4f}/"
            f"{val['deterministic_sam']:.4f} | "
            f"final={val['final_psnr']:.4f}/"
            f"{val['final_sam']:.4f} | "
            f"diff=({val['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['diffusion_sam_gain_over_deterministic']:+.4f} deg) | "
            f"oracle_mix={oracle_mix:.3f} | "
            f"corr=({val['basis_uncertainty_error_correlation']:.3f}, "
            f"{val['orthogonal_uncertainty_error_correlation']:.3f}).",
        )
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "oracle_mix": oracle_mix,
            "det_lr": det_lr,
            "diff_lr": diff_lr,
            **{name: val[name] for name in metric_fields},
        }
        row.update({f"train_{name}": train_result[name] for name in LOSS_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "stage3_uncertainty_guided_diffusion_v2",
            "phase": phase,
            "oracle_mix": oracle_mix,
            "source_checkpoint": cfg.stage3_initial_checkpoint,
            "validation": val,
            "train": train_result,
        }
        if val["selection"] < best_selection:
            best_selection = val["selection"]
            save_checkpoint(model, optimizer, epoch + 1, best_selection, best_path, extra)
        if val["final_psnr"] > best_psnr:
            best_psnr = val["final_psnr"]
            save_checkpoint(model, optimizer, epoch + 1, best_psnr, best_psnr_path, extra)
        if val["final_sam"] < best_sam:
            best_sam = val["final_sam"]
            save_checkpoint(model, optimizer, epoch + 1, best_sam, best_sam_path, extra)
        save_checkpoint(model, optimizer, epoch + 1, best_selection, last_path, extra)

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    final = evaluate(model, test_loader, cfg, device)
    export_outputs(model, test_loader, output_dir, cfg, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"V2 complete | PSNR={final['final_psnr']:.4f}, "
        f"SAM={final['final_sam']:.4f} deg, "
        f"diffusion gain=({final['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
