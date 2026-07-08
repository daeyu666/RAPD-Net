"""Train the corrected Stage-3 uncertainty-guided local diffusion refiner.

Training is deliberately split into three phases:

1. deterministic warm-up: learn basis/orthogonal residual means and calibrated
   heteroscedastic uncertainty; diffusion is disabled;
2. local diffusion: freeze deterministic heads and learn only the remaining
   error inside detached high-uncertainty masks;
3. joint fine-tuning: update both parts with smaller learning rates while the
   frozen Stage-2 backbone remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage3_uncertainty_guided_diffusion import (
    UncertaintyGuidedDualDomainDiffusionRefiner,
)
from train_stage2_coefficients import (
    first_spectral_difference,
    second_spectral_difference,
)
from train_stage3_dual_domain_diffusion import (
    build_stage2_model,
    estimate_residual_scales,
)
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def parse_uncertainty_guided_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        default=(
            "./checkpoints/stage2_multiscale_residual_pyramid/PaviaU/"
            "residual_pyramid_best_psnr.pth"
        ),
    )

    # Frozen Stage-2 reconstruction settings.
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--anchor_normalized_clip", type=float, default=0.0)
    parser.add_argument("--projector_tolerance", type=float, default=1e-6)
    parser.add_argument("--stage2_feature_channels", type=int, default=64)
    parser.add_argument("--stage2_encoder_blocks", type=int, default=3)
    parser.add_argument("--stage2_fusion_channels", type=int, default=96)
    parser.add_argument("--stage2_fusion_blocks", type=int, default=4)
    parser.add_argument("--stage2_max_normalized_residual", type=float, default=6.0)
    parser.add_argument("--stage2_coefficient_scale_floor", type=float, default=1e-4)
    parser.add_argument("--stage2_num_frequency_bands", type=int, default=20)
    parser.add_argument("--stage2_init_low_boundary", type=float, default=5.0)
    parser.add_argument("--stage2_init_high_boundary", type=float, default=18.0)
    parser.add_argument("--stage2_boundary_temperature", type=float, default=0.5)
    parser.add_argument("--stage2_soft_frequency_partition", action="store_true")
    parser.add_argument(
        "--stage2_edge_threshold_mode",
        type=str,
        default="relative",
        choices=["fixed", "relative", "quantile"],
    )
    parser.add_argument("--stage2_edge_mask_threshold", type=float, default=0.1)
    parser.add_argument("--stage2_edge_reference_quantile", type=float, default=0.9)
    parser.add_argument("--stage2_noise_quantile", type=float, default=0.2)
    parser.add_argument("--pyramid_quarter_scale", type=float, default=0.25)
    parser.add_argument("--pyramid_half_scale", type=float, default=0.5)

    # Corrected Stage-3 architecture.
    parser.add_argument("--stage3_det_hidden_channels", type=int, default=96)
    parser.add_argument("--stage3_det_blocks", type=int, default=5)
    parser.add_argument("--stage3_diff_hidden_channels", type=int, default=96)
    parser.add_argument("--stage3_diff_blocks", type=int, default=6)
    parser.add_argument("--stage3_time_channels", type=int, default=192)
    parser.add_argument("--stage3_diffusion_timesteps", type=int, default=100)
    parser.add_argument("--stage3_beta_start", type=float, default=1e-4)
    parser.add_argument("--stage3_beta_end", type=float, default=2e-2)
    parser.add_argument("--stage3_max_normalized_residual", type=float, default=6.0)
    parser.add_argument("--stage3_clean_clip", type=float, default=8.0)
    parser.add_argument("--stage3_log_variance_min", type=float, default=-6.0)
    parser.add_argument("--stage3_log_variance_max", type=float, default=3.0)
    parser.add_argument("--stage3_basis_mask_threshold", type=float, default=0.5)
    parser.add_argument("--stage3_orthogonal_mask_threshold", type=float, default=0.5)
    parser.add_argument("--stage3_mask_temperature", type=float, default=0.25)
    parser.add_argument("--stage3_mask_spread_floor", type=float, default=1e-4)
    parser.add_argument("--stage3_msi_residual_gain", type=float, default=10.0)
    parser.add_argument("--stage3_residual_scale_floor", type=float, default=1e-5)
    parser.add_argument("--stage3_inference_steps", type=int, default=12)
    parser.add_argument(
        "--stage3_initial_noise",
        type=str,
        default="zero",
        choices=["zero", "random"],
    )

    # Three-phase optimization.
    parser.add_argument("--stage3_det_warmup_epochs", type=int, default=20)
    parser.add_argument("--stage3_joint_start_epoch", type=int, default=70)
    parser.add_argument("--stage3_det_lr", type=float, default=5e-5)
    parser.add_argument("--stage3_diff_lr", type=float, default=1e-4)
    parser.add_argument("--stage3_joint_det_lr", type=float, default=1e-5)
    parser.add_argument("--stage3_joint_diff_lr", type=float, default=2e-5)
    parser.add_argument("--stage3_grad_clip", type=float, default=1.0)
    parser.add_argument("--stage3_scale_estimation_batches", type=int, default=0)

    # Deterministic residual and uncertainty losses.
    parser.add_argument("--lambda_det_basis_nll", type=float, default=1.0)
    parser.add_argument("--lambda_det_orthogonal_nll", type=float, default=1.0)
    parser.add_argument("--lambda_det_basis_x0", type=float, default=0.2)
    parser.add_argument("--lambda_det_orthogonal_x0", type=float, default=0.2)
    parser.add_argument("--lambda_uncertainty_calibration", type=float, default=0.1)
    parser.add_argument("--lambda_mask_calibration", type=float, default=0.1)
    parser.add_argument("--lambda_det_hsi_l1", type=float, default=0.5)
    parser.add_argument("--lambda_det_sam", type=float, default=0.2)
    parser.add_argument("--lambda_det_sgrad1", type=float, default=0.05)
    parser.add_argument("--lambda_det_sgrad2", type=float, default=0.02)

    # Local diffusion and final reconstruction losses.
    parser.add_argument("--lambda_diff_basis_noise", type=float, default=1.0)
    parser.add_argument("--lambda_diff_orthogonal_noise", type=float, default=1.0)
    parser.add_argument("--lambda_diff_basis_x0", type=float, default=0.2)
    parser.add_argument("--lambda_diff_orthogonal_x0", type=float, default=0.2)
    parser.add_argument("--lambda_final_hsi_l1", type=float, default=0.5)
    parser.add_argument("--lambda_final_sam", type=float, default=0.2)
    parser.add_argument("--lambda_final_sgrad1", type=float, default=0.05)
    parser.add_argument("--lambda_final_sgrad2", type=float, default=0.02)
    parser.add_argument("--lambda_orthogonality", type=float, default=0.05)
    parser.add_argument("--stage3_joint_det_objective_weight", type=float, default=0.5)
    parser.add_argument("--stage3_selection_sam_weight", type=float, default=0.5)

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.stage3_det_warmup_epochs < 0:
        raise ValueError("stage3_det_warmup_epochs must be non-negative")
    if cfg.stage3_joint_start_epoch < cfg.stage3_det_warmup_epochs:
        raise ValueError("joint_start_epoch must not precede deterministic warm-up")
    if min(
        cfg.stage3_det_lr,
        cfg.stage3_diff_lr,
        cfg.stage3_joint_det_lr,
        cfg.stage3_joint_diff_lr,
    ) < 0:
        raise ValueError("Stage-3 learning rates must be non-negative")
    if cfg.stage3_inference_steps <= 0:
        raise ValueError("stage3_inference_steps must be positive")
    if cfg.stage3_scale_estimation_batches < 0:
        raise ValueError("stage3_scale_estimation_batches must be non-negative")

    if cfg.dataset != "PaviaU":
        if cfg.stage1_basis_checkpoint.endswith(
            "stage1_basis/PaviaU/basis_for_stage2.pth"
        ):
            cfg.stage1_basis_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "stage1_basis",
                cfg.dataset,
                "basis_for_stage2.pth",
            )
        if cfg.stage2_checkpoint.endswith(
            "stage2_multiscale_residual_pyramid/PaviaU/"
            "residual_pyramid_best_psnr.pth"
        ):
            cfg.stage2_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "stage2_multiscale_residual_pyramid",
                cfg.dataset,
                "residual_pyramid_best_psnr.pth",
            )
    return cfg


def build_model(cfg, stage2, device: torch.device):
    return UncertaintyGuidedDualDomainDiffusionRefiner(
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
    ).to(device)


def heteroscedastic_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target).square()
    return (torch.exp(-log_variance) * squared_error + log_variance).mean()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.25,
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(
        prediction,
        target,
        beta=beta,
        reduction="none",
    )
    return masked_mean(elementwise, mask)


def reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sam_loss: SAMLoss,
) -> Dict[str, torch.Tensor]:
    return {
        "hsi_l1": F.l1_loss(prediction, target),
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


def compute_losses(
    model: UncertaintyGuidedDualDomainDiffusionRefiner,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    sam_loss: SAMLoss,
    cfg,
    run_diffusion: bool,
    phase: str,
) -> Dict[str, torch.Tensor]:
    target_coefficient = outputs["target_coefficient_normalized"]
    target_orthogonal = outputs["target_orthogonal_normalized"]
    predicted_coefficient = outputs["deterministic_coefficient_normalized"]
    predicted_orthogonal = outputs["deterministic_orthogonal_normalized"]
    coefficient_log_variance = outputs[
        "deterministic_coefficient_log_variance"
    ]
    orthogonal_log_variance = outputs[
        "deterministic_orthogonal_log_variance"
    ]

    losses: Dict[str, torch.Tensor] = {}
    losses["det_basis_nll"] = heteroscedastic_nll(
        predicted_coefficient,
        target_coefficient,
        coefficient_log_variance,
    )
    losses["det_orthogonal_nll"] = heteroscedastic_nll(
        predicted_orthogonal,
        target_orthogonal,
        orthogonal_log_variance,
    )
    losses["det_basis_x0"] = F.smooth_l1_loss(
        predicted_coefficient,
        target_coefficient,
        beta=0.25,
    )
    losses["det_orthogonal_x0"] = F.smooth_l1_loss(
        predicted_orthogonal,
        target_orthogonal,
        beta=0.25,
    )

    basis_error = (predicted_coefficient - target_coefficient).detach().abs()
    orthogonal_error = (predicted_orthogonal - target_orthogonal).detach().abs()
    basis_standard_deviation = torch.exp(0.5 * coefficient_log_variance)
    orthogonal_standard_deviation = torch.exp(0.5 * orthogonal_log_variance)
    losses["basis_uncertainty_calibration"] = F.smooth_l1_loss(
        basis_standard_deviation,
        basis_error,
        beta=0.25,
    )
    losses["orthogonal_uncertainty_calibration"] = F.smooth_l1_loss(
        orthogonal_standard_deviation,
        orthogonal_error,
        beta=0.25,
    )
    losses["basis_mask_calibration"] = F.smooth_l1_loss(
        outputs["basis_mask"],
        outputs["basis_oracle_mask"],
        beta=0.1,
    )
    losses["orthogonal_mask_calibration"] = F.smooth_l1_loss(
        outputs["orthogonal_mask"],
        outputs["orthogonal_oracle_mask"],
        beta=0.1,
    )

    deterministic_reconstruction = reconstruction_losses(
        outputs["deterministic_hsi"],
        gt,
        sam_loss,
    )
    losses.update(
        {f"det_{name}": value for name, value in deterministic_reconstruction.items()}
    )

    zero = outputs["stage2_hsi"].new_zeros(())
    losses["diff_basis_noise"] = zero
    losses["diff_orthogonal_noise"] = zero
    losses["diff_basis_x0"] = zero
    losses["diff_orthogonal_x0"] = zero
    if run_diffusion:
        basis_mask = outputs["basis_mask_for_diffusion"]
        orthogonal_mask = outputs["orthogonal_mask_for_diffusion"]
        target_basis_noise = outputs["coefficient_noise"] * basis_mask
        target_orthogonal_noise = outputs["orthogonal_noise"] * orthogonal_mask
        losses["diff_basis_noise"] = masked_mean(
            (
                outputs["predicted_coefficient_noise"]
                - target_basis_noise
            ).square(),
            basis_mask,
        )
        losses["diff_orthogonal_noise"] = masked_mean(
            (
                outputs["predicted_orthogonal_noise"]
                - target_orthogonal_noise
            ).square(),
            orthogonal_mask,
        )
        target_basis_clean = (
            outputs["remaining_coefficient_normalized"] * basis_mask
        )
        target_orthogonal_clean = (
            outputs["remaining_orthogonal_normalized"] * orthogonal_mask
        )
        losses["diff_basis_x0"] = masked_smooth_l1(
            outputs["predicted_coefficient_clean"],
            target_basis_clean,
            basis_mask,
        )
        losses["diff_orthogonal_x0"] = masked_smooth_l1(
            outputs["predicted_orthogonal_clean"],
            target_orthogonal_clean,
            orthogonal_mask,
        )

    final_reconstruction = reconstruction_losses(
        outputs["refined_hsi"],
        gt,
        sam_loss,
    )
    losses.update(
        {f"final_{name}": value for name, value in final_reconstruction.items()}
    )
    orthogonal_leakage = model.project_to_basis_coefficients(
        outputs["deterministic_orthogonal_residual"]
        + outputs["diffusion_orthogonal_residual"],
        outputs["basis"],
    )
    losses["orthogonality"] = orthogonal_leakage.abs().mean()

    losses["det_objective"] = (
        cfg.lambda_det_basis_nll * losses["det_basis_nll"]
        + cfg.lambda_det_orthogonal_nll * losses["det_orthogonal_nll"]
        + cfg.lambda_det_basis_x0 * losses["det_basis_x0"]
        + cfg.lambda_det_orthogonal_x0 * losses["det_orthogonal_x0"]
        + cfg.lambda_uncertainty_calibration
        * (
            losses["basis_uncertainty_calibration"]
            + losses["orthogonal_uncertainty_calibration"]
        )
        + cfg.lambda_mask_calibration
        * (
            losses["basis_mask_calibration"]
            + losses["orthogonal_mask_calibration"]
        )
        + cfg.lambda_det_hsi_l1 * losses["det_hsi_l1"]
        + cfg.lambda_det_sam * losses["det_sam"]
        + cfg.lambda_det_sgrad1 * losses["det_sgrad1"]
        + cfg.lambda_det_sgrad2 * losses["det_sgrad2"]
    )
    losses["diff_objective"] = (
        cfg.lambda_diff_basis_noise * losses["diff_basis_noise"]
        + cfg.lambda_diff_orthogonal_noise * losses["diff_orthogonal_noise"]
        + cfg.lambda_diff_basis_x0 * losses["diff_basis_x0"]
        + cfg.lambda_diff_orthogonal_x0 * losses["diff_orthogonal_x0"]
    )
    losses["final_objective"] = (
        cfg.lambda_final_hsi_l1 * losses["final_hsi_l1"]
        + cfg.lambda_final_sam * losses["final_sam"]
        + cfg.lambda_final_sgrad1 * losses["final_sgrad1"]
        + cfg.lambda_final_sgrad2 * losses["final_sgrad2"]
        + cfg.lambda_orthogonality * losses["orthogonality"]
    )

    if phase == "deterministic":
        losses["total"] = losses["det_objective"]
    elif phase == "diffusion":
        losses["total"] = losses["diff_objective"] + losses["final_objective"]
    elif phase == "joint":
        losses["total"] = (
            cfg.stage3_joint_det_objective_weight * losses["det_objective"]
            + losses["diff_objective"]
            + losses["final_objective"]
        )
    else:
        raise ValueError(f"Unknown Stage-3 phase: {phase}")

    losses["basis_mask_mean"] = outputs["basis_mask"].mean().detach()
    losses["orthogonal_mask_mean"] = outputs["orthogonal_mask"].mean().detach()
    losses["basis_high_fraction"] = (
        outputs["basis_mask"] > 0.5
    ).float().mean().detach()
    losses["orthogonal_high_fraction"] = (
        outputs["orthogonal_mask"] > 0.5
    ).float().mean().detach()
    losses["det_basis_abs"] = outputs[
        "deterministic_parallel_residual"
    ].abs().mean().detach()
    losses["det_orthogonal_abs"] = outputs[
        "deterministic_orthogonal_residual"
    ].abs().mean().detach()
    losses["diff_basis_abs"] = outputs[
        "diffusion_parallel_residual"
    ].abs().mean().detach()
    losses["diff_orthogonal_abs"] = outputs[
        "diffusion_orthogonal_residual"
    ].abs().mean().detach()
    return losses


LOSS_NAMES = [
    "total",
    "det_objective",
    "diff_objective",
    "final_objective",
    "det_basis_nll",
    "det_orthogonal_nll",
    "det_basis_x0",
    "det_orthogonal_x0",
    "basis_uncertainty_calibration",
    "orthogonal_uncertainty_calibration",
    "basis_mask_calibration",
    "orthogonal_mask_calibration",
    "det_hsi_l1",
    "det_sam",
    "det_sgrad1",
    "det_sgrad2",
    "diff_basis_noise",
    "diff_orthogonal_noise",
    "diff_basis_x0",
    "diff_orthogonal_x0",
    "final_hsi_l1",
    "final_sam",
    "final_sgrad1",
    "final_sgrad2",
    "orthogonality",
    "basis_mask_mean",
    "orthogonal_mask_mean",
    "basis_high_fraction",
    "orthogonal_high_fraction",
    "det_basis_abs",
    "det_orthogonal_abs",
    "diff_basis_abs",
    "diff_orthogonal_abs",
]


def set_requires_grad(parameters: Sequence[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def configure_phase(
    model: UncertaintyGuidedDualDomainDiffusionRefiner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg,
) -> Tuple[str, bool, float, float]:
    deterministic_parameters = list(model.deterministic_parameters())
    diffusion_parameters = list(model.diffusion_parameters())

    if epoch < cfg.stage3_det_warmup_epochs:
        phase = "deterministic"
        det_enabled = True
        diff_enabled = False
        det_lr = cfg.stage3_det_lr
        diff_lr = 0.0
    elif epoch < cfg.stage3_joint_start_epoch:
        phase = "diffusion"
        det_enabled = False
        diff_enabled = True
        det_lr = 0.0
        diff_lr = cfg.stage3_diff_lr
    else:
        phase = "joint"
        det_enabled = True
        diff_enabled = True
        det_lr = cfg.stage3_joint_det_lr
        diff_lr = cfg.stage3_joint_diff_lr

    set_requires_grad(deterministic_parameters, det_enabled)
    set_requires_grad(diffusion_parameters, diff_enabled)
    for group in optimizer.param_groups:
        if group.get("group_name") == "deterministic":
            group["lr"] = det_lr
        elif group.get("group_name") == "diffusion":
            group["lr"] = diff_lr
    return phase, diff_enabled, det_lr, diff_lr


def train_one_epoch(
    model: UncertaintyGuidedDualDomainDiffusionRefiner,
    loader,
    optimizer: torch.optim.Optimizer,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
    phase: str,
    run_diffusion: bool,
) -> Dict[str, float]:
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
        )
        losses = compute_losses(
            model,
            outputs,
            batch["gt"],
            sam_loss,
            cfg,
            run_diffusion,
            phase,
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(
                f"Non-finite Stage-3 total loss in phase {phase}: "
                f"{float(losses['total'].detach().item())}"
            )
        losses["total"].backward()
        if cfg.stage3_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                cfg.stage3_grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite Stage-3 gradient norm in phase {phase}"
                )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
    return {name: meter.avg for name, meter in meters.items()}


def spatial_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().flatten(1)
    y = y.detach().float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).mean(dim=1)
    denominator = torch.sqrt(
        x.square().mean(dim=1) * y.square().mean(dim=1) + 1e-12
    )
    correlation = numerator / denominator
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    return float(correlation.mean().item())


@torch.no_grad()
def evaluate(
    model: UncertaintyGuidedDualDomainDiffusionRefiner,
    loader,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metric_sets = {
        name: MetricAverager()
        for name in (
            "stage2",
            "det_basis",
            "det_orthogonal",
            "deterministic",
            "final_basis",
            "final_orthogonal",
            "final",
        )
    }
    diagnostic_names = (
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_high_fraction",
        "orthogonal_high_fraction",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
        "basis_mask_oracle_mae",
        "orthogonal_mask_oracle_mae",
        "det_basis_abs",
        "det_orthogonal_abs",
        "diff_basis_abs",
        "diff_orthogonal_abs",
        "orthogonality_leakage",
        "out_of_range_ratio",
    )
    diagnostics = {name: AverageMeter() for name in diagnostic_names}

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model.sample(
            batch["lr_hsi"],
            batch["hr_msi"],
            inference_steps=cfg.stage3_inference_steps,
            initial_noise=cfg.stage3_initial_noise,
        )
        stage2_hsi = outputs["stage2_hsi"]
        det_basis_hsi = stage2_hsi + outputs["deterministic_parallel_residual"]
        det_orthogonal_hsi = (
            stage2_hsi + outputs["deterministic_orthogonal_residual"]
        )
        deterministic_hsi = outputs["stage3_deterministic_hsi"]
        final_basis_hsi = (
            stage2_hsi
            + outputs["deterministic_parallel_residual"]
            + outputs["stage3_diffusion_parallel_residual"]
        )
        final_orthogonal_hsi = (
            stage2_hsi
            + outputs["deterministic_orthogonal_residual"]
            + outputs["stage3_diffusion_orthogonal_residual"]
        )
        predictions = {
            "stage2": stage2_hsi,
            "det_basis": det_basis_hsi,
            "det_orthogonal": det_orthogonal_hsi,
            "deterministic": deterministic_hsi,
            "final_basis": final_basis_hsi,
            "final_orthogonal": final_orthogonal_hsi,
            "final": outputs["refined_hsi"],
        }
        for name, prediction in predictions.items():
            metric_sets[name].update(
                calc_metrics(prediction, batch["gt"], cfg.scale_ratio)
            )

        target_residual = batch["gt"] - stage2_hsi
        target_coefficient, _, target_orthogonal = model.decompose_residual(
            target_residual,
            outputs["basis"],
        )
        coefficient_scale = model.coefficient_residual_scale.view(1, -1, 1, 1)
        orthogonal_scale = model.orthogonal_residual_scale.view(1, 1, 1, 1)
        coefficient_error = (
            target_coefficient / coefficient_scale
            - outputs["deterministic_coefficient_normalized"]
        )
        orthogonal_error = model.project_orthogonal(
            target_orthogonal / orthogonal_scale
            - outputs["deterministic_orthogonal_normalized"],
            outputs["basis"],
        )
        basis_error_map = coefficient_error.abs().mean(dim=1, keepdim=True)
        orthogonal_error_map = orthogonal_error.abs().mean(dim=1, keepdim=True)
        basis_oracle_mask = model.error_to_oracle_mask(
            coefficient_error,
            model.basis_mask_threshold,
        )
        orthogonal_oracle_mask = model.error_to_oracle_mask(
            orthogonal_error,
            model.orthogonal_mask_threshold,
        )
        orthogonal_leakage = model.project_to_basis_coefficients(
            outputs["deterministic_orthogonal_residual"]
            + outputs["stage3_diffusion_orthogonal_residual"],
            outputs["basis"],
        ).abs().mean()
        out_of_range = (
            (outputs["refined_hsi"] < 0.0)
            | (outputs["refined_hsi"] > 1.0)
        ).float().mean()
        values = {
            "basis_mask_mean": float(outputs["basis_mask"].mean().item()),
            "orthogonal_mask_mean": float(outputs["orthogonal_mask"].mean().item()),
            "basis_high_fraction": float(
                (outputs["basis_mask"] > 0.5).float().mean().item()
            ),
            "orthogonal_high_fraction": float(
                (outputs["orthogonal_mask"] > 0.5).float().mean().item()
            ),
            "basis_uncertainty_error_correlation": spatial_correlation(
                outputs["basis_uncertainty"],
                basis_error_map,
            ),
            "orthogonal_uncertainty_error_correlation": spatial_correlation(
                outputs["orthogonal_uncertainty"],
                orthogonal_error_map,
            ),
            "basis_mask_oracle_mae": float(
                (outputs["basis_mask"] - basis_oracle_mask).abs().mean().item()
            ),
            "orthogonal_mask_oracle_mae": float(
                (
                    outputs["orthogonal_mask"] - orthogonal_oracle_mask
                ).abs().mean().item()
            ),
            "det_basis_abs": float(
                outputs["deterministic_parallel_residual"].abs().mean().item()
            ),
            "det_orthogonal_abs": float(
                outputs["deterministic_orthogonal_residual"].abs().mean().item()
            ),
            "diff_basis_abs": float(
                outputs["stage3_diffusion_parallel_residual"].abs().mean().item()
            ),
            "diff_orthogonal_abs": float(
                outputs["stage3_diffusion_orthogonal_residual"].abs().mean().item()
            ),
            "orthogonality_leakage": float(orthogonal_leakage.item()),
            "out_of_range_ratio": float(out_of_range.item()),
        }
        batch_size = batch["lr_hsi"].size(0)
        for name, value in values.items():
            diagnostics[name].update(value, batch_size)

    averages = {name: meter.average() for name, meter in metric_sets.items()}
    result: Dict[str, float] = {}
    for name, metrics in averages.items():
        result[f"{name}_psnr"] = metrics["PSNR"]
        result[f"{name}_sam"] = metrics["SAM"]
    final_metrics = averages["final"]
    result.update(
        {
            "final_rmse": final_metrics["RMSE"],
            "final_ergas": final_metrics["ERGAS"],
            "final_ssim": final_metrics["SSIM"],
            "final_cc": final_metrics["CC"],
            "psnr_gain_over_stage2": (
                result["final_psnr"] - result["stage2_psnr"]
            ),
            "sam_gain_over_stage2": (
                result["stage2_sam"] - result["final_sam"]
            ),
            "diffusion_psnr_gain_over_deterministic": (
                result["final_psnr"] - result["deterministic_psnr"]
            ),
            "diffusion_sam_gain_over_deterministic": (
                result["deterministic_sam"] - result["final_sam"]
            ),
            **{name: meter.avg for name, meter in diagnostics.items()},
        }
    )
    result["selection"] = (
        -result["final_psnr"]
        + cfg.stage3_selection_sam_weight * result["final_sam"]
    )
    return result


@torch.no_grad()
def export_outputs(
    model: UncertaintyGuidedDualDomainDiffusionRefiner,
    loader,
    output_dir: str,
    cfg,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    batch = move_to_device(next(iter(loader)), device)
    outputs = model.sample(
        batch["lr_hsi"],
        batch["hr_msi"],
        inference_steps=cfg.stage3_inference_steps,
        initial_noise=cfg.stage3_initial_noise,
    )
    np.savez_compressed(
        os.path.join(output_dir, "stage3_uncertainty_guided_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        stage2_hsi=outputs["stage2_hsi"].detach().cpu().numpy(),
        deterministic_hsi=outputs["stage3_deterministic_hsi"].detach().cpu().numpy(),
        final_hsi=outputs["refined_hsi"].detach().cpu().numpy(),
        deterministic_coefficient_residual=outputs[
            "deterministic_coefficient_residual"
        ].detach().cpu().numpy(),
        deterministic_parallel_residual=outputs[
            "deterministic_parallel_residual"
        ].detach().cpu().numpy(),
        deterministic_orthogonal_residual=outputs[
            "deterministic_orthogonal_residual"
        ].detach().cpu().numpy(),
        diffusion_coefficient_residual=outputs[
            "stage3_diffusion_coefficient_residual"
        ].detach().cpu().numpy(),
        diffusion_parallel_residual=outputs[
            "stage3_diffusion_parallel_residual"
        ].detach().cpu().numpy(),
        diffusion_orthogonal_residual=outputs[
            "stage3_diffusion_orthogonal_residual"
        ].detach().cpu().numpy(),
        basis_uncertainty=outputs["basis_uncertainty"].detach().cpu().numpy(),
        orthogonal_uncertainty=outputs[
            "orthogonal_uncertainty"
        ].detach().cpu().numpy(),
        basis_mask=outputs["basis_mask"].detach().cpu().numpy(),
        orthogonal_mask=outputs["orthogonal_mask"].detach().cpu().numpy(),
        reliability_map=outputs["reliability_map"].detach().cpu().numpy(),
        coefficient_residual_scale=model.coefficient_residual_scale.detach().cpu().numpy(),
        orthogonal_residual_scale=model.orthogonal_residual_scale.detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_uncertainty_guided_args()
    cfg.stage = "stage3_uncertainty_guided_diffusion"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_model(cfg, stage2, device)
    deterministic_parameters = list(model.deterministic_parameters())
    diffusion_parameters = list(model.diffusion_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": deterministic_parameters,
                "lr": cfg.stage3_det_lr,
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
        "stage3_uncertainty_guided_diffusion",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage3_uncertainty_guided_diffusion",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage3_uncertainty_guided_diffusion")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "uncertainty_guided_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "uncertainty_guided_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "uncertainty_guided_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "uncertainty_guided_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=True,
            map_location=str(device),
        )
    else:
        scales = estimate_residual_scales(
            model,
            train_loader,
            device,
            max_batches=cfg.stage3_scale_estimation_batches,
        )
        coefficient_scale = scales["coefficient_scale"]
        write_log(
            log_path,
            f"Estimated residual scales | coefficient min/median/max="
            f"({coefficient_scale.min().item():.6e}, "
            f"{coefficient_scale.median().item():.6e}, "
            f"{coefficient_scale.max().item():.6e}), "
            f"orthogonal={scales['orthogonal_scale'].item():.6e}.",
        )

    sam_loss = SAMLoss()
    initial = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"Corrected Stage 3 start | Stage2={initial['stage2_psnr']:.4f} dB/"
        f"{initial['stage2_sam']:.4f} deg | "
        f"det={initial['deterministic_psnr']:.4f} dB/"
        f"{initial['deterministic_sam']:.4f} deg | "
        f"final={initial['final_psnr']:.4f} dB/"
        f"{initial['final_sam']:.4f} deg | "
        f"masks=({initial['basis_mask_mean']:.3f}, "
        f"{initial['orthogonal_mask_mean']:.3f}) | "
        f"trainable={count_parameters(model):.3f} M.",
    )

    metric_fields = [
        "stage2_psnr",
        "stage2_sam",
        "det_basis_psnr",
        "det_basis_sam",
        "det_orthogonal_psnr",
        "det_orthogonal_sam",
        "deterministic_psnr",
        "deterministic_sam",
        "final_basis_psnr",
        "final_basis_sam",
        "final_orthogonal_psnr",
        "final_orthogonal_sam",
        "final_psnr",
        "final_sam",
        "final_rmse",
        "final_ergas",
        "final_ssim",
        "final_cc",
        "psnr_gain_over_stage2",
        "sam_gain_over_stage2",
        "diffusion_psnr_gain_over_deterministic",
        "diffusion_sam_gain_over_deterministic",
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_high_fraction",
        "orthogonal_high_fraction",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
        "basis_mask_oracle_mae",
        "orthogonal_mask_oracle_mae",
        "det_basis_abs",
        "det_orthogonal_abs",
        "diff_basis_abs",
        "diff_orthogonal_abs",
        "orthogonality_leakage",
        "out_of_range_ratio",
    ]
    csv_fields = [
        "epoch",
        "phase",
        "det_lr",
        "diff_lr",
        *metric_fields,
        *[f"train_{name}" for name in LOSS_NAMES],
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    best_selection = initial["selection"]
    best_psnr = initial["final_psnr"]
    best_sam = initial["final_sam"]
    initial_extra = {
        "stage": "stage3_uncertainty_guided_diffusion",
        "dataset": cfg.dataset,
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "stage2_checkpoint": cfg.stage2_checkpoint,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_selection,
        best_path,
        extra=initial_extra,
    )
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_psnr,
        best_psnr_path,
        extra=initial_extra,
    )
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_sam,
        best_sam_path,
        extra=initial_extra,
    )

    for epoch in range(start_epoch, cfg.epochs):
        phase, run_diffusion, det_lr, diff_lr = configure_phase(
            model,
            optimizer,
            epoch,
            cfg,
        )
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            phase,
            run_diffusion,
        )

        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={
                    "stage": "stage3_uncertainty_guided_diffusion",
                    "dataset": cfg.dataset,
                    "phase": phase,
                    "train": train_result,
                },
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | phase={phase} | "
            f"det={val['deterministic_psnr']:.4f} dB/"
            f"{val['deterministic_sam']:.4f} deg | "
            f"final={val['final_psnr']:.4f} dB/"
            f"{val['final_sam']:.4f} deg | "
            f"stage2 gain=({val['psnr_gain_over_stage2']:+.4f} dB, "
            f"{val['sam_gain_over_stage2']:+.4f} deg) | "
            f"diff gain=({val['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['diffusion_sam_gain_over_deterministic']:+.4f} deg) | "
            f"mask=({val['basis_mask_mean']:.3f}, "
            f"{val['orthogonal_mask_mean']:.3f}) | "
            f"corr=({val['basis_uncertainty_error_correlation']:.3f}, "
            f"{val['orthogonal_uncertainty_error_correlation']:.3f}).",
        )

        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "det_lr": det_lr,
            "diff_lr": diff_lr,
            **{name: val[name] for name in metric_fields},
        }
        row.update({f"train_{name}": train_result[name] for name in LOSS_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "stage3_uncertainty_guided_diffusion",
            "dataset": cfg.dataset,
            "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
            "stage2_checkpoint": cfg.stage2_checkpoint,
            "stage2_epoch": stage2_epoch,
            "phase": phase,
            "validation": val,
            "train": train_result,
        }
        if val["selection"] < best_selection:
            best_selection = val["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                best_path,
                extra=extra,
            )
        if val["final_psnr"] > best_psnr:
            best_psnr = val["final_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )
        if val["final_sam"] < best_sam:
            best_sam = val["final_sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_sam,
                best_sam_path,
                extra=extra,
            )
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            best_selection,
            last_path,
            extra=extra,
        )

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
        f"Corrected Stage 3 complete | PSNR={final['final_psnr']:.4f}, "
        f"SAM={final['final_sam']:.4f} deg, "
        f"stage2 gain=({final['psnr_gain_over_stage2']:+.4f} dB, "
        f"{final['sam_gain_over_stage2']:+.4f} deg), "
        f"diffusion gain=({final['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
