"""Train Stage 3: dual-domain uncertainty-aware residual diffusion refinement.

Stage 2 is frozen. Its remaining HR-HSI error is decomposed into an affine-basis
coefficient residual and a spectral orthogonal-complement residual. Two
conditional diffusion denoisers learn the domains separately and predict both
noise and log variance. Validation uses deterministic zero-latent DDIM so the
untrained Stage 3 starts exactly from the Stage-2 result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage2_multiscale_pyramid import Stage2MultiScalePyramidNet
from models.stage3_dual_domain_diffusion import (
    BasisOrthogonalResidualDiffusionRefiner,
)
from train_stage2_coefficients import (
    build_spectral_response,
    first_spectral_difference,
    load_stage1_basis_checkpoint,
    second_spectral_difference,
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


def parse_stage3_args():
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

    # Stage-2 architecture required for strict checkpoint reconstruction.
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--anchor_normalized_clip", type=float, default=0.0)
    parser.add_argument("--projector_tolerance", type=float, default=1e-6)
    parser.add_argument("--stage2_feature_channels", type=int, default=64)
    parser.add_argument("--stage2_encoder_blocks", type=int, default=3)
    parser.add_argument("--stage2_fusion_channels", type=int, default=96)
    parser.add_argument("--stage2_fusion_blocks", type=int, default=4)
    parser.add_argument(
        "--stage2_max_normalized_residual",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--stage2_coefficient_scale_floor",
        type=float,
        default=1e-4,
    )
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

    # Stage-3 model.
    parser.add_argument("--stage3_hidden_channels", type=int, default=96)
    parser.add_argument("--stage3_num_blocks", type=int, default=6)
    parser.add_argument("--stage3_time_channels", type=int, default=192)
    parser.add_argument("--stage3_diffusion_timesteps", type=int, default=100)
    parser.add_argument("--stage3_beta_start", type=float, default=1e-4)
    parser.add_argument("--stage3_beta_end", type=float, default=2e-2)
    parser.add_argument("--stage3_log_variance_min", type=float, default=-6.0)
    parser.add_argument("--stage3_log_variance_max", type=float, default=3.0)
    parser.add_argument("--stage3_clean_clip", type=float, default=8.0)
    parser.add_argument("--stage3_msi_residual_gain", type=float, default=10.0)
    parser.add_argument("--stage3_residual_scale_floor", type=float, default=1e-5)
    parser.add_argument("--stage3_inference_steps", type=int, default=12)
    parser.add_argument(
        "--stage3_initial_noise",
        type=str,
        default="zero",
        choices=["zero", "random"],
    )

    # Stage-3 optimization and losses.
    parser.add_argument("--stage3_lr", type=float, default=1e-4)
    parser.add_argument("--stage3_grad_clip", type=float, default=1.0)
    parser.add_argument("--stage3_scale_estimation_batches", type=int, default=0)
    parser.add_argument("--stage3_diffusion_warmup_epochs", type=int, default=5)
    parser.add_argument("--stage3_reconstruction_ramp_epochs", type=int, default=5)
    parser.add_argument("--stage3_lambda_basis_diffusion", type=float, default=1.0)
    parser.add_argument("--stage3_lambda_orthogonal_diffusion", type=float, default=1.0)
    parser.add_argument("--stage3_lambda_basis_x0", type=float, default=0.2)
    parser.add_argument("--stage3_lambda_orthogonal_x0", type=float, default=0.2)
    parser.add_argument("--stage3_lambda_hsi_l1", type=float, default=0.5)
    parser.add_argument("--stage3_lambda_sam", type=float, default=0.2)
    parser.add_argument("--stage3_lambda_sgrad1", type=float, default=0.05)
    parser.add_argument("--stage3_lambda_sgrad2", type=float, default=0.02)
    parser.add_argument("--stage3_lambda_orthogonality", type=float, default=0.05)
    parser.add_argument("--stage3_selection_sam_weight", type=float, default=0.5)

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.stage3_inference_steps <= 0:
        raise ValueError("stage3_inference_steps must be positive")
    if cfg.stage3_lr <= 0:
        raise ValueError("stage3_lr must be positive")
    if cfg.stage3_scale_estimation_batches < 0:
        raise ValueError("stage3_scale_estimation_batches must be non-negative")
    if cfg.stage3_diffusion_warmup_epochs < 0:
        raise ValueError("stage3_diffusion_warmup_epochs must be non-negative")
    if cfg.stage3_reconstruction_ramp_epochs < 0:
        raise ValueError("stage3_reconstruction_ramp_epochs must be non-negative")

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


def build_stage2_model(cfg, info: dict, device: torch.device):
    stage1, stage1_state = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    stage2 = Stage2MultiScalePyramidNet(
        stage1_model=stage1,
        spectral_response=build_spectral_response(info).to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        anchor_normalized_clip=cfg.anchor_normalized_clip,
        projector_tolerance=cfg.projector_tolerance,
        feature_channels=cfg.stage2_feature_channels,
        encoder_blocks=cfg.stage2_encoder_blocks,
        fusion_channels=cfg.stage2_fusion_channels,
        fusion_blocks=cfg.stage2_fusion_blocks,
        max_normalized_residual=cfg.stage2_max_normalized_residual,
        coefficient_scale_floor=cfg.stage2_coefficient_scale_floor,
        num_frequency_bands=cfg.stage2_num_frequency_bands,
        init_low_boundary=cfg.stage2_init_low_boundary,
        init_high_boundary=cfg.stage2_init_high_boundary,
        boundary_temperature=cfg.stage2_boundary_temperature,
        edge_threshold_mode=cfg.stage2_edge_threshold_mode,
        edge_mask_threshold=cfg.stage2_edge_mask_threshold,
        edge_reference_quantile=cfg.stage2_edge_reference_quantile,
        noise_quantile=cfg.stage2_noise_quantile,
        hard_partition=not cfg.stage2_soft_frequency_partition,
        pyramid_quarter_scale=cfg.pyramid_quarter_scale,
        pyramid_half_scale=cfg.pyramid_half_scale,
    ).to(device)
    stage2_epoch, _ = load_checkpoint(
        stage2,
        cfg.stage2_checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    stage2.eval()
    for parameter in stage2.parameters():
        parameter.requires_grad_(False)
    return stage2, stage1_state, stage2_epoch


def build_stage3_model(
    cfg,
    stage2: Stage2MultiScalePyramidNet,
    device: torch.device,
) -> BasisOrthogonalResidualDiffusionRefiner:
    return BasisOrthogonalResidualDiffusionRefiner(
        stage2_model=stage2,
        hidden_channels=cfg.stage3_hidden_channels,
        num_blocks=cfg.stage3_num_blocks,
        time_channels=cfg.stage3_time_channels,
        diffusion_timesteps=cfg.stage3_diffusion_timesteps,
        beta_start=cfg.stage3_beta_start,
        beta_end=cfg.stage3_beta_end,
        log_variance_min=cfg.stage3_log_variance_min,
        log_variance_max=cfg.stage3_log_variance_max,
        clean_clip=cfg.stage3_clean_clip,
        msi_residual_gain=cfg.stage3_msi_residual_gain,
        residual_scale_floor=cfg.stage3_residual_scale_floor,
    ).to(device)


@torch.no_grad()
def estimate_residual_scales(
    model: BasisOrthogonalResidualDiffusionRefiner,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> Dict[str, torch.Tensor]:
    model.eval()
    coefficient_square_sum = torch.zeros(
        model.basis_rank,
        device=device,
        dtype=torch.float64,
    )
    coefficient_count = 0
    orthogonal_square_sum = torch.zeros((), device=device, dtype=torch.float64)
    orthogonal_count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_to_device(batch, device)
        stage2_outputs = model.stage2_forward(batch["lr_hsi"], batch["hr_msi"])
        residual = batch["gt"] - stage2_outputs["reconstructed_hsi"]
        coefficient, _, orthogonal = model.decompose_residual(
            residual,
            stage2_outputs["basis"],
        )
        coefficient_square_sum += coefficient.double().square().sum(dim=(0, 2, 3))
        coefficient_count += coefficient.size(0) * coefficient.size(2) * coefficient.size(3)
        orthogonal_square_sum += orthogonal.double().square().sum()
        orthogonal_count += orthogonal.numel()

    if coefficient_count == 0 or orthogonal_count == 0:
        raise RuntimeError("No samples were available for Stage-3 scale estimation")
    coefficient_scale = torch.sqrt(
        coefficient_square_sum / float(coefficient_count)
    ).float()
    orthogonal_scale = torch.sqrt(
        orthogonal_square_sum / float(orthogonal_count)
    ).float()
    model.set_residual_scales(coefficient_scale, orthogonal_scale)
    return {
        "coefficient_scale": model.coefficient_residual_scale.detach().clone(),
        "orthogonal_scale": model.orthogonal_residual_scale.detach().clone(),
    }


def heteroscedastic_noise_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    squared_error = (predicted_noise - target_noise).square()
    return (torch.exp(-log_variance) * squared_error + log_variance).mean()


def reconstruction_factor(epoch: int, cfg) -> float:
    if epoch < cfg.stage3_diffusion_warmup_epochs:
        return 0.0
    if cfg.stage3_reconstruction_ramp_epochs == 0:
        return 1.0
    progress = epoch - cfg.stage3_diffusion_warmup_epochs + 1
    return min(progress / float(cfg.stage3_reconstruction_ramp_epochs), 1.0)


def compute_stage3_losses(
    model: BasisOrthogonalResidualDiffusionRefiner,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    sam_loss: SAMLoss,
    cfg,
    reconstruction_weight: float,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    losses["basis_diffusion"] = heteroscedastic_noise_loss(
        outputs["predicted_coefficient_noise"],
        outputs["coefficient_noise"],
        outputs["coefficient_log_variance"],
    )
    losses["orthogonal_diffusion"] = heteroscedastic_noise_loss(
        outputs["predicted_orthogonal_noise"],
        outputs["orthogonal_noise"],
        outputs["orthogonal_log_variance"],
    )
    losses["basis_x0"] = F.smooth_l1_loss(
        outputs["predicted_coefficient_clean"],
        outputs["coefficient_clean"],
        beta=0.25,
    )
    losses["orthogonal_x0"] = F.smooth_l1_loss(
        outputs["predicted_orthogonal_clean"],
        outputs["orthogonal_clean"],
        beta=0.25,
    )

    prediction = outputs["refined_hsi"]
    losses["hsi_l1"] = F.l1_loss(prediction, gt)
    losses["sam"] = sam_loss(prediction, gt)
    losses["sgrad1"] = F.l1_loss(
        first_spectral_difference(prediction),
        first_spectral_difference(gt),
    )
    losses["sgrad2"] = F.l1_loss(
        second_spectral_difference(prediction),
        second_spectral_difference(gt),
    )
    orthogonal_leakage = model.project_to_basis_coefficients(
        outputs["predicted_orthogonal_residual"],
        outputs["basis"],
    )
    losses["orthogonality"] = orthogonal_leakage.abs().mean()

    losses["total"] = (
        cfg.stage3_lambda_basis_diffusion * losses["basis_diffusion"]
        + cfg.stage3_lambda_orthogonal_diffusion
        * losses["orthogonal_diffusion"]
        + cfg.stage3_lambda_basis_x0 * losses["basis_x0"]
        + cfg.stage3_lambda_orthogonal_x0 * losses["orthogonal_x0"]
        + reconstruction_weight
        * (
            cfg.stage3_lambda_hsi_l1 * losses["hsi_l1"]
            + cfg.stage3_lambda_sam * losses["sam"]
            + cfg.stage3_lambda_sgrad1 * losses["sgrad1"]
            + cfg.stage3_lambda_sgrad2 * losses["sgrad2"]
            + cfg.stage3_lambda_orthogonality * losses["orthogonality"]
        )
    )

    coefficient_error = (
        outputs["predicted_coefficient_noise"] - outputs["coefficient_noise"]
    ).square()
    orthogonal_error = (
        outputs["predicted_orthogonal_noise"] - outputs["orthogonal_noise"]
    ).square()
    losses["basis_uncertainty_mean"] = torch.exp(
        0.5 * outputs["coefficient_log_variance"]
    ).mean().detach()
    losses["orthogonal_uncertainty_mean"] = torch.exp(
        0.5 * outputs["orthogonal_log_variance"]
    ).mean().detach()
    losses["basis_noise_mse"] = coefficient_error.mean().detach()
    losses["orthogonal_noise_mse"] = orthogonal_error.mean().detach()
    losses["target_basis_abs"] = outputs[
        "target_coefficient_residual"
    ].abs().mean().detach()
    losses["target_orthogonal_abs"] = outputs[
        "target_orthogonal_residual"
    ].abs().mean().detach()
    losses["predicted_basis_abs"] = outputs[
        "predicted_parallel_residual"
    ].abs().mean().detach()
    losses["predicted_orthogonal_abs"] = outputs[
        "predicted_orthogonal_residual"
    ].abs().mean().detach()
    return losses


STAGE3_LOSS_NAMES = [
    "total",
    "basis_diffusion",
    "orthogonal_diffusion",
    "basis_x0",
    "orthogonal_x0",
    "hsi_l1",
    "sam",
    "sgrad1",
    "sgrad2",
    "orthogonality",
    "basis_uncertainty_mean",
    "orthogonal_uncertainty_mean",
    "basis_noise_mse",
    "orthogonal_noise_mse",
    "target_basis_abs",
    "target_orthogonal_abs",
    "predicted_basis_abs",
    "predicted_orthogonal_abs",
]


def train_one_epoch(
    model: BasisOrthogonalResidualDiffusionRefiner,
    loader,
    optimizer: torch.optim.Optimizer,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
    reconstruction_weight: float,
) -> Dict[str, float]:
    model.train()
    meters = {name: AverageMeter() for name in STAGE3_LOSS_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_forward(
            batch["lr_hsi"],
            batch["hr_msi"],
            batch["gt"],
        )
        losses = compute_stage3_losses(
            model,
            outputs,
            batch["gt"],
            sam_loss,
            cfg,
            reconstruction_weight,
        )
        losses["total"].backward()
        if cfg.stage3_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                cfg.stage3_grad_clip,
            )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        for name in STAGE3_LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: BasisOrthogonalResidualDiffusionRefiner,
    loader,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    stage2_metrics = MetricAverager()
    stage3_metrics = MetricAverager()
    diagnostics = {
        name: AverageMeter()
        for name in (
            "basis_correction_abs",
            "orthogonal_correction_abs",
            "basis_uncertainty",
            "orthogonal_uncertainty",
            "orthogonality_leakage",
            "out_of_range_ratio",
        )
    }

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model.sample(
            batch["lr_hsi"],
            batch["hr_msi"],
            inference_steps=cfg.stage3_inference_steps,
            initial_noise=cfg.stage3_initial_noise,
        )
        stage2_metrics.update(
            calc_metrics(outputs["stage2_hsi"], batch["gt"], cfg.scale_ratio)
        )
        stage3_metrics.update(
            calc_metrics(outputs["refined_hsi"], batch["gt"], cfg.scale_ratio)
        )
        leakage = model.project_to_basis_coefficients(
            outputs["stage3_orthogonal_residual"],
            outputs["basis"],
        ).abs().mean()
        out_of_range = (
            (outputs["refined_hsi"] < 0.0)
            | (outputs["refined_hsi"] > 1.0)
        ).float().mean()
        values = {
            "basis_correction_abs": outputs[
                "stage3_parallel_residual"
            ].abs().mean(),
            "orthogonal_correction_abs": outputs[
                "stage3_orthogonal_residual"
            ].abs().mean(),
            "basis_uncertainty": outputs[
                "stage3_coefficient_uncertainty"
            ].mean(),
            "orthogonal_uncertainty": outputs[
                "stage3_orthogonal_uncertainty"
            ].mean(),
            "orthogonality_leakage": leakage,
            "out_of_range_ratio": out_of_range,
        }
        batch_size = batch["lr_hsi"].size(0)
        for name, value in values.items():
            diagnostics[name].update(float(value.item()), batch_size)

    stage2 = stage2_metrics.average()
    stage3 = stage3_metrics.average()
    result = {
        "stage2_psnr": stage2["PSNR"],
        "stage2_sam": stage2["SAM"],
        "stage2_rmse": stage2["RMSE"],
        "stage3_psnr": stage3["PSNR"],
        "stage3_sam": stage3["SAM"],
        "stage3_rmse": stage3["RMSE"],
        "stage3_ergas": stage3["ERGAS"],
        "stage3_ssim": stage3["SSIM"],
        "stage3_cc": stage3["CC"],
        "psnr_gain_over_stage2": stage3["PSNR"] - stage2["PSNR"],
        "sam_gain_over_stage2": stage2["SAM"] - stage3["SAM"],
        **{name: meter.avg for name, meter in diagnostics.items()},
    }
    result["selection"] = (
        -result["stage3_psnr"]
        + cfg.stage3_selection_sam_weight * result["stage3_sam"]
    )
    return result


@torch.no_grad()
def export_outputs(
    model: BasisOrthogonalResidualDiffusionRefiner,
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
        os.path.join(output_dir, "stage3_dual_domain_diffusion_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        stage2_hsi=outputs["stage2_hsi"].detach().cpu().numpy(),
        stage3_hsi=outputs["refined_hsi"].detach().cpu().numpy(),
        coefficient_residual=outputs[
            "stage3_coefficient_residual"
        ].detach().cpu().numpy(),
        parallel_residual=outputs[
            "stage3_parallel_residual"
        ].detach().cpu().numpy(),
        orthogonal_residual=outputs[
            "stage3_orthogonal_residual"
        ].detach().cpu().numpy(),
        coefficient_uncertainty=outputs[
            "stage3_coefficient_uncertainty"
        ].detach().cpu().numpy(),
        orthogonal_uncertainty=outputs[
            "stage3_orthogonal_uncertainty"
        ].detach().cpu().numpy(),
        coefficient_uncertainty_map=outputs[
            "stage3_coefficient_uncertainty_map"
        ].detach().cpu().numpy(),
        orthogonal_uncertainty_map=outputs[
            "stage3_orthogonal_uncertainty_map"
        ].detach().cpu().numpy(),
        reliability_map=outputs["reliability_map"].detach().cpu().numpy(),
        coefficient_residual_scale=model.coefficient_residual_scale.detach().cpu().numpy(),
        orthogonal_residual_scale=model.orthogonal_residual_scale.detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_stage3_args()
    cfg.stage = "stage3_dual_domain_diffusion"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage2, stage1_state, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_stage3_model(cfg, stage2, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.stage3_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.stage3_lr * 0.05,
    )

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage3_dual_domain_diffusion",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage3_dual_domain_diffusion",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage3_dual_domain_diffusion")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "stage3_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "stage3_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "stage3_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "stage3_last.pth")
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
        f"Stage 3 start | Stage2={initial['stage2_psnr']:.4f} dB/"
        f"{initial['stage2_sam']:.4f} deg | "
        f"Stage3={initial['stage3_psnr']:.4f} dB/"
        f"{initial['stage3_sam']:.4f} deg | "
        f"correction=({initial['basis_correction_abs']:.3e}, "
        f"{initial['orthogonal_correction_abs']:.3e}) | "
        f"trainable={count_parameters(model):.3f} M.",
    )

    csv_fields = [
        "epoch",
        "lr",
        "reconstruction_weight",
        "stage2_psnr",
        "stage2_sam",
        "stage3_psnr",
        "stage3_sam",
        "stage3_rmse",
        "stage3_ergas",
        "stage3_ssim",
        "stage3_cc",
        "psnr_gain_over_stage2",
        "sam_gain_over_stage2",
        "basis_correction_abs",
        "orthogonal_correction_abs",
        "basis_uncertainty",
        "orthogonal_uncertainty",
        "orthogonality_leakage",
        "out_of_range_ratio",
        *[f"train_{name}" for name in STAGE3_LOSS_NAMES],
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    best_selection = initial["selection"]
    best_psnr = initial["stage3_psnr"]
    best_sam = initial["stage3_sam"]
    initial_extra = {
        "stage": "stage3_dual_domain_diffusion",
        "dataset": cfg.dataset,
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "stage2_checkpoint": cfg.stage2_checkpoint,
        "stage2_epoch": stage2_epoch,
        "basis_rank": model.basis_rank,
        "n_bands": model.n_bands,
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
        current_reconstruction_weight = reconstruction_factor(epoch, cfg)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            current_reconstruction_weight,
        )
        scheduler.step()

        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={
                    "stage": "stage3_dual_domain_diffusion",
                    "dataset": cfg.dataset,
                    "train": train_result,
                },
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"PSNR={val['stage3_psnr']:.4f}, SAM={val['stage3_sam']:.4f} deg | "
            f"gain=({val['psnr_gain_over_stage2']:+.4f} dB, "
            f"{val['sam_gain_over_stage2']:+.4f} deg) | "
            f"corr=({val['basis_correction_abs']:.3e}, "
            f"{val['orthogonal_correction_abs']:.3e}) | "
            f"unc=({val['basis_uncertainty']:.3f}, "
            f"{val['orthogonal_uncertainty']:.3f}) | "
            f"recon_weight={current_reconstruction_weight:.2f}.",
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "reconstruction_weight": current_reconstruction_weight,
            **{key: val[key] for key in csv_fields if key in val},
        }
        row.update({f"train_{name}": train_result[name] for name in STAGE3_LOSS_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "stage3_dual_domain_diffusion",
            "dataset": cfg.dataset,
            "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
            "stage2_checkpoint": cfg.stage2_checkpoint,
            "stage2_epoch": stage2_epoch,
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
        if val["stage3_psnr"] > best_psnr:
            best_psnr = val["stage3_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )
        if val["stage3_sam"] < best_sam:
            best_sam = val["stage3_sam"]
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
        f"Stage 3 complete | PSNR={final['stage3_psnr']:.4f}, "
        f"SAM={final['stage3_sam']:.4f} deg, "
        f"gain=({final['psnr_gain_over_stage2']:+.4f} dB, "
        f"{final['sam_gain_over_stage2']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
