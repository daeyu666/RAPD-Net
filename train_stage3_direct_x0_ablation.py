"""Train the direct x0 Stage-3 residual ablation.

This control keeps the fitted deterministic Stage-3 branches and removes the
entire diffusion denoising/sampling process. The only trainable part is a pair
of direct clean-residual heads, using the original Stage-3 residual scales and
an explicitly selected mask mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage3_direct_x0_ablation import DirectX0Stage3AblationRefiner
from train_stage2_coefficients import (
    first_spectral_difference,
    second_spectral_difference,
)
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import (
    parse_uncertainty_guided_args,
    spatial_correlation,
)
from train_stage3_uncertainty_guided_diffusion_v2 import (
    load_deterministic_initialization,
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


def parse_direct_args():
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
    parser.add_argument("--direct_x0_epochs", type=int, default=120)
    parser.add_argument("--direct_x0_lr", type=float, default=5e-5)
    parser.add_argument("--direct_x0_hidden_channels", type=int, default=96)
    parser.add_argument("--direct_x0_blocks", type=int, default=6)
    parser.add_argument(
        "--direct_x0_mask_mode",
        type=str,
        default="predicted",
        choices=["predicted", "oracle", "none"],
    )
    parser.add_argument("--direct_x0_grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_direct_x0", type=float, default=1.0)
    parser.add_argument("--lambda_direct_final_l1", type=float, default=0.5)
    parser.add_argument("--lambda_direct_final_sam", type=float, default=0.2)
    parser.add_argument("--lambda_direct_sgrad1", type=float, default=0.05)
    parser.add_argument("--lambda_direct_sgrad2", type=float, default=0.02)
    parser.add_argument("--lambda_direct_orthogonality", type=float, default=0.05)

    specific, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_uncertainty_guided_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    if cfg.direct_x0_epochs <= 0:
        raise ValueError("direct_x0_epochs must be positive")
    cfg.epochs = cfg.direct_x0_epochs
    cfg.stage3_inference_steps = 1
    cfg.stage3_initial_noise = "zero"
    return cfg


def build_model(cfg, stage2, device: torch.device):
    return DirectX0Stage3AblationRefiner(
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
        direct_hidden_channels=cfg.direct_x0_hidden_channels,
        direct_blocks=cfg.direct_x0_blocks,
    ).to(device)


def set_requires_grad(parameters: Iterable[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    expanded = weight.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1e-6)


def weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 0.25,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return weighted_mean(loss, weight)


def spectral_losses(prediction: torch.Tensor, target: torch.Tensor, sam_loss: SAMLoss):
    l1 = F.l1_loss(prediction, target)
    sam = sam_loss(prediction, target)
    sgrad1 = F.l1_loss(
        first_spectral_difference(prediction),
        first_spectral_difference(target),
    )
    sgrad2 = F.l1_loss(
        second_spectral_difference(prediction),
        second_spectral_difference(target),
    )
    return l1, sam, sgrad1, sgrad2


def compute_losses(model, outputs, gt, sam_loss, cfg):
    basis_mask = outputs["basis_train_mask"]
    orthogonal_mask = outputs["orthogonal_train_mask"]
    basis_x0 = weighted_smooth_l1(
        outputs["predicted_coefficient_clean"],
        outputs["remaining_coefficient_normalized"] * basis_mask,
        basis_mask,
    )
    orthogonal_x0 = weighted_smooth_l1(
        outputs["predicted_orthogonal_clean"],
        outputs["remaining_orthogonal_normalized"] * orthogonal_mask,
        orthogonal_mask,
    )
    final_l1, final_sam, final_sgrad1, final_sgrad2 = spectral_losses(
        outputs["refined_hsi"],
        gt,
        sam_loss,
    )
    orthogonality = model.project_to_basis_coefficients(
        outputs["diffusion_orthogonal_residual"],
        outputs["basis"],
    ).abs().mean()
    total = (
        cfg.lambda_direct_x0 * (basis_x0 + orthogonal_x0)
        + cfg.lambda_direct_final_l1 * final_l1
        + cfg.lambda_direct_final_sam * final_sam
        + cfg.lambda_direct_sgrad1 * final_sgrad1
        + cfg.lambda_direct_sgrad2 * final_sgrad2
        + cfg.lambda_direct_orthogonality * orthogonality
    )
    return {
        "total": total,
        "basis_x0": basis_x0,
        "orthogonal_x0": orthogonal_x0,
        "final_l1": final_l1,
        "final_sam": final_sam,
        "final_sgrad1": final_sgrad1,
        "final_sgrad2": final_sgrad2,
        "orthogonality": orthogonality,
        "basis_mask_mean": basis_mask.mean().detach(),
        "orthogonal_mask_mean": orthogonal_mask.mean().detach(),
        "direct_basis_abs": outputs["diffusion_parallel_residual"].abs().mean().detach(),
        "direct_orthogonal_abs": outputs["diffusion_orthogonal_residual"].abs().mean().detach(),
    }


LOSS_NAMES = [
    "total",
    "basis_x0",
    "orthogonal_x0",
    "final_l1",
    "final_sam",
    "final_sgrad1",
    "final_sgrad2",
    "orthogonality",
    "basis_mask_mean",
    "orthogonal_mask_mean",
    "direct_basis_abs",
    "direct_orthogonal_abs",
]


def train_one_epoch(model, loader, optimizer, sam_loss, cfg, device):
    model.train()
    meters = {name: AverageMeter() for name in LOSS_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_forward(
            batch["lr_hsi"],
            batch["hr_msi"],
            batch["gt"],
            mask_mode=cfg.direct_x0_mask_mode,
        )
        losses = compute_losses(model, outputs, batch["gt"], sam_loss, cfg)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("Non-finite direct x0 loss")
        losses["total"].backward()
        if cfg.direct_x0_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                cfg.direct_x0_grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite direct x0 gradient")
        optimizer.step()
        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_direct(model, loader, cfg, device):
    model.eval()
    metric_sets = {name: MetricAverager() for name in ("stage2", "deterministic", "final")}
    diagnostics = {name: AverageMeter() for name in (
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
        "direct_basis_abs",
        "direct_orthogonal_abs",
    )}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model.sample(
            batch["lr_hsi"],
            batch["hr_msi"],
            mask_mode="predicted",
        )
        metric_sets["stage2"].update(calc_metrics(outputs["stage2_hsi"], batch["gt"], cfg.scale_ratio))
        metric_sets["deterministic"].update(calc_metrics(outputs["stage3_deterministic_hsi"], batch["gt"], cfg.scale_ratio))
        metric_sets["final"].update(calc_metrics(outputs["refined_hsi"], batch["gt"], cfg.scale_ratio))

        target_residual = batch["gt"] - outputs["stage2_hsi"]
        target_coefficient, _, target_orthogonal = model.decompose_residual(
            target_residual,
            outputs["basis"],
        )
        coefficient_error = (
            target_coefficient / model.coefficient_residual_scale.view(1, -1, 1, 1)
            - outputs["deterministic_coefficient_normalized"]
        )
        orthogonal_error = model.project_orthogonal(
            target_orthogonal / model.orthogonal_residual_scale.view(1, 1, 1, 1)
            - outputs["deterministic_orthogonal_normalized"],
            outputs["basis"],
        )
        batch_size = batch["lr_hsi"].size(0)
        values = {
            "basis_mask_mean": float(outputs["basis_mask"].mean().item()),
            "orthogonal_mask_mean": float(outputs["orthogonal_mask"].mean().item()),
            "basis_uncertainty_error_correlation": spatial_correlation(
                outputs["basis_uncertainty"],
                coefficient_error.abs().mean(dim=1, keepdim=True),
            ),
            "orthogonal_uncertainty_error_correlation": spatial_correlation(
                outputs["orthogonal_uncertainty"],
                orthogonal_error.abs().mean(dim=1, keepdim=True),
            ),
            "direct_basis_abs": float(outputs["stage3_diffusion_parallel_residual"].abs().mean().item()),
            "direct_orthogonal_abs": float(outputs["stage3_diffusion_orthogonal_residual"].abs().mean().item()),
        }
        for name, value in values.items():
            diagnostics[name].update(value, batch_size)
    averages = {name: meter.average() for name, meter in metric_sets.items()}
    result = {}
    for name, metrics in averages.items():
        result[f"{name}_psnr"] = metrics["PSNR"]
        result[f"{name}_sam"] = metrics["SAM"]
        if name == "final":
            result["final_rmse"] = metrics["RMSE"]
            result["final_ergas"] = metrics["ERGAS"]
            result["final_ssim"] = metrics["SSIM"]
            result["final_cc"] = metrics["CC"]
    result["psnr_gain_over_stage2"] = result["final_psnr"] - result["stage2_psnr"]
    result["sam_gain_over_stage2"] = result["stage2_sam"] - result["final_sam"]
    result["direct_psnr_gain_over_deterministic"] = result["final_psnr"] - result["deterministic_psnr"]
    result["direct_sam_gain_over_deterministic"] = result["deterministic_sam"] - result["final_sam"]
    result["diffusion_psnr_gain_over_deterministic"] = result[
        "direct_psnr_gain_over_deterministic"
    ]
    result["diffusion_sam_gain_over_deterministic"] = result[
        "direct_sam_gain_over_deterministic"
    ]
    for name, meter in diagnostics.items():
        result[name] = meter.avg
    result["selection"] = -result["final_psnr"] + cfg.stage3_selection_sam_weight * result["final_sam"]
    return result


@torch.no_grad()
def export_outputs(model, loader, output_dir, cfg, device):
    ensure_dir(output_dir)
    batch = move_to_device(next(iter(loader)), device)
    outputs = model.sample(batch["lr_hsi"], batch["hr_msi"], mask_mode="predicted")
    np.savez_compressed(
        os.path.join(output_dir, "stage3_direct_x0_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        stage2_hsi=outputs["stage2_hsi"].detach().cpu().numpy(),
        deterministic_hsi=outputs["stage3_deterministic_hsi"].detach().cpu().numpy(),
        final_hsi=outputs["refined_hsi"].detach().cpu().numpy(),
        basis_mask=outputs["basis_mask"].detach().cpu().numpy(),
        orthogonal_mask=outputs["orthogonal_mask"].detach().cpu().numpy(),
        direct_parallel=outputs["stage3_diffusion_parallel_residual"].detach().cpu().numpy(),
        direct_orthogonal=outputs["stage3_diffusion_orthogonal_residual"].detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_direct_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_model(cfg, stage2, device)

    if cfg.resume:
        optimizer = torch.optim.AdamW(model.direct_parameters(), lr=cfg.direct_x0_lr, weight_decay=cfg.weight_decay)
        start_epoch, _ = load_checkpoint(model, cfg.resume, optimizer=optimizer, strict=True, map_location=str(device))
        source_epoch = -1
    else:
        source_epoch = load_deterministic_initialization(model, cfg.stage3_initial_checkpoint, device)
        start_epoch = 0
        optimizer = torch.optim.AdamW(model.direct_parameters(), lr=cfg.direct_x0_lr, weight_decay=cfg.weight_decay)

    set_requires_grad(model.stage2.parameters(), False)
    set_requires_grad(model.deterministic_parameters(), False)
    set_requires_grad(model.direct_parameters(), True)

    checkpoint_dir = os.path.join(cfg.checkpoint_root, "stage3_direct_x0_ablation", cfg.dataset, cfg.direct_x0_mask_mode)
    output_dir = os.path.join(cfg.output_root, "stage3_direct_x0_ablation", cfg.dataset, cfg.direct_x0_mask_mode)
    log_dir = os.path.join(cfg.log_root, "stage3_direct_x0_ablation", cfg.direct_x0_mask_mode)
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "direct_x0_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "direct_x0_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "direct_x0_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "direct_x0_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    initial = evaluate_direct(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"Direct x0 start | source_epoch={source_epoch} | mask={cfg.direct_x0_mask_mode} | "
        f"det={initial['deterministic_psnr']:.4f}/{initial['deterministic_sam']:.4f} | "
        f"final={initial['final_psnr']:.4f}/{initial['final_sam']:.4f}."
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
        "direct_psnr_gain_over_deterministic",
        "direct_sam_gain_over_deterministic",
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
        "direct_basis_abs",
        "direct_orthogonal_abs",
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        ["epoch", *metric_fields, *[f"train_{name}" for name in LOSS_NAMES]],
    )
    best_selection = initial["selection"]
    best_psnr = initial["final_psnr"]
    best_sam = initial["final_sam"]
    initial_extra = {
        "stage": "stage3_direct_x0_ablation",
        "mask_mode": cfg.direct_x0_mask_mode,
        "source_checkpoint": cfg.stage3_initial_checkpoint,
        "source_epoch": source_epoch,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, initial_extra)

    sam_loss = SAMLoss()
    for epoch in range(start_epoch, cfg.direct_x0_epochs):
        train_result = train_one_epoch(model, train_loader, optimizer, sam_loss, cfg, device)
        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(model, optimizer, epoch + 1, best_selection, last_path, {"train": train_result})
            continue
        val = evaluate_direct(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.direct_x0_epochs:03d} | "
            f"final={val['final_psnr']:.4f}/{val['final_sam']:.4f} | "
            f"direct=({val['direct_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['direct_sam_gain_over_deterministic']:+.4f} deg) | "
            f"abs=({val['direct_basis_abs']:.3e}, {val['direct_orthogonal_abs']:.3e})."
        )
        row = {"epoch": epoch + 1, **{name: val[name] for name in metric_fields}}
        row.update({f"train_{name}": train_result[name] for name in LOSS_NAMES})
        csv_logger.write(row)
        extra = {
            "stage": "stage3_direct_x0_ablation",
            "mask_mode": cfg.direct_x0_mask_mode,
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

    load_checkpoint(model, best_path, optimizer=None, strict=True, map_location=str(device), load_optimizer=False)
    final = evaluate_direct(model, test_loader, cfg, device)
    export_outputs(model, test_loader, output_dir, cfg, device)
    with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(final, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Direct x0 complete | PSNR={final['final_psnr']:.4f}, "
        f"SAM={final['final_sam']:.4f}, "
        f"direct gain=({final['direct_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['direct_sam_gain_over_deterministic']:+.4f} deg)."
    )


if __name__ == "__main__":
    main()
