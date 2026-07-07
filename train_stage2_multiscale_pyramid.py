"""Train the residual-of-residual multiscale coefficient pyramid.

A trained symmetric-frequency model remains the source predictor. New
quarter/half/full correction branches are initialized with copied trunks and
zero output heads, then learn only the coefficient error left by the source.

Training schedule:
1. correction-only warm-up: source and SSP boundaries are frozen;
2. low-rate joint fine-tuning: source is unfrozen at a much smaller LR;
3. SSP boundaries remain frozen throughout to avoid destroying the mature
   symmetric frequency partition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_multiscale_pyramid import (
    Stage2MultiScalePyramidNet,
    resize_antialiased,
)
from train_stage2_coefficients import (
    MONITOR_NAMES,
    FixedSpatialDegradation,
    build_spectral_response,
    compute_losses as compute_base_losses,
    load_stage1_basis_checkpoint,
    monitor_values,
)
from train_stage2_dual_space import DUAL_NAMES, dual_losses
from train_stage2_symmetric_frequency import (
    SYMMETRIC_NAMES,
    evaluate_symmetric,
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


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def parse_multiscale_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument(
        "--symmetric_frequency_checkpoint",
        type=str,
        default=(
            "./checkpoints/stage2_symmetric_frequency/PaviaU/"
            "symmetric_frequency_best_psnr.pth"
        ),
    )
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
    parser.add_argument("--stage2_grad_clip", type=float, default=1.0)

    parser.add_argument("--stage2_lambda_l1", type=float, default=1.0)
    parser.add_argument("--stage2_lambda_sam", type=float, default=0.3)
    parser.add_argument("--stage2_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument(
        "--stage2_lambda_coefficient_residual",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--stage2_lambda_coefficient_reconstruction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--stage2_lambda_lr_hsi", type=float, default=0.2)
    parser.add_argument("--stage2_lambda_lr_coefficient", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_msi", type=float, default=0.2)
    parser.add_argument("--stage2_lambda_residual_l1", type=float, default=0.001)
    parser.add_argument("--stage2_lambda_residual_tv", type=float, default=0.001)
    parser.add_argument("--stage2_lambda_lf_alignment", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_noise", type=float, default=0.01)
    parser.add_argument("--stage2_lambda_partition", type=float, default=0.01)
    parser.add_argument("--stage2_lambda_improvement", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_msi_usage", type=float, default=0.05)
    parser.add_argument("--stage2_improvement_margin", type=float, default=1e-4)
    parser.add_argument("--stage2_msi_usage_margin", type=float, default=1e-4)
    parser.add_argument("--stage2_selection_sam_weight", type=float, default=0.5)
    parser.add_argument("--stage2_selection_sgrad1_weight", type=float, default=0.1)
    parser.add_argument("--stage2_selection_sgrad2_weight", type=float, default=0.05)

    parser.add_argument("--dual_lambda_observable", type=float, default=0.1)
    parser.add_argument("--dual_lambda_null", type=float, default=0.2)
    parser.add_argument("--dual_lambda_null_msi_leakage", type=float, default=0.05)

    parser.add_argument("--pyramid_quarter_scale", type=float, default=0.25)
    parser.add_argument("--pyramid_half_scale", type=float, default=0.5)
    parser.add_argument("--pyramid_lambda_quarter", type=float, default=0.1)
    parser.add_argument("--pyramid_lambda_half", type=float, default=0.25)
    parser.add_argument("--pyramid_lambda_full", type=float, default=0.5)
    parser.add_argument("--pyramid_lambda_observable", type=float, default=0.1)
    parser.add_argument("--pyramid_lambda_null", type=float, default=0.2)

    parser.add_argument("--pyramid_warmup_epochs", type=int, default=10)
    parser.add_argument("--pyramid_new_lr", type=float, default=5e-5)
    parser.add_argument("--pyramid_source_lr", type=float, default=1e-6)
    parser.add_argument("--pyramid_min_lr_ratio", type=float, default=0.05)

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    if cfg.pyramid_warmup_epochs < 0:
        raise ValueError("pyramid_warmup_epochs must be non-negative")
    if cfg.pyramid_new_lr <= 0 or cfg.pyramid_source_lr < 0:
        raise ValueError("Pyramid learning rates are invalid")
    if not 0.0 < cfg.pyramid_min_lr_ratio <= 1.0:
        raise ValueError("pyramid_min_lr_ratio must lie in (0, 1]")

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
        if cfg.symmetric_frequency_checkpoint.endswith(
            "stage2_symmetric_frequency/PaviaU/"
            "symmetric_frequency_best_psnr.pth"
        ):
            cfg.symmetric_frequency_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "stage2_symmetric_frequency",
                cfg.dataset,
                "symmetric_frequency_best_psnr.pth",
            )
    return cfg


def load_symmetric_warm_start(
    model: Stage2MultiScalePyramidNet,
    path: str,
    device: torch.device,
) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Symmetric-frequency checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    source = state.get("model", state)
    destination = model.state_dict()
    transferable = {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    allowed_missing = (
        "quarter_branch.",
        "half_branch.",
        "full_correction_branch.",
    )
    problematic_missing = [
        key for key in missing if not key.startswith(allowed_missing)
    ]
    skipped_source = [key for key in source if key not in transferable]
    if unexpected or problematic_missing or skipped_source:
        raise RuntimeError(
            "Residual-pyramid warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"skipped_source={skipped_source}"
        )
    model.initialize_pyramid_from_full()
    return state


def project(
    projector: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


def pyramid_losses(
    model: Stage2MultiScalePyramidNet,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    scale = outputs["coefficient_scale"].view(1, -1, 1, 1)
    quarter_size = tuple(
        outputs["pyramid_quarter_normalized_residual"].shape[-2:]
    )
    half_size = tuple(
        outputs["pyramid_half_cumulative_normalized_residual"].shape[-2:]
    )

    with torch.no_grad():
        target_coefficients = model.stage1.encode(gt, basis=outputs["basis"])
        target_full_residual = (
            target_coefficients - outputs["anchor_coefficients"]
        )
        source_residual = outputs[
            "pyramid_source_coefficient_residual"
        ].detach()
        remaining_full = target_full_residual - source_residual
        remaining_quarter = resize_antialiased(
            remaining_full,
            quarter_size,
            mode="bicubic",
        )
        remaining_half = resize_antialiased(
            remaining_full,
            half_size,
            mode="bicubic",
        )

        quarter_observable_target = project(
            model.exact_observable_projector.to(remaining_quarter),
            remaining_quarter,
        )
        quarter_null_target = project(
            model.exact_null_projector.to(remaining_quarter),
            remaining_quarter,
        )
        half_observable_target = project(
            model.exact_observable_projector.to(remaining_half),
            remaining_half,
        )
        half_null_target = project(
            model.exact_null_projector.to(remaining_half),
            remaining_half,
        )
        full_observable_target = project(
            model.exact_observable_projector.to(remaining_full),
            remaining_full,
        )
        full_null_target = project(
            model.exact_null_projector.to(remaining_full),
            remaining_full,
        )

    quarter_observable = outputs["pyramid_quarter_observable_residual"]
    quarter_null = outputs["pyramid_quarter_null_residual"]
    half_observable = resize_antialiased(
        quarter_observable,
        half_size,
        mode="bicubic",
    ) + outputs["pyramid_half_increment_observable_residual"]
    half_null = resize_antialiased(
        quarter_null,
        half_size,
        mode="bicubic",
    ) + outputs["pyramid_half_increment_null_residual"]

    full_observable = outputs["pyramid_correction_observable_residual"]
    full_null = outputs["pyramid_correction_null_residual"]
    full_correction = outputs["pyramid_correction_coefficient_residual"]

    losses = {
        "pyramid_quarter_total": F.smooth_l1_loss(
            outputs["pyramid_quarter_normalized_residual"],
            remaining_quarter / scale,
            beta=0.25,
        ),
        "pyramid_half_total": F.smooth_l1_loss(
            outputs["pyramid_half_cumulative_normalized_residual"],
            remaining_half / scale,
            beta=0.25,
        ),
        "pyramid_full_total": F.smooth_l1_loss(
            full_correction / scale,
            remaining_full / scale,
            beta=0.25,
        ),
        "pyramid_quarter_observable": F.smooth_l1_loss(
            quarter_observable / scale,
            quarter_observable_target / scale,
            beta=0.25,
        ),
        "pyramid_quarter_null": F.smooth_l1_loss(
            quarter_null / scale,
            quarter_null_target / scale,
            beta=0.25,
        ),
        "pyramid_half_observable": F.smooth_l1_loss(
            half_observable / scale,
            half_observable_target / scale,
            beta=0.25,
        ),
        "pyramid_half_null": F.smooth_l1_loss(
            half_null / scale,
            half_null_target / scale,
            beta=0.25,
        ),
        "pyramid_full_observable": F.smooth_l1_loss(
            full_observable / scale,
            full_observable_target / scale,
            beta=0.25,
        ),
        "pyramid_full_null": F.smooth_l1_loss(
            full_null / scale,
            full_null_target / scale,
            beta=0.25,
        ),
        "pyramid_source_residual_abs": outputs[
            "pyramid_source_normalized_residual"
        ].abs().mean().detach(),
        "pyramid_remaining_target_abs": (
            remaining_full / scale
        ).abs().mean().detach(),
        "pyramid_quarter_increment_abs": outputs[
            "pyramid_quarter_normalized_residual"
        ].abs().mean().detach(),
        "pyramid_half_increment_abs": outputs[
            "pyramid_half_increment_normalized_residual"
        ].abs().mean().detach(),
        "pyramid_full_increment_abs": outputs[
            "pyramid_full_correction_normalized_residual"
        ].abs().mean().detach(),
        "pyramid_correction_abs": outputs[
            "pyramid_correction_normalized_residual"
        ].abs().mean().detach(),
    }
    return losses


PYRAMID_NAMES = [
    "pyramid_quarter_total",
    "pyramid_half_total",
    "pyramid_full_total",
    "pyramid_quarter_observable",
    "pyramid_quarter_null",
    "pyramid_half_observable",
    "pyramid_half_null",
    "pyramid_full_observable",
    "pyramid_full_null",
    "pyramid_source_residual_abs",
    "pyramid_remaining_target_abs",
    "pyramid_quarter_increment_abs",
    "pyramid_half_increment_abs",
    "pyramid_full_increment_abs",
    "pyramid_correction_abs",
]


def train_one_epoch_multiscale(
    model: Stage2MultiScalePyramidNet,
    loader,
    optimizer: torch.optim.Optimizer,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    names = [
        "total",
        "base_total",
        "hsi_l1",
        "sam",
        *DUAL_NAMES,
        *PYRAMID_NAMES,
        *MONITOR_NAMES,
    ]
    meters = {name: AverageMeter() for name in names}

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=cfg.stage2_lambda_msi_usage > 0,
        )
        base = compute_base_losses(
            model,
            outputs,
            batch,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
        )
        dual = dual_losses(model, outputs, batch["gt"])
        pyramid = pyramid_losses(model, outputs, batch["gt"])

        quarter_objective = (
            pyramid["pyramid_quarter_total"]
            + cfg.pyramid_lambda_observable
            * pyramid["pyramid_quarter_observable"]
            + cfg.pyramid_lambda_null * pyramid["pyramid_quarter_null"]
        )
        half_objective = (
            pyramid["pyramid_half_total"]
            + cfg.pyramid_lambda_observable
            * pyramid["pyramid_half_observable"]
            + cfg.pyramid_lambda_null * pyramid["pyramid_half_null"]
        )
        full_objective = (
            pyramid["pyramid_full_total"]
            + cfg.pyramid_lambda_observable
            * pyramid["pyramid_full_observable"]
            + cfg.pyramid_lambda_null * pyramid["pyramid_full_null"]
        )
        total = (
            base["total"]
            + cfg.dual_lambda_observable * dual["dual_observable_loss"]
            + cfg.dual_lambda_null * dual["dual_null_loss"]
            + cfg.dual_lambda_null_msi_leakage
            * dual["dual_null_msi_leakage"]
            + cfg.pyramid_lambda_quarter * quarter_objective
            + cfg.pyramid_lambda_half * half_objective
            + cfg.pyramid_lambda_full * full_objective
        )
        total.backward()
        if cfg.stage2_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                cfg.stage2_grad_clip,
            )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        values = {
            "total": float(total.detach().item()),
            "base_total": float(base["total"].detach().item()),
            "hsi_l1": float(base["hsi_l1"].detach().item()),
            "sam": float(base["sam"].detach().item()),
            **{name: float(dual[name].detach().item()) for name in DUAL_NAMES},
            **{
                name: float(pyramid[name].detach().item())
                for name in PYRAMID_NAMES
            },
            **monitor_values(model, outputs),
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_multiscale(
    model: Stage2MultiScalePyramidNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    result = evaluate_symmetric(
        model,
        loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    meters = {name: AverageMeter() for name in PYRAMID_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        values = pyramid_losses(model, outputs, batch["gt"])
        batch_size = batch["lr_hsi"].size(0)
        for name in PYRAMID_NAMES:
            meters[name].update(float(values[name].item()), batch_size)
    result.update({name: meter.avg for name, meter in meters.items()})
    return result


def correction_parameters(
    model: Stage2MultiScalePyramidNet,
) -> List[torch.nn.Parameter]:
    parameters: List[torch.nn.Parameter] = []
    for module in (
        model.quarter_branch,
        model.half_branch,
        model.full_correction_branch,
    ):
        parameters.extend(module.parameters())
    return parameters


def split_trainable_parameters(
    model: Stage2MultiScalePyramidNet,
) -> tuple[
    List[torch.nn.Parameter],
    List[torch.nn.Parameter],
    List[torch.nn.Parameter],
]:
    correction = correction_parameters(model)
    correction_ids = {id(parameter) for parameter in correction}

    boundaries = list(model.spectral_boundary_parameters())
    boundary_ids = {id(parameter) for parameter in boundaries}
    for parameter in boundaries:
        parameter.requires_grad_(False)

    source = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in correction_ids
        and id(parameter) not in boundary_ids
    ]
    return correction, source, boundaries


def set_requires_grad(
    parameters: Sequence[torch.nn.Parameter],
    enabled: bool,
) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def cosine_lr(
    base_lr: float,
    progress: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if total_steps <= 1:
        return base_lr
    ratio = min(max(progress / float(total_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def configure_epoch_training(
    optimizer: torch.optim.Optimizer,
    source_parameters: Sequence[torch.nn.Parameter],
    epoch: int,
    cfg,
) -> tuple[float, float, bool]:
    warmup = min(cfg.pyramid_warmup_epochs, cfg.epochs)
    source_enabled = epoch >= warmup
    set_requires_grad(source_parameters, source_enabled)

    if epoch < warmup:
        correction_lr = cfg.pyramid_new_lr
        source_lr = 0.0
    else:
        post_total = max(cfg.epochs - warmup, 1)
        post_epoch = epoch - warmup
        correction_lr = cosine_lr(
            cfg.pyramid_new_lr,
            post_epoch,
            post_total,
            cfg.pyramid_min_lr_ratio,
        )
        source_lr = cosine_lr(
            cfg.pyramid_source_lr,
            post_epoch,
            post_total,
            cfg.pyramid_min_lr_ratio,
        )

    for group in optimizer.param_groups:
        if group.get("group_name") == "correction":
            group["lr"] = correction_lr
        elif group.get("group_name") == "source":
            group["lr"] = source_lr
    return correction_lr, source_lr, source_enabled


@torch.no_grad()
def export_outputs(
    model: Stage2MultiScalePyramidNet,
    loader,
    output_dir: str,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    batch = move_to_device(next(iter(loader)), device)
    outputs = model(
        batch["lr_hsi"],
        batch["hr_msi"],
        compute_zero_msi=True,
    )
    np.savez_compressed(
        os.path.join(output_dir, "stage2_residual_pyramid_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        base_hsi=outputs["base_hsi"].detach().cpu().numpy(),
        anchor_hsi=outputs["anchor_hsi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        source_residual=outputs[
            "pyramid_source_normalized_residual"
        ].detach().cpu().numpy(),
        quarter_correction=outputs[
            "pyramid_quarter_normalized_residual"
        ].detach().cpu().numpy(),
        half_cumulative_correction=outputs[
            "pyramid_half_cumulative_normalized_residual"
        ].detach().cpu().numpy(),
        full_correction=outputs[
            "pyramid_full_correction_normalized_residual"
        ].detach().cpu().numpy(),
        cumulative_correction=outputs[
            "pyramid_correction_normalized_residual"
        ].detach().cpu().numpy(),
        total_residual=outputs[
            "normalized_coefficient_residual"
        ].detach().cpu().numpy(),
        observable_residual=outputs[
            "observable_coefficient_residual"
        ].detach().cpu().numpy(),
        null_residual=outputs["null_coefficient_residual"].detach().cpu().numpy(),
        reliability_map=outputs["reliability_map"].detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_multiscale_args()
    cfg.stage = "multiscale_residual_pyramid"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2MultiScalePyramidNet(
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

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage2_multiscale_residual_pyramid",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_multiscale_residual_pyramid",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage2_multiscale_residual_pyramid",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "residual_pyramid_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "residual_pyramid_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "residual_pyramid_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "residual_pyramid_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    if not cfg.resume:
        source_state = load_symmetric_warm_start(
            model,
            cfg.symmetric_frequency_checkpoint,
            device,
        )
        write_log(
            log_path,
            f"Loaded symmetric-frequency source "
            f"{cfg.symmetric_frequency_checkpoint} at epoch "
            f"{source_state.get('epoch', -1)}; all correction heads are zero.",
        )

    correction_params, source_params, boundary_params = (
        split_trainable_parameters(model)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": correction_params,
                "lr": cfg.pyramid_new_lr,
                "group_name": "correction",
            },
            {
                "params": source_params,
                "lr": 0.0,
                "group_name": "source",
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )

    sam_loss = SAMLoss()
    hsi_degrader = FixedSpatialDegradation(info["n_bands"]).to(device)
    coefficient_degrader = FixedSpatialDegradation(stage1.basis_rank).to(device)

    initial = evaluate_multiscale(
        model,
        test_loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    write_log(
        log_path,
        f"Residual pyramid start | PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg, "
        f"source={initial['pyramid_source_residual_abs']:.5f}, "
        f"remaining={initial['pyramid_remaining_target_abs']:.5f}, "
        f"correction={initial['pyramid_correction_abs']:.5f}, "
        f"source_params={sum(p.numel() for p in source_params) / 1e6:.3f} M, "
        f"correction_params={sum(p.numel() for p in correction_params) / 1e6:.3f} M, "
        f"frozen_boundaries={sum(p.numel() for p in boundary_params)}.",
    )

    csv_fields = [
        "epoch",
        "correction_lr",
        "source_lr",
        "source_enabled",
        "stage2_psnr",
        "stage2_sam",
        "anchor_psnr",
        "anchor_sam",
        "oracle_psnr",
        "oracle_sam",
        "psnr_gain_over_base",
        "stage2_psnr_gain_over_anchor",
        "zero_msi_psnr_drop",
        "remaining_psnr_to_oracle",
        "recoverable_error_fraction",
        *DUAL_NAMES,
        *PYRAMID_NAMES,
        *SYMMETRIC_NAMES,
        *MONITOR_NAMES,
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    best_selection = initial["selection"]
    best_psnr = initial["stage2_psnr"]
    best_sam = initial["stage2_sam"]
    initial_extra = {
        "stage": "multiscale_residual_pyramid",
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.symmetric_frequency_checkpoint,
        "warmup_epochs": cfg.pyramid_warmup_epochs,
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
        correction_lr, source_lr, source_enabled = configure_epoch_training(
            optimizer,
            source_params,
            epoch,
            cfg,
        )
        train_result = train_one_epoch_multiscale(
            model,
            train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val = evaluate_multiscale(
            model,
            test_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )

        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"phase={'joint' if source_enabled else 'correction-only'} | "
            f"PSNR={val['stage2_psnr']:.4f}, SAM={val['stage2_sam']:.4f} deg | "
            f"source gain={val['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB | "
            f"lr=({correction_lr:.2e}, {source_lr:.2e}) | "
            f"remaining={val['pyramid_remaining_target_abs']:.4f}, "
            f"correction={val['pyramid_correction_abs']:.4f} | "
            f"increments=({val['pyramid_quarter_increment_abs']:.4f}, "
            f"{val['pyramid_half_increment_abs']:.4f}, "
            f"{val['pyramid_full_increment_abs']:.4f}).",
        )

        row = {
            "epoch": epoch + 1,
            "correction_lr": correction_lr,
            "source_lr": source_lr,
            "source_enabled": int(source_enabled),
            "stage2_psnr": val["stage2_psnr"],
            "stage2_sam": val["stage2_sam"],
            "anchor_psnr": val["anchor_psnr"],
            "anchor_sam": val["anchor_sam"],
            "oracle_psnr": val["oracle_psnr"],
            "oracle_sam": val["oracle_sam"],
            "psnr_gain_over_base": val["psnr_gain_over_base"],
            "stage2_psnr_gain_over_anchor": val[
                "stage2_psnr_gain_over_anchor"
            ],
            "zero_msi_psnr_drop": val["zero_msi_psnr_drop"],
            "remaining_psnr_to_oracle": val["remaining_psnr_to_oracle"],
            "recoverable_error_fraction": val["recoverable_error_fraction"],
        }
        row.update({name: val[name] for name in DUAL_NAMES})
        row.update({name: val[name] for name in PYRAMID_NAMES})
        row.update({name: val[name] for name in SYMMETRIC_NAMES})
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "multiscale_residual_pyramid",
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.symmetric_frequency_checkpoint,
            "warmup_epochs": cfg.pyramid_warmup_epochs,
            "source_enabled": source_enabled,
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
        if val["stage2_psnr"] > best_psnr:
            best_psnr = val["stage2_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )
        if val["stage2_sam"] < best_sam:
            best_sam = val["stage2_sam"]
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
        map_location=str(device),
        load_optimizer=False,
    )
    final = evaluate_multiscale(
        model,
        test_loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    export_outputs(model, test_loader, output_dir, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Residual pyramid complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg, "
        f"gain over initial={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
