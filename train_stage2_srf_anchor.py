"""Train the controlled Stage-2 SRF analytical-anchor variant.

Only one architectural change is introduced relative to
``train_stage2_coefficients.py``: the deterministic coefficient starting point
is changed from bicubic coefficients to an SRF analytical coefficient anchor.
The frequency encoder, SSP, NSP, single-scale fusion trunk, losses, and training
schedule remain unchanged. This isolates the value of analytical SRF
backprojection before observable/null-space splitting or multi-scale fusion is
introduced.
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
from models.stage2_srf_anchor import Stage2SRFAnchorNet
from train_stage2_coefficients import (
    MONITOR_NAMES,
    FixedSpatialDegradation,
    build_spectral_response,
    evaluate as evaluate_base,
    load_stage1_basis_checkpoint,
    train_one_epoch,
)
from utils import (
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


def parse_anchor_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--anchor_normalized_clip", type=float, default=0.0)

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
    parser.add_argument("--stage2_boundary_lr_multiplier", type=float, default=10.0)
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

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    default_path = "./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth"
    if cfg.stage1_basis_checkpoint == default_path and cfg.dataset != "PaviaU":
        cfg.stage1_basis_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage1_basis",
            cfg.dataset,
            "basis_for_stage2.pth",
        )
    return cfg


@torch.no_grad()
def evaluate_anchor(
    model: Stage2SRFAnchorNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    result = evaluate_base(
        model,
        loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )

    anchor_metrics = MetricAverager()
    total_pixels = 0
    analytic_abs_sum = 0.0
    analytic_max = 0.0
    anchor_msi_mse_sum = 0.0
    base_msi_mse_sum = 0.0
    out_of_range_sum = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        anchor_metrics.update(
            calc_metrics(outputs["anchor_hsi"], batch["gt"], cfg.scale_ratio)
        )
        pixels = batch["gt"].size(0) * batch["gt"].size(2) * batch["gt"].size(3)
        total_pixels += pixels
        normalized = outputs[
            "normalized_analytic_coefficient_residual"
        ].detach()
        analytic_abs_sum += float(normalized.abs().mean().item()) * pixels
        analytic_max = max(analytic_max, float(normalized.abs().max().item()))
        anchor_msi_mse_sum += float(
            F.mse_loss(outputs["anchor_msi"], batch["hr_msi"]).item()
        ) * pixels
        base_msi_mse_sum += float(
            F.mse_loss(outputs["base_msi"], batch["hr_msi"]).item()
        ) * pixels
        out_of_range_sum += float(
            (
                (outputs["anchor_hsi"] < 0.0)
                | (outputs["anchor_hsi"] > 1.0)
            ).float().mean().item()
        ) * pixels

    for name, value in anchor_metrics.average().items():
        result[f"anchor_{name.lower()}"] = value
    total_pixels = max(total_pixels, 1)
    result["analytic_normalized_abs_mean"] = analytic_abs_sum / total_pixels
    result["analytic_normalized_abs_max"] = analytic_max
    result["anchor_out_of_range_ratio"] = out_of_range_sum / total_pixels
    result["anchor_msi_residual_reduction"] = 1.0 - (
        anchor_msi_mse_sum / max(base_msi_mse_sum, 1e-12)
    )
    result["stage2_psnr_gain_over_anchor"] = (
        result["stage2_psnr"] - result["anchor_psnr"]
    )
    result["stage2_sam_gain_over_anchor"] = (
        result["anchor_sam"] - result["stage2_sam"]
    )
    result["anchor_psnr_gain_over_base"] = (
        result["anchor_psnr"] - result["base_psnr"]
    )
    result["anchor_sam_gain_over_base"] = (
        result["base_sam"] - result["anchor_sam"]
    )
    return result


@torch.no_grad()
def export_anchor_artifacts(
    model: Stage2SRFAnchorNet,
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
        os.path.join(output_dir, "stage2_srf_anchor_outputs.npz"),
        lr_hsi=batch["lr_hsi"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        gt=batch["gt"].detach().cpu().numpy(),
        base_hsi=outputs["base_hsi"].detach().cpu().numpy(),
        anchor_hsi=outputs["anchor_hsi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        bicubic_coefficients=outputs["bicubic_coefficients"].detach().cpu().numpy(),
        anchor_coefficients=outputs["anchor_coefficients"].detach().cpu().numpy(),
        corrected_coefficients=outputs["corrected_coefficients"].detach().cpu().numpy(),
        analytic_coefficient_residual=outputs[
            "analytic_coefficient_residual"
        ].detach().cpu().numpy(),
        learned_coefficient_residual=outputs[
            "coefficient_residual"
        ].detach().cpu().numpy(),
        normalized_analytic_coefficient_residual=outputs[
            "normalized_analytic_coefficient_residual"
        ].detach().cpu().numpy(),
        normalized_learned_coefficient_residual=outputs[
            "normalized_coefficient_residual"
        ].detach().cpu().numpy(),
        base_msi=outputs["base_msi"].detach().cpu().numpy(),
        anchor_msi=outputs["anchor_msi"].detach().cpu().numpy(),
        projected_msi=outputs["projected_msi"].detach().cpu().numpy(),
        reliability_map=outputs["reliability_map"].detach().cpu().numpy(),
        reduced_response=model.reduced_response.detach().cpu().numpy(),
        observable_projector=model.observable_projector.detach().cpu().numpy(),
        actual_anchor_ridge=np.asarray(
            float(model.actual_anchor_ridge.detach().item()),
            dtype=np.float32,
        ),
    )


def checkpoint_extra(
    cfg,
    info: dict,
    stage1_state: dict,
    result: Dict[str, float],
) -> dict:
    return {
        "stage": "srf_coefficient_anchor",
        "dataset": cfg.dataset,
        "n_bands": int(info["n_bands"]),
        "n_msi_bands": int(info["n_select_bands"]),
        "basis_rank": int(stage1_state.get("extra", {}).get("basis_rank", -1)),
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "stage1_epoch": int(stage1_state.get("epoch", -1)),
        "anchor_ridge_ratio": cfg.anchor_ridge_ratio,
        "anchor_normalized_clip": cfg.anchor_normalized_clip,
        "feature_channels": cfg.stage2_feature_channels,
        "encoder_blocks": cfg.stage2_encoder_blocks,
        "fusion_channels": cfg.stage2_fusion_channels,
        "fusion_blocks": cfg.stage2_fusion_blocks,
        "validation": result,
    }


def main() -> None:
    cfg = parse_anchor_args()
    cfg.stage = "srf_coefficient_anchor"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, stage1_state = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    spectral_response = build_spectral_response(info).to(device)
    model = Stage2SRFAnchorNet(
        stage1_model=stage1,
        spectral_response=spectral_response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        anchor_normalized_clip=cfg.anchor_normalized_clip,
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
    ).to(device)

    boundary_lr = cfg.lr * cfg.stage2_boundary_lr_multiplier
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.regular_parameters()), "lr": cfg.lr},
            {
                "params": list(model.spectral_boundary_parameters()),
                "lr": boundary_lr,
                "weight_decay": 0.0,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * 0.05,
    )
    sam_loss = SAMLoss()
    hsi_degrader = FixedSpatialDegradation(info["n_bands"]).to(device)
    coefficient_degrader = FixedSpatialDegradation(stage1.basis_rank).to(device)

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage2_srf_anchor",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_srf_anchor",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage2_srf_anchor")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "srf_anchor_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "srf_anchor_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "srf_anchor_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "srf_anchor_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    csv_fields = [
        "epoch",
        "lr",
        "train_l1",
        "train_sam_deg",
        "base_psnr",
        "base_sam",
        "anchor_psnr",
        "anchor_sam",
        "stage2_psnr",
        "stage2_sam",
        "oracle_psnr",
        "oracle_sam",
        "anchor_psnr_gain_over_base",
        "anchor_sam_gain_over_base",
        "stage2_psnr_gain_over_anchor",
        "stage2_sam_gain_over_anchor",
        "psnr_gain_over_base",
        "sam_gain_over_base",
        "zero_msi_psnr_drop",
        "zero_msi_sam_drop",
        "remaining_psnr_to_oracle",
        "remaining_sam_to_oracle",
        "recoverable_error_fraction",
        "anchor_msi_residual_reduction",
        "analytic_normalized_abs_mean",
        "analytic_normalized_abs_max",
        "anchor_out_of_range_ratio",
        "selection",
        *MONITOR_NAMES,
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    start_epoch = 0
    best_selection = float("inf")
    best_psnr = -float("inf")
    best_sam = float("inf")
    if cfg.resume:
        start_epoch, best_selection = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        optimizer.param_groups[0]["lr"] = cfg.lr
        optimizer.param_groups[1]["lr"] = boundary_lr

    write_log(
        log_path,
        f"Stage-2 SRF anchor start | dataset={cfg.dataset}, "
        f"basis_rank={stage1.basis_rank}, ridge_ratio={cfg.anchor_ridge_ratio:.3e}, "
        f"actual_ridge={float(model.actual_anchor_ridge.item()):.6e}, "
        f"clip={cfg.anchor_normalized_clip}, trainable={count_parameters(model):.3f} M.",
    )

    for epoch in range(start_epoch, cfg.epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val_result = evaluate_anchor(
            model,
            test_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        scheduler.step()

        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"anchor=({val_result['anchor_psnr']:.4f} dB, "
            f"{val_result['anchor_sam']:.4f} deg) | "
            f"stage2=({val_result['stage2_psnr']:.4f} dB, "
            f"{val_result['stage2_sam']:.4f} deg) | "
            f"gain over anchor=({val_result['stage2_psnr_gain_over_anchor']:+.4f} dB, "
            f"{val_result['stage2_sam_gain_over_anchor']:+.4f} deg) | "
            f"base gain={val_result['psnr_gain_over_base']:+.4f} dB | "
            f"Zero-MSI drop={val_result['zero_msi_psnr_drop']:+.4f} dB | "
            f"oracle gap={val_result['remaining_psnr_to_oracle']:.4f} dB | "
            f"noise={val_result['noise_ratio']:.4f}, "
            f"sat={val_result['residual_saturation_ratio']:.4f}.",
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_l1": train_result["hsi_l1"],
            "train_sam_deg": train_result["sam"] * 180.0 / math.pi,
            "base_psnr": val_result["base_psnr"],
            "base_sam": val_result["base_sam"],
            "anchor_psnr": val_result["anchor_psnr"],
            "anchor_sam": val_result["anchor_sam"],
            "stage2_psnr": val_result["stage2_psnr"],
            "stage2_sam": val_result["stage2_sam"],
            "oracle_psnr": val_result["oracle_psnr"],
            "oracle_sam": val_result["oracle_sam"],
            "anchor_psnr_gain_over_base": val_result[
                "anchor_psnr_gain_over_base"
            ],
            "anchor_sam_gain_over_base": val_result[
                "anchor_sam_gain_over_base"
            ],
            "stage2_psnr_gain_over_anchor": val_result[
                "stage2_psnr_gain_over_anchor"
            ],
            "stage2_sam_gain_over_anchor": val_result[
                "stage2_sam_gain_over_anchor"
            ],
            "psnr_gain_over_base": val_result["psnr_gain_over_base"],
            "sam_gain_over_base": val_result["sam_gain_over_base"],
            "zero_msi_psnr_drop": val_result["zero_msi_psnr_drop"],
            "zero_msi_sam_drop": val_result["zero_msi_sam_drop"],
            "remaining_psnr_to_oracle": val_result[
                "remaining_psnr_to_oracle"
            ],
            "remaining_sam_to_oracle": val_result[
                "remaining_sam_to_oracle"
            ],
            "recoverable_error_fraction": val_result[
                "recoverable_error_fraction"
            ],
            "anchor_msi_residual_reduction": val_result[
                "anchor_msi_residual_reduction"
            ],
            "analytic_normalized_abs_mean": val_result[
                "analytic_normalized_abs_mean"
            ],
            "analytic_normalized_abs_max": val_result[
                "analytic_normalized_abs_max"
            ],
            "anchor_out_of_range_ratio": val_result[
                "anchor_out_of_range_ratio"
            ],
            "selection": val_result["selection"],
        }
        row.update({name: val_result[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = checkpoint_extra(cfg, info, stage1_state, val_result)
        if val_result["selection"] < best_selection:
            best_selection = val_result["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                best_path,
                extra=extra,
            )
        if val_result["stage2_psnr"] > best_psnr:
            best_psnr = val_result["stage2_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )
        if val_result["stage2_sam"] < best_sam:
            best_sam = val_result["stage2_sam"]
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
    final_result = evaluate_anchor(
        model,
        test_loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    export_anchor_artifacts(model, test_loader, output_dir, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_result, file, indent=2, ensure_ascii=False)

    write_log(
        log_path,
        f"Stage-2 SRF anchor complete | PSNR={final_result['stage2_psnr']:.4f}, "
        f"SAM={final_result['stage2_sam']:.4f} deg, "
        f"gain over anchor={final_result['stage2_psnr_gain_over_anchor']:+.4f} dB, "
        f"gain over base={final_result['psnr_gain_over_base']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
