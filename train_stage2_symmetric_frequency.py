"""Fine-tune Stage 2 with dual-source symmetric SSP frequency differences.

The script strictly loads a trained dual-space checkpoint and changes only the
frequency representation. Both MSI sources use the same encoder and SSP, and
the coefficient predictor receives low/mid/high same-band differences. The SRF
anchor, exact observable/null projectors, dual heads, fusion trunk, and loss
weights are retained.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_symmetric_frequency import Stage2SymmetricFrequencyNet
from train_stage2_coefficients import (
    MONITOR_NAMES,
    FixedSpatialDegradation,
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from train_stage2_dual_space import (
    DUAL_NAMES,
    evaluate_dual,
    train_one_epoch_dual,
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


def parse_symmetric_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument(
        "--dual_space_checkpoint",
        type=str,
        default="./checkpoints/stage2_dual_space/PaviaU/dual_space_best_psnr.pth",
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

    parser.add_argument("--dual_lambda_observable", type=float, default=0.1)
    parser.add_argument("--dual_lambda_null", type=float, default=0.2)
    parser.add_argument("--dual_lambda_null_msi_leakage", type=float, default=0.05)

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

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
        if cfg.dual_space_checkpoint.endswith(
            "stage2_dual_space/PaviaU/dual_space_best_psnr.pth"
        ):
            cfg.dual_space_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "stage2_dual_space",
                cfg.dataset,
                "dual_space_best_psnr.pth",
            )
    return cfg


SYMMETRIC_NAMES = [
    "symmetric_low_abs",
    "symmetric_mid_abs",
    "symmetric_high_abs",
    "symmetric_reliable_high_abs",
    "symmetric_low_share",
    "symmetric_mid_share",
    "symmetric_high_share",
    "physical_freq_low",
    "physical_freq_mid",
    "physical_freq_high",
    "reference_freq_low",
    "reference_freq_mid",
    "reference_freq_high",
    "physical_partition_loss",
    "reference_partition_loss",
]


@torch.no_grad()
def symmetric_diagnostics(
    model: Stage2SymmetricFrequencyNet,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in SYMMETRIC_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        low = float(outputs["low_difference_feature"].abs().mean().item())
        mid = float(outputs["mid_difference_feature"].abs().mean().item())
        high = float(outputs["high_difference_feature"].abs().mean().item())
        reliable_high = float(
            outputs["reliable_high_difference_feature"].abs().mean().item()
        )
        total = max(low + mid + reliable_high, 1e-12)
        physical = outputs["physical_frequency_activation_ratio"].detach()
        reference = outputs["reference_frequency_activation_ratio"].detach()
        values = {
            "symmetric_low_abs": low,
            "symmetric_mid_abs": mid,
            "symmetric_high_abs": high,
            "symmetric_reliable_high_abs": reliable_high,
            "symmetric_low_share": low / total,
            "symmetric_mid_share": mid / total,
            "symmetric_high_share": reliable_high / total,
            "physical_freq_low": float(physical[0].item()),
            "physical_freq_mid": float(physical[1].item()),
            "physical_freq_high": float(physical[2].item()),
            "reference_freq_low": float(reference[0].item()),
            "reference_freq_mid": float(reference[1].item()),
            "reference_freq_high": float(reference[2].item()),
            "physical_partition_loss": float(
                outputs["physical_partition_reconstruction_loss"].item()
            ),
            "reference_partition_loss": float(
                outputs["reference_partition_reconstruction_loss"].item()
            ),
        }
        batch_size = batch["lr_hsi"].size(0)
        for name, value in values.items():
            meters[name].update(value, batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_symmetric(
    model: Stage2SymmetricFrequencyNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    result = evaluate_dual(
        model,
        loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    result.update(symmetric_diagnostics(model, loader, device))
    return result


@torch.no_grad()
def export_outputs(
    model: Stage2SymmetricFrequencyNet,
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
        os.path.join(output_dir, "stage2_symmetric_frequency_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        base_hsi=outputs["base_hsi"].detach().cpu().numpy(),
        anchor_hsi=outputs["anchor_hsi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        physical_low=outputs["physical_low_feature"].detach().cpu().numpy(),
        physical_mid=outputs["physical_mid_feature"].detach().cpu().numpy(),
        physical_high=outputs["physical_high_feature"].detach().cpu().numpy(),
        reference_low=outputs["reference_low_feature"].detach().cpu().numpy(),
        reference_mid=outputs["reference_mid_feature"].detach().cpu().numpy(),
        reference_high=outputs["reference_high_feature"].detach().cpu().numpy(),
        low_difference=outputs["low_difference_feature"].detach().cpu().numpy(),
        mid_difference=outputs["mid_difference_feature"].detach().cpu().numpy(),
        high_difference=outputs["high_difference_feature"].detach().cpu().numpy(),
        reliable_high_difference=outputs[
            "reliable_high_difference_feature"
        ].detach().cpu().numpy(),
        reliability_map=outputs["reliability_map"].detach().cpu().numpy(),
        observable_residual=outputs[
            "observable_coefficient_residual"
        ].detach().cpu().numpy(),
        null_residual=outputs["null_coefficient_residual"].detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_symmetric_args()
    cfg.stage = "symmetric_frequency_coefficient"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2SymmetricFrequencyNet(
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
        "stage2_symmetric_frequency",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_symmetric_frequency",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage2_symmetric_frequency")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "symmetric_frequency_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "symmetric_frequency_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "symmetric_frequency_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "symmetric_frequency_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
    else:
        source_epoch, _ = load_checkpoint(
            model,
            cfg.dual_space_checkpoint,
            optimizer=None,
            strict=True,
            map_location=str(device),
            load_optimizer=False,
        )
        write_log(
            log_path,
            f"Loaded dual-space source {cfg.dual_space_checkpoint} "
            f"at epoch {source_epoch}; state_dict matched strictly.",
        )

    initial = evaluate_symmetric(
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
        f"Symmetric SSP start | PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg, "
        f"diff share=({initial['symmetric_low_share']:.3f}, "
        f"{initial['symmetric_mid_share']:.3f}, "
        f"{initial['symmetric_high_share']:.3f}), "
        f"noise={initial['noise_ratio']:.4f}, "
        f"trainable={count_parameters(model):.3f} M.",
    )

    csv_fields = [
        "epoch",
        "lr",
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
        "stage": "symmetric_frequency_coefficient",
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.dual_space_checkpoint,
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
        train_one_epoch_dual(
            model,
            train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val = evaluate_symmetric(
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
            f"PSNR={val['stage2_psnr']:.4f}, SAM={val['stage2_sam']:.4f} deg | "
            f"source gain={val['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB | "
            f"diff share=({val['symmetric_low_share']:.3f}, "
            f"{val['symmetric_mid_share']:.3f}, "
            f"{val['symmetric_high_share']:.3f}) | "
            f"noise={val['noise_ratio']:.4f}, "
            f"obs/null loss=({val['dual_observable_loss']:.5f}, "
            f"{val['dual_null_loss']:.5f}).",
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
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
        row.update({name: val[name] for name in SYMMETRIC_NAMES})
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "symmetric_frequency_coefficient",
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.dual_space_checkpoint,
            "validation": val,
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
    final = evaluate_symmetric(
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
        f"Symmetric SSP complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg, "
        f"gain over initial={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
