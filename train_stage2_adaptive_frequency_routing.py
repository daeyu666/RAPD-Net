"""Controlled Stage-2 experiment: adaptive frequency routing.

The experiment warm-starts from a trained Stage2SymmetricFrequencyNet and
changes only the frequency routing rule:

    static/channel-wise boundaries
        -> sample-conditioned boundary offsets driven by
           physical MSI, reference MSI, and |reference - physical| features.

NSP is bypassed and all symmetric frequency differences are preserved. SRF
anchor, exact observable/null-space projectors, dual coefficient heads, fusion
trunk, training losses, optimizer settings, and data pipeline remain unchanged.

For a fair ablation, use the exact same source checkpoint, epoch count, learning
rate, and scheduler as the continued symmetric-frequency baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_adaptive_frequency_routing import (
    Stage2AdaptiveFrequencyRoutingNet,
)
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
from train_stage2_symmetric_frequency import (
    SYMMETRIC_NAMES,
    parse_symmetric_args,
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


ADAPTIVE_NAMES = [
    "adaptive_low_shift_abs",
    "adaptive_high_shift_abs",
    "adaptive_low_shift_channel_std",
    "adaptive_high_shift_channel_std",
    "adaptive_tau_low_sample_std",
    "adaptive_tau_high_sample_std",
    "adaptive_gap_mean",
    "adaptive_low_clamp_ratio",
    "adaptive_high_clamp_ratio",
]


def parse_adaptive_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--adaptive_source_checkpoint",
        type=str,
        default="",
        help=(
            "Exact symmetric-frequency checkpoint used to start the fair "
            "continued-training baseline. If omitted, use the standard "
            "symmetric_frequency_best_psnr.pth path."
        ),
    )
    parser.add_argument(
        "--adaptive_boundary_hidden_channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--adaptive_boundary_max_shift",
        type=float,
        default=2.0,
        help="Maximum absolute per-sample boundary offset in radial-band units.",
    )

    adaptive, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_symmetric_args()
    finally:
        sys.argv = original_argv

    cfg.adaptive_boundary_hidden_channels = (
        adaptive.adaptive_boundary_hidden_channels
    )
    cfg.adaptive_boundary_max_shift = adaptive.adaptive_boundary_max_shift
    if adaptive.adaptive_source_checkpoint:
        cfg.adaptive_source_checkpoint = adaptive.adaptive_source_checkpoint
    else:
        cfg.adaptive_source_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage2_symmetric_frequency",
            cfg.dataset,
            "symmetric_frequency_best_psnr.pth",
        )
    return cfg


def load_symmetric_warm_start(
    model: Stage2AdaptiveFrequencyRoutingNet,
    path: str,
    device: torch.device,
) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Symmetric source checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)

    source = state.get("model", state)
    destination = model.state_dict()
    transferable = {}
    shape_mismatch = []
    for key, value in source.items():
        if key in destination and destination[key].shape == value.shape:
            transferable[key] = value
        else:
            shape_mismatch.append(key)

    missing, unexpected = model.load_state_dict(transferable, strict=False)
    allowed_missing_prefix = "reliability.boundary_predictor."
    problematic_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefix)
    ]
    if unexpected or problematic_missing or shape_mismatch:
        raise RuntimeError(
            "Adaptive warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"shape_mismatch={shape_mismatch}"
        )

    final = model.reliability.boundary_predictor.net[-1]
    if final.weight.detach().abs().max().item() != 0.0:
        raise RuntimeError("Adaptive boundary predictor did not start at zero")
    if final.bias.detach().abs().max().item() != 0.0:
        raise RuntimeError("Adaptive boundary predictor bias did not start at zero")
    return state


@torch.no_grad()
def adaptive_frequency_diagnostics(
    model: Stage2AdaptiveFrequencyRoutingNet,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    names = [*SYMMETRIC_NAMES, *ADAPTIVE_NAMES]
    meters = {name: AverageMeter() for name in names}
    tau_low_sample_means: List[float] = []
    tau_high_sample_means: List[float] = []

    max_low = float(model.reliability.spectral_splitter.num_frequency_bands - 2)
    max_high = float(model.reliability.spectral_splitter.num_frequency_bands - 1)

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        batch_size = batch["lr_hsi"].size(0)

        low = float(outputs["low_difference_feature"].abs().mean().item())
        mid = float(outputs["mid_difference_feature"].abs().mean().item())
        high = float(outputs["high_difference_feature"].abs().mean().item())
        total = max(low + mid + high, 1e-12)

        physical = outputs["physical_frequency_activation_ratio"].detach()
        reference = outputs["reference_frequency_activation_ratio"].detach()
        low_shift = outputs["boundary_low_shift"].detach().float()
        high_shift = outputs["boundary_high_shift"].detach().float()
        tau_low = outputs["tau_low"].detach().float()
        tau_high = outputs["tau_high"].detach().float()
        gap = outputs["adaptive_boundary_gap"].detach().float()

        tau_low_sample_means.extend(
            tau_low.mean(dim=1).detach().cpu().tolist()
        )
        tau_high_sample_means.extend(
            tau_high.mean(dim=1).detach().cpu().tolist()
        )

        low_clamped = ((tau_low <= 1e-6) | (tau_low >= max_low - 1e-6)).float()
        high_clamped = (
            (tau_high <= tau_low + 1.0 + 1e-6)
            | (tau_high >= max_high - 1e-6)
        ).float()

        values = {
            "symmetric_low_abs": low,
            "symmetric_mid_abs": mid,
            "symmetric_high_abs": high,
            "symmetric_reliable_high_abs": high,
            "symmetric_low_share": low / total,
            "symmetric_mid_share": mid / total,
            "symmetric_high_share": high / total,
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
            "adaptive_low_shift_abs": float(low_shift.abs().mean().item()),
            "adaptive_high_shift_abs": float(high_shift.abs().mean().item()),
            "adaptive_low_shift_channel_std": float(
                low_shift.std(dim=1, unbiased=False).mean().item()
            ),
            "adaptive_high_shift_channel_std": float(
                high_shift.std(dim=1, unbiased=False).mean().item()
            ),
            "adaptive_gap_mean": float(gap.mean().item()),
            "adaptive_low_clamp_ratio": float(low_clamped.mean().item()),
            "adaptive_high_clamp_ratio": float(high_clamped.mean().item()),
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)

    result = {name: meter.avg for name, meter in meters.items()}
    if tau_low_sample_means:
        result["adaptive_tau_low_sample_std"] = float(
            np.std(np.asarray(tau_low_sample_means, dtype=np.float64))
        )
        result["adaptive_tau_high_sample_std"] = float(
            np.std(np.asarray(tau_high_sample_means, dtype=np.float64))
        )
    else:
        result["adaptive_tau_low_sample_std"] = 0.0
        result["adaptive_tau_high_sample_std"] = 0.0
    return result


@torch.no_grad()
def evaluate_adaptive(
    model: Stage2AdaptiveFrequencyRoutingNet,
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
    result.update(adaptive_frequency_diagnostics(model, loader, device))
    return result


@torch.no_grad()
def export_outputs(
    model: Stage2AdaptiveFrequencyRoutingNet,
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
        os.path.join(output_dir, "stage2_adaptive_frequency_routing_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        base_msi=outputs["base_msi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        low_difference=outputs["low_difference_feature"].detach().cpu().numpy(),
        mid_difference=outputs["mid_difference_feature"].detach().cpu().numpy(),
        high_difference=outputs["high_difference_feature"].detach().cpu().numpy(),
        base_tau_low=outputs["base_tau_low"].detach().cpu().numpy(),
        base_tau_high=outputs["base_tau_high"].detach().cpu().numpy(),
        tau_low=outputs["tau_low"].detach().cpu().numpy(),
        tau_high=outputs["tau_high"].detach().cpu().numpy(),
        low_shift=outputs["boundary_low_shift"].detach().cpu().numpy(),
        high_shift=outputs["boundary_high_shift"].detach().cpu().numpy(),
        observable_residual=outputs[
            "observable_coefficient_residual"
        ].detach().cpu().numpy(),
        null_residual=outputs["null_coefficient_residual"].detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_adaptive_args()
    cfg.stage = "adaptive_frequency_routing"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2AdaptiveFrequencyRoutingNet(
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
        adaptive_boundary_hidden_channels=cfg.adaptive_boundary_hidden_channels,
        adaptive_boundary_max_shift=cfg.adaptive_boundary_max_shift,
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
        "stage2_adaptive_frequency_routing",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_adaptive_frequency_routing",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage2_adaptive_frequency_routing")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "adaptive_frequency_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "adaptive_frequency_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "adaptive_frequency_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "adaptive_frequency_last.pth")
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
        source_state = load_symmetric_warm_start(
            model,
            cfg.adaptive_source_checkpoint,
            device,
        )
        source_epoch = int(source_state.get("epoch", 0))
        write_log(
            log_path,
            f"Loaded symmetric source {cfg.adaptive_source_checkpoint} "
            f"at source epoch {source_epoch}; new boundary predictor starts "
            "with exactly zero offsets.",
        )

    initial = evaluate_adaptive(
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
        f"Adaptive routing start | PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg | "
        f"shift=({initial['adaptive_low_shift_abs']:.4f}, "
        f"{initial['adaptive_high_shift_abs']:.4f}) bands | "
        f"sample std=({initial['adaptive_tau_low_sample_std']:.4f}, "
        f"{initial['adaptive_tau_high_sample_std']:.4f}) | "
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
        *ADAPTIVE_NAMES,
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
        "stage": cfg.stage,
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.adaptive_source_checkpoint,
        "adaptive_boundary_hidden_channels": cfg.adaptive_boundary_hidden_channels,
        "adaptive_boundary_max_shift": cfg.adaptive_boundary_max_shift,
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
        val = evaluate_adaptive(
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
            f"shift=({val['adaptive_low_shift_abs']:.3f}, "
            f"{val['adaptive_high_shift_abs']:.3f}) | "
            f"sample std=({val['adaptive_tau_low_sample_std']:.3f}, "
            f"{val['adaptive_tau_high_sample_std']:.3f}) | "
            f"obs/null=({val['dual_observable_loss']:.5f}, "
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
        row.update({name: val[name] for name in ADAPTIVE_NAMES})
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": cfg.stage,
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.adaptive_source_checkpoint,
            "adaptive_boundary_hidden_channels": cfg.adaptive_boundary_hidden_channels,
            "adaptive_boundary_max_shift": cfg.adaptive_boundary_max_shift,
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
    final = evaluate_adaptive(
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
        f"Adaptive routing complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg, "
        f"gain over source={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB | "
        f"final sample boundary std=("
        f"{final['adaptive_tau_low_sample_std']:.4f}, "
        f"{final['adaptive_tau_high_sample_std']:.4f}).",
    )


if __name__ == "__main__":
    main()
