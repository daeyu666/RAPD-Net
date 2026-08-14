"""Controlled Stage-2 experiment: observable-guided masked null-space completion.

Warm-start from a trained Stage2SymmetricFrequencyNet. The observable branch,
symmetric-frequency representation, SRF anchor, exact projectors, losses, and
optimizer settings remain unchanged. During training only the null-space route
receives blockwise-masked symmetric-frequency difference evidence and is
conditioned on the projected observable coefficient residual through pooled
cross-attention.

For a fair ablation, use the same symmetric-frequency source checkpoint and
training schedule as the continued baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_observable_guided_masked_null import (
    Stage2ObservableGuidedMaskedNullNet,
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
    symmetric_diagnostics,
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


NULL_ROUTE_NAMES = [
    "null_configured_mask_ratio",
    "null_cross_gate",
    "null_cross_attention_abs",
    "observable_context_abs",
    "null_hidden_abs",
    "null_effective_cross_ratio",
]


def parse_masked_null_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--masked_null_source_checkpoint",
        type=str,
        default="",
        help=(
            "Trained Stage2SymmetricFrequencyNet checkpoint used as the exact "
            "source for both the continued baseline and this ablation."
        ),
    )
    parser.add_argument("--null_mask_ratio", type=float, default=0.5)
    parser.add_argument("--null_mask_block_size", type=int, default=4)
    parser.add_argument("--null_attention_heads", type=int, default=4)
    parser.add_argument("--null_attention_pool_size", type=int, default=8)

    specific, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_symmetric_args()
    finally:
        sys.argv = original_argv

    cfg.null_mask_ratio = specific.null_mask_ratio
    cfg.null_mask_block_size = specific.null_mask_block_size
    cfg.null_attention_heads = specific.null_attention_heads
    cfg.null_attention_pool_size = specific.null_attention_pool_size
    if specific.masked_null_source_checkpoint:
        cfg.masked_null_source_checkpoint = specific.masked_null_source_checkpoint
    else:
        cfg.masked_null_source_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage2_symmetric_frequency",
            cfg.dataset,
            "symmetric_frequency_best_psnr.pth",
        )
    return cfg


def load_symmetric_warm_start(
    model: Stage2ObservableGuidedMaskedNullNet,
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
    skipped = []
    for key, value in source.items():
        if key in destination:
            if destination[key].shape == value.shape:
                transferable[key] = value
            else:
                shape_mismatch.append(
                    (key, tuple(value.shape), tuple(destination[key].shape))
                )
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(transferable, strict=False)
    allowed_missing_prefix = "observable_to_null."
    problematic_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefix)
    ]
    if unexpected or problematic_missing or shape_mismatch or skipped:
        raise RuntimeError(
            "Masked-null warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"shape_mismatch={shape_mismatch}, skipped={skipped}"
        )

    if float(model.observable_to_null.gate_logit.detach().abs().item()) != 0.0:
        raise RuntimeError("Observable-to-null attention gate must start at zero")
    return state


@torch.no_grad()
def null_route_diagnostics(
    model: Stage2ObservableGuidedMaskedNullNet,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in NULL_ROUTE_NAMES}
    model.eval()
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        gate = float(outputs["null_cross_gate"].item())
        attention_abs = float(outputs["null_cross_attention_abs"].item())
        hidden_abs = float(outputs["null_hidden_abs"].item())
        effective = abs(gate) * attention_abs / max(hidden_abs, 1e-12)
        values = {
            "null_configured_mask_ratio": float(model.null_mask_ratio),
            "null_cross_gate": gate,
            "null_cross_attention_abs": attention_abs,
            "observable_context_abs": float(
                outputs["observable_context_abs"].item()
            ),
            "null_hidden_abs": hidden_abs,
            "null_effective_cross_ratio": effective,
        }
        batch_size = batch["lr_hsi"].size(0)
        for name, value in values.items():
            meters[name].update(value, batch_size)
    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_masked_null(
    model: Stage2ObservableGuidedMaskedNullNet,
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
    result.update(null_route_diagnostics(model, loader, device))
    return result


@torch.no_grad()
def export_outputs(
    model: Stage2ObservableGuidedMaskedNullNet,
    loader,
    output_dir: str,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    batch = move_to_device(next(iter(loader)), device)
    model.eval()
    outputs = model(
        batch["lr_hsi"],
        batch["hr_msi"],
        compute_zero_msi=True,
    )
    np.savez_compressed(
        os.path.join(
            output_dir,
            "stage2_observable_guided_masked_null_outputs.npz",
        ),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        observable_residual=outputs[
            "observable_coefficient_residual"
        ].detach().cpu().numpy(),
        null_residual=outputs[
            "null_coefficient_residual"
        ].detach().cpu().numpy(),
        low_difference=outputs["low_difference_feature"].detach().cpu().numpy(),
        mid_difference=outputs["mid_difference_feature"].detach().cpu().numpy(),
        high_difference=outputs["high_difference_feature"].detach().cpu().numpy(),
        cross_gate=np.asarray(
            [float(outputs["null_cross_gate"].detach().cpu().item())],
            dtype=np.float32,
        ),
    )


def main() -> None:
    cfg = parse_masked_null_args()
    cfg.stage = "observable_guided_masked_null"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2ObservableGuidedMaskedNullNet(
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
        null_mask_ratio=cfg.null_mask_ratio,
        null_mask_block_size=cfg.null_mask_block_size,
        null_attention_heads=cfg.null_attention_heads,
        null_attention_pool_size=cfg.null_attention_pool_size,
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
        "stage2_observable_guided_masked_null",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_observable_guided_masked_null",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage2_observable_guided_masked_null",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "masked_null_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "masked_null_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "masked_null_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "masked_null_last.pth")
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
            cfg.masked_null_source_checkpoint,
            device,
        )
        write_log(
            log_path,
            f"Loaded symmetric source {cfg.masked_null_source_checkpoint} "
            f"at source epoch {int(source_state.get('epoch', 0))}; "
            "cross-attention gate starts at exactly zero and evaluation masking "
            "is disabled.",
        )

    initial = evaluate_masked_null(
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
        f"Masked-null start | PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg, "
        f"mask={cfg.null_mask_ratio:.2f}, block={cfg.null_mask_block_size}, "
        f"cross_gate={initial['null_cross_gate']:+.6f}, "
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
        *NULL_ROUTE_NAMES,
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
        "source_checkpoint": cfg.masked_null_source_checkpoint,
        "null_mask_ratio": cfg.null_mask_ratio,
        "null_mask_block_size": cfg.null_mask_block_size,
        "null_attention_heads": cfg.null_attention_heads,
        "null_attention_pool_size": cfg.null_attention_pool_size,
        "validation": initial,
    }
    save_checkpoint(
        model, optimizer, start_epoch, best_selection, best_path, extra=initial_extra
    )
    save_checkpoint(
        model, optimizer, start_epoch, best_psnr, best_psnr_path, extra=initial_extra
    )
    save_checkpoint(
        model, optimizer, start_epoch, best_sam, best_sam_path, extra=initial_extra
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
        val = evaluate_masked_null(
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
            f"PSNR={val['stage2_psnr']:.4f}, "
            f"SAM={val['stage2_sam']:.4f} deg | "
            f"source gain={val['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB | "
            f"obs/null loss=({val['dual_observable_loss']:.5f}, "
            f"{val['dual_null_loss']:.5f}) | "
            f"cross_gate={val['null_cross_gate']:+.4f}, "
            f"cross_ratio={val['null_effective_cross_ratio']:.4f}.",
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
        row.update({name: val[name] for name in NULL_ROUTE_NAMES})
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": cfg.stage,
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.masked_null_source_checkpoint,
            "null_mask_ratio": cfg.null_mask_ratio,
            "null_mask_block_size": cfg.null_mask_block_size,
            "null_attention_heads": cfg.null_attention_heads,
            "null_attention_pool_size": cfg.null_attention_pool_size,
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
    final = evaluate_masked_null(
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
        f"Masked-null complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg, "
        f"gain over source={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB, "
        f"cross_gate={final['null_cross_gate']:+.4f}.",
    )


if __name__ == "__main__":
    main()
