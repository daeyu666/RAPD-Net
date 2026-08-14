"""Train observable-to-null cross-space routing for RAPD-Net Stage 2.

This is a controlled ablation of the information route only. It warm-starts from
an exact Stage2SymmetricFrequencyNet checkpoint and keeps the symmetric-frequency
representation, SRF anchor, exact projectors, losses, data pipeline, and
observable branch unchanged. The new null branch has an independent fusion trunk
and receives detached observable coefficient context through pooled
cross-attention. No masking is used.
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
from models.stage2_observable_to_null_routing import (
    Stage2ObservableToNullRoutingNet,
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


ROUTING_NAMES = [
    "null_cross_gate",
    "null_cross_attention_abs",
    "observable_context_abs",
    "null_hidden_abs",
    "null_effective_cross_ratio",
    "cross_trunk_parameter_l1",
    "cross_trunk_parameter_max",
]


def parse_routing_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cross_space_source_checkpoint",
        type=str,
        default="",
        help=(
            "Exact trained Stage2SymmetricFrequencyNet checkpoint used by the "
            "continued-training baseline."
        ),
    )
    parser.add_argument("--null_attention_heads", type=int, default=4)
    parser.add_argument("--null_attention_pool_size", type=int, default=8)
    parser.add_argument("--null_cross_init_gate", type=float, default=0.1)

    specific, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_symmetric_args()
    finally:
        sys.argv = original_argv

    cfg.null_attention_heads = specific.null_attention_heads
    cfg.null_attention_pool_size = specific.null_attention_pool_size
    cfg.null_cross_init_gate = specific.null_cross_init_gate
    if specific.cross_space_source_checkpoint:
        cfg.cross_space_source_checkpoint = specific.cross_space_source_checkpoint
    else:
        cfg.cross_space_source_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage2_symmetric_frequency",
            cfg.dataset,
            "symmetric_frequency_best_psnr.pth",
        )
    return cfg


def load_symmetric_warm_start(
    model: Stage2ObservableToNullRoutingNet,
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
    allowed_missing_prefixes = (
        "null_fusion_trunk.",
        "observable_to_null.",
    )
    problematic_missing = [
        key
        for key in missing
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or problematic_missing or shape_mismatch or skipped:
        raise RuntimeError(
            "Cross-space warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"shape_mismatch={shape_mismatch}, skipped={skipped}"
        )

    model.synchronize_null_trunk_from_observable()

    actual_gate = float(model.observable_to_null.gate_value().detach().item())
    expected_gate = float(model.observable_to_null.init_gate)
    if abs(actual_gate - expected_gate) > 1e-6:
        raise RuntimeError(
            f"Cross gate initialized to {actual_gate}, expected {expected_gate}"
        )

    divergence = model.trunk_divergence()
    if float(divergence["cross_trunk_parameter_max"].item()) > 1e-8:
        raise RuntimeError("Null fusion trunk did not exactly match source trunk")
    return state


@torch.no_grad()
def routing_diagnostics(
    model: Stage2ObservableToNullRoutingNet,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in ROUTING_NAMES[:-2]}
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

    result = {name: meter.avg for name, meter in meters.items()}
    result.update(
        {
            name: float(value.detach().item())
            for name, value in model.trunk_divergence().items()
        }
    )
    return result


@torch.no_grad()
def evaluate_routing(
    model: Stage2ObservableToNullRoutingNet,
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
    result.update(routing_diagnostics(model, loader, device))
    return result


@torch.no_grad()
def evaluate_zero_gate_equivalence(
    model: Stage2ObservableToNullRoutingNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    saved_logit = model.observable_to_null.gate_logit.detach().clone()
    model.observable_to_null.gate_logit.zero_()
    try:
        result = evaluate_dual(
            model,
            loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
    finally:
        model.observable_to_null.gate_logit.copy_(saved_logit)
    return result


@torch.no_grad()
def export_outputs(
    model: Stage2ObservableToNullRoutingNet,
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
            "stage2_observable_to_null_routing_outputs.npz",
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
        low_difference=outputs[
            "low_difference_feature"
        ].detach().cpu().numpy(),
        mid_difference=outputs[
            "mid_difference_feature"
        ].detach().cpu().numpy(),
        high_difference=outputs[
            "high_difference_feature"
        ].detach().cpu().numpy(),
        cross_gate=np.asarray(
            [float(outputs["null_cross_gate"].detach().cpu().item())],
            dtype=np.float32,
        ),
    )


def main() -> None:
    cfg = parse_routing_args()
    cfg.stage = "observable_to_null_routing"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2ObservableToNullRoutingNet(
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
        null_attention_heads=cfg.null_attention_heads,
        null_attention_pool_size=cfg.null_attention_pool_size,
        null_cross_init_gate=cfg.null_cross_init_gate,
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
        "stage2_observable_to_null_routing",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_observable_to_null_routing",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage2_observable_to_null_routing",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "cross_space_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "cross_space_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "cross_space_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "cross_space_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    start_epoch = 0
    source_equivalence = None
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
            cfg.cross_space_source_checkpoint,
            device,
        )
        source_equivalence = evaluate_zero_gate_equivalence(
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
            f"Loaded symmetric source {cfg.cross_space_source_checkpoint} "
            f"at source epoch {int(source_state.get('epoch', 0))}; "
            f"zero-gate equivalence PSNR={source_equivalence['stage2_psnr']:.4f}, "
            f"SAM={source_equivalence['stage2_sam']:.4f} deg; "
            f"training gate starts at {cfg.null_cross_init_gate:.3f}.",
        )

    initial = evaluate_routing(
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
        f"Cross-space start | PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg | "
        f"gate={initial['null_cross_gate']:+.6f}, "
        f"effective={initial['null_effective_cross_ratio']:.6f}, "
        f"trunk_l1={initial['cross_trunk_parameter_l1']:.3e}, "
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
        *ROUTING_NAMES,
        *MONITOR_NAMES,
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    best_selection = initial["selection"]
    best_psnr = initial["stage2_psnr"]
    best_sam = initial["stage2_sam"]
    source_psnr = (
        source_equivalence["stage2_psnr"]
        if source_equivalence is not None
        else initial["stage2_psnr"]
    )
    source_sam = (
        source_equivalence["stage2_sam"]
        if source_equivalence is not None
        else initial["stage2_sam"]
    )
    initial_extra = {
        "stage": cfg.stage,
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.cross_space_source_checkpoint,
        "source_equivalence_psnr": source_psnr,
        "source_equivalence_sam": source_sam,
        "null_attention_heads": cfg.null_attention_heads,
        "null_attention_pool_size": cfg.null_attention_pool_size,
        "null_cross_init_gate": cfg.null_cross_init_gate,
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
        val = evaluate_routing(
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
            f"source gain={val['stage2_psnr'] - source_psnr:+.4f} dB, "
            f"SAM change={val['stage2_sam'] - source_sam:+.4f} deg | "
            f"obs/null loss=({val['dual_observable_loss']:.5f}, "
            f"{val['dual_null_loss']:.5f}) | "
            f"gate={val['null_cross_gate']:+.6f}, "
            f"effective={val['null_effective_cross_ratio']:.6f}, "
            f"trunk_l1={val['cross_trunk_parameter_l1']:.3e}.",
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
        row.update({name: val[name] for name in ROUTING_NAMES})
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": cfg.stage,
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.cross_space_source_checkpoint,
            "source_equivalence_psnr": source_psnr,
            "source_equivalence_sam": source_sam,
            "null_attention_heads": cfg.null_attention_heads,
            "null_attention_pool_size": cfg.null_attention_pool_size,
            "null_cross_init_gate": cfg.null_cross_init_gate,
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
    final = evaluate_routing(
        model,
        test_loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    final["source_equivalence_psnr"] = source_psnr
    final["source_equivalence_sam"] = source_sam
    final["psnr_gain_over_source"] = final["stage2_psnr"] - source_psnr
    final["sam_change_from_source"] = final["stage2_sam"] - source_sam

    export_outputs(model, test_loader, output_dir, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final, file, indent=2, ensure_ascii=False)

    write_log(
        log_path,
        f"Cross-space complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg | "
        f"gain over source={final['psnr_gain_over_source']:+.4f} dB, "
        f"SAM change={final['sam_change_from_source']:+.4f} deg | "
        f"gate={final['null_cross_gate']:+.6f}, "
        f"effective={final['null_effective_cross_ratio']:.6f}.",
    )


if __name__ == "__main__":
    main()
