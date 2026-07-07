"""Fine-tune Stage 2 with observable/null-space dual coefficient heads.

The script warm-starts from a trained single-head SRF-anchor checkpoint. The
single residual head is cloned into both new branches. Because their outputs are
projected by complementary exact projectors, the initial total residual matches
the trained single-head prediction up to floating-point error.

No multi-scale fusion or branch-specific feature routing is introduced here.
This is a controlled test of coefficient-space decomposition only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_dual_space import Stage2DualSpaceNet
from train_stage2_coefficients import (
    MONITOR_NAMES,
    FixedSpatialDegradation,
    build_spectral_response,
    compute_losses as compute_base_losses,
    load_stage1_basis_checkpoint,
    monitor_values,
)
from train_stage2_srf_anchor import evaluate_anchor
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


def parse_dual_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument(
        "--single_head_checkpoint",
        type=str,
        default="./checkpoints/stage2_srf_anchor/PaviaU/srf_anchor_best_psnr.pth",
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
        if cfg.single_head_checkpoint.endswith(
            "stage2_srf_anchor/PaviaU/srf_anchor_best_psnr.pth"
        ):
            cfg.single_head_checkpoint = os.path.join(
                cfg.checkpoint_root,
                "stage2_srf_anchor",
                cfg.dataset,
                "srf_anchor_best_psnr.pth",
            )
    return cfg


def load_single_head_warm_start(
    model: Stage2DualSpaceNet,
    path: str,
    device: torch.device,
) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Single-head checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    source = state.get("model", state)

    head_weight = source.get("normalized_residual_head.weight")
    head_bias = source.get("normalized_residual_head.bias")
    if head_weight is None:
        raise KeyError(
            "Checkpoint does not contain normalized_residual_head.weight; "
            "use a Stage2SRFAnchorNet checkpoint"
        )

    destination = model.state_dict()
    transferable = {}
    skipped = []
    for key, value in source.items():
        if key.startswith("normalized_residual_head."):
            continue
        if key in destination and destination[key].shape == value.shape:
            transferable[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    model.initialize_dual_heads_from_single(head_weight, head_bias)

    allowed_missing_prefixes = (
        "observable_normalized_residual_head.",
        "null_normalized_residual_head.",
        "exact_observable_projector",
        "exact_null_projector",
        "observable_singular_values",
        "observable_rank",
    )
    problematic_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or problematic_missing:
        raise RuntimeError(
            "Warm-start mismatch: "
            f"unexpected={unexpected}, missing={problematic_missing}, "
            f"skipped={skipped}"
        )
    return state


def project(
    projector: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


def dual_losses(
    model: Stage2DualSpaceNet,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    scale = outputs["coefficient_scale"].view(1, -1, 1, 1)
    with torch.no_grad():
        target_coefficients = model.stage1.encode(
            gt,
            basis=outputs["basis"],
        )
        target_residual = target_coefficients - outputs["anchor_coefficients"]
        target_observable = project(
            model.exact_observable_projector.to(target_residual),
            target_residual,
        )
        target_null = project(
            model.exact_null_projector.to(target_residual),
            target_residual,
        )
        target_observable_normalized = target_observable / scale
        target_null_normalized = target_null / scale

    observable_loss = F.smooth_l1_loss(
        outputs["normalized_observable_coefficient_residual"],
        target_observable_normalized,
        beta=0.25,
    )
    null_loss = F.smooth_l1_loss(
        outputs["normalized_null_coefficient_residual"],
        target_null_normalized,
        beta=0.25,
    )
    null_msi = torch.einsum(
        "mr,nrhw->nmhw",
        model.reduced_response.to(outputs["null_coefficient_residual"]),
        outputs["null_coefficient_residual"],
    )
    null_msi_leakage = null_msi.abs().mean()

    target_energy = target_residual.square().mean().clamp_min(1e-12)
    observable_energy_share = target_observable.square().mean() / target_energy
    null_energy_share = target_null.square().mean() / target_energy
    prediction_energy = outputs["coefficient_residual"].square().mean().clamp_min(
        1e-12
    )
    predicted_observable_share = (
        outputs["observable_coefficient_residual"].square().mean()
        / prediction_energy
    )
    predicted_null_share = (
        outputs["null_coefficient_residual"].square().mean()
        / prediction_energy
    )

    return {
        "dual_observable_loss": observable_loss,
        "dual_null_loss": null_loss,
        "dual_null_msi_leakage": null_msi_leakage,
        "target_observable_energy_share": observable_energy_share.detach(),
        "target_null_energy_share": null_energy_share.detach(),
        "predicted_observable_energy_share": predicted_observable_share.detach(),
        "predicted_null_energy_share": predicted_null_share.detach(),
    }


DUAL_NAMES = [
    "dual_observable_loss",
    "dual_null_loss",
    "dual_null_msi_leakage",
    "target_observable_energy_share",
    "target_null_energy_share",
    "predicted_observable_energy_share",
    "predicted_null_energy_share",
]


def train_one_epoch_dual(
    model: Stage2DualSpaceNet,
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
        total = (
            base["total"]
            + cfg.dual_lambda_observable * dual["dual_observable_loss"]
            + cfg.dual_lambda_null * dual["dual_null_loss"]
            + cfg.dual_lambda_null_msi_leakage
            * dual["dual_null_msi_leakage"]
        )
        total.backward()
        if cfg.stage2_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
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
            **monitor_values(model, outputs),
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate_dual(
    model: Stage2DualSpaceNet,
    loader,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    result = evaluate_anchor(
        model,
        loader,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
    )
    meters = {name: AverageMeter() for name in DUAL_NAMES}
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=False,
        )
        losses = dual_losses(model, outputs, batch["gt"])
        batch_size = batch["lr_hsi"].size(0)
        for name in DUAL_NAMES:
            meters[name].update(float(losses[name].item()), batch_size)
    result.update({name: meter.avg for name, meter in meters.items()})
    result.update(
        {
            name: float(value.detach().item())
            for name, value in model.projector_statistics().items()
        }
    )
    result["observable_rank"] = int(model.observable_rank.item())
    return result


@torch.no_grad()
def export_outputs(
    model: Stage2DualSpaceNet,
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
        os.path.join(output_dir, "stage2_dual_space_outputs.npz"),
        gt=batch["gt"].detach().cpu().numpy(),
        hr_msi=batch["hr_msi"].detach().cpu().numpy(),
        base_hsi=outputs["base_hsi"].detach().cpu().numpy(),
        anchor_hsi=outputs["anchor_hsi"].detach().cpu().numpy(),
        stage2_hsi=outputs["reconstructed_hsi"].detach().cpu().numpy(),
        zero_msi_hsi=outputs["zero_msi_hsi"].detach().cpu().numpy(),
        observable_residual=outputs[
            "observable_coefficient_residual"
        ].detach().cpu().numpy(),
        null_residual=outputs[
            "null_coefficient_residual"
        ].detach().cpu().numpy(),
        total_residual=outputs["coefficient_residual"].detach().cpu().numpy(),
        observable_projector=model.exact_observable_projector.detach().cpu().numpy(),
        null_projector=model.exact_null_projector.detach().cpu().numpy(),
        singular_values=model.observable_singular_values.detach().cpu().numpy(),
    )


def main() -> None:
    cfg = parse_dual_args()
    cfg.stage = "dual_space_coefficient"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, stage1_state = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2DualSpaceNet(
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
    single_state = load_single_head_warm_start(
        model,
        cfg.single_head_checkpoint,
        device,
    )

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
        "stage2_dual_space",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_dual_space",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage2_dual_space")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "dual_space_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "dual_space_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "dual_space_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "dual_space_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    initial = evaluate_dual(
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
        f"Dual-space warm start | source={cfg.single_head_checkpoint} "
        f"(epoch {single_state.get('epoch', -1)}), "
        f"rank={initial['observable_rank']}/{stage1.basis_rank}, "
        f"PSNR={initial['stage2_psnr']:.4f}, SAM={initial['stage2_sam']:.4f} deg, "
        f"complement_error={initial['projector_complement_error']:.3e}, "
        f"orthogonality_error={initial['projector_orthogonality_error']:.3e}.",
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
        *MONITOR_NAMES,
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        csv_fields,
    )

    best_selection = float("inf")
    best_psnr = initial["stage2_psnr"]
    best_sam = initial["stage2_sam"]
    for epoch in range(cfg.epochs):
        train_result = train_one_epoch_dual(
            model,
            train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val = evaluate_dual(
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
            f"anchor gain={val['stage2_psnr_gain_over_anchor']:+.4f} dB | "
            f"obs/null loss=({val['dual_observable_loss']:.5f}, "
            f"{val['dual_null_loss']:.5f}) | "
            f"target energy=({val['target_observable_energy_share']:.3f}, "
            f"{val['target_null_energy_share']:.3f}) | "
            f"pred energy=({val['predicted_observable_energy_share']:.3f}, "
            f"{val['predicted_null_energy_share']:.3f}) | "
            f"null MSI leak={val['dual_null_msi_leakage']:.3e}.",
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
        row.update({name: val[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "dual_space_coefficient",
            "dataset": cfg.dataset,
            "basis_rank": stage1.basis_rank,
            "observable_rank": int(model.observable_rank.item()),
            "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
            "single_head_checkpoint": cfg.single_head_checkpoint,
            "single_head_epoch": int(single_state.get("epoch", -1)),
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
    final = evaluate_dual(
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
        f"Dual-space complete | PSNR={final['stage2_psnr']:.4f}, "
        f"SAM={final['stage2_sam']:.4f} deg, "
        f"source gain={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
