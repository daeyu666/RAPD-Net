"""Train RAPD-Net Stage 2 with LR-null local tangent-manifold constraints.

The experiment follows the GT tangent-oracle diagnosis.  The analytical SRF
anchor handles the MSI-observable coefficient subspace.  The remaining null
component starts from bicubic LR-HSI coefficients.  Local LR-null differences
supply a fixed d-dimensional spectral tangent basis at each HR pixel; the
trainable network predicts only bounded coordinates inside that basis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage2_null_tangent_manifold import Stage2NullTangentManifoldNet
from train_stage2_coefficients import (
    FixedSpatialDegradation,
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def first_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def parse_tangent_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--projector_tolerance", type=float, default=1e-6)

    parser.add_argument("--tangent_dimension", type=int, default=4)
    parser.add_argument("--tangent_kernel_size", type=int, default=5)
    parser.add_argument("--tangent_dilation", type=int, default=2)
    parser.add_argument("--tangent_chunk_pixels", type=int, default=2048)
    parser.add_argument("--tangent_amplitude_multiplier", type=float, default=8.0)
    parser.add_argument("--tangent_predictor_hidden", type=int, default=96)
    parser.add_argument("--tangent_predictor_blocks", type=int, default=4)
    parser.add_argument("--tangent_grad_clip", type=float, default=1.0)
    parser.add_argument("--tangent_diagnose_only", action="store_true")

    parser.add_argument("--tangent_lambda_l1", type=float, default=1.0)
    parser.add_argument("--tangent_lambda_sam", type=float, default=0.3)
    parser.add_argument("--tangent_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--tangent_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--tangent_lambda_coordinate", type=float, default=0.5)
    parser.add_argument("--tangent_lambda_residual", type=float, default=0.3)
    parser.add_argument("--tangent_lambda_lr_hsi", type=float, default=0.2)
    parser.add_argument("--tangent_lambda_lr_null", type=float, default=0.1)

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

    if cfg.tangent_dimension < 1:
        raise ValueError("tangent_dimension must be positive")
    if cfg.tangent_kernel_size < 3 or cfg.tangent_kernel_size % 2 == 0:
        raise ValueError("tangent_kernel_size must be an odd integer >= 3")
    if cfg.tangent_dilation < 1:
        raise ValueError("tangent_dilation must be >= 1")
    if cfg.tangent_chunk_pixels < 1:
        raise ValueError("tangent_chunk_pixels must be >= 1")
    if cfg.tangent_amplitude_multiplier <= 0:
        raise ValueError("tangent_amplitude_multiplier must be positive")
    return cfg


def project(projector: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


@torch.no_grad()
def tangent_targets(
    model: Stage2NullTangentManifoldNet,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    target_coefficients = model.stage1.encode(gt, basis=outputs["basis"])
    target_null = project(
        model.exact_null_projector.to(target_coefficients),
        target_coefficients,
    )
    missing_null = target_null - outputs["null_seed_coefficients"]
    tangent_basis = outputs["tangent_basis"]
    target_coordinates = torch.einsum(
        "nrdhw,nrhw->ndhw",
        tangent_basis,
        missing_null,
    )
    limit = outputs["tangent_coordinate_limit"]
    bounded_coordinates = torch.maximum(
        torch.minimum(target_coordinates, limit),
        -limit,
    )
    unbounded_residual = torch.einsum(
        "nrdhw,ndhw->nrhw",
        tangent_basis,
        target_coordinates,
    )
    bounded_residual = torch.einsum(
        "nrdhw,ndhw->nrhw",
        tangent_basis,
        bounded_coordinates,
    )
    unbounded_residual = project(
        model.exact_null_projector.to(unbounded_residual),
        unbounded_residual,
    )
    bounded_residual = project(
        model.exact_null_projector.to(bounded_residual),
        bounded_residual,
    )
    clip_mask = target_coordinates.abs() > limit
    return {
        "target_coefficients": target_coefficients,
        "target_null": target_null,
        "missing_null": missing_null,
        "target_coordinates": target_coordinates,
        "bounded_target_coordinates": bounded_coordinates,
        "unbounded_tangent_residual": unbounded_residual,
        "bounded_tangent_residual": bounded_residual,
        "coordinate_clip_ratio": clip_mask.float().mean(),
    }


def compute_losses(
    model: Stage2NullTangentManifoldNet,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
) -> Dict[str, torch.Tensor]:
    gt = batch["gt"]
    lr_hsi = batch["lr_hsi"]
    reconstructed = outputs["reconstructed_hsi"]

    hsi_l1 = F.l1_loss(reconstructed, gt)
    sam = sam_loss(reconstructed, gt)
    sgrad1 = F.l1_loss(
        first_spectral_difference(reconstructed),
        first_spectral_difference(gt),
    )
    sgrad2 = F.l1_loss(
        second_spectral_difference(reconstructed),
        second_spectral_difference(gt),
    )

    targets = tangent_targets(model, outputs, gt)
    global_scale = outputs["coefficient_scale"].mean().clamp_min(1e-8)
    coordinate_loss = F.smooth_l1_loss(
        outputs["tangent_coordinates"] / global_scale,
        targets["bounded_target_coordinates"] / global_scale,
        beta=0.25,
    )

    coefficient_scale = outputs["coefficient_scale"].view(1, -1, 1, 1)
    tangent_residual_loss = F.smooth_l1_loss(
        outputs["tangent_residual"] / coefficient_scale,
        targets["bounded_tangent_residual"] / coefficient_scale,
        beta=0.25,
    )

    degraded_hsi = hsi_degrader(reconstructed, target_size=lr_hsi.shape[-2:])
    lr_hsi_loss = F.l1_loss(degraded_hsi, lr_hsi)

    corrected_null = outputs["null_seed_coefficients"] + outputs["tangent_residual"]
    lr_target_null = project(
        model.exact_null_projector.to(outputs["lr_coefficients"]),
        outputs["lr_coefficients"],
    )
    degraded_null = coefficient_degrader(
        corrected_null,
        target_size=lr_target_null.shape[-2:],
    )
    lr_null_loss = F.smooth_l1_loss(
        degraded_null / coefficient_scale,
        lr_target_null / coefficient_scale,
        beta=0.25,
    )

    total = (
        cfg.tangent_lambda_l1 * hsi_l1
        + cfg.tangent_lambda_sam * sam
        + cfg.tangent_lambda_sgrad1 * sgrad1
        + cfg.tangent_lambda_sgrad2 * sgrad2
        + cfg.tangent_lambda_coordinate * coordinate_loss
        + cfg.tangent_lambda_residual * tangent_residual_loss
        + cfg.tangent_lambda_lr_hsi * lr_hsi_loss
        + cfg.tangent_lambda_lr_null * lr_null_loss
    )

    return {
        "total": total,
        "hsi_l1": hsi_l1,
        "sam": sam,
        "sgrad1": sgrad1,
        "sgrad2": sgrad2,
        "coordinate_loss": coordinate_loss,
        "tangent_residual_loss": tangent_residual_loss,
        "lr_hsi_loss": lr_hsi_loss,
        "lr_null_loss": lr_null_loss,
        "coordinate_clip_ratio": targets["coordinate_clip_ratio"].detach(),
    }


LOSS_NAMES = [
    "total",
    "hsi_l1",
    "sam",
    "sgrad1",
    "sgrad2",
    "coordinate_loss",
    "tangent_residual_loss",
    "lr_hsi_loss",
    "lr_null_loss",
    "coordinate_clip_ratio",
]

DIAGNOSTIC_NAMES = [
    "normalized_coordinate_abs",
    "coordinate_abs",
    "coordinate_limit_mean",
    "tangent_scale_mean",
    "tangent_residual_abs",
]


def forward_diagnostics(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        "normalized_coordinate_abs": float(
            outputs["normalized_tangent_coordinates"].detach().abs().mean().item()
        ),
        "coordinate_abs": float(
            outputs["tangent_coordinates"].detach().abs().mean().item()
        ),
        "coordinate_limit_mean": float(
            outputs["tangent_coordinate_limit"].detach().mean().item()
        ),
        "tangent_scale_mean": float(
            outputs["tangent_scale"].detach().mean().item()
        ),
        "tangent_residual_abs": float(
            outputs["tangent_residual"].detach().abs().mean().item()
        ),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    hsi_degrader,
    coefficient_degrader,
    sam_loss,
    cfg,
    device,
):
    model.train()
    model.stage1.eval()
    meters = {name: AverageMeter() for name in LOSS_NAMES + DIAGNOSTIC_NAMES}

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["lr_hsi"], batch["hr_msi"])
        losses = compute_losses(
            model,
            outputs,
            batch,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
        )
        losses["total"].backward()
        if cfg.tangent_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.coordinate_predictor.parameters(),
                cfg.tangent_grad_clip,
            )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
        for name, value in forward_diagnostics(outputs).items():
            meters[name].update(value, batch_size)

    return {name: meter.avg for name, meter in meters.items()}


class OnlinePearson:
    def __init__(self):
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sx2 = 0.0
        self.sy2 = 0.0
        self.sxy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().double().reshape(-1)
        y = y.detach().double().reshape(-1)
        self.n += int(x.numel())
        self.sx += float(x.sum().item())
        self.sy += float(y.sum().item())
        self.sx2 += float(x.square().sum().item())
        self.sy2 += float(y.square().sum().item())
        self.sxy += float((x * y).sum().item())

    def value(self) -> float:
        if self.n <= 1:
            return 0.0
        n = float(self.n)
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sx2 - self.sx * self.sx / n
        vy = self.sy2 - self.sy * self.sy / n
        denom = math.sqrt(max(vx * vy, 0.0))
        return cov / denom if denom > 1e-30 else 0.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    hsi_degrader,
    coefficient_degrader,
    sam_loss,
    cfg,
    device,
):
    model.eval()
    metric_sets = {
        "tangent": MetricAverager(),
        "anchor": MetricAverager(),
        "bounded_oracle": MetricAverager(),
        "unbounded_oracle": MetricAverager(),
        "basis_oracle": MetricAverager(),
    }
    loss_meters = {name: AverageMeter() for name in LOSS_NAMES}
    diagnostic_meters = {name: AverageMeter() for name in DIAGNOSTIC_NAMES}

    missing_energy = 0.0
    trained_error_energy = 0.0
    bounded_error_energy = 0.0
    unbounded_error_energy = 0.0
    null_pearson = OnlinePearson()
    max_null_leakage = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch["lr_hsi"], batch["hr_msi"])
        losses = compute_losses(
            model,
            outputs,
            batch,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
        )
        targets = tangent_targets(model, outputs, batch["gt"])

        bounded_hsi = model.stage1.decode(
            outputs["anchor_coefficients"] + targets["bounded_tangent_residual"],
            basis=outputs["basis"],
        )
        unbounded_hsi = model.stage1.decode(
            outputs["anchor_coefficients"] + targets["unbounded_tangent_residual"],
            basis=outputs["basis"],
        )
        basis_oracle_hsi = model.stage1.decode(
            targets["target_coefficients"],
            basis=outputs["basis"],
        )

        pairs = {
            "tangent": outputs["reconstructed_hsi"],
            "anchor": outputs["anchor_hsi"],
            "bounded_oracle": bounded_hsi,
            "unbounded_oracle": unbounded_hsi,
            "basis_oracle": basis_oracle_hsi,
        }
        for name, prediction in pairs.items():
            metric_sets[name].update(
                calc_metrics(prediction, batch["gt"], cfg.scale_ratio)
            )

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            loss_meters[name].update(float(losses[name].item()), batch_size)
        for name, value in forward_diagnostics(outputs).items():
            diagnostic_meters[name].update(value, batch_size)

        missing = targets["missing_null"].double()
        trained_remaining = (
            targets["missing_null"] - outputs["tangent_residual"]
        ).double()
        bounded_remaining = (
            targets["missing_null"] - targets["bounded_tangent_residual"]
        ).double()
        unbounded_remaining = (
            targets["missing_null"] - targets["unbounded_tangent_residual"]
        ).double()
        missing_energy += float(missing.square().sum().item())
        trained_error_energy += float(trained_remaining.square().sum().item())
        bounded_error_energy += float(bounded_remaining.square().sum().item())
        unbounded_error_energy += float(unbounded_remaining.square().sum().item())

        predicted_null = outputs["null_seed_coefficients"] + outputs["tangent_residual"]
        null_pearson.update(predicted_null, targets["target_null"])
        null_msi = torch.einsum(
            "mr,nrhw->nmhw",
            model.reduced_response.to(outputs["tangent_residual"]),
            outputs["tangent_residual"],
        )
        max_null_leakage = max(
            max_null_leakage,
            float(null_msi.abs().max().item()),
        )

    result = {}
    for prefix, meter in metric_sets.items():
        for name, value in meter.average().items():
            result[f"{prefix}_{name.lower()}"] = value
    result.update({name: meter.avg for name, meter in loss_meters.items()})
    result.update({name: meter.avg for name, meter in diagnostic_meters.items()})

    missing_energy = max(missing_energy, 1e-30)
    result["trained_missing_mse_capture"] = 1.0 - trained_error_energy / missing_energy
    result["bounded_oracle_missing_mse_capture"] = 1.0 - bounded_error_energy / missing_energy
    result["unbounded_oracle_missing_mse_capture"] = 1.0 - unbounded_error_energy / missing_energy
    result["null_relative_rmse"] = math.sqrt(trained_error_energy / missing_energy)
    result["null_pearson"] = null_pearson.value()
    result["tangent_null_leakage_max"] = max_null_leakage
    result["tangent_psnr_gain_over_anchor"] = result["tangent_psnr"] - result["anchor_psnr"]
    result["tangent_sam_gain_over_anchor"] = result["anchor_sam"] - result["tangent_sam"]
    result["bounded_oracle_gap_from_unbounded_psnr"] = (
        result["unbounded_oracle_psnr"] - result["bounded_oracle_psnr"]
    )
    result["observable_rank"] = int(model.observable_rank.item())
    return result


def main() -> None:
    cfg = parse_tangent_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    spectral_response = build_spectral_response(info).to(device)
    model = Stage2NullTangentManifoldNet(
        stage1_model=stage1,
        spectral_response=spectral_response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
        tangent_dimension=cfg.tangent_dimension,
        tangent_kernel_size=cfg.tangent_kernel_size,
        tangent_dilation=cfg.tangent_dilation,
        tangent_chunk_pixels=cfg.tangent_chunk_pixels,
        tangent_amplitude_multiplier=cfg.tangent_amplitude_multiplier,
        predictor_hidden_channels=cfg.tangent_predictor_hidden,
        predictor_blocks=cfg.tangent_predictor_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.coordinate_predictor.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * 0.05,
    )
    hsi_degrader = FixedSpatialDegradation(
        channels=info["n_bands"],
        kernel_size=5,
        sigma=2.0,
    ).to(device)
    coefficient_degrader = FixedSpatialDegradation(
        channels=stage1.basis_rank,
        kernel_size=5,
        sigma=2.0,
    ).to(device)
    sam_loss = SAMLoss()

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage2_null_tangent_manifold",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_null_tangent_manifold",
        cfg.dataset,
    )
    log_path = os.path.join(
        cfg.log_root,
        "stage2_null_tangent_manifold",
        f"{cfg.dataset}.log",
    )
    csv_path = os.path.join(
        cfg.log_root,
        "stage2_null_tangent_manifold",
        f"{cfg.dataset}.csv",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)

    csv_fields = [
        "epoch",
        "lr",
        "train_total",
        "train_coordinate_loss",
        "train_tangent_residual_loss",
        "tangent_psnr",
        "tangent_sam",
        "anchor_psnr",
        "anchor_sam",
        "bounded_oracle_psnr",
        "unbounded_oracle_psnr",
        "basis_oracle_psnr",
        "trained_missing_mse_capture",
        "bounded_oracle_missing_mse_capture",
        "unbounded_oracle_missing_mse_capture",
        "null_relative_rmse",
        "null_pearson",
        "coordinate_clip_ratio",
        *DIAGNOSTIC_NAMES,
    ]
    csv_logger = CSVLogger(csv_path, csv_fields)

    start = evaluate(
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
        "Tangent start | "
        f"PSNR={start['tangent_psnr']:.4f} SAM={start['tangent_sam']:.4f} | "
        f"anchor={start['anchor_psnr']:.4f}/{start['anchor_sam']:.4f} | "
        f"oracle bounded/unbounded={start['bounded_oracle_psnr']:.4f}/"
        f"{start['unbounded_oracle_psnr']:.4f} | "
        f"basis={start['basis_oracle_psnr']:.4f} | "
        f"clip={100.0 * start['coordinate_clip_ratio']:.2f}% | "
        f"null rRMSE={start['null_relative_rmse']:.4f} r={start['null_pearson']:.4f}"
    )
    write_log(
        log_path,
        f"Model | params={count_parameters(model.coordinate_predictor):.3f}M | "
        f"d={cfg.tangent_dimension} | geometry={cfg.tangent_kernel_size}x"
        f"{cfg.tangent_kernel_size} dilation={cfg.tangent_dilation} | "
        f"amplitude={cfg.tangent_amplitude_multiplier:.3f} | "
        f"observable_rank={start['observable_rank']}/{stage1.basis_rank}"
    )

    if start["bounded_oracle_gap_from_unbounded_psnr"] > 0.5:
        write_log(
            log_path,
            "WARNING | bounded tangent oracle is more than 0.5 dB below the "
            "unbounded oracle; consider increasing --tangent_amplitude_multiplier."
        )

    if cfg.tangent_diagnose_only:
        summary_path = os.path.join(output_dir, "diagnose_only.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(start, file, indent=2, ensure_ascii=False)
        print(f"Saved diagnostic: {summary_path}")
        return

    best_psnr = start["tangent_psnr"]
    best_epoch = 0
    best_path = os.path.join(checkpoint_dir, "tangent_manifold_best_psnr.pth")
    save_checkpoint(
        model,
        optimizer,
        epoch=0,
        best_metric=best_psnr,
        path=best_path,
        extra={
            "dataset": cfg.dataset,
            "tangent_dimension": cfg.tangent_dimension,
            "tangent_kernel_size": cfg.tangent_kernel_size,
            "tangent_dilation": cfg.tangent_dilation,
            "tangent_amplitude_multiplier": cfg.tangent_amplitude_multiplier,
        },
    )

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        evaluation = evaluate(
            model,
            test_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_total": train_stats["total"],
            "train_coordinate_loss": train_stats["coordinate_loss"],
            "train_tangent_residual_loss": train_stats["tangent_residual_loss"],
            **{key: evaluation.get(key, "") for key in csv_fields},
        }
        row["epoch"] = epoch
        row["lr"] = current_lr
        row["train_total"] = train_stats["total"]
        row["train_coordinate_loss"] = train_stats["coordinate_loss"]
        row["train_tangent_residual_loss"] = train_stats["tangent_residual_loss"]
        csv_logger.write(row)

        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"PSNR={evaluation['tangent_psnr']:.4f} "
            f"SAM={evaluation['tangent_sam']:.4f} | "
            f"gain={evaluation['tangent_psnr_gain_over_anchor']:+.4f} dB/"
            f"{evaluation['tangent_sam_gain_over_anchor']:+.4f} deg | "
            f"capture={100.0 * evaluation['trained_missing_mse_capture']:.2f}% | "
            f"null={evaluation['null_relative_rmse']:.4f}, r={evaluation['null_pearson']:.4f} | "
            f"coord={evaluation['normalized_coordinate_abs']:.4f} | "
            f"clip={100.0 * evaluation['coordinate_clip_ratio']:.2f}%"
        )

        if evaluation["tangent_psnr"] > best_psnr:
            best_psnr = evaluation["tangent_psnr"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                best_metric=best_psnr,
                path=best_path,
                extra={
                    "dataset": cfg.dataset,
                    "tangent_dimension": cfg.tangent_dimension,
                    "tangent_kernel_size": cfg.tangent_kernel_size,
                    "tangent_dilation": cfg.tangent_dilation,
                    "tangent_amplitude_multiplier": cfg.tangent_amplitude_multiplier,
                },
            )

    summary = {
        "dataset": cfg.dataset,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "start": start,
        "checkpoint": best_path,
        "tangent_dimension": cfg.tangent_dimension,
        "tangent_kernel_size": cfg.tangent_kernel_size,
        "tangent_dilation": cfg.tangent_dilation,
        "tangent_amplitude_multiplier": cfg.tangent_amplitude_multiplier,
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Best tangent PSNR={best_psnr:.4f} at epoch {best_epoch}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
