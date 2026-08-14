"""Train the constrained MSI-only spatial transport Stage-2 experiment.

The experiment tests one hypothesis only: HR-MSI supplies spatial geometry for
redistributing LR-HSI-derived null-space coefficient vectors, while it is not
allowed to synthesize null-space spectral values directly.
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
from models.stage2_msi_spatial_transport import Stage2MSISpatialTransportNet
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


def parse_transport_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--projector_tolerance", type=float, default=1e-6)

    parser.add_argument("--transport_hidden_channels", type=int, default=48)
    parser.add_argument("--transport_blocks", type=int, default=3)
    parser.add_argument("--transport_kernel_size", type=int, default=5)
    parser.add_argument("--transport_dilation", type=int, default=2)
    parser.add_argument("--transport_identity_logit", type=float, default=6.0)
    parser.add_argument("--transport_grad_clip", type=float, default=1.0)

    parser.add_argument("--transport_lambda_l1", type=float, default=1.0)
    parser.add_argument("--transport_lambda_sam", type=float, default=0.3)
    parser.add_argument("--transport_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--transport_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--transport_lambda_null", type=float, default=0.3)
    parser.add_argument("--transport_lambda_lr_hsi", type=float, default=0.2)
    parser.add_argument("--transport_lambda_lr_null", type=float, default=0.2)

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


def project(projector: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


def compute_losses(
    model: Stage2MSISpatialTransportNet,
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

    scale = outputs["coefficient_scale"].view(1, -1, 1, 1)
    with torch.no_grad():
        target_coefficients = model.stage1.encode(
            gt,
            basis=outputs["basis"],
        )
        target_null = project(
            model.exact_null_projector.to(target_coefficients),
            target_coefficients,
        )
        normalized_target_null = target_null / scale
        lr_target_null = project(
            model.exact_null_projector.to(outputs["lr_coefficients"]),
            outputs["lr_coefficients"],
        )

    normalized_transport_null = outputs["transported_null_coefficients"] / scale
    null_loss = F.smooth_l1_loss(
        normalized_transport_null,
        normalized_target_null,
        beta=0.25,
    )

    degraded_hsi = hsi_degrader(reconstructed, target_size=lr_hsi.shape[-2:])
    lr_hsi_loss = F.l1_loss(degraded_hsi, lr_hsi)

    degraded_null = coefficient_degrader(
        outputs["transported_null_coefficients"],
        target_size=lr_target_null.shape[-2:],
    )
    lr_null_loss = F.smooth_l1_loss(
        degraded_null / scale,
        lr_target_null / scale,
        beta=0.25,
    )

    total = (
        cfg.transport_lambda_l1 * hsi_l1
        + cfg.transport_lambda_sam * sam
        + cfg.transport_lambda_sgrad1 * sgrad1
        + cfg.transport_lambda_sgrad2 * sgrad2
        + cfg.transport_lambda_null * null_loss
        + cfg.transport_lambda_lr_hsi * lr_hsi_loss
        + cfg.transport_lambda_lr_null * lr_null_loss
    )
    return {
        "total": total,
        "hsi_l1": hsi_l1,
        "sam": sam,
        "sgrad1": sgrad1,
        "sgrad2": sgrad2,
        "null_loss": null_loss,
        "lr_hsi_loss": lr_hsi_loss,
        "lr_null_loss": lr_null_loss,
    }


LOSS_NAMES = [
    "total",
    "hsi_l1",
    "sam",
    "sgrad1",
    "sgrad2",
    "null_loss",
    "lr_hsi_loss",
    "lr_null_loss",
]

DIAGNOSTIC_NAMES = [
    "transport_center_weight",
    "transport_max_weight",
    "transport_entropy",
    "transport_expected_radius",
]


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
        if cfg.transport_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.transport.parameters(),
                cfg.transport_grad_clip,
            )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
        for name in DIAGNOSTIC_NAMES:
            meters[name].update(float(outputs[name].detach().item()), batch_size)

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, hsi_degrader, coefficient_degrader, sam_loss, cfg, device):
    model.eval()
    metric_sets = {
        "transport": MetricAverager(),
        "anchor": MetricAverager(),
        "base": MetricAverager(),
    }
    loss_meters = {name: AverageMeter() for name in LOSS_NAMES}
    diagnostic_meters = {name: AverageMeter() for name in DIAGNOSTIC_NAMES}

    null_error_sq = 0.0
    null_target_sq = 0.0
    null_dot = 0.0
    null_pred_center_sq = 0.0
    null_target_center_sq = 0.0

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
        metric_sets["transport"].update(
            calc_metrics(outputs["reconstructed_hsi"], batch["gt"], cfg.scale_ratio)
        )
        metric_sets["anchor"].update(
            calc_metrics(outputs["anchor_hsi"], batch["gt"], cfg.scale_ratio)
        )
        metric_sets["base"].update(
            calc_metrics(outputs["base_hsi"], batch["gt"], cfg.scale_ratio)
        )

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            loss_meters[name].update(float(losses[name].item()), batch_size)
        for name in DIAGNOSTIC_NAMES:
            diagnostic_meters[name].update(float(outputs[name].item()), batch_size)

        target_coeff = model.stage1.encode(batch["gt"], basis=outputs["basis"])
        target_null = project(
            model.exact_null_projector.to(target_coeff),
            target_coeff,
        )
        pred_null = outputs["transported_null_coefficients"]
        error = (pred_null - target_null).double()
        target64 = target_null.double()
        pred64 = pred_null.double()
        null_error_sq += float(error.square().sum().item())
        null_target_sq += float(target64.square().sum().item())

        target_centered = target64 - target64.mean()
        pred_centered = pred64 - pred64.mean()
        null_dot += float((target_centered * pred_centered).sum().item())
        null_pred_center_sq += float(pred_centered.square().sum().item())
        null_target_center_sq += float(target_centered.square().sum().item())

    result = {}
    for prefix, meter in metric_sets.items():
        for name, value in meter.average().items():
            result[f"{prefix}_{name.lower()}"] = value
    result.update({name: meter.avg for name, meter in loss_meters.items()})
    result.update({name: meter.avg for name, meter in diagnostic_meters.items()})
    result["transport_psnr_gain_over_anchor"] = (
        result["transport_psnr"] - result["anchor_psnr"]
    )
    result["transport_sam_gain_over_anchor"] = (
        result["anchor_sam"] - result["transport_sam"]
    )
    result["null_relative_rmse"] = math.sqrt(
        null_error_sq / max(null_target_sq, 1e-30)
    )
    result["null_pearson"] = null_dot / math.sqrt(
        max(null_pred_center_sq * null_target_center_sq, 1e-30)
    )
    result["observable_rank"] = int(model.observable_rank.item())
    result.update(
        {
            name: float(value.detach().item())
            for name, value in model.projector_statistics().items()
        }
    )
    return result


def main() -> None:
    cfg = parse_transport_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    spectral_response = build_spectral_response(info).to(device)
    model = Stage2MSISpatialTransportNet(
        stage1_model=stage1,
        spectral_response=spectral_response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
        transport_hidden_channels=cfg.transport_hidden_channels,
        transport_blocks=cfg.transport_blocks,
        transport_kernel_size=cfg.transport_kernel_size,
        transport_dilation=cfg.transport_dilation,
        transport_identity_logit=cfg.transport_identity_logit,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.transport.parameters(),
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
        "stage2_msi_spatial_transport",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_msi_spatial_transport",
        cfg.dataset,
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    log_path = os.path.join(
        cfg.log_root,
        "stage2_msi_spatial_transport",
        f"{cfg.dataset}.log",
    )
    csv_path = os.path.join(
        cfg.log_root,
        "stage2_msi_spatial_transport",
        f"{cfg.dataset}.csv",
    )
    csv_fields = [
        "epoch",
        "lr",
        "train_total",
        "train_null_loss",
        "train_center_weight",
        "train_entropy",
        "transport_psnr",
        "transport_sam",
        "anchor_psnr",
        "anchor_sam",
        "transport_psnr_gain_over_anchor",
        "transport_sam_gain_over_anchor",
        "null_relative_rmse",
        "null_pearson",
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
        "Transport start | "
        f"PSNR={start['transport_psnr']:.4f} SAM={start['transport_sam']:.4f} | "
        f"anchor={start['anchor_psnr']:.4f}/{start['anchor_sam']:.4f} | "
        f"gain={start['transport_psnr_gain_over_anchor']:+.4f} dB | "
        f"null rRMSE={start['null_relative_rmse']:.4f} r={start['null_pearson']:.4f} | "
        f"center={start['transport_center_weight']:.4f} "
        f"entropy={start['transport_entropy']:.4f} "
        f"radius={start['transport_expected_radius']:.4f}",
    )
    write_log(
        log_path,
        f"Trainable parameters={count_parameters(model):.4f} M, "
        f"rank(RU)={int(model.observable_rank.item())}/{stage1.basis_rank}, "
        f"null_dim={stage1.basis_rank-int(model.observable_rank.item())}",
    )

    best_psnr = start["transport_psnr"]
    best_sam = start["transport_sam"]
    best_path = os.path.join(checkpoint_dir, "spatial_transport_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "spatial_transport_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "spatial_transport_last.pth")

    for epoch in range(1, cfg.epochs + 1):
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
        scheduler.step()

        result = evaluate(
            model,
            test_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        lr = optimizer.param_groups[0]["lr"]
        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"PSNR={result['transport_psnr']:.4f} SAM={result['transport_sam']:.4f} | "
            f"gain={result['transport_psnr_gain_over_anchor']:+.4f} dB/"
            f"{result['transport_sam_gain_over_anchor']:+.4f} deg | "
            f"null={result['null_relative_rmse']:.4f}, r={result['null_pearson']:.4f} | "
            f"center={result['transport_center_weight']:.4f} "
            f"max={result['transport_max_weight']:.4f} "
            f"entropy={result['transport_entropy']:.4f} "
            f"radius={result['transport_expected_radius']:.4f}",
        )
        csv_logger.write(
            {
                "epoch": epoch,
                "lr": lr,
                "train_total": train_result["total"],
                "train_null_loss": train_result["null_loss"],
                "train_center_weight": train_result["transport_center_weight"],
                "train_entropy": train_result["transport_entropy"],
                **{name: result[name] for name in csv_fields if name in result},
            }
        )

        extra = {
            "dataset": cfg.dataset,
            "n_bands": int(info["n_bands"]),
            "n_msi": int(info["n_select_bands"]),
            "basis_rank": int(stage1.basis_rank),
            "observable_rank": int(model.observable_rank.item()),
            "transport_hidden_channels": cfg.transport_hidden_channels,
            "transport_blocks": cfg.transport_blocks,
            "transport_kernel_size": cfg.transport_kernel_size,
            "transport_dilation": cfg.transport_dilation,
            "transport_identity_logit": cfg.transport_identity_logit,
            "msi_mode": cfg.msi_mode,
            "srf_band_set": cfg.srf_band_set,
            "metrics": result,
        }
        if result["transport_psnr"] > best_psnr:
            best_psnr = result["transport_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                best_path,
                extra=extra,
            )
        if result["transport_sam"] < best_sam:
            best_sam = result["transport_sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_sam,
                best_sam_path,
                extra=extra,
            )
        if epoch == cfg.epochs or epoch % max(cfg.save_interval, 1) == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                last_path,
                extra=extra,
            )

    summary = {
        "dataset": cfg.dataset,
        "start": start,
        "best_psnr": best_psnr,
        "best_sam": best_sam,
        "checkpoint": best_path,
    }
    with open(
        os.path.join(output_dir, "spatial_transport_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    write_log(
        log_path,
        f"Finished | best PSNR={best_psnr:.4f}, best SAM={best_sam:.4f} | "
        f"saved={best_path}",
    )


if __name__ == "__main__":
    main()
