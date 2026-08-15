"""Train symmetric-frequency guided tangent-projected Stage-2 proposal."""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage2_symmetric_frequency_tangent_proposal import (
    Stage2SymmetricFrequencyTangentProposalNet,
)
from train_stage2_coefficients import (
    FixedSpatialDegradation,
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from train_stage2_null_tangent_manifold import OnlinePearson
from train_stage2_tangent_projected_proposal import (
    compute_losses,
    diagnostics,
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


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def parse_specific_args():
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
    parser.add_argument("--proposal_amplitude_multiplier", type=float, default=8.0)
    parser.add_argument("--proposal_predictor_hidden", type=int, default=96)
    parser.add_argument("--proposal_predictor_blocks", type=int, default=4)
    parser.add_argument("--proposal_grad_clip", type=float, default=1.0)
    parser.add_argument("--proposal_diagnose_only", action="store_true")

    parser.add_argument("--proposal_lambda_l1", type=float, default=1.0)
    parser.add_argument("--proposal_lambda_sam", type=float, default=0.3)
    parser.add_argument("--proposal_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--proposal_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--proposal_lambda_residual", type=float, default=0.8)
    parser.add_argument("--proposal_lambda_lr_hsi", type=float, default=0.2)
    parser.add_argument("--proposal_lambda_lr_null", type=float, default=0.1)
    parser.add_argument("--proposal_lambda_off_tangent", type=float, default=0.0)

    # Symmetric-frequency representation only: no NSP/reliability branch.
    parser.add_argument("--frequency_feature_channels", type=int, default=64)
    parser.add_argument("--frequency_encoder_blocks", type=int, default=3)
    parser.add_argument("--frequency_num_bands", type=int, default=20)
    parser.add_argument("--frequency_init_low_boundary", type=float, default=5.0)
    parser.add_argument("--frequency_init_high_boundary", type=float, default=18.0)
    parser.add_argument("--frequency_boundary_temperature", type=float, default=0.5)
    parser.add_argument("--frequency_soft_partition", action="store_true")
    parser.add_argument("--frequency_boundary_lr_multiplier", type=float, default=1.0)

    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    default = "./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth"
    if cfg.stage1_basis_checkpoint == default and cfg.dataset != "PaviaU":
        cfg.stage1_basis_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage1_basis",
            cfg.dataset,
            "basis_for_stage2.pth",
        )

    if cfg.tangent_dimension < 1:
        raise ValueError("tangent_dimension must be positive")
    if cfg.tangent_kernel_size < 3 or cfg.tangent_kernel_size % 2 == 0:
        raise ValueError("tangent_kernel_size must be odd and >= 3")
    if cfg.tangent_dilation < 1 or cfg.tangent_chunk_pixels < 1:
        raise ValueError("invalid tangent geometry")
    if cfg.proposal_amplitude_multiplier <= 0:
        raise ValueError("proposal_amplitude_multiplier must be positive")
    if cfg.frequency_feature_channels < 8:
        raise ValueError("frequency_feature_channels must be >= 8")
    if cfg.frequency_encoder_blocks < 1:
        raise ValueError("frequency_encoder_blocks must be >= 1")
    if cfg.frequency_boundary_lr_multiplier <= 0:
        raise ValueError("frequency_boundary_lr_multiplier must be positive")
    return cfg


def frequency_diagnostics(out: Dict[str, torch.Tensor]) -> Dict[str, float]:
    difference_share = out["difference_activation_share"].detach()
    physical_share = out["physical_activation_share"].detach()
    reference_share = out["reference_activation_share"].detach()
    return {
        "diff_low_abs": float(out["low_difference_feature"].detach().abs().mean().item()),
        "diff_mid_abs": float(out["mid_difference_feature"].detach().abs().mean().item()),
        "diff_high_abs": float(out["high_difference_feature"].detach().abs().mean().item()),
        "diff_low_share": float(difference_share[0].item()),
        "diff_mid_share": float(difference_share[1].item()),
        "diff_high_share": float(difference_share[2].item()),
        "physical_low_share": float(physical_share[0].item()),
        "physical_mid_share": float(physical_share[1].item()),
        "physical_high_share": float(physical_share[2].item()),
        "reference_low_share": float(reference_share[0].item()),
        "reference_mid_share": float(reference_share[1].item()),
        "reference_high_share": float(reference_share[2].item()),
        "tau_low_mean": float(out["tau_low"].detach().mean().item()),
        "tau_high_mean": float(out["tau_high"].detach().mean().item()),
    }


FREQUENCY_DIAGNOSTICS = [
    "diff_low_abs",
    "diff_mid_abs",
    "diff_high_abs",
    "diff_low_share",
    "diff_mid_share",
    "diff_high_share",
    "physical_low_share",
    "physical_mid_share",
    "physical_high_share",
    "reference_low_share",
    "reference_mid_share",
    "reference_high_share",
    "tau_low_mean",
    "tau_high_mean",
]


def train_epoch(model, loader, optimizer, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.train()
    model.stage1.eval()
    names = [
        "total",
        "residual",
        "off",
        "rho_tan",
        "rho_off",
        "sat",
        *FREQUENCY_DIAGNOSTICS,
    ]
    meters = {name: AverageMeter() for name in names}

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        losses, _ = compute_losses(
            model,
            out,
            batch,
            hsi_deg,
            coeff_deg,
            sam_loss,
            cfg,
        )
        losses["total"].backward()
        if cfg.proposal_grad_clip > 0:
            parameters = list(model.regular_trainable_parameters()) + list(
                model.frequency_boundary_parameters()
            )
            torch.nn.utils.clip_grad_norm_(parameters, cfg.proposal_grad_clip)
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        base_diag = diagnostics(out)
        freq_diag = frequency_diagnostics(out)
        for name in ["total", "residual", "off"]:
            meters[name].update(float(losses[name].detach().item()), batch_size)
        for name in ["rho_tan", "rho_off", "sat"]:
            meters[name].update(base_diag[name], batch_size)
        for name, value in freq_diag.items():
            meters[name].update(value, batch_size)

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.eval()
    metric_sets = {
        name: MetricAverager()
        for name in ["proposal", "anchor", "oracle", "basis"]
    }
    diagnostic_names = [
        "rho_tan",
        "rho_off",
        "sat",
        "proposal_abs",
        "tangent_abs",
        *FREQUENCY_DIAGNOSTICS,
    ]
    meters = {name: AverageMeter() for name in diagnostic_names}

    missing_energy = 0.0
    predicted_error_energy = 0.0
    oracle_error_energy = 0.0
    correlation = OnlinePearson()
    leakage = 0.0

    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        _, target = compute_losses(
            model,
            out,
            batch,
            hsi_deg,
            coeff_deg,
            sam_loss,
            cfg,
        )
        oracle_hsi = model.stage1.decode(
            out["anchor_coefficients"] + target["tangent"],
            basis=out["basis"],
        )
        basis_hsi = model.stage1.decode(target["coeff"], basis=out["basis"])

        predictions = {
            "proposal": out["reconstructed_hsi"],
            "anchor": out["anchor_hsi"],
            "oracle": oracle_hsi,
            "basis": basis_hsi,
        }
        for name, prediction in predictions.items():
            metric_sets[name].update(
                calc_metrics(prediction, batch["gt"], cfg.scale_ratio)
            )

        batch_size = batch["lr_hsi"].size(0)
        values = diagnostics(out)
        values.update(frequency_diagnostics(out))
        for name in diagnostic_names:
            meters[name].update(values[name], batch_size)

        missing = target["missing"].double()
        predicted_remaining = (
            target["missing"] - out["tangent_residual"]
        ).double()
        oracle_remaining = (
            target["missing"] - target["tangent"]
        ).double()
        missing_energy += float(missing.square().sum().item())
        predicted_error_energy += float(predicted_remaining.square().sum().item())
        oracle_error_energy += float(oracle_remaining.square().sum().item())
        correlation.update(
            out["null_seed_coefficients"] + out["tangent_residual"],
            target["null"],
        )
        null_msi = torch.einsum(
            "mr,nrhw->nmhw",
            model.reduced_response.to(out["tangent_residual"]),
            out["tangent_residual"],
        )
        leakage = max(leakage, float(null_msi.abs().max().item()))

    result = {}
    for prefix, meter in metric_sets.items():
        for name, value in meter.average().items():
            result[f"{prefix}_{name.lower()}"] = value
    for name, meter in meters.items():
        result[name] = meter.avg

    missing_energy = max(missing_energy, 1e-30)
    result["capture"] = 1.0 - predicted_error_energy / missing_energy
    result["oracle_capture"] = 1.0 - oracle_error_energy / missing_energy
    result["null_rrmse"] = math.sqrt(predicted_error_energy / missing_energy)
    result["null_pearson"] = correlation.value()
    result["null_leakage"] = leakage
    result["gain_psnr"] = result["proposal_psnr"] - result["anchor_psnr"]
    result["gain_sam"] = result["anchor_sam"] - result["proposal_sam"]
    result["oracle_gap"] = result["oracle_psnr"] - result["proposal_psnr"]
    result["observable_rank"] = int(model.observable_rank.item())
    return result


def main() -> None:
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2SymmetricFrequencyTangentProposalNet(
        stage1_model=stage1,
        spectral_response=build_spectral_response(info).to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
        tangent_dimension=cfg.tangent_dimension,
        tangent_kernel_size=cfg.tangent_kernel_size,
        tangent_dilation=cfg.tangent_dilation,
        tangent_chunk_pixels=cfg.tangent_chunk_pixels,
        proposal_amplitude_multiplier=cfg.proposal_amplitude_multiplier,
        predictor_hidden_channels=cfg.proposal_predictor_hidden,
        predictor_blocks=cfg.proposal_predictor_blocks,
        frequency_feature_channels=cfg.frequency_feature_channels,
        frequency_encoder_blocks=cfg.frequency_encoder_blocks,
        num_frequency_bands=cfg.frequency_num_bands,
        init_low_boundary=cfg.frequency_init_low_boundary,
        init_high_boundary=cfg.frequency_init_high_boundary,
        boundary_temperature=cfg.frequency_boundary_temperature,
        hard_frequency_partition=not cfg.frequency_soft_partition,
    ).to(device)

    regular_parameters = list(model.regular_trainable_parameters())
    boundary_parameters = list(model.frequency_boundary_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": regular_parameters, "lr": cfg.lr},
            {
                "params": boundary_parameters,
                "lr": cfg.lr * cfg.frequency_boundary_lr_multiplier,
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
    hsi_deg = FixedSpatialDegradation(
        channels=info["n_bands"], kernel_size=5, sigma=2.0
    ).to(device)
    coeff_deg = FixedSpatialDegradation(
        channels=stage1.basis_rank, kernel_size=5, sigma=2.0
    ).to(device)
    sam_loss = SAMLoss()

    root = "stage2_symmetric_frequency_tangent_proposal"
    checkpoint_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset)
    output_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    log_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log")
    csv_path = os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)

    fields = [
        "epoch",
        "lr",
        "boundary_lr",
        "train_total",
        "proposal_psnr",
        "proposal_sam",
        "anchor_psnr",
        "anchor_sam",
        "oracle_psnr",
        "basis_psnr",
        "capture",
        "oracle_capture",
        "null_rrmse",
        "null_pearson",
        "rho_tan",
        "rho_off",
        "sat",
        "oracle_gap",
        "diff_low_share",
        "diff_mid_share",
        "diff_high_share",
        "tau_low_mean",
        "tau_high_mean",
    ]
    logger = CSVLogger(csv_path, fields)

    start = evaluate(model, test_loader, hsi_deg, coeff_deg, sam_loss, cfg, device)
    write_log(
        log_path,
        "SF-tangent proposal start | "
        f"PSNR={start['proposal_psnr']:.4f} SAM={start['proposal_sam']:.4f} | "
        f"anchor={start['anchor_psnr']:.4f}/{start['anchor_sam']:.4f} | "
        f"oracle={start['oracle_psnr']:.4f} basis={start['basis_psnr']:.4f} | "
        f"diff=({start['diff_low_share']:.3f},{start['diff_mid_share']:.3f},"
        f"{start['diff_high_share']:.3f}) | "
        f"tau=({start['tau_low_mean']:.2f},{start['tau_high_mean']:.2f})"
    )
    write_log(
        log_path,
        "Model | "
        f"trainable={count_parameters(model):.3f}M | d={cfg.tangent_dimension} | "
        f"geometry={cfg.tangent_kernel_size}x{cfg.tangent_kernel_size} "
        f"dilation={cfg.tangent_dilation} | frequency_channels="
        f"{cfg.frequency_feature_channels} | NSP=off reliability=off"
    )

    if cfg.proposal_diagnose_only:
        with open(
            os.path.join(output_dir, "diagnose_only.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(start, file, indent=2, ensure_ascii=False)
        return

    best_psnr = start["proposal_psnr"]
    best_sam = start["proposal_sam"]
    best_epoch = 0
    best_path = os.path.join(checkpoint_dir, "sf_tangent_proposal_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "sf_tangent_proposal_best_sam.pth")
    save_checkpoint(
        model,
        optimizer,
        0,
        best_psnr,
        best_path,
        extra={"dataset": cfg.dataset, "tangent_dimension": cfg.tangent_dimension},
    )
    save_checkpoint(
        model,
        optimizer,
        0,
        -best_sam,
        best_sam_path,
        extra={"dataset": cfg.dataset, "tangent_dimension": cfg.tangent_dimension},
    )

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            hsi_deg,
            coeff_deg,
            sam_loss,
            cfg,
            device,
        )
        evaluation = evaluate(
            model,
            test_loader,
            hsi_deg,
            coeff_deg,
            sam_loss,
            cfg,
            device,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        current_boundary_lr = optimizer.param_groups[1]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "boundary_lr": current_boundary_lr,
            "train_total": train_stats["total"],
        }
        for key in fields:
            if key in evaluation:
                row[key] = evaluation[key]
        logger.write(row)

        write_log(
            log_path,
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"PSNR={evaluation['proposal_psnr']:.4f} "
            f"SAM={evaluation['proposal_sam']:.4f} | "
            f"capture={100.0 * evaluation['capture']:.2f}% | "
            f"rho_tan={evaluation['rho_tan']:.4f} "
            f"off={evaluation['rho_off']:.4f} "
            f"sat={100.0 * evaluation['sat']:.2f}% | "
            f"diff=({evaluation['diff_low_share']:.3f},"
            f"{evaluation['diff_mid_share']:.3f},"
            f"{evaluation['diff_high_share']:.3f}) | "
            f"null={evaluation['null_rrmse']:.4f}, "
            f"r={evaluation['null_pearson']:.4f}"
        )

        if evaluation["proposal_psnr"] > best_psnr:
            best_psnr = evaluation["proposal_psnr"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                best_path,
                extra={
                    "dataset": cfg.dataset,
                    "tangent_dimension": cfg.tangent_dimension,
                    "frequency_feature_channels": cfg.frequency_feature_channels,
                },
            )
        if evaluation["proposal_sam"] < best_sam:
            best_sam = evaluation["proposal_sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                -best_sam,
                best_sam_path,
                extra={
                    "dataset": cfg.dataset,
                    "tangent_dimension": cfg.tangent_dimension,
                    "frequency_feature_channels": cfg.frequency_feature_channels,
                },
            )

    summary = {
        "dataset": cfg.dataset,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "best_sam": best_sam,
        "start": start,
        "checkpoint": best_path,
        "best_sam_checkpoint": best_sam_path,
        "tangent_dimension": cfg.tangent_dimension,
        "tangent_kernel_size": cfg.tangent_kernel_size,
        "tangent_dilation": cfg.tangent_dilation,
        "frequency_feature_channels": cfg.frequency_feature_channels,
        "nsp": False,
        "reliability": False,
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Best PSNR={best_psnr:.4f} at epoch {best_epoch}; best SAM={best_sam:.4f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
