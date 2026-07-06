"""Diagnose the representation ceiling of Stage 2.

The script compares four increasingly relaxed reconstruction spaces:

1. ``simplex``: frozen endmembers with non-negative abundances summing to one;
2. ``cone``: frozen endmembers with non-negative coefficients but no sum-to-one
   constraint, equivalent to adding a positive per-pixel illumination/gain map;
3. ``linear_span``: unconstrained projection onto the frozen endmember span;
4. ``pca_rank_k``: a GT-derived rank-K spectral upper bound, used only to test
   whether K dimensions are sufficient in principle.

These ceilings separate four possible bottlenecks: the Stage-2 predictor, the
sum-to-one illumination assumption, the learned endmember span, and the chosen
spectral rank. They are diagnostics only and do not alter model training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from train_stage2_physical import build_stage1_from_checkpoint
from utils import ensure_dir, get_device, move_to_device, set_seed


def inverse_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x).clamp_min(eps))


def parse_ceiling_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="./checkpoints/stage1_unmix/PaviaU/unmixing_best.pth",
    )
    parser.add_argument("--oracle_steps", type=int, default=500)
    parser.add_argument("--oracle_lr", type=float, default=0.05)
    parser.add_argument("--oracle_log_interval", type=int, default=100)
    parser.add_argument("--current_stage2_psnr", type=float, default=float("nan"))
    parser.add_argument("--current_stage2_sam", type=float, default=float("nan"))
    ceiling_args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(ceiling_args).items():
        setattr(cfg, key, value)

    default_path = "./checkpoints/stage1_unmix/PaviaU/unmixing_best.pth"
    if cfg.stage1_checkpoint == default_path and cfg.dataset != "PaviaU":
        cfg.stage1_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage1_unmix",
            cfg.dataset,
            "unmixing_best.pth",
        )
    return cfg


@torch.no_grad()
def stage1_baseline(
    stage1,
    lr_hsi: torch.Tensor,
    target_size,
):
    outputs = stage1(lr_hsi)
    endmembers = outputs["endmembers"].detach()
    logits = F.interpolate(
        outputs["abundance_logits"].detach(),
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    abundance = torch.softmax(logits, dim=1)
    reconstruction = torch.einsum("bk,nkhw->nbhw", endmembers, abundance)
    return endmembers, logits, abundance, reconstruction


def optimize_simplex_oracle(
    endmembers: torch.Tensor,
    initial_logits: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    lr: float,
    log_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Best MSE reconstruction with A>=0 and sum_k A_k=1."""
    logits = initial_logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=lr * 0.01,
    )

    best_loss = float("inf")
    best_logits = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        abundance = torch.softmax(logits, dim=1)
        reconstruction = torch.einsum(
            "bk,nkhw->nbhw", endmembers, abundance
        )
        loss = F.mse_loss(reconstruction, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_logits = logits.detach().clone()
        if log_interval > 0 and (
            step == 0 or (step + 1) % log_interval == 0 or step + 1 == steps
        ):
            print(
                f"  simplex step {step + 1:04d}/{steps:04d}: "
                f"MSE={value:.8f}, PSNR={-10.0 * math.log10(max(value, 1e-12)):.4f}"
            )

    with torch.no_grad():
        abundance = torch.softmax(best_logits, dim=1)
        reconstruction = torch.einsum(
            "bk,nkhw->nbhw", endmembers, abundance
        )
    return abundance, reconstruction, best_loss


def optimize_cone_oracle(
    endmembers: torch.Tensor,
    initial_abundance: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    lr: float,
    log_interval: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Best non-negative coefficients without the abundance sum constraint.

    Any coefficient vector C>=0 can be written as C=g*A, where A is simplex
    abundance and g=sum(C) is a positive per-pixel gain/illumination factor.
    """
    raw_coefficients = inverse_softplus(
        initial_abundance.detach().clamp_min(1e-6)
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([raw_coefficients], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(steps, 1),
        eta_min=lr * 0.01,
    )

    best_loss = float("inf")
    best_raw = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        coefficients = F.softplus(raw_coefficients)
        reconstruction = torch.einsum(
            "bk,nkhw->nbhw", endmembers, coefficients
        )
        loss = F.mse_loss(reconstruction, gt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_raw = raw_coefficients.detach().clone()
        if log_interval > 0 and (
            step == 0 or (step + 1) % log_interval == 0 or step + 1 == steps
        ):
            print(
                f"  cone    step {step + 1:04d}/{steps:04d}: "
                f"MSE={value:.8f}, PSNR={-10.0 * math.log10(max(value, 1e-12)):.4f}"
            )

    with torch.no_grad():
        coefficients = F.softplus(best_raw)
        gain = coefficients.sum(dim=1, keepdim=True).clamp_min(1e-8)
        abundance = coefficients / gain
        reconstruction = torch.einsum(
            "bk,nkhw->nbhw", endmembers, coefficients
        )
    return abundance, gain, reconstruction, best_loss


@torch.no_grad()
def linear_span_oracle(
    endmembers: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unconstrained orthogonal projection onto span(E)."""
    batch, bands, height, width = gt.shape
    basis = endmembers.float()
    pseudo_inverse = torch.linalg.pinv(basis)
    spectra = gt.float().permute(0, 2, 3, 1).reshape(-1, bands).transpose(0, 1)
    coefficients = pseudo_inverse @ spectra
    reconstruction = basis @ coefficients
    raw = reconstruction.transpose(0, 1).reshape(batch, height, width, bands)
    raw = raw.permute(0, 3, 1, 2).contiguous()
    diagnostics = {
        "negative_value_ratio": float((raw < 0).float().mean().item()),
        "above_one_ratio": float((raw > 1).float().mean().item()),
        "coefficient_negative_ratio": float(
            (coefficients < 0).float().mean().item()
        ),
    }
    return raw.clamp(0.0, 1.0), diagnostics


@torch.no_grad()
def pca_rank_oracle(
    gt: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """GT-derived affine rank-r spectral reconstruction upper bound."""
    batch, bands, height, width = gt.shape
    spectra = gt.float().permute(0, 2, 3, 1).reshape(-1, bands).transpose(0, 1)
    mean = spectra.mean(dim=1, keepdim=True)
    centered = spectra - mean
    covariance = centered @ centered.transpose(0, 1)
    covariance = covariance / max(centered.size(1), 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    rank = min(int(rank), bands)
    basis = eigenvectors[:, order[:rank]]
    reconstruction = mean + basis @ (basis.transpose(0, 1) @ centered)
    reconstruction = reconstruction.transpose(0, 1).reshape(
        batch, height, width, bands
    )
    return reconstruction.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)


def main() -> None:
    cfg = parse_ceiling_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    stage1, stage1_state = build_stage1_from_checkpoint(
        cfg.stage1_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )

    metric_sets = {
        "base": MetricAverager(),
        "simplex": MetricAverager(),
        "cone": MetricAverager(),
        "linear_span": MetricAverager(),
        "pca_rank_k": MetricAverager(),
    }
    abundance_change_values = []
    cone_gain_values = []
    span_diagnostics = []
    patch_records = []

    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        with torch.no_grad():
            endmembers, logits, base_abundance, base_hsi = stage1_baseline(
                stage1,
                batch["lr_hsi"],
                gt.shape[-2:],
            )

        print(f"Test patch {batch_index}: optimizing representation ceilings")
        simplex_abundance, simplex_hsi, simplex_mse = optimize_simplex_oracle(
            endmembers=endmembers,
            initial_logits=logits,
            gt=gt,
            steps=cfg.oracle_steps,
            lr=cfg.oracle_lr,
            log_interval=cfg.oracle_log_interval,
        )
        cone_abundance, cone_gain, cone_hsi, cone_mse = optimize_cone_oracle(
            endmembers=endmembers,
            initial_abundance=base_abundance,
            gt=gt,
            steps=cfg.oracle_steps,
            lr=cfg.oracle_lr,
            log_interval=cfg.oracle_log_interval,
        )
        span_hsi, span_info = linear_span_oracle(endmembers, gt)
        pca_hsi = pca_rank_oracle(gt, rank=stage1.num_endmembers)

        reconstructions = {
            "base": base_hsi,
            "simplex": simplex_hsi,
            "cone": cone_hsi,
            "linear_span": span_hsi,
            "pca_rank_k": pca_hsi,
        }
        patch_metrics = {}
        for name, reconstruction in reconstructions.items():
            values = calc_metrics(reconstruction, gt, cfg.scale_ratio)
            metric_sets[name].update(values)
            patch_metrics[name] = values

        abundance_change = float(
            (simplex_abundance - base_abundance).abs().mean().item()
        )
        abundance_change_values.append(abundance_change)
        cone_gain_values.append(
            {
                "mean": float(cone_gain.mean().item()),
                "std": float(cone_gain.std(unbiased=False).item()),
                "min": float(cone_gain.min().item()),
                "max": float(cone_gain.max().item()),
            }
        )
        span_diagnostics.append(span_info)
        patch_records.append(
            {
                "patch": batch_index,
                "metrics": patch_metrics,
                "simplex_mse": simplex_mse,
                "cone_mse": cone_mse,
                "mean_absolute_simplex_abundance_change": abundance_change,
                "cone_gain": cone_gain_values[-1],
                "linear_span_diagnostics": span_info,
            }
        )

    averages = {
        name: averager.average() for name, averager in metric_sets.items()
    }
    base = averages["base"]
    simplex = averages["simplex"]
    cone = averages["cone"]
    span = averages["linear_span"]
    pca = averages["pca_rank_k"]

    summary = {
        "dataset": cfg.dataset,
        "stage1_checkpoint": cfg.stage1_checkpoint,
        "stage1_epoch": int(stage1_state.get("epoch", -1)),
        "num_endmembers": int(stage1.num_endmembers),
        "oracle_steps": cfg.oracle_steps,
        "oracle_lr": cfg.oracle_lr,
        "metrics": averages,
        "headroom": {
            "simplex_psnr_over_base": simplex["PSNR"] - base["PSNR"],
            "simplex_sam_over_base": base["SAM"] - simplex["SAM"],
            "cone_psnr_over_simplex": cone["PSNR"] - simplex["PSNR"],
            "cone_sam_over_simplex": simplex["SAM"] - cone["SAM"],
            "span_psnr_over_cone": span["PSNR"] - cone["PSNR"],
            "span_sam_over_cone": cone["SAM"] - span["SAM"],
            "pca_psnr_over_span": pca["PSNR"] - span["PSNR"],
            "pca_sam_over_span": span["SAM"] - pca["SAM"],
        },
        "mean_absolute_simplex_abundance_change": float(
            np.mean(abundance_change_values)
        ),
        "cone_gain_mean": float(
            np.mean([item["mean"] for item in cone_gain_values])
        ),
        "cone_gain_std_mean": float(
            np.mean([item["std"] for item in cone_gain_values])
        ),
        "linear_span_negative_value_ratio": float(
            np.mean([item["negative_value_ratio"] for item in span_diagnostics])
        ),
        "linear_span_above_one_ratio": float(
            np.mean([item["above_one_ratio"] for item in span_diagnostics])
        ),
        "linear_span_coefficient_negative_ratio": float(
            np.mean(
                [item["coefficient_negative_ratio"] for item in span_diagnostics]
            )
        ),
        "patches": patch_records,
    }

    if np.isfinite(cfg.current_stage2_psnr):
        summary["remaining_psnr_to_simplex"] = (
            simplex["PSNR"] - cfg.current_stage2_psnr
        )
    if np.isfinite(cfg.current_stage2_sam):
        summary["remaining_sam_to_simplex"] = (
            cfg.current_stage2_sam - simplex["SAM"]
        )

    output_dir = os.path.join(
        cfg.output_root,
        "stage2_physical_ceiling",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "representation_ceiling.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 88)
    for name, label in (
        ("base", "Base abundance upsampling"),
        ("simplex", "Simplex abundance oracle"),
        ("cone", "Non-negative cone oracle"),
        ("linear_span", "Frozen-E linear span"),
        ("pca_rank_k", f"GT PCA rank-{stage1.num_endmembers}"),
    ):
        values = averages[name]
        print(
            f"{label:28s}: PSNR={values['PSNR']:.4f}, "
            f"SAM={values['SAM']:.4f} deg"
        )
    print("-" * 88)
    print(
        f"Predictor headroom to simplex : "
        f"{simplex['PSNR'] - cfg.current_stage2_psnr:+.4f} dB"
        if np.isfinite(cfg.current_stage2_psnr)
        else "Predictor headroom not computed: current Stage-2 PSNR not supplied."
    )
    print(
        f"Gain/sum-to-one bottleneck    : "
        f"{cone['PSNR'] - simplex['PSNR']:+.4f} dB"
    )
    print(
        f"Nonnegative/E-placement gap   : "
        f"{span['PSNR'] - cone['PSNR']:+.4f} dB"
    )
    print(
        f"Learned-span/rank gap         : "
        f"{pca['PSNR'] - span['PSNR']:+.4f} dB"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
