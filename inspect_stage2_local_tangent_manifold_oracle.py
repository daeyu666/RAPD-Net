"""GT local tangent-manifold oracle for RAPD-Net null-space coefficients.

No network is trained. The tangent basis at every HR pixel is constructed only
from local differences in the bicubic-upsampled LR-HSI null coefficient field.
GT is used only to project the missing HR null residual onto that LR-derived
local tangent space, yielding an upper bound for signed manifold-constrained
null residual generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from inspect_stage2_observability_ceiling import (
    build_observability_operators,
    project_coefficients,
)
from metrics import MetricAverager, calc_metrics
from train_stage2_coefficients import (
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from utils import ensure_dir, get_device, move_to_device, set_seed


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def _parse_dimensions(text: str) -> List[int]:
    dims = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not dims or dims[0] < 1:
        raise ValueError("tangent_dimensions must contain positive integers")
    return dims


def parse_specific_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    parser.add_argument("--projector_tolerance", type=float, default=1e-6)
    parser.add_argument("--tangent_kernel_size", type=int, default=5)
    parser.add_argument("--tangent_dilation", type=int, default=2)
    parser.add_argument("--tangent_dimensions", type=str, default="2,4,6,8")
    parser.add_argument(
        "--tangent_chunk_pixels",
        type=int,
        default=2048,
        help="Maximum HR pixels processed by one batched SVD call.",
    )
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

    if cfg.anchor_ridge_ratio <= 0:
        raise ValueError("anchor_ridge_ratio must be positive")
    if cfg.projector_tolerance <= 0:
        raise ValueError("projector_tolerance must be positive")
    if cfg.tangent_kernel_size < 3 or cfg.tangent_kernel_size % 2 == 0:
        raise ValueError("tangent_kernel_size must be an odd integer >= 3")
    if cfg.tangent_dilation < 1:
        raise ValueError("tangent_dilation must be >= 1")
    if cfg.tangent_chunk_pixels < 1:
        raise ValueError("tangent_chunk_pixels must be >= 1")
    cfg.tangent_dimensions = _parse_dimensions(cfg.tangent_dimensions)
    return cfg


def local_difference_patches(
    source: torch.Tensor,
    kernel_size: int,
    dilation: int,
) -> torch.Tensor:
    """Return local LR-null neighbor-minus-center differences as [P,C,K]."""
    if source.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tuple(source.shape)}")
    radius = dilation * (kernel_size - 1) // 2
    if source.size(-2) <= radius or source.size(-1) <= radius:
        raise ValueError(
            f"Spatial size {tuple(source.shape[-2:])} too small for radius={radius}"
        )

    padded = F.pad(
        source,
        (radius, radius, radius, radius),
        mode="reflect",
    )
    patches = F.unfold(
        padded,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=0,
        stride=1,
    )
    n, channels, height, width = source.shape
    elements = kernel_size * kernel_size
    patches = patches.view(n, channels, elements, height, width)
    differences = patches - source.unsqueeze(2)
    return (
        differences.permute(0, 3, 4, 1, 2)
        .reshape(n * height * width, channels, elements)
        .contiguous()
    )


@torch.no_grad()
def tangent_project_missing_residual(
    null_seed: torch.Tensor,
    missing_null: torch.Tensor,
    dimensions: Sequence[int],
    kernel_size: int,
    dilation: int,
    chunk_pixels: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
    """Project GT missing null onto LR-derived local tangent spaces."""
    if null_seed.shape != missing_null.shape:
        raise ValueError("null_seed and missing_null must have identical shapes")

    n, channels, height, width = null_seed.shape
    differences = local_difference_patches(
        null_seed,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    targets = (
        missing_null.permute(0, 2, 3, 1)
        .reshape(n * height * width, channels)
        .contiguous()
    )

    max_available = min(channels, kernel_size * kernel_size)
    projections = {d: torch.zeros_like(targets) for d in dimensions}
    singular_capture_sum = {d: 0.0 for d in dimensions}
    singular_total_sum = 0.0

    for start in range(0, targets.size(0), chunk_pixels):
        stop = min(start + chunk_pixels, targets.size(0))
        matrix = differences[start:stop].float()
        target = targets[start:stop].float()

        # D is [P,C,K]. U contains LR-null local spectral tangent directions.
        u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        singular_sq = singular_values.double().square()
        singular_total_sum += float(singular_sq.sum().item())

        for d in dimensions:
            actual_dim = min(int(d), max_available, u.size(-1))
            tangent = u[:, :, :actual_dim]
            coordinates = torch.einsum("pcd,pc->pd", tangent, target)
            projected = torch.einsum("pcd,pd->pc", tangent, coordinates)
            projections[d][start:stop] = projected.to(targets.dtype)
            singular_capture_sum[d] += float(
                singular_sq[:, :actual_dim].sum().item()
            )

    singular_total_sum = max(singular_total_sum, 1e-30)
    projected_fields = {}
    local_variation_capture = {}
    for d in dimensions:
        projected_fields[d] = (
            projections[d]
            .reshape(n, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        local_variation_capture[d] = singular_capture_sum[d] / singular_total_sum

    return projected_fields, local_variation_capture


class OnlinePearson:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().double().reshape(-1)
        y = y.detach().double().reshape(-1)
        self.n += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_x2 += float(x.square().sum().item())
        self.sum_y2 += float(y.square().sum().item())
        self.sum_xy += float((x * y).sum().item())

    def value(self) -> float:
        if self.n <= 1:
            return 0.0
        n = float(self.n)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        vx = self.sum_x2 - self.sum_x * self.sum_x / n
        vy = self.sum_y2 - self.sum_y * self.sum_y / n
        denom = math.sqrt(max(vx * vy, 0.0))
        return cov / denom if denom > 1e-30 else 0.0


@torch.no_grad()
def evaluate(stage1, response, loader, cfg, device) -> Dict[str, object]:
    basis = stage1.get_basis().detach()
    response = response.to(device)
    operators = build_observability_operators(
        basis=basis,
        spectral_response=response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
    )
    null = operators["null_projector"]
    backprojector = operators["backprojector"]

    metric_sets = {
        "analytical_anchor": MetricAverager(),
        "full_null_oracle": MetricAverager(),
        "hr_basis_oracle": MetricAverager(),
    }
    for d in cfg.tangent_dimensions:
        metric_sets[f"tangent_d{d}"] = MetricAverager()

    total_missing_energy = 0.0
    total_remaining_energy = {d: 0.0 for d in cfg.tangent_dimensions}
    total_local_variation_capture = {d: 0.0 for d in cfg.tangent_dimensions}
    null_pearson = {d: OnlinePearson() for d in cfg.tangent_dimensions}
    max_tangent_null_leakage = {d: 0.0 for d in cfg.tangent_dimensions}
    max_anchor_null_seed_difference = 0.0
    batch_count = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        lr_coefficients = stage1.encode(lr_hsi, basis=basis)
        upsampled_coefficients = F.interpolate(
            lr_coefficients,
            size=gt.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        base_hsi = stage1.decode(upsampled_coefficients, basis=basis)
        base_msi = torch.einsum("mb,nbhw->nmhw", response, base_hsi)

        msi_residual = hr_msi - base_msi
        analytic_residual = torch.einsum(
            "rm,nmhw->nrhw",
            backprojector.to(msi_residual),
            msi_residual,
        )
        anchor_coefficients = upsampled_coefficients + analytic_residual
        anchor_hsi = stage1.decode(anchor_coefficients, basis=basis)

        gt_coefficients = stage1.encode(gt, basis=basis)
        gt_null = project_coefficients(null.to(gt_coefficients), gt_coefficients)
        null_seed = project_coefficients(
            null.to(upsampled_coefficients),
            upsampled_coefficients,
        )
        anchor_null = project_coefficients(
            null.to(anchor_coefficients),
            anchor_coefficients,
        )
        max_anchor_null_seed_difference = max(
            max_anchor_null_seed_difference,
            float((anchor_null - null_seed).abs().max().item()),
        )

        missing_null = gt_null - null_seed
        tangent_residuals, local_capture = tangent_project_missing_residual(
            null_seed=null_seed,
            missing_null=missing_null,
            dimensions=cfg.tangent_dimensions,
            kernel_size=cfg.tangent_kernel_size,
            dilation=cfg.tangent_dilation,
            chunk_pixels=cfg.tangent_chunk_pixels,
        )

        full_null_hsi = stage1.decode(
            anchor_coefficients + missing_null,
            basis=basis,
        )
        basis_oracle_hsi = stage1.decode(gt_coefficients, basis=basis)

        metric_sets["analytical_anchor"].update(
            calc_metrics(anchor_hsi, gt, cfg.scale_ratio)
        )
        metric_sets["full_null_oracle"].update(
            calc_metrics(full_null_hsi, gt, cfg.scale_ratio)
        )
        metric_sets["hr_basis_oracle"].update(
            calc_metrics(basis_oracle_hsi, gt, cfg.scale_ratio)
        )

        total_missing_energy += float(missing_null.double().square().sum().item())

        for d in cfg.tangent_dimensions:
            tangent = project_coefficients(
                null.to(tangent_residuals[d]),
                tangent_residuals[d],
            )
            tangent_hsi = stage1.decode(
                anchor_coefficients + tangent,
                basis=basis,
            )
            metric_sets[f"tangent_d{d}"].update(
                calc_metrics(tangent_hsi, gt, cfg.scale_ratio)
            )

            remaining = missing_null - tangent
            total_remaining_energy[d] += float(
                remaining.double().square().sum().item()
            )
            total_local_variation_capture[d] += float(local_capture[d])
            null_pearson[d].update(null_seed + tangent, gt_null)
            max_tangent_null_leakage[d] = max(
                max_tangent_null_leakage[d],
                float(
                    torch.einsum(
                        "mr,nrhw->nmhw",
                        operators["reduced_response"].to(tangent),
                        tangent,
                    ).abs().max().item()
                ),
            )
        batch_count += 1

    metrics = {name: meter.average() for name, meter in metric_sets.items()}
    total_missing_energy = max(total_missing_energy, 1e-30)

    tangent_diagnostics = {}
    for d in cfg.tangent_dimensions:
        tangent_diagnostics[str(d)] = {
            "missing_null_mse_captured": (
                1.0 - total_remaining_energy[d] / total_missing_energy
            ),
            "null_relative_rmse": math.sqrt(
                total_remaining_energy[d] / total_missing_energy
            ),
            "null_pearson": null_pearson[d].value(),
            "mean_local_variation_singular_energy_captured": (
                total_local_variation_capture[d] / max(batch_count, 1)
            ),
            "null_leakage_max": max_tangent_null_leakage[d],
            "psnr_gain_over_anchor": (
                metrics[f"tangent_d{d}"]["PSNR"]
                - metrics["analytical_anchor"]["PSNR"]
            ),
            "sam_improvement_over_anchor": (
                metrics["analytical_anchor"]["SAM"]
                - metrics[f"tangent_d{d}"]["SAM"]
            ),
        }

    return {
        "metrics": metrics,
        "diagnostics": {
            "basis_rank": int(stage1.basis_rank),
            "observable_rank": int(operators["observable_rank"].item()),
            "null_dimension": int(
                stage1.basis_rank - int(operators["observable_rank"].item())
            ),
            "tangent_kernel_size": int(cfg.tangent_kernel_size),
            "tangent_dilation": int(cfg.tangent_dilation),
            "tangent_dimensions": [int(d) for d in cfg.tangent_dimensions],
            "anchor_null_seed_max_abs_difference": max_anchor_null_seed_difference,
            "tangent": tangent_diagnostics,
        },
    }


def print_report(result: Dict[str, object]) -> None:
    metrics = result["metrics"]
    diagnostics = result["diagnostics"]
    print("=" * 108)
    print("RAPD-Net GT Local Tangent-Manifold Oracle")
    print("=" * 108)
    print(
        "Geometry: "
        f"{diagnostics['tangent_kernel_size']}x{diagnostics['tangent_kernel_size']}, "
        f"dilation={diagnostics['tangent_dilation']} | "
        f"rank(RU)={diagnostics['observable_rank']}/{diagnostics['basis_rank']}, "
        f"null_dim={diagnostics['null_dimension']}"
    )
    print("-" * 108)

    anchor = metrics["analytical_anchor"]
    print(
        f"{'Analytical SRF anchor':32s}: "
        f"PSNR={anchor['PSNR']:.4f} dB, SAM={anchor['SAM']:.4f} deg"
    )
    for d in diagnostics["tangent_dimensions"]:
        values = metrics[f"tangent_d{d}"]
        diag = diagnostics["tangent"][str(d)]
        print(
            f"{('GT tangent oracle d=' + str(d)):32s}: "
            f"PSNR={values['PSNR']:.4f} dB, SAM={values['SAM']:.4f} deg | "
            f"gain={diag['psnr_gain_over_anchor']:+.4f} dB | "
            f"null rRMSE={diag['null_relative_rmse']:.4f}, "
            f"r={diag['null_pearson']:.4f}"
        )

    full_null = metrics["full_null_oracle"]
    basis = metrics["hr_basis_oracle"]
    print(
        f"{'GT full-null oracle':32s}: "
        f"PSNR={full_null['PSNR']:.4f} dB, SAM={full_null['SAM']:.4f} deg"
    )
    print(
        f"{'HR basis oracle':32s}: "
        f"PSNR={basis['PSNR']:.4f} dB, SAM={basis['SAM']:.4f} deg"
    )
    print("-" * 108)

    for d in diagnostics["tangent_dimensions"]:
        diag = diagnostics["tangent"][str(d)]
        print(
            f"d={d:2d} | missing-null MSE captured="
            f"{100.0 * diag['missing_null_mse_captured']:.2f}% | "
            f"LR local variation energy represented="
            f"{100.0 * diag['mean_local_variation_singular_energy_captured']:.2f}% | "
            f"null leak={diag['null_leakage_max']:.3e}"
        )

    print(
        "Numerical check | anchor-null vs LR-null seed="
        f"{diagnostics['anchor_null_seed_max_abs_difference']:.3e}"
    )
    print("=" * 108)


def main() -> None:
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    stage1, state = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    response = build_spectral_response(info).to(device)

    max_dim = min(
        stage1.basis_rank,
        cfg.tangent_kernel_size * cfg.tangent_kernel_size,
    )
    if max(cfg.tangent_dimensions) > max_dim:
        raise ValueError(
            f"Requested tangent dimension exceeds local matrix rank bound {max_dim}"
        )

    result = evaluate(stage1, response, test_loader, cfg, device)
    print_report(result)

    output_dir = os.path.join(
        cfg.output_root,
        "stage2_local_tangent_manifold_oracle",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "local_tangent_manifold_oracle.json")
    payload = {
        "dataset": cfg.dataset,
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "msi_mode": cfg.msi_mode,
        "srf_band_set": cfg.srf_band_set,
        **result,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
