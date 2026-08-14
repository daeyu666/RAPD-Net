"""Diagnose how predictable high-resolution null-space coefficients are from LR-HSI.

This is a no-training diagnostic for the current RAPD-Net Stage 2. It keeps the
frozen Stage-1 affine spectral basis and the configured MSI spectral response,
then asks three questions:

1. How much of the GT null-space coefficient field is already explained by
   bicubic upsampling of LR-HSI coefficients?
2. Of the remaining null-space detail, how much energy lies below versus above
   the LR Nyquist limit implied by the spatial scale ratio?
3. Is the missing null-space detail spatially aligned with HR-MSI edges strongly
   enough for MSI-guided sharpening to be plausible?

Two frequency oracles are also reported:
- anchor + only the GT missing-null low-frequency component;
- anchor + only the GT missing-null high-frequency component.

No learned Stage-2 weights are loaded or optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

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


ORACLE_NAMES = (
    "analytical_anchor",
    "null_low_frequency_oracle",
    "null_high_frequency_oracle",
    "full_null_oracle",
    "hr_basis_oracle",
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
    parser.add_argument(
        "--edge_quantile",
        type=float,
        default=0.9,
        help="Quantile used for null-detail/MSI-edge overlap diagnostics.",
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
    if cfg.scale_ratio <= 1:
        raise ValueError("scale_ratio must be greater than 1")
    if not 0.5 < cfg.edge_quantile < 1.0:
        raise ValueError("edge_quantile must be in (0.5, 1.0)")
    return cfg


@torch.no_grad()
def split_at_lr_nyquist(
    x: torch.Tensor,
    scale_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Orthogonally split HR spatial frequencies at the LR Nyquist box.

    HR frequencies are measured in cycles / HR pixel. Downsampling by s makes
    the LR Nyquist limit equal to 0.5 / s along each HR frequency axis.
    A rectangular passband is used because Cartesian sampling has a square
    Nyquist region.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
    height, width = x.shape[-2:]
    cutoff = 0.5 / float(scale_ratio)

    fy = torch.fft.fftfreq(height, d=1.0, device=x.device)
    fx = torch.fft.fftfreq(width, d=1.0, device=x.device)
    low_mask = (
        (fy.abs()[:, None] <= cutoff)
        & (fx.abs()[None, :] <= cutoff)
    ).to(dtype=x.dtype)
    low_mask = low_mask.view(1, 1, height, width)

    spectrum = torch.fft.fft2(x, norm="ortho")
    low = torch.fft.ifft2(spectrum * low_mask, norm="ortho").real
    high = torch.fft.ifft2(spectrum * (1.0 - low_mask), norm="ortho").real
    return low, high, cutoff


def spatial_gradient_energy(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.double().square().sum() + dy.double().square().sum()


def sobel_edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Return a single HR spatial edge-magnitude map averaged over MSI bands."""
    if x.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")
    kernel_x = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    kernel_y = kernel_x.transpose(0, 1)
    channels = x.size(1)
    weight_x = kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    weight_y = kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, weight_x, padding=1, groups=channels)
    gy = F.conv2d(x, weight_y, padding=1, groups=channels)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
    return magnitude.mean(dim=1, keepdim=True)


def coefficient_detail_magnitude(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.square().sum(dim=1, keepdim=True) + 1e-12)


class OnlinePearson:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x64 = x.detach().double().reshape(-1)
        y64 = y.detach().double().reshape(-1)
        if x64.numel() != y64.numel():
            raise ValueError("Pearson inputs must contain the same number of values")
        self.n += int(x64.numel())
        self.sum_x += float(x64.sum().item())
        self.sum_y += float(y64.sum().item())
        self.sum_x2 += float(x64.square().sum().item())
        self.sum_y2 += float(y64.square().sum().item())
        self.sum_xy += float((x64 * y64).sum().item())

    def value(self) -> float:
        if self.n <= 1:
            return float("nan")
        n = float(self.n)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        var_x = self.sum_x2 - self.sum_x * self.sum_x / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        denom = math.sqrt(max(var_x * var_y, 0.0))
        if denom <= 1e-30:
            return 0.0
        return cov / denom


@torch.no_grad()
def top_quantile_overlap(
    detail_map: torch.Tensor,
    edge_map: torch.Tensor,
    quantile: float,
) -> Tuple[float, float]:
    """Return P(MSI edge | null detail) and IoU for per-sample top-quantile sets."""
    conditional_values = []
    iou_values = []
    for sample in range(detail_map.size(0)):
        d = detail_map[sample].reshape(-1)
        e = edge_map[sample].reshape(-1)
        d_threshold = torch.quantile(d.float(), quantile)
        e_threshold = torch.quantile(e.float(), quantile)
        d_mask = d >= d_threshold
        e_mask = e >= e_threshold
        intersection = (d_mask & e_mask).sum().float()
        detail_count = d_mask.sum().float().clamp_min(1.0)
        union = (d_mask | e_mask).sum().float().clamp_min(1.0)
        conditional_values.append(float((intersection / detail_count).item()))
        iou_values.append(float((intersection / union).item()))
    return (
        sum(conditional_values) / max(len(conditional_values), 1),
        sum(iou_values) / max(len(iou_values), 1),
    )


@torch.no_grad()
def evaluate(
    stage1,
    spectral_response: torch.Tensor,
    loader,
    cfg,
    device: torch.device,
) -> Dict[str, object]:
    basis = stage1.get_basis().detach()
    response = spectral_response.to(device)
    operators = build_observability_operators(
        basis=basis,
        spectral_response=response,
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
    )
    observable = operators["observable_projector"]
    null = operators["null_projector"]
    backprojector = operators["backprojector"]

    metric_sets = {name: MetricAverager() for name in ORACLE_NAMES}

    total_gt_null_energy = 0.0
    total_lr_null_energy = 0.0
    total_missing_null_energy = 0.0
    total_missing_low_energy = 0.0
    total_missing_high_energy = 0.0
    total_missing_gradient_energy = 0.0
    total_gt_null_gradient_energy = 0.0
    total_lr_null_gradient_energy = 0.0
    total_gt_null_sum = 0.0
    total_lr_null_sum = 0.0
    total_gt_null_sum2 = 0.0
    total_lr_null_sum2 = 0.0
    total_gt_null_lr_product = 0.0
    total_coeff_count = 0

    missing_edge_corr = OnlinePearson()
    high_missing_edge_corr = OnlinePearson()
    low_missing_edge_corr = OnlinePearson()

    conditional_overlap_sum = 0.0
    high_conditional_overlap_sum = 0.0
    iou_sum = 0.0
    high_iou_sum = 0.0
    overlap_batches = 0

    max_missing_equivalence_error = 0.0
    max_null_leakage = 0.0
    cutoff = 0.5 / float(cfg.scale_ratio)

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
        lr_null_up = project_coefficients(
            null.to(upsampled_coefficients),
            upsampled_coefficients,
        )

        # Since the analytical SRF correction lies in the observable row space,
        # this equals the residual null component left after the anchor.
        missing_null = gt_null - lr_null_up
        target_residual = gt_coefficients - anchor_coefficients
        projected_target_null = project_coefficients(
            null.to(target_residual),
            target_residual,
        )
        max_missing_equivalence_error = max(
            max_missing_equivalence_error,
            float((missing_null - projected_target_null).abs().max().item()),
        )

        missing_low, missing_high, cutoff = split_at_lr_nyquist(
            missing_null,
            cfg.scale_ratio,
        )

        low_oracle_hsi = stage1.decode(
            anchor_coefficients + missing_low,
            basis=basis,
        )
        high_oracle_hsi = stage1.decode(
            anchor_coefficients + missing_high,
            basis=basis,
        )
        full_null_hsi = stage1.decode(
            anchor_coefficients + missing_null,
            basis=basis,
        )
        basis_oracle_hsi = stage1.decode(gt_coefficients, basis=basis)

        predictions = {
            "analytical_anchor": anchor_hsi,
            "null_low_frequency_oracle": low_oracle_hsi,
            "null_high_frequency_oracle": high_oracle_hsi,
            "full_null_oracle": full_null_hsi,
            "hr_basis_oracle": basis_oracle_hsi,
        }
        for name, prediction in predictions.items():
            metric_sets[name].update(
                calc_metrics(prediction, gt, cfg.scale_ratio)
            )

        gt64 = gt_null.double()
        lr64 = lr_null_up.double()
        missing64 = missing_null.double()
        low64 = missing_low.double()
        high64 = missing_high.double()

        total_gt_null_energy += float(gt64.square().sum().item())
        total_lr_null_energy += float(lr64.square().sum().item())
        total_missing_null_energy += float(missing64.square().sum().item())
        total_missing_low_energy += float(low64.square().sum().item())
        total_missing_high_energy += float(high64.square().sum().item())
        total_missing_gradient_energy += float(
            spatial_gradient_energy(missing_null).item()
        )
        total_gt_null_gradient_energy += float(
            spatial_gradient_energy(gt_null).item()
        )
        total_lr_null_gradient_energy += float(
            spatial_gradient_energy(lr_null_up).item()
        )

        total_gt_null_sum += float(gt64.sum().item())
        total_lr_null_sum += float(lr64.sum().item())
        total_gt_null_sum2 += float(gt64.square().sum().item())
        total_lr_null_sum2 += float(lr64.square().sum().item())
        total_gt_null_lr_product += float((gt64 * lr64).sum().item())
        total_coeff_count += int(gt_null.numel())

        msi_edges = sobel_edge_magnitude(hr_msi)
        missing_map = coefficient_detail_magnitude(missing_null)
        low_missing_map = coefficient_detail_magnitude(missing_low)
        high_missing_map = coefficient_detail_magnitude(missing_high)

        missing_edge_corr.update(missing_map, msi_edges)
        low_missing_edge_corr.update(low_missing_map, msi_edges)
        high_missing_edge_corr.update(high_missing_map, msi_edges)

        conditional, iou = top_quantile_overlap(
            missing_map,
            msi_edges,
            cfg.edge_quantile,
        )
        high_conditional, high_iou = top_quantile_overlap(
            high_missing_map,
            msi_edges,
            cfg.edge_quantile,
        )
        conditional_overlap_sum += conditional
        high_conditional_overlap_sum += high_conditional
        iou_sum += iou
        high_iou_sum += high_iou
        overlap_batches += 1

        null_msi = torch.einsum(
            "mr,nrhw->nmhw",
            operators["reduced_response"].to(missing_null),
            missing_null,
        )
        max_null_leakage = max(
            max_null_leakage,
            float(null_msi.abs().max().item()),
        )

    metrics = {name: metric_sets[name].average() for name in ORACLE_NAMES}

    gt_energy = max(total_gt_null_energy, 1e-30)
    missing_energy = max(total_missing_null_energy, 1e-30)
    band_energy = max(total_missing_low_energy + total_missing_high_energy, 1e-30)

    n = float(max(total_coeff_count, 1))
    gt_mean = total_gt_null_sum / n
    lr_mean = total_lr_null_sum / n
    covariance = total_gt_null_lr_product / n - gt_mean * lr_mean
    gt_variance = total_gt_null_sum2 / n - gt_mean * gt_mean
    lr_variance = total_lr_null_sum2 / n - lr_mean * lr_mean
    coeff_pearson = covariance / math.sqrt(
        max(gt_variance * lr_variance, 1e-30)
    )

    rank = int(operators["observable_rank"].item())
    basis_rank = int(stage1.basis_rank)
    null_dimension = basis_rank - rank

    diagnostics: Dict[str, object] = {
        "basis_rank": basis_rank,
        "msi_channels": int(response.size(0)),
        "observable_rank": rank,
        "null_dimension": null_dimension,
        "scale_ratio": int(cfg.scale_ratio),
        "lr_nyquist_cutoff_cycles_per_hr_pixel": float(cutoff),
        "gt_null_coefficient_energy": total_gt_null_energy,
        "upsampled_lr_null_coefficient_energy": total_lr_null_energy,
        "missing_null_coefficient_energy": total_missing_null_energy,
        "lr_null_energy_capture_fraction": 1.0 - total_missing_null_energy / gt_energy,
        "lr_null_relative_rmse": math.sqrt(total_missing_null_energy / gt_energy),
        "lr_vs_gt_null_coefficient_pearson": coeff_pearson,
        "missing_null_low_frequency_energy_share": total_missing_low_energy / band_energy,
        "missing_null_high_frequency_energy_share": total_missing_high_energy / band_energy,
        "missing_null_fft_energy_closure": (
            total_missing_low_energy + total_missing_high_energy
        ) / missing_energy,
        "missing_null_gradient_energy": total_missing_gradient_energy,
        "gt_null_gradient_energy": total_gt_null_gradient_energy,
        "upsampled_lr_null_gradient_energy": total_lr_null_gradient_energy,
        "missing_vs_gt_null_gradient_energy_fraction": (
            total_missing_gradient_energy
            / max(total_gt_null_gradient_energy, 1e-30)
        ),
        "upsampled_lr_vs_gt_null_gradient_energy_fraction": (
            total_lr_null_gradient_energy
            / max(total_gt_null_gradient_energy, 1e-30)
        ),
        "missing_null_magnitude_vs_msi_edge_pearson": missing_edge_corr.value(),
        "missing_null_lowfreq_magnitude_vs_msi_edge_pearson": low_missing_edge_corr.value(),
        "missing_null_highfreq_magnitude_vs_msi_edge_pearson": high_missing_edge_corr.value(),
        "edge_quantile": float(cfg.edge_quantile),
        "top_null_detail_msi_edge_conditional_overlap": (
            conditional_overlap_sum / max(overlap_batches, 1)
        ),
        "top_high_null_detail_msi_edge_conditional_overlap": (
            high_conditional_overlap_sum / max(overlap_batches, 1)
        ),
        "top_null_detail_msi_edge_iou": iou_sum / max(overlap_batches, 1),
        "top_high_null_detail_msi_edge_iou": high_iou_sum / max(overlap_batches, 1),
        "missing_null_vs_projected_target_null_max_abs_error": (
            max_missing_equivalence_error
        ),
        "missing_null_reduced_response_leakage_max": max_null_leakage,
    }

    gaps = {
        "low_frequency_oracle_gain_over_anchor_psnr": (
            metrics["null_low_frequency_oracle"]["PSNR"]
            - metrics["analytical_anchor"]["PSNR"]
        ),
        "high_frequency_oracle_gain_over_anchor_psnr": (
            metrics["null_high_frequency_oracle"]["PSNR"]
            - metrics["analytical_anchor"]["PSNR"]
        ),
        "full_null_oracle_gain_over_anchor_psnr": (
            metrics["full_null_oracle"]["PSNR"]
            - metrics["analytical_anchor"]["PSNR"]
        ),
        "basis_oracle_headroom_over_anchor_psnr": (
            metrics["hr_basis_oracle"]["PSNR"]
            - metrics["analytical_anchor"]["PSNR"]
        ),
        "full_null_vs_basis_oracle_psnr_gap": (
            metrics["hr_basis_oracle"]["PSNR"]
            - metrics["full_null_oracle"]["PSNR"]
        ),
        "low_frequency_oracle_sam_improvement": (
            metrics["analytical_anchor"]["SAM"]
            - metrics["null_low_frequency_oracle"]["SAM"]
        ),
        "high_frequency_oracle_sam_improvement": (
            metrics["analytical_anchor"]["SAM"]
            - metrics["null_high_frequency_oracle"]["SAM"]
        ),
    }
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "gaps": gaps,
    }


def print_report(result: Dict[str, object]) -> None:
    metrics = result["metrics"]
    diagnostics = result["diagnostics"]
    gaps = result["gaps"]

    print("=" * 108)
    print("RAPD-Net null-space LR predictability diagnosis")
    print("=" * 108)
    print(
        "Coefficient observability : "
        f"rank(RU)={diagnostics['observable_rank']}/{diagnostics['basis_rank']}, "
        f"null_dim={diagnostics['null_dimension']}, "
        f"MSI channels={diagnostics['msi_channels']}"
    )
    print(
        "LR spatial limit          : "
        f"scale={diagnostics['scale_ratio']}x, "
        f"Nyquist cutoff={diagnostics['lr_nyquist_cutoff_cycles_per_hr_pixel']:.4f} "
        "cycles/HR-pixel"
    )
    print("-" * 108)
    print(
        "LR -> HR null coefficient : "
        f"energy capture={100.0 * diagnostics['lr_null_energy_capture_fraction']:.2f}%, "
        f"relative RMSE={diagnostics['lr_null_relative_rmse']:.4f}, "
        f"Pearson={diagnostics['lr_vs_gt_null_coefficient_pearson']:.4f}"
    )
    print(
        "Missing null spectrum     : "
        f"low={100.0 * diagnostics['missing_null_low_frequency_energy_share']:.2f}%, "
        f"high={100.0 * diagnostics['missing_null_high_frequency_energy_share']:.2f}%, "
        f"closure={diagnostics['missing_null_fft_energy_closure']:.6f}"
    )
    print(
        "Spatial gradient energy   : "
        f"LR-up/GT={100.0 * diagnostics['upsampled_lr_vs_gt_null_gradient_energy_fraction']:.2f}%, "
        f"missing/GT={100.0 * diagnostics['missing_vs_gt_null_gradient_energy_fraction']:.2f}%"
    )
    print(
        "MSI edge Pearson          : "
        f"missing={diagnostics['missing_null_magnitude_vs_msi_edge_pearson']:.4f}, "
        f"low={diagnostics['missing_null_lowfreq_magnitude_vs_msi_edge_pearson']:.4f}, "
        f"high={diagnostics['missing_null_highfreq_magnitude_vs_msi_edge_pearson']:.4f}"
    )
    expected = 1.0 - diagnostics["edge_quantile"]
    print(
        f"Top-{100.0 * expected:.0f}% edge overlap       : "
        f"missing->MSI={100.0 * diagnostics['top_null_detail_msi_edge_conditional_overlap']:.2f}% "
        f"(chance~{100.0 * expected:.1f}%), "
        f"high->MSI={100.0 * diagnostics['top_high_null_detail_msi_edge_conditional_overlap']:.2f}%"
    )
    print("-" * 108)

    labels = {
        "analytical_anchor": "Analytical SRF anchor",
        "null_low_frequency_oracle": "GT null low-frequency oracle",
        "null_high_frequency_oracle": "GT null high-frequency oracle",
        "full_null_oracle": "GT full-null oracle",
        "hr_basis_oracle": "HR basis oracle",
    }
    for name in ORACLE_NAMES:
        values = metrics[name]
        print(
            f"{labels[name]:34s}: "
            f"PSNR={values['PSNR']:.4f} dB, "
            f"SAM={values['SAM']:.4f} deg, "
            f"RMSE={values['RMSE']:.8f}"
        )

    print("-" * 108)
    print(
        "Low-frequency null gain   : "
        f"{gaps['low_frequency_oracle_gain_over_anchor_psnr']:+.4f} dB"
    )
    print(
        "High-frequency null gain  : "
        f"{gaps['high_frequency_oracle_gain_over_anchor_psnr']:+.4f} dB"
    )
    print(
        "Full null gain            : "
        f"{gaps['full_null_oracle_gain_over_anchor_psnr']:+.4f} dB"
    )
    print(
        "Numerical checks          : "
        f"null-target={diagnostics['missing_null_vs_projected_target_null_max_abs_error']:.3e}, "
        f"null-leak={diagnostics['missing_null_reduced_response_leakage_max']:.3e}"
    )
    print("=" * 108)


def main() -> None:
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    stage1, stage1_state = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    spectral_response = build_spectral_response(info).to(device)

    result = evaluate(
        stage1,
        spectral_response,
        test_loader,
        cfg,
        device,
    )

    output_dir = os.path.join(
        cfg.output_root,
        "stage2_null_lr_predictability",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "null_lr_predictability.json")
    payload = {
        "dataset": cfg.dataset,
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "stage1_checkpoint_epoch": int(stage1_state.get("epoch", -1)),
        "msi_mode": cfg.msi_mode,
        "srf_band_set": cfg.srf_band_set,
        **result,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print_report(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
