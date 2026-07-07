"""Diagnose SRF-observable coefficient backprojection before changing Stage 2.

This is the first controlled Stage-2 improvement step. It does not modify or
train the current coefficient network. It only tests the analytical mapping

    S = R U_r
    Delta C_obs = S^T (S S^T + lambda I)^(-1) (Z - R X_base)

where ``R`` is the MSI spectral response and ``U_r`` is the frozen Stage-1
spectral basis.

The script compares:

* coefficient bicubic base;
* SRF analytical backprojection anchors under several ridge ratios;
* HR basis oracle.

The diagnostic answers one question only: can the known SRF-basis relation
supply a better deterministic coefficient starting point before any new neural
network structure is introduced?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage1_spectral_basis import Stage1SpectralBasisNet
from train_stage2_coefficients import (
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from utils import ensure_dir, get_device, move_to_device, set_seed


def parse_ridge_ratios(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("All ridge ratios must be positive")
        values.append(value)
    if not values:
        raise ValueError("At least one ridge ratio is required")
    return values


def spectral_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sam_loss: SAMLoss,
) -> Dict[str, float]:
    mse = float(F.mse_loss(prediction, target).item())
    l1 = float(F.l1_loss(prediction, target).item())
    sam_rad = float(sam_loss(prediction, target).item())
    return {
        "l1": l1,
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "sam_deg": sam_rad * 180.0 / math.pi,
    }


def accumulate(
    totals: Dict[str, float],
    values: Dict[str, float],
    weight: int,
) -> None:
    for key, value in values.items():
        totals[key] = totals.get(key, 0.0) + float(value) * weight
    totals["count"] = totals.get("count", 0.0) + weight


def average_totals(totals: Dict[str, float]) -> Dict[str, float]:
    count = max(totals.get("count", 0.0), 1.0)
    return {
        key: value / count
        for key, value in totals.items()
        if key != "count"
    }


def analytical_backprojection(
    msi_residual: torch.Tensor,
    reduced_response: torch.Tensor,
    ridge_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Map an MSI residual to the minimum-energy ridge coefficient residual."""
    if msi_residual.ndim != 4:
        raise ValueError(
            f"Expected MSI residual [N, M, H, W], got {tuple(msi_residual.shape)}"
        )
    if reduced_response.ndim != 2:
        raise ValueError("reduced_response must be [M, r]")
    if msi_residual.size(1) != reduced_response.size(0):
        raise ValueError("MSI channels do not match reduced response rows")

    msi_channels = reduced_response.size(0)
    gram = reduced_response @ reduced_response.transpose(0, 1)
    scale = torch.trace(gram) / max(msi_channels, 1)
    ridge = float(ridge_ratio) * float(scale.detach().item())
    ridge = max(ridge, 1e-12)

    regularized = gram + ridge * torch.eye(
        msi_channels,
        device=gram.device,
        dtype=gram.dtype,
    )
    flat_residual = msi_residual.permute(1, 0, 2, 3).reshape(
        msi_channels,
        -1,
    )
    solved = torch.linalg.solve(regularized, flat_residual)
    flat_coefficient = reduced_response.transpose(0, 1) @ solved
    coefficient_residual = flat_coefficient.reshape(
        reduced_response.size(1),
        msi_residual.size(0),
        msi_residual.size(2),
        msi_residual.size(3),
    ).permute(1, 0, 2, 3).contiguous()

    projected_flat = reduced_response @ flat_coefficient
    projected_msi_residual = projected_flat.reshape(
        msi_channels,
        msi_residual.size(0),
        msi_residual.size(2),
        msi_residual.size(3),
    ).permute(1, 0, 2, 3).contiguous()
    return coefficient_residual, projected_msi_residual, ridge


@torch.no_grad()
def evaluate(
    stage1: Stage1SpectralBasisNet,
    spectral_response: torch.Tensor,
    loader,
    ridge_ratios: List[float],
    device: torch.device,
) -> Dict:
    sam_loss = SAMLoss()
    basis = stage1.get_basis().detach()
    reduced_response = spectral_response @ basis
    singular_values = torch.linalg.svdvals(reduced_response)
    condition_number = float(
        (singular_values.max() / singular_values.min().clamp_min(1e-12)).item()
    )

    totals: Dict[str, Dict[str, float]] = {
        "base": {},
        "oracle": {},
    }
    for ratio in ridge_ratios:
        totals[f"ridge_{ratio:.0e}"] = {}

    diagnostic_totals: Dict[str, Dict[str, float]] = {
        f"ridge_{ratio:.0e}": {} for ratio in ridge_ratios
    }
    first_arrays = None

    coefficient_scale = stage1.coefficient_scale.detach().clamp_min(1e-8)
    scale_view = coefficient_scale.view(1, -1, 1, 1)

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]
        pixel_count = gt.size(0) * gt.size(2) * gt.size(3)

        lr_coefficients = stage1.encode(lr_hsi, basis=basis)
        upsampled_coefficients = F.interpolate(
            lr_coefficients,
            size=gt.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        base_hsi = stage1.decode(upsampled_coefficients, basis=basis)
        base_msi = torch.einsum(
            "mb,nbhw->nmhw",
            spectral_response,
            base_hsi,
        )
        msi_residual = hr_msi - base_msi

        oracle_coefficients = stage1.encode(gt, basis=basis)
        oracle_hsi = stage1.decode(oracle_coefficients, basis=basis)

        accumulate(
            totals["base"],
            spectral_metrics(base_hsi, gt, sam_loss),
            pixel_count,
        )
        accumulate(
            totals["oracle"],
            spectral_metrics(oracle_hsi, gt, sam_loss),
            pixel_count,
        )

        base_msi_mse = float(F.mse_loss(base_msi, hr_msi).item())
        for ratio in ridge_ratios:
            name = f"ridge_{ratio:.0e}"
            delta_coefficients, projected_residual, actual_ridge = (
                analytical_backprojection(
                    msi_residual,
                    reduced_response,
                    ratio,
                )
            )
            anchor_coefficients = upsampled_coefficients + delta_coefficients
            anchor_hsi = stage1.decode(anchor_coefficients, basis=basis)
            anchor_msi = torch.einsum(
                "mb,nbhw->nmhw",
                spectral_response,
                anchor_hsi,
            )
            accumulate(
                totals[name],
                spectral_metrics(anchor_hsi, gt, sam_loss),
                pixel_count,
            )

            anchor_msi_mse = float(F.mse_loss(anchor_msi, hr_msi).item())
            normalized_delta = delta_coefficients / scale_view
            out_of_range = (
                (anchor_hsi < 0.0) | (anchor_hsi > 1.0)
            ).float().mean()
            diagnostics = {
                "actual_ridge": actual_ridge,
                "msi_residual_reduction": 1.0
                - anchor_msi_mse / max(base_msi_mse, 1e-12),
                "normalized_delta_abs_mean": float(
                    normalized_delta.abs().mean().item()
                ),
                "normalized_delta_abs_max": float(
                    normalized_delta.abs().max().item()
                ),
                "out_of_range_ratio": float(out_of_range.item()),
                "projected_residual_fit": 1.0
                - float(
                    F.mse_loss(projected_residual, msi_residual).item()
                )
                / max(float(msi_residual.square().mean().item()), 1e-12),
            }
            accumulate(
                diagnostic_totals[name],
                diagnostics,
                pixel_count,
            )

            if first_arrays is None and ratio == ridge_ratios[0]:
                first_arrays = {
                    "base_hsi": base_hsi.detach().cpu().numpy(),
                    "anchor_hsi": anchor_hsi.detach().cpu().numpy(),
                    "gt": gt.detach().cpu().numpy(),
                    "base_msi": base_msi.detach().cpu().numpy(),
                    "hr_msi": hr_msi.detach().cpu().numpy(),
                    "msi_residual": msi_residual.detach().cpu().numpy(),
                    "delta_coefficients": delta_coefficients.detach().cpu().numpy(),
                    "normalized_delta_coefficients": normalized_delta.detach().cpu().numpy(),
                    "reduced_response": reduced_response.detach().cpu().numpy(),
                }

    results = {
        name: average_totals(values)
        for name, values in totals.items()
    }
    diagnostics = {
        name: average_totals(values)
        for name, values in diagnostic_totals.items()
    }

    base_psnr = results["base"]["psnr"]
    base_sam = results["base"]["sam_deg"]
    for ratio in ridge_ratios:
        name = f"ridge_{ratio:.0e}"
        results[name]["psnr_gain_over_base"] = (
            results[name]["psnr"] - base_psnr
        )
        results[name]["sam_gain_over_base"] = (
            base_sam - results[name]["sam_deg"]
        )
        results[name].update(diagnostics[name])

    best_psnr_name = max(
        (f"ridge_{ratio:.0e}" for ratio in ridge_ratios),
        key=lambda key: results[key]["psnr"],
    )
    best_sam_name = min(
        (f"ridge_{ratio:.0e}" for ratio in ridge_ratios),
        key=lambda key: results[key]["sam_deg"],
    )
    return {
        "reduced_response_shape": list(reduced_response.shape),
        "reduced_response_rank": int(
            torch.linalg.matrix_rank(reduced_response).item()
        ),
        "singular_values": singular_values.detach().cpu().tolist(),
        "condition_number": condition_number,
        "results": results,
        "best_psnr_setting": best_psnr_name,
        "best_sam_setting": best_sam_name,
        "first_arrays": first_arrays,
    }


def write_csv(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    fields = [
        "setting",
        "l1",
        "mse",
        "rmse",
        "psnr",
        "sam_deg",
        "psnr_gain_over_base",
        "sam_gain_over_base",
        "actual_ridge",
        "msi_residual_reduction",
        "projected_residual_fit",
        "normalized_delta_abs_mean",
        "normalized_delta_abs_max",
        "out_of_range_ratio",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for setting, values in payload["results"].items():
            writer.writerow(
                {"setting": setting, **{field: values.get(field, "") for field in fields[1:]}}
            )


def parse_specific_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_basis_checkpoint",
        type=str,
        default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth",
    )
    parser.add_argument(
        "--ridge_ratios",
        type=str,
        default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1",
    )
    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    cfg.stage1_basis_checkpoint = specific.stage1_basis_checkpoint
    cfg.ridge_ratios = parse_ridge_ratios(specific.ridge_ratios)
    if cfg.dataset != "PaviaU" and (
        specific.stage1_basis_checkpoint
        == "./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth"
    ):
        cfg.stage1_basis_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage1_basis",
            cfg.dataset,
            "basis_for_stage2.pth",
        )
    return cfg


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
    spectral_response = build_spectral_response(info).to(device)
    payload = evaluate(
        stage1,
        spectral_response,
        test_loader,
        cfg.ridge_ratios,
        device,
    )
    payload.update(
        {
            "dataset": cfg.dataset,
            "stage1_checkpoint": cfg.stage1_basis_checkpoint,
            "stage1_epoch": int(state.get("epoch", -1)),
            "ridge_ratios": cfg.ridge_ratios,
        }
    )

    output_dir = os.path.join(
        cfg.output_root,
        "stage2_srf_backprojection",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    arrays = payload.pop("first_arrays")
    with open(
        os.path.join(output_dir, "backprojection_sweep.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    write_csv(
        os.path.join(output_dir, "backprojection_sweep.csv"),
        payload,
    )
    if arrays is not None:
        np.savez_compressed(
            os.path.join(output_dir, "backprojection_first_sample.npz"),
            **arrays,
        )

    print("=" * 108)
    print(
        f"S=R@U shape={payload['reduced_response_shape']}, "
        f"rank={payload['reduced_response_rank']}, "
        f"condition={payload['condition_number']:.4e}"
    )
    print("-" * 108)
    for setting, values in payload["results"].items():
        print(
            f"{setting:16s} | PSNR={values['psnr']:.4f} | "
            f"SAM={values['sam_deg']:.4f} deg | "
            f"gain=({values.get('psnr_gain_over_base', 0.0):+.4f} dB, "
            f"{values.get('sam_gain_over_base', 0.0):+.4f} deg) | "
            f"MSI-fit={values.get('msi_residual_reduction', 0.0):.4f} | "
            f"|dC/s|={values.get('normalized_delta_abs_mean', 0.0):.4f}"
        )
    print("-" * 108)
    print(f"Best PSNR setting: {payload['best_psnr_setting']}")
    print(f"Best SAM setting : {payload['best_sam_setting']}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
