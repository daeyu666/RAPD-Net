"""Diagnose the observability ceiling of the current RAPD-Net Stage 2.

No network is trained. The script uses the frozen Stage-1 affine spectral basis
and configured MSI spectral response to decompose the ground-truth coefficient
residual relative to the analytical SRF anchor into observable and null parts.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from train_stage2_coefficients import build_spectral_response, load_stage1_basis_checkpoint
from utils import ensure_dir, get_device, move_to_device, set_seed

ORACLE_NAMES = (
    "coefficient_upsampling_base",
    "analytical_anchor",
    "gt_observable_oracle",
    "gt_null_oracle",
    "full_coefficient_oracle",
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
            cfg.checkpoint_root, "stage1_basis", cfg.dataset, "basis_for_stage2.pth"
        )
    if cfg.anchor_ridge_ratio <= 0:
        raise ValueError("anchor_ridge_ratio must be positive")
    if cfg.projector_tolerance <= 0:
        raise ValueError("projector_tolerance must be positive")
    return cfg


@torch.no_grad()
def build_observability_operators(
    basis: torch.Tensor,
    spectral_response: torch.Tensor,
    anchor_ridge_ratio: float,
    projector_tolerance: float,
) -> Dict[str, torch.Tensor]:
    basis = basis.detach().float()
    response = spectral_response.detach().float()
    reduced = response @ basis

    _, singular_values, vh = torch.linalg.svd(reduced, full_matrices=True)
    threshold = projector_tolerance * singular_values.max().clamp_min(1e-12)
    rank = int((singular_values > threshold).sum().item())

    row_basis = vh[:rank].transpose(0, 1).contiguous()
    observable = row_basis @ row_basis.transpose(0, 1)
    identity = torch.eye(basis.size(1), device=basis.device, dtype=basis.dtype)
    null = identity - observable

    gram = reduced @ reduced.transpose(0, 1)
    gram_scale = torch.trace(gram) / max(reduced.size(0), 1)
    actual_ridge = anchor_ridge_ratio * gram_scale
    regularized = gram + actual_ridge * torch.eye(
        reduced.size(0), device=reduced.device, dtype=reduced.dtype
    )
    inverse = torch.linalg.solve(
        regularized,
        torch.eye(reduced.size(0), device=reduced.device, dtype=reduced.dtype),
    )
    backprojector = reduced.transpose(0, 1) @ inverse

    return {
        "reduced_response": reduced,
        "singular_values": singular_values,
        "rank_threshold": threshold.reshape(()),
        "observable_rank": torch.tensor(rank, device=basis.device, dtype=torch.int64),
        "observable_projector": observable,
        "null_projector": null,
        "backprojector": backprojector,
        "actual_anchor_ridge": actual_ridge.reshape(()),
    }


def project_coefficients(projector: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rk,nkhw->nrhw", projector, coefficients)


@torch.no_grad()
def evaluate(stage1, spectral_response: torch.Tensor, loader, cfg, device: torch.device) -> Dict[str, object]:
    basis = stage1.get_basis().detach()
    operators = build_observability_operators(
        basis=basis,
        spectral_response=spectral_response.to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        projector_tolerance=cfg.projector_tolerance,
    )
    response = spectral_response.to(device)
    backprojector = operators["backprojector"]
    observable = operators["observable_projector"]
    null = operators["null_projector"]

    metric_sets = {name: MetricAverager() for name in ORACLE_NAMES}
    total_target_energy = 0.0
    total_observable_energy = 0.0
    total_null_energy = 0.0
    total_cross_inner = 0.0
    total_residual_abs = 0.0
    total_observable_abs = 0.0
    total_null_abs = 0.0
    total_coeff_values = 0
    max_full_basis_difference = 0.0

    identity = torch.eye(stage1.basis_rank, device=device, dtype=observable.dtype)
    projector_checks = {
        "observable_projector_idempotence_error": float((observable @ observable - observable).abs().max().item()),
        "null_projector_idempotence_error": float((null @ null - null).abs().max().item()),
        "projector_complement_error": float((observable + null - identity).abs().max().item()),
        "projector_orthogonality_error": float((observable @ null).abs().max().item()),
        "reduced_response_null_leakage_max": float((operators["reduced_response"] @ null).abs().max().item()),
    }

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        hr_msi = batch["hr_msi"]
        gt = batch["gt"]

        lr_coefficients = stage1.encode(lr_hsi, basis=basis)
        upsampled_coefficients = F.interpolate(
            lr_coefficients, size=gt.shape[-2:], mode="bicubic", align_corners=False
        )
        base_hsi = stage1.decode(upsampled_coefficients, basis=basis)
        base_msi = torch.einsum("mb,nbhw->nmhw", response, base_hsi)

        msi_residual = hr_msi - base_msi
        analytic_residual = torch.einsum(
            "rm,nmhw->nrhw", backprojector.to(msi_residual), msi_residual
        )
        anchor_coefficients = upsampled_coefficients + analytic_residual
        anchor_hsi = stage1.decode(anchor_coefficients, basis=basis)

        gt_coefficients = stage1.encode(gt, basis=basis)
        target_residual = gt_coefficients - anchor_coefficients
        target_observable = project_coefficients(observable.to(target_residual), target_residual)
        target_null = project_coefficients(null.to(target_residual), target_residual)

        observable_hsi = stage1.decode(anchor_coefficients + target_observable, basis=basis)
        null_hsi = stage1.decode(anchor_coefficients + target_null, basis=basis)
        full_coefficients = anchor_coefficients + target_observable + target_null
        full_hsi = stage1.decode(full_coefficients, basis=basis)
        basis_oracle_hsi = stage1.decode(gt_coefficients, basis=basis)

        predictions = {
            "coefficient_upsampling_base": base_hsi,
            "analytical_anchor": anchor_hsi,
            "gt_observable_oracle": observable_hsi,
            "gt_null_oracle": null_hsi,
            "full_coefficient_oracle": full_hsi,
            "hr_basis_oracle": basis_oracle_hsi,
        }
        for name, prediction in predictions.items():
            metric_sets[name].update(calc_metrics(prediction, gt, cfg.scale_ratio))

        target64 = target_residual.double()
        observable64 = target_observable.double()
        null64 = target_null.double()
        total_target_energy += float(target64.square().sum().item())
        total_observable_energy += float(observable64.square().sum().item())
        total_null_energy += float(null64.square().sum().item())
        total_cross_inner += float((observable64 * null64).sum().item())
        total_residual_abs += float(target64.abs().sum().item())
        total_observable_abs += float(observable64.abs().sum().item())
        total_null_abs += float(null64.abs().sum().item())
        total_coeff_values += target_residual.numel()
        max_full_basis_difference = max(
            max_full_basis_difference,
            float((full_hsi - basis_oracle_hsi).abs().max().item()),
        )

    metrics = {name: metric_sets[name].average() for name in ORACLE_NAMES}
    target_energy = max(total_target_energy, 1e-30)
    observable_energy_share = total_observable_energy / target_energy
    null_energy_share = total_null_energy / target_energy

    anchor_mse = metrics["analytical_anchor"]["RMSE"] ** 2
    observable_mse = metrics["gt_observable_oracle"]["RMSE"] ** 2
    null_mse = metrics["gt_null_oracle"]["RMSE"] ** 2
    full_mse = metrics["full_coefficient_oracle"]["RMSE"] ** 2
    reducible_mse = max(anchor_mse - full_mse, 1e-30)

    rank = int(operators["observable_rank"].item())
    basis_rank = int(stage1.basis_rank)
    null_dimension = basis_rank - rank

    diagnostics: Dict[str, object] = {
        "basis_rank": basis_rank,
        "msi_channels": int(response.size(0)),
        "observable_rank": rank,
        "null_dimension": null_dimension,
        "observable_dimension_fraction": rank / max(basis_rank, 1),
        "null_dimension_fraction": null_dimension / max(basis_rank, 1),
        "singular_values": [float(x) for x in operators["singular_values"].detach().cpu().tolist()],
        "rank_threshold": float(operators["rank_threshold"].item()),
        "actual_anchor_ridge": float(operators["actual_anchor_ridge"].item()),
        "gt_residual_observable_energy_share": observable_energy_share,
        "gt_residual_null_energy_share": null_energy_share,
        "gt_residual_energy_closure": (total_observable_energy + total_null_energy) / target_energy,
        "gt_residual_normalized_obs_null_inner_product": total_cross_inner / target_energy,
        "gt_residual_observable_l1_share": total_observable_abs / max(total_residual_abs, 1e-30),
        "gt_residual_null_l1_share": total_null_abs / max(total_residual_abs, 1e-30),
        "gt_residual_abs_mean": total_residual_abs / max(total_coeff_values, 1),
        "observable_oracle_recoverable_mse_fraction": (anchor_mse - observable_mse) / reducible_mse,
        "null_oracle_recoverable_mse_fraction": (anchor_mse - null_mse) / reducible_mse,
        "full_vs_basis_oracle_max_abs_difference": max_full_basis_difference,
        **projector_checks,
    }

    gaps = {
        "basis_oracle_headroom_over_base_psnr": metrics["hr_basis_oracle"]["PSNR"] - metrics["coefficient_upsampling_base"]["PSNR"],
        "basis_oracle_headroom_over_anchor_psnr": metrics["hr_basis_oracle"]["PSNR"] - metrics["analytical_anchor"]["PSNR"],
        "observable_oracle_gain_over_anchor_psnr": metrics["gt_observable_oracle"]["PSNR"] - metrics["analytical_anchor"]["PSNR"],
        "null_oracle_gain_over_anchor_psnr": metrics["gt_null_oracle"]["PSNR"] - metrics["analytical_anchor"]["PSNR"],
        "full_oracle_gain_over_observable_psnr": metrics["full_coefficient_oracle"]["PSNR"] - metrics["gt_observable_oracle"]["PSNR"],
        "full_oracle_gain_over_null_psnr": metrics["full_coefficient_oracle"]["PSNR"] - metrics["gt_null_oracle"]["PSNR"],
        "observable_oracle_sam_improvement_over_anchor": metrics["analytical_anchor"]["SAM"] - metrics["gt_observable_oracle"]["SAM"],
        "null_oracle_sam_improvement_over_anchor": metrics["analytical_anchor"]["SAM"] - metrics["gt_null_oracle"]["SAM"],
    }
    return {"metrics": metrics, "diagnostics": diagnostics, "gaps": gaps}


def print_report(result: Dict[str, object]) -> None:
    metrics = result["metrics"]
    diagnostics = result["diagnostics"]
    gaps = result["gaps"]
    print("=" * 100)
    print("RAPD-Net Stage-2 observability ceiling diagnosis")
    print("=" * 100)
    print(
        "Coefficient observability: "
        f"rank(RU)={diagnostics['observable_rank']}/{diagnostics['basis_rank']}, "
        f"null_dim={diagnostics['null_dimension']} "
        f"({100.0 * diagnostics['null_dimension_fraction']:.2f}%), "
        f"MSI channels={diagnostics['msi_channels']}"
    )
    print(
        "GT residual energy: "
        f"observable={100.0 * diagnostics['gt_residual_observable_energy_share']:.2f}%, "
        f"null={100.0 * diagnostics['gt_residual_null_energy_share']:.2f}%, "
        f"closure={diagnostics['gt_residual_energy_closure']:.6f}"
    )
    print("-" * 100)
    labels = {
        "coefficient_upsampling_base": "Coefficient upsampling base",
        "analytical_anchor": "Analytical SRF anchor",
        "gt_observable_oracle": "GT observable oracle",
        "gt_null_oracle": "GT null oracle",
        "full_coefficient_oracle": "Full coefficient oracle",
        "hr_basis_oracle": "HR basis oracle",
    }
    for name in ORACLE_NAMES:
        values = metrics[name]
        print(
            f"{labels[name]:30s}: PSNR={values['PSNR']:.4f} dB, "
            f"SAM={values['SAM']:.4f} deg, RMSE={values['RMSE']:.8f}"
        )
    print("-" * 100)
    print(f"Observable GT gain over anchor : {gaps['observable_oracle_gain_over_anchor_psnr']:+.4f} dB")
    print(f"Null GT gain over anchor       : {gaps['null_oracle_gain_over_anchor_psnr']:+.4f} dB")
    print(f"Full oracle over observable    : {gaps['full_oracle_gain_over_observable_psnr']:+.4f} dB")
    print(f"Basis headroom over anchor     : {gaps['basis_oracle_headroom_over_anchor_psnr']:+.4f} dB")
    print(
        "Recoverable MSE fraction       : "
        f"observable={100.0 * diagnostics['observable_oracle_recoverable_mse_fraction']:.2f}%, "
        f"null={100.0 * diagnostics['null_oracle_recoverable_mse_fraction']:.2f}%"
    )
    print(
        "Numerical checks               : "
        f"null_leak={diagnostics['reduced_response_null_leakage_max']:.3e}, "
        f"full-vs-basis={diagnostics['full_vs_basis_oracle_max_abs_difference']:.3e}"
    )
    print("=" * 100)


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
    result = evaluate(stage1, spectral_response, test_loader, cfg, device)

    payload = {
        "dataset": cfg.dataset,
        "stage1_basis_checkpoint": cfg.stage1_basis_checkpoint,
        "checkpoint_epoch": int(stage1_state.get("epoch", -1)),
        "msi_mode": cfg.msi_mode,
        "srf_band_set": cfg.srf_band_set,
        "anchor_ridge_ratio": cfg.anchor_ridge_ratio,
        "projector_tolerance": cfg.projector_tolerance,
        **result,
    }
    output_dir = os.path.join(cfg.output_root, "stage2_observability_ceiling", cfg.dataset)
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "observability_ceiling.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print_report(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
