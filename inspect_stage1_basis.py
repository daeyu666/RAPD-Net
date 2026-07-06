"""Inspect the new Stage-1 spectral basis checkpoint.

The script is headless-server safe and checks:

* LR-HSI reconstruction L1 / PSNR / SAM;
* basis orthogonality and projector idempotence;
* residual leakage back into the learned basis;
* signed coefficient scale, energy share, and channel activity;
* basis, coefficient-map, and per-band residual visualizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage1_spectral_basis import Stage1SpectralBasisNet
from utils import ensure_dir, get_device, move_to_device, set_seed


def first_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def load_basis_checkpoint(
    path: str,
    expected_n_bands: int,
    device: torch.device,
) -> Tuple[Stage1SpectralBasisNet, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)

    model_state = state.get("model", state)
    extra = state.get("extra", {})
    raw_basis = model_state["raw_basis"]
    n_bands = int(extra.get("n_bands", raw_basis.shape[0]))
    rank = int(extra.get("basis_rank", raw_basis.shape[1]))
    if n_bands != expected_n_bands:
        raise ValueError(
            f"Checkpoint bands={n_bands}, dataset bands={expected_n_bands}"
        )

    model = Stage1SpectralBasisNet(n_bands=n_bands, basis_rank=rank).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, state


@torch.no_grad()
def evaluate_checkpoint(
    model: Stage1SpectralBasisNet,
    loader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    sam_loss = SAMLoss()
    total_pixels = 0
    sums = {
        "l1": 0.0,
        "mse": 0.0,
        "sam_rad": 0.0,
        "sgrad1": 0.0,
        "sgrad2": 0.0,
    }
    coefficient_sum = torch.zeros(
        model.basis_rank, dtype=torch.float64, device=device
    )
    coefficient_square = torch.zeros_like(coefficient_sum)
    coefficient_abs_max = torch.zeros_like(coefficient_sum)
    coefficient_nontrivial = torch.zeros_like(coefficient_sum)
    residual_band_square = torch.zeros(
        model.n_bands, dtype=torch.float64, device=device
    )
    residual_count = 0
    residual_leak_numerator = 0.0
    residual_leak_denominator = 0.0
    centered_energy = 0.0
    residual_energy = 0.0
    first_arrays = None

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        outputs = model(lr_hsi)
        reconstruction = outputs["reconstruction"]
        residual = outputs["residual"]
        coefficients = outputs["coefficients"]

        pixels = lr_hsi.size(0) * lr_hsi.size(2) * lr_hsi.size(3)
        total_pixels += pixels
        sums["l1"] += float(F.l1_loss(reconstruction, lr_hsi).item()) * pixels
        sums["mse"] += float(F.mse_loss(reconstruction, lr_hsi).item()) * pixels
        sums["sam_rad"] += float(sam_loss(reconstruction, lr_hsi).item()) * pixels
        sums["sgrad1"] += float(
            F.l1_loss(first_difference(reconstruction), first_difference(lr_hsi)).item()
        ) * pixels
        sums["sgrad2"] += float(
            F.l1_loss(second_difference(reconstruction), second_difference(lr_hsi)).item()
        ) * pixels

        flat_coeff = coefficients.permute(1, 0, 2, 3).reshape(
            model.basis_rank, -1
        ).double()
        coefficient_sum += flat_coeff.sum(dim=1)
        coefficient_square += flat_coeff.square().sum(dim=1)
        coefficient_abs_max = torch.maximum(
            coefficient_abs_max, flat_coeff.abs().max(dim=1).values
        )
        coefficient_nontrivial += (flat_coeff.abs() > 1e-3).double().sum(dim=1)

        residual_band_square += residual.double().square().sum(dim=(0, 2, 3))
        residual_count += residual.size(0) * residual.size(2) * residual.size(3)
        leakage = torch.einsum(
            "br,nbhw->nrhw", outputs["basis"], residual
        )
        residual_leak_numerator += float(leakage.double().square().sum().item())
        residual_leak_denominator += float(residual.double().square().sum().item())

        centered = lr_hsi - model.mean_spectrum.view(1, -1, 1, 1)
        centered_energy += float(centered.double().square().sum().item())
        residual_energy += float(residual.double().square().sum().item())

        if first_arrays is None:
            first_arrays = {
                "lr_hsi": lr_hsi.detach().cpu().numpy(),
                "reconstruction": reconstruction.detach().cpu().numpy(),
                "residual": residual.detach().cpu().numpy(),
                "coefficients": coefficients.detach().cpu().numpy(),
                "normalized_coefficients": outputs[
                    "normalized_coefficients"
                ].detach().cpu().numpy(),
            }

    if total_pixels == 0:
        raise RuntimeError("Empty evaluation loader")

    mean = coefficient_sum / total_pixels
    variance = (coefficient_square / total_pixels - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    energy = coefficient_square / coefficient_square.sum().clamp_min(1e-12)
    active_ratio = coefficient_nontrivial / total_pixels
    per_band_rmse = (residual_band_square / max(residual_count, 1)).sqrt()

    statistics = model.subspace_statistics()
    mse = sums["mse"] / total_pixels
    summary = {
        "l1": sums["l1"] / total_pixels,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "sam_deg": sums["sam_rad"] / total_pixels * 180.0 / math.pi,
        "sgrad1": sums["sgrad1"] / total_pixels,
        "sgrad2": sums["sgrad2"] / total_pixels,
        "orthogonality_error": float(
            statistics["orthogonality_error"].item()
        ),
        "projector_idempotence_error": float(
            statistics["projector_idempotence_error"].item()
        ),
        "projector_drift": float(statistics["projector_drift"].item()),
        "pca_explained_variance_ratio": float(
            statistics["pca_explained_variance_ratio"].item()
        ),
        "test_centered_energy_retained": 1.0
        - residual_energy / max(centered_energy, 1e-12),
        "residual_basis_leakage_ratio": residual_leak_numerator
        / max(residual_leak_denominator, 1e-12),
        "coefficient_std_min": float(std.min().item()),
        "coefficient_std_max": float(std.max().item()),
        "coefficient_active_ratio_min": float(active_ratio.min().item()),
    }

    arrays = {
        **first_arrays,
        "basis": model.get_basis().detach().cpu().numpy(),
        "mean_spectrum": model.mean_spectrum.detach().cpu().numpy(),
        "coefficient_scale": model.coefficient_scale.detach().cpu().numpy(),
        "coefficient_mean": mean.float().cpu().numpy(),
        "coefficient_std": std.float().cpu().numpy(),
        "coefficient_abs_max": coefficient_abs_max.float().cpu().numpy(),
        "coefficient_active_ratio": active_ratio.float().cpu().numpy(),
        "coefficient_energy_share": energy.float().cpu().numpy(),
        "per_band_rmse": per_band_rmse.float().cpu().numpy(),
        "pca_eigenvalues": model.pca_eigenvalues.detach().cpu().numpy(),
    }
    return summary, arrays


def write_csv(path: str, rows: List[Dict], fields: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_basis(
    arrays: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
    path: str,
    max_vectors: int,
) -> None:
    basis = arrays["basis"]
    mean = arrays["mean_spectrum"]
    count = min(max_vectors, basis.shape[1])
    figure = plt.figure(figsize=(12, 7))
    axis = figure.add_subplot(111)
    axis.plot(wavelengths, mean, linewidth=2.5, label="scene mean")
    for index in range(count):
        axis.plot(
            wavelengths,
            basis[:, index],
            linewidth=1.0,
            alpha=0.8,
            label=f"basis {index}" if index < 8 else None,
        )
    axis.set_title(
        "Affine spectral mean and orthogonal basis vectors\n"
        "(basis vectors are mathematical coordinates, not endmember spectra)"
    )
    axis.set_xlabel("Wavelength / band index")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_coefficient_statistics(arrays: Dict[str, np.ndarray], path: str) -> None:
    energy = arrays["coefficient_energy_share"]
    std = arrays["coefficient_std"]
    active = arrays["coefficient_active_ratio"]
    indices = np.arange(len(energy))

    figure = plt.figure(figsize=(12, 7))
    axis = figure.add_subplot(111)
    axis.bar(indices, energy, label="energy share")
    axis.plot(indices, std / max(std.max(), 1e-12), marker="o", label="normalized std")
    axis.plot(indices, active, marker="x", label="active ratio |C|>1e-3")
    axis.set_xlabel("Basis coordinate")
    axis.set_ylabel("Ratio")
    axis.set_title("Signed spectral coefficient usage")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_coefficient_maps(
    arrays: Dict[str, np.ndarray],
    path: str,
    max_maps: int,
) -> None:
    coefficients = arrays["normalized_coefficients"][0]
    energy = arrays["coefficient_energy_share"]
    selected = np.argsort(-energy)[: min(max_maps, coefficients.shape[0])]
    columns = 4
    rows = int(math.ceil(len(selected) / columns))
    figure = plt.figure(figsize=(4 * columns, 3.5 * rows))
    for plot_index, channel in enumerate(selected, start=1):
        axis = figure.add_subplot(rows, columns, plot_index)
        image = coefficients[channel]
        limit = max(np.percentile(np.abs(image), 99), 1e-6)
        handle = axis.imshow(image, vmin=-limit, vmax=limit, cmap="coolwarm")
        axis.set_title(f"C{channel}, energy={energy[channel]:.3f}")
        axis.axis("off")
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_band_error(
    arrays: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
    path: str,
) -> None:
    figure = plt.figure(figsize=(11, 5))
    axis = figure.add_subplot(111)
    axis.plot(wavelengths, arrays["per_band_rmse"], linewidth=2)
    axis.set_title("LR-HSI projection residual RMSE by spectral band")
    axis.set_xlabel("Wavelength / band index")
    axis.set_ylabel("RMSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def inspect_one(
    checkpoint: str,
    cfg,
    loader,
    info: dict,
    root: str,
    device: torch.device,
) -> Dict[str, float]:
    model, state = load_basis_checkpoint(checkpoint, info["n_bands"], device)
    summary, arrays = evaluate_checkpoint(model, loader, device)
    summary.update(
        {
            "checkpoint": checkpoint,
            "checkpoint_name": os.path.basename(checkpoint),
            "epoch": int(state.get("epoch", -1)),
            "basis_rank": model.basis_rank,
        }
    )

    name = os.path.splitext(os.path.basename(checkpoint))[0]
    output_dir = os.path.join(root, name)
    ensure_dir(output_dir)
    with open(
        os.path.join(output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(output_dir, "inspection_arrays.npz"),
        **arrays,
    )

    coefficient_rows = []
    for index in range(model.basis_rank):
        coefficient_rows.append(
            {
                "coordinate": index,
                "mean": float(arrays["coefficient_mean"][index]),
                "std": float(arrays["coefficient_std"][index]),
                "abs_max": float(arrays["coefficient_abs_max"][index]),
                "active_ratio": float(arrays["coefficient_active_ratio"][index]),
                "energy_share": float(arrays["coefficient_energy_share"][index]),
                "pca_eigenvalue": float(arrays["pca_eigenvalues"][index]),
            }
        )
    write_csv(
        os.path.join(output_dir, "coefficient_statistics.csv"),
        coefficient_rows,
        [
            "coordinate",
            "mean",
            "std",
            "abs_max",
            "active_ratio",
            "energy_share",
            "pca_eigenvalue",
        ],
    )

    wavelengths = info.get("hsi_wavelengths")
    if wavelengths is None:
        wavelengths = np.arange(model.n_bands, dtype=np.float32)
    else:
        wavelengths = np.asarray(wavelengths, dtype=np.float32)
    plot_basis(
        arrays,
        wavelengths,
        os.path.join(output_dir, "spectral_basis.png"),
        cfg.max_basis_vectors,
    )
    plot_coefficient_statistics(
        arrays,
        os.path.join(output_dir, "coefficient_statistics.png"),
    )
    plot_coefficient_maps(
        arrays,
        os.path.join(output_dir, "coefficient_maps.png"),
        cfg.max_coefficient_maps,
    )
    plot_band_error(
        arrays,
        wavelengths,
        os.path.join(output_dir, "per_band_rmse.png"),
    )

    print(
        f"{summary['checkpoint_name']}: epoch={summary['epoch']}, "
        f"PSNR={summary['psnr']:.4f}, SAM={summary['sam_deg']:.4f} deg, "
        f"L1={summary['l1']:.8f}, orth={summary['orthogonality_error']:.3e}, "
        f"leak={summary['residual_basis_leakage_ratio']:.3e}"
    )
    return summary


def parse_inspection_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--max_basis_vectors", type=int, default=12)
    parser.add_argument("--max_coefficient_maps", type=int, default=12)
    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(specific).items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    cfg = parse_inspection_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root, "stage1_basis", cfg.dataset
    )
    if cfg.compare_all:
        names = [
            "basis_pca_init.pth",
            "basis_best.pth",
            "basis_best_sam.pth",
            "basis_best_psnr.pth",
            "basis_last.pth",
            "basis_for_stage2.pth",
        ]
        checkpoints = [
            os.path.join(checkpoint_dir, name)
            for name in names
            if os.path.exists(os.path.join(checkpoint_dir, name))
        ]
    else:
        checkpoint = cfg.checkpoint or os.path.join(
            checkpoint_dir, "basis_for_stage2.pth"
        )
        checkpoints = [checkpoint]

    if not checkpoints:
        raise FileNotFoundError(
            f"No Stage-1 basis checkpoints found under {checkpoint_dir}"
        )

    root = os.path.join(
        cfg.output_root, "stage1_basis_inspection", cfg.dataset
    )
    ensure_dir(root)
    summaries = [
        inspect_one(
            checkpoint,
            cfg,
            test_loader,
            info,
            root,
            device,
        )
        for checkpoint in checkpoints
    ]
    fields = [
        "checkpoint_name",
        "epoch",
        "basis_rank",
        "l1",
        "mse",
        "psnr",
        "sam_deg",
        "sgrad1",
        "sgrad2",
        "orthogonality_error",
        "projector_idempotence_error",
        "projector_drift",
        "pca_explained_variance_ratio",
        "test_centered_energy_retained",
        "residual_basis_leakage_ratio",
        "coefficient_std_min",
        "coefficient_std_max",
        "coefficient_active_ratio_min",
    ]
    write_csv(
        os.path.join(root, "checkpoint_comparison.csv"),
        summaries,
        fields,
    )


if __name__ == "__main__":
    main()
