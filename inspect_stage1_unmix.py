"""Inspect Stage-1 unmixing checkpoints.

The script checks four things that reconstruction L1/SAM alone cannot reveal:
1. whether endmember spectra are duplicated;
2. whether endmember curves are abnormally jagged or saturated;
3. whether abundance channels are dead or collapsed;
4. whether the physical reconstruction preserves LR-HSI spectra.

It intentionally stays in one file so Stage-1 diagnostics remain easy to run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List

# Headless Linux server: do not initialize Qt/XCB.
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
from models.stage1_unmixing import Stage1UnmixingNet
from utils import ensure_dir, get_device, move_to_device, set_seed


def load_state(path: str, device: torch.device) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    checkpoint_path: str,
    fallback_n_bands: int,
    device: torch.device,
) -> tuple[Stage1UnmixingNet, dict]:
    state = load_state(checkpoint_path, device)
    extra = state.get("extra", {})
    model_state = state.get("model", state)

    n_bands = int(extra.get("n_bands", fallback_n_bands))
    num_endmembers = int(
        extra.get("num_endmembers", model_state["endmember_logits"].shape[0])
    )
    hidden_channels = int(
        extra.get(
            "hidden_channels",
            model_state["spectral_stem.0.weight"].shape[0],
        )
    )

    block_indices = []
    for key in model_state:
        if key.startswith("spatial_blocks."):
            block_indices.append(int(key.split(".")[1]))
    num_blocks = int(extra.get("num_blocks", max(block_indices) + 1))

    model = Stage1UnmixingNet(
        n_bands=n_bands,
        num_endmembers=num_endmembers,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model, state


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float(-10.0 * math.log10(max(mse, 1e-12)))


@torch.no_grad()
def collect_reconstruction_and_abundance(
    model: Stage1UnmixingNet,
    loader,
    device: torch.device,
    active_threshold: float,
) -> tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    sam_loss = SAMLoss()
    total_pixels = 0
    sum_l1 = 0.0
    sum_mse = 0.0
    sum_sam = 0.0

    k = model.num_endmembers
    abundance_sum = torch.zeros(k, dtype=torch.float64)
    abundance_max = torch.zeros(k, dtype=torch.float64)
    active_count = torch.zeros(k, dtype=torch.float64)
    dominant_count = torch.zeros(k, dtype=torch.float64)
    entropy_sum = 0.0

    visual_batch = None
    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        outputs = model(lr_hsi)
        reconstruction = outputs["reconstruction"]
        abundance = outputs["abundance"]

        n_pixels = lr_hsi.size(0) * lr_hsi.size(2) * lr_hsi.size(3)
        total_pixels += n_pixels
        sum_l1 += F.l1_loss(reconstruction, lr_hsi, reduction="mean").item() * n_pixels
        sum_mse += F.mse_loss(reconstruction, lr_hsi, reduction="mean").item() * n_pixels
        sum_sam += sam_loss(reconstruction, lr_hsi).item() * n_pixels

        abundance_cpu = abundance.detach().double().cpu()
        abundance_sum += abundance_cpu.sum(dim=(0, 2, 3))
        abundance_max = torch.maximum(
            abundance_max,
            abundance_cpu.amax(dim=(0, 2, 3)),
        )
        active_count += (abundance_cpu > active_threshold).sum(dim=(0, 2, 3))

        dominant = abundance_cpu.argmax(dim=1)
        dominant_count += torch.bincount(
            dominant.reshape(-1), minlength=k
        ).double()

        safe = abundance_cpu.clamp_min(1e-12)
        entropy_sum += float((-(safe * safe.log()).sum(dim=1)).sum().item())

        if visual_batch is None:
            visual_batch = {
                "lr_hsi": lr_hsi[:1].detach().cpu(),
                "reconstruction": reconstruction[:1].detach().cpu(),
                "abundance": abundance[:1].detach().cpu(),
            }

    if total_pixels == 0:
        raise RuntimeError("The loader produced no samples.")

    metrics = {
        "l1": sum_l1 / total_pixels,
        "rmse": math.sqrt(sum_mse / total_pixels),
        "psnr": -10.0 * math.log10(max(sum_mse / total_pixels, 1e-12)),
        "sam_deg": (sum_sam / total_pixels) * 180.0 / math.pi,
        "abundance_entropy": entropy_sum / total_pixels,
    }
    abundance_stats = {
        "mean": (abundance_sum / total_pixels).numpy(),
        "max": abundance_max.numpy(),
        "active_ratio": (active_count / total_pixels).numpy(),
        "dominant_ratio": (dominant_count / total_pixels).numpy(),
    }
    return metrics, abundance_stats, visual_batch


@torch.no_grad()
def inspect_endmembers(
    model: Stage1UnmixingNet,
    duplicate_threshold: float,
) -> tuple[Dict[str, float], np.ndarray, List[dict]]:
    endmembers = model.get_endmembers().detach().cpu()  # [B, K]
    spectra = F.normalize(endmembers.transpose(0, 1), dim=1, eps=1e-8)
    cosine = spectra @ spectra.transpose(0, 1)
    k = cosine.size(0)
    off_diag = ~torch.eye(k, dtype=torch.bool)
    values = cosine[off_diag]

    first_diff = endmembers[1:] - endmembers[:-1]
    second_diff = endmembers[2:] - 2.0 * endmembers[1:-1] + endmembers[:-2]
    spectral_tv = first_diff.abs().mean(dim=0)
    spectral_curvature = second_diff.abs().mean(dim=0)
    low_saturation = (endmembers <= 1e-3).float().mean(dim=0)
    high_saturation = (endmembers >= 1.0 - 1e-3).float().mean(dim=0)

    duplicate_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            score = float(cosine[i, j].item())
            if score >= duplicate_threshold:
                duplicate_pairs.append(
                    {"endmember_i": i, "endmember_j": j, "cosine": score}
                )
    duplicate_pairs.sort(key=lambda item: item["cosine"], reverse=True)

    summary = {
        "endmember_cosine_mean": float(values.mean().item()),
        "endmember_cosine_max": float(values.max().item()),
        "duplicate_pair_count": len(duplicate_pairs),
        "spectral_tv_mean": float(spectral_tv.mean().item()),
        "spectral_tv_max": float(spectral_tv.max().item()),
        "spectral_curvature_mean": float(spectral_curvature.mean().item()),
        "spectral_curvature_max": float(spectral_curvature.max().item()),
        "low_saturation_max_ratio": float(low_saturation.max().item()),
        "high_saturation_max_ratio": float(high_saturation.max().item()),
    }
    per_endmember = np.stack(
        [
            spectral_tv.numpy(),
            spectral_curvature.numpy(),
            low_saturation.numpy(),
            high_saturation.numpy(),
        ],
        axis=1,
    )
    return summary, per_endmember, duplicate_pairs


def save_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_endmembers(
    endmembers: np.ndarray,
    wavelengths: np.ndarray,
    path: str,
) -> None:
    plt.figure(figsize=(11, 7))
    for index in range(endmembers.shape[1]):
        plt.plot(wavelengths, endmembers[:, index], linewidth=1.0, alpha=0.8)
    plt.xlabel("Wavelength / band index")
    plt.ylabel("Reflectance")
    plt.title("Stage-1 endmember spectra")
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_abundance_usage(stats: Dict[str, np.ndarray], path: str) -> None:
    indices = np.arange(stats["mean"].size)
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(indices - width, stats["mean"], width=width, label="mean")
    plt.bar(indices, stats["active_ratio"], width=width, label="active ratio")
    plt.bar(indices + width, stats["dominant_ratio"], width=width, label="dominant ratio")
    plt.xlabel("Endmember index")
    plt.ylabel("Ratio")
    plt.title("Abundance channel usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_abundance_maps(abundance: torch.Tensor, path: str) -> None:
    abundance = abundance[0].numpy()
    k = abundance.shape[0]
    columns = 8
    rows = math.ceil(k / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for index in range(rows * columns):
        axes[index].axis("off")
        if index < k:
            image = axes[index].imshow(abundance[index], vmin=0.0, vmax=abundance[index].max())
            axes[index].set_title(f"A{index}: max={abundance[index].max():.3f}")
            fig.colorbar(image, ax=axes[index], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def checkpoint_list(cfg, compare_all: bool, checkpoint: str) -> List[str]:
    if checkpoint:
        return [checkpoint]
    root = os.path.join(cfg.checkpoint_root, "stage1_unmix", cfg.dataset)
    if compare_all:
        names = [
            "unmixing_best.pth",
            "unmixing_best_sam.pth",
            "unmixing_best_l1.pth",
            "unmixing_last.pth",
        ]
        paths = [os.path.join(root, name) for name in names]
        return [path for path in paths if os.path.exists(path)]
    return [os.path.join(root, "unmixing_best.pth")]


def parse_inspection_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--active_threshold", type=float, default=0.01)
    parser.add_argument("--dead_mean_threshold", type=float, default=1e-4)
    parser.add_argument("--dead_max_threshold", type=float, default=0.01)
    parser.add_argument("--duplicate_threshold", type=float, default=0.999)
    inspection_args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(inspection_args).items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    cfg = parse_inspection_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    checkpoints = checkpoint_list(cfg, cfg.compare_all, cfg.checkpoint)
    if not checkpoints:
        raise FileNotFoundError("No Stage-1 checkpoints were found.")

    root = os.path.join(cfg.output_root, "stage1_inspection", cfg.dataset)
    ensure_dir(root)
    comparison_rows = []

    for checkpoint_path in checkpoints:
        name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_dir = os.path.join(root, name)
        ensure_dir(output_dir)

        model, checkpoint_state = build_model_from_checkpoint(
            checkpoint_path,
            fallback_n_bands=info["n_bands"],
            device=device,
        )
        reconstruction, abundance_stats, visual_batch = (
            collect_reconstruction_and_abundance(
                model,
                test_loader,
                device,
                active_threshold=cfg.active_threshold,
            )
        )
        endmember_summary, per_endmember, duplicate_pairs = inspect_endmembers(
            model,
            duplicate_threshold=cfg.duplicate_threshold,
        )

        dead_mask = (
            (abundance_stats["mean"] < cfg.dead_mean_threshold)
            | (abundance_stats["max"] < cfg.dead_max_threshold)
        )
        dead_indices = np.flatnonzero(dead_mask).tolist()
        dominant_indices = np.flatnonzero(
            abundance_stats["dominant_ratio"] > 0.5
        ).tolist()

        endmembers = model.get_endmembers().detach().cpu().numpy()
        wavelengths = info.get("hsi_wavelengths")
        if wavelengths is None:
            wavelengths = np.arange(endmembers.shape[0], dtype=np.float32)
        else:
            wavelengths = np.asarray(wavelengths, dtype=np.float32)

        abundance_rows = []
        for index in range(model.num_endmembers):
            abundance_rows.append(
                {
                    "index": index,
                    "mean": float(abundance_stats["mean"][index]),
                    "max": float(abundance_stats["max"][index]),
                    "active_ratio": float(abundance_stats["active_ratio"][index]),
                    "dominant_ratio": float(abundance_stats["dominant_ratio"][index]),
                    "spectral_tv": float(per_endmember[index, 0]),
                    "spectral_curvature": float(per_endmember[index, 1]),
                    "low_saturation_ratio": float(per_endmember[index, 2]),
                    "high_saturation_ratio": float(per_endmember[index, 3]),
                    "is_dead": bool(dead_mask[index]),
                }
            )

        summary = {
            "checkpoint": checkpoint_path,
            "checkpoint_epoch": int(checkpoint_state.get("epoch", -1)),
            **reconstruction,
            **endmember_summary,
            "dead_endmember_count": len(dead_indices),
            "dead_endmember_indices": dead_indices,
            "dominant_over_50_percent_indices": dominant_indices,
            "abundance_mean_min": float(abundance_stats["mean"].min()),
            "abundance_mean_max": float(abundance_stats["mean"].max()),
            "abundance_active_ratio_min": float(abundance_stats["active_ratio"].min()),
        }

        with open(
            os.path.join(output_dir, "summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        save_csv(
            os.path.join(output_dir, "abundance_and_endmember_stats.csv"),
            abundance_rows,
            fieldnames=list(abundance_rows[0].keys()),
        )
        save_csv(
            os.path.join(output_dir, "duplicate_pairs.csv"),
            duplicate_pairs,
            fieldnames=["endmember_i", "endmember_j", "cosine"],
        )
        plot_endmembers(
            endmembers,
            wavelengths,
            os.path.join(output_dir, "endmember_curves.png"),
        )
        plot_abundance_usage(
            abundance_stats,
            os.path.join(output_dir, "abundance_usage.png"),
        )
        plot_abundance_maps(
            visual_batch["abundance"],
            os.path.join(output_dir, "abundance_maps.png"),
        )

        comparison_rows.append(
            {
                "checkpoint": name,
                "epoch": summary["checkpoint_epoch"],
                "l1": reconstruction["l1"],
                "rmse": reconstruction["rmse"],
                "psnr": reconstruction["psnr"],
                "sam_deg": reconstruction["sam_deg"],
                "dead_count": len(dead_indices),
                "duplicate_pair_count": endmember_summary["duplicate_pair_count"],
                "max_endmember_cosine": endmember_summary["endmember_cosine_max"],
                "abundance_entropy": reconstruction["abundance_entropy"],
            }
        )

        print("=" * 78)
        print(f"Checkpoint : {checkpoint_path}")
        print(f"Epoch      : {summary['checkpoint_epoch']}")
        print(
            f"LR recon   : L1={reconstruction['l1']:.6f}, "
            f"PSNR={reconstruction['psnr']:.3f}, "
            f"SAM={reconstruction['sam_deg']:.4f} deg"
        )
        print(
            f"Endmembers : max cosine={endmember_summary['endmember_cosine_max']:.6f}, "
            f"duplicate pairs(>={cfg.duplicate_threshold})="
            f"{endmember_summary['duplicate_pair_count']}"
        )
        print(
            f"Abundance  : dead={len(dead_indices)}/{model.num_endmembers}, "
            f"dead indices={dead_indices}, dominant>50%={dominant_indices}"
        )
        print(f"Saved to  : {output_dir}")

    save_csv(
        os.path.join(root, "checkpoint_comparison.csv"),
        comparison_rows,
        fieldnames=list(comparison_rows[0].keys()),
    )


if __name__ == "__main__":
    main()
