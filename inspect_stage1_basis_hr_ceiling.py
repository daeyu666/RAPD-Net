"""Diagnose what a very high Stage-1 LR projection score actually means.

The new Stage 1 is an LR-HSI spectral auto-projection. A high LR score does not
by itself measure HSI super-resolution quality. This script reports three
separate quantities for the selected Stage-1 basis:

1. LR self-projection:
       Y_lr -> U^T(Y_lr-mu) -> mu + U C_lr
2. HR basis oracle:
       X_gt -> U^T(X_gt-mu) -> mu + U C_hr
   This is the representation ceiling of the learned spectral subspace.
3. LR-coefficient upsampling baseline:
       C_lr -> bicubic upsampling -> mu + U C_up
   This is the deterministic no-MSI starting point for the new Stage 2.

The HR oracle uses GT only for diagnosis and is never a training input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage1_spectral_basis import Stage1SpectralBasisNet
from utils import ensure_dir, get_device, move_to_device, set_seed


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


@torch.no_grad()
def evaluate(
    model: Stage1SpectralBasisNet,
    loader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    sam_loss = SAMLoss()
    totals = {
        "lr_self_projection": {"l1": 0.0, "mse": 0.0, "sam_rad": 0.0},
        "hr_basis_oracle": {"l1": 0.0, "mse": 0.0, "sam_rad": 0.0},
        "coefficient_upsampling_base": {
            "l1": 0.0,
            "mse": 0.0,
            "sam_rad": 0.0,
        },
    }
    counts = {name: 0 for name in totals}

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        gt = batch["gt"]
        basis = model.get_basis()

        lr_coefficients = model.encode(lr_hsi, basis=basis)
        lr_reconstruction = model.decode(lr_coefficients, basis=basis)

        hr_coefficients = model.encode(gt, basis=basis)
        hr_oracle = model.decode(hr_coefficients, basis=basis)

        upsampled_coefficients = F.interpolate(
            lr_coefficients,
            size=gt.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        upsampled_base = model.decode(upsampled_coefficients, basis=basis)

        items = {
            "lr_self_projection": (lr_reconstruction, lr_hsi),
            "hr_basis_oracle": (hr_oracle, gt),
            "coefficient_upsampling_base": (upsampled_base, gt),
        }
        for name, (prediction, target) in items.items():
            pixel_count = target.size(0) * target.size(2) * target.size(3)
            totals[name]["l1"] += float(
                F.l1_loss(prediction, target).item()
            ) * pixel_count
            totals[name]["mse"] += float(
                F.mse_loss(prediction, target).item()
            ) * pixel_count
            totals[name]["sam_rad"] += float(
                sam_loss(prediction, target).item()
            ) * pixel_count
            counts[name] += pixel_count

    results: Dict[str, Dict[str, float]] = {}
    for name, values in totals.items():
        count = max(counts[name], 1)
        mse = values["mse"] / count
        results[name] = {
            "l1": values["l1"] / count,
            "mse": mse,
            "rmse": math.sqrt(max(mse, 0.0)),
            "psnr": -10.0 * math.log10(max(mse, 1e-12)),
            "sam_deg": values["sam_rad"] / count * 180.0 / math.pi,
        }

    results["gaps"] = {
        "hr_oracle_minus_lr_psnr": (
            results["hr_basis_oracle"]["psnr"]
            - results["lr_self_projection"]["psnr"]
        ),
        "hr_oracle_minus_coefficient_base_psnr": (
            results["hr_basis_oracle"]["psnr"]
            - results["coefficient_upsampling_base"]["psnr"]
        ),
        "coefficient_base_sam_minus_hr_oracle": (
            results["coefficient_upsampling_base"]["sam_deg"]
            - results["hr_basis_oracle"]["sam_deg"]
        ),
    }
    return results


def parse_specific_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default="")
    specific, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    cfg.checkpoint = specific.checkpoint
    return cfg


def main() -> None:
    cfg = parse_specific_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)

    checkpoint = cfg.checkpoint or os.path.join(
        cfg.checkpoint_root,
        "stage1_basis",
        cfg.dataset,
        "basis_for_stage2.pth",
    )
    model, state = load_basis_checkpoint(
        checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    results = evaluate(model, test_loader, device)

    output_dir = os.path.join(
        cfg.output_root,
        "stage1_basis_hr_ceiling",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "hr_ceiling.json")
    payload = {
        "dataset": cfg.dataset,
        "checkpoint": checkpoint,
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "basis_rank": model.basis_rank,
        **results,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print("=" * 88)
    for name, label in (
        ("lr_self_projection", "LR self-projection"),
        ("hr_basis_oracle", "HR basis oracle"),
        ("coefficient_upsampling_base", "LR-coefficient upsampling base"),
    ):
        values = results[name]
        print(
            f"{label:32s}: PSNR={values['psnr']:.4f}, "
            f"SAM={values['sam_deg']:.4f} deg, "
            f"RMSE={values['rmse']:.8f}"
        )
    print("-" * 88)
    print(
        "Stage-2 coefficient headroom: "
        f"{results['gaps']['hr_oracle_minus_coefficient_base_psnr']:+.4f} dB"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
