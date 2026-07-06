"""Estimate the Stage-2 abundance-only physical reconstruction ceiling.

Stage 2 is restricted to the frozen endmember span. A low PSNR can therefore
come from two very different causes:

1. the abundance-residual network has not reached the best abundance map;
2. the frozen endmember dictionary itself cannot represent the remaining
   HR-HSI residual, which should be handled by Stage 3.

This script freezes the Stage-1 endmembers and directly optimizes one simplex
abundance vector per HR pixel against the test GT. The resulting reconstruction
is an approximate oracle upper bound for any Stage-2 method that only changes
abundance while keeping the same endmember bank.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from metrics import MetricAverager, calc_metrics
from train_stage2_physical import build_stage1_from_checkpoint
from utils import ensure_dir, get_device, move_to_device, set_seed


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


def optimize_oracle_abundance(
    endmembers: torch.Tensor,
    initial_logits: torch.Tensor,
    gt: torch.Tensor,
    steps: int,
    lr: float,
    log_interval: int,
):
    """Optimize simplex abundance logits for the best in-span MSE solution."""
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
                f"  oracle step {step + 1:04d}/{steps:04d}: "
                f"MSE={value:.8f}, PSNR={-10.0 * np.log10(max(value, 1e-12)):.4f}"
            )

    with torch.no_grad():
        abundance = torch.softmax(best_logits, dim=1)
        reconstruction = torch.einsum(
            "bk,nkhw->nbhw", endmembers, abundance
        )
    return abundance, reconstruction, best_loss


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

    baseline_metrics = MetricAverager()
    oracle_metrics = MetricAverager()
    abundance_change_values = []
    oracle_records = []

    for batch_index, batch in enumerate(test_loader):
        batch = move_to_device(batch, device)
        gt = batch["gt"]
        with torch.no_grad():
            endmembers, logits, base_abundance, base_hsi = stage1_baseline(
                stage1,
                batch["lr_hsi"],
                gt.shape[-2:],
            )

        print(f"Test patch {batch_index}: optimizing abundance-only oracle")
        oracle_abundance, oracle_hsi, oracle_mse = optimize_oracle_abundance(
            endmembers=endmembers,
            initial_logits=logits,
            gt=gt,
            steps=cfg.oracle_steps,
            lr=cfg.oracle_lr,
            log_interval=cfg.oracle_log_interval,
        )

        base_result = calc_metrics(base_hsi, gt, cfg.scale_ratio)
        oracle_result = calc_metrics(oracle_hsi, gt, cfg.scale_ratio)
        baseline_metrics.update(base_result)
        oracle_metrics.update(oracle_result)
        abundance_change = float(
            (oracle_abundance - base_abundance).abs().mean().item()
        )
        abundance_change_values.append(abundance_change)
        oracle_records.append(
            {
                "patch": batch_index,
                "base": base_result,
                "oracle": oracle_result,
                "oracle_mse": oracle_mse,
                "mean_absolute_abundance_change": abundance_change,
            }
        )

    base = baseline_metrics.average()
    oracle = oracle_metrics.average()
    summary = {
        "dataset": cfg.dataset,
        "stage1_checkpoint": cfg.stage1_checkpoint,
        "stage1_epoch": int(stage1_state.get("epoch", -1)),
        "oracle_steps": cfg.oracle_steps,
        "oracle_lr": cfg.oracle_lr,
        "base_metrics": base,
        "abundance_oracle_metrics": oracle,
        "oracle_psnr_headroom_over_base": oracle["PSNR"] - base["PSNR"],
        "oracle_sam_headroom_over_base": base["SAM"] - oracle["SAM"],
        "mean_absolute_abundance_change": float(
            np.mean(abundance_change_values)
        ),
        "patches": oracle_records,
    }

    if np.isfinite(cfg.current_stage2_psnr):
        summary["remaining_psnr_headroom_over_current_stage2"] = (
            oracle["PSNR"] - cfg.current_stage2_psnr
        )
    if np.isfinite(cfg.current_stage2_sam):
        summary["remaining_sam_headroom_over_current_stage2"] = (
            cfg.current_stage2_sam - oracle["SAM"]
        )

    output_dir = os.path.join(
        cfg.output_root,
        "stage2_physical_ceiling",
        cfg.dataset,
    )
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "abundance_oracle.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(
        f"Base abundance upsampling : PSNR={base['PSNR']:.4f}, "
        f"SAM={base['SAM']:.4f} deg"
    )
    print(
        f"Abundance-only oracle     : PSNR={oracle['PSNR']:.4f}, "
        f"SAM={oracle['SAM']:.4f} deg"
    )
    print(
        f"Available Stage-2 headroom: "
        f"{summary['oracle_psnr_headroom_over_base']:+.4f} dB, "
        f"{summary['oracle_sam_headroom_over_base']:+.4f} deg"
    )
    if "remaining_psnr_headroom_over_current_stage2" in summary:
        print(
            f"Remaining over current Stage 2: "
            f"{summary['remaining_psnr_headroom_over_current_stage2']:+.4f} dB"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
