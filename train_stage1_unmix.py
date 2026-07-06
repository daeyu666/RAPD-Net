"""Train Stage 1 of RAPD-Net: physical unmixing on LR-HSI only.

This stage learns a scene-level endmember bank and LR abundance estimator.
The best checkpoint is later frozen by the reliable abundance-injection stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage1_unmixing import Stage1UnmixingNet
from utils import (
    CSVLogger,
    AverageMeter,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match first-order spectral differences."""
    return F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def abundance_tv_loss(abundance: torch.Tensor) -> torch.Tensor:
    """Spatial total variation on abundance maps."""
    loss_h = torch.mean(torch.abs(abundance[:, :, 1:] - abundance[:, :, :-1]))
    loss_w = torch.mean(torch.abs(abundance[:, :, :, 1:] - abundance[:, :, :, :-1]))
    return loss_h + loss_w


def abundance_entropy_loss(abundance: torch.Tensor) -> torch.Tensor:
    """Low-weight entropy penalty that discourages uniformly dense mixtures."""
    safe = abundance.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=1).mean()


def endmember_diversity_loss(
    endmembers: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Penalize only nearly duplicated endmember spectra."""
    spectra = F.normalize(endmembers.transpose(0, 1), dim=1, eps=1e-8)
    cosine = spectra @ spectra.transpose(0, 1)
    k = cosine.size(0)
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=cosine.device)
    return F.relu(cosine[off_diagonal] - margin).pow(2).mean()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    sam_loss: SAMLoss,
    cfg,
) -> Dict[str, torch.Tensor]:
    reconstruction = outputs["reconstruction"]
    abundance = outputs["abundance"]
    endmembers = outputs["endmembers"]

    losses = {
        "l1": F.l1_loss(reconstruction, target),
        "sam": sam_loss(reconstruction, target),
        "sgrad": spectral_gradient_loss(reconstruction, target),
        "div": endmember_diversity_loss(
            endmembers,
            margin=cfg.unmix_diversity_margin,
        ),
        "tv": abundance_tv_loss(abundance),
        "entropy": abundance_entropy_loss(abundance),
    }
    losses["total"] = (
        cfg.lambda_l1 * losses["l1"]
        + cfg.lambda_sam * losses["sam"]
        + cfg.lambda_sgrad * losses["sgrad"]
        + cfg.lambda_endmember_div * losses["div"]
        + cfg.lambda_abundance_tv * losses["tv"]
        + cfg.lambda_abundance_entropy * losses["entropy"]
    )
    return losses


@torch.no_grad()
def collect_initialization_pixels(
    loader: Iterable,
    max_pixels: int,
) -> torch.Tensor:
    """Collect representative LR-HSI spectra without loading the full scene."""
    collected = []
    total = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].float()
        pixels = lr_hsi.permute(0, 2, 3, 1).reshape(-1, lr_hsi.size(1))
        pixels = pixels[pixels.mean(dim=1) > 1e-4]
        if pixels.numel() == 0:
            continue

        remaining = max_pixels - total
        if pixels.size(0) > remaining:
            indices = torch.randperm(pixels.size(0))[:remaining]
            pixels = pixels[indices]
        collected.append(pixels.cpu())
        total += pixels.size(0)
        if total >= max_pixels:
            break

    if not collected:
        raise RuntimeError("No valid LR-HSI pixels were found for initialization.")
    return torch.cat(collected, dim=0)


@torch.no_grad()
def farthest_spectral_initialization(
    pixels: torch.Tensor,
    num_endmembers: int,
) -> torch.Tensor:
    """Select real spectra by farthest-point sampling under cosine distance."""
    if pixels.ndim != 2 or pixels.size(0) < num_endmembers:
        raise ValueError(
            f"Need [P, B] pixels with P >= {num_endmembers}, got {tuple(pixels.shape)}"
        )

    normalized = F.normalize(pixels, dim=1, eps=1e-8)
    mean_spectrum = F.normalize(pixels.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
    first_index = torch.argmax(
        1.0 - normalized @ mean_spectrum.transpose(0, 1)
    ).item()

    selected_indices = [first_index]
    min_distance = 1.0 - normalized @ normalized[first_index]
    for _ in range(1, num_endmembers):
        index = torch.argmax(min_distance).item()
        selected_indices.append(index)
        min_distance = torch.minimum(
            min_distance,
            1.0 - normalized @ normalized[index],
        )

    return pixels[selected_indices].transpose(0, 1).contiguous()


def create_meters() -> Dict[str, AverageMeter]:
    return {
        name: AverageMeter()
        for name in ("total", "l1", "sam", "sgrad", "div", "tv", "entropy")
    }


def update_meters(
    meters: Dict[str, AverageMeter],
    losses: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, meter in meters.items():
        meter.update(losses[name].item(), batch_size)


def train_one_epoch(
    model: Stage1UnmixingNet,
    loader,
    optimizer: torch.optim.Optimizer,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    meters = create_meters()

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(lr_hsi)
        losses = compute_losses(outputs, lr_hsi, sam_loss, cfg)
        losses["total"].backward()
        if cfg.unmix_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.unmix_grad_clip)
        optimizer.step()
        update_meters(meters, losses, lr_hsi.size(0))

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: Stage1UnmixingNet,
    loader,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    meters = create_meters()

    for batch in loader:
        batch = move_to_device(batch, device)
        lr_hsi = batch["lr_hsi"]
        losses = compute_losses(model(lr_hsi), lr_hsi, sam_loss, cfg)
        update_meters(meters, losses, lr_hsi.size(0))

    result = {name: meter.avg for name, meter in meters.items()}
    result["sam_deg"] = result["sam"] * 180.0 / math.pi
    result["selection"] = (
        cfg.lambda_l1 * result["l1"]
        + cfg.lambda_sam * result["sam"]
        + cfg.lambda_sgrad * result["sgrad"]
    )
    return result


@torch.no_grad()
def endmember_statistics(endmembers: torch.Tensor) -> Dict[str, float]:
    spectra = F.normalize(endmembers.transpose(0, 1), dim=1, eps=1e-8)
    cosine = spectra @ spectra.transpose(0, 1)
    k = cosine.size(0)
    values = cosine[~torch.eye(k, dtype=torch.bool, device=cosine.device)]
    return {
        "endmember_cosine_mean": float(values.mean().item()),
        "endmember_cosine_max": float(values.max().item()),
        "endmember_spectral_tv": float(
            (endmembers[1:] - endmembers[:-1]).abs().mean().item()
        ),
    }


@torch.no_grad()
def export_stage1_artifacts(
    model: Stage1UnmixingNet,
    test_loader,
    info: dict,
    output_dir: str,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    model.eval()

    endmembers = model.get_endmembers().detach().cpu()
    wavelengths = info.get("hsi_wavelengths")
    if wavelengths is None:
        wavelengths = np.arange(endmembers.size(0), dtype=np.float32)
    else:
        wavelengths = np.asarray(wavelengths, dtype=np.float32)

    wavelength_tensor = torch.from_numpy(wavelengths).view(-1, 1)
    centroids = (
        (endmembers * wavelength_tensor).sum(dim=0)
        / endmembers.sum(dim=0).clamp_min(1e-8)
    )
    permutation = torch.argsort(centroids)

    np.save(
        os.path.join(output_dir, "endmembers_model_order.npy"),
        endmembers.numpy(),
    )
    np.save(
        os.path.join(output_dir, "endmembers_sorted.npy"),
        endmembers[:, permutation].transpose(0, 1).numpy(),
    )

    order_records = [
        {
            "sorted_index": sorted_index,
            "model_index": original_index,
            "spectral_centroid": float(centroids[original_index].item()),
        }
        for sorted_index, original_index in enumerate(permutation.tolist())
    ]
    with open(
        os.path.join(output_dir, "endmember_order.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(order_records, file, indent=2, ensure_ascii=False)

    first_batch = move_to_device(next(iter(test_loader)), device)
    lr_hsi = first_batch["lr_hsi"]
    outputs = model(lr_hsi)
    abundance = outputs["abundance"].detach().cpu()
    reconstruction = outputs["reconstruction"].detach().cpu()
    np.savez_compressed(
        os.path.join(output_dir, "stage1_test_outputs.npz"),
        lr_hsi=lr_hsi.detach().cpu().numpy(),
        reconstruction=reconstruction.numpy(),
        abundance_model_order=abundance.numpy(),
        abundance_sorted=abundance[:, permutation].numpy(),
        endmembers_model_order=endmembers.numpy(),
        endmembers_sorted=endmembers[:, permutation].transpose(0, 1).numpy(),
        wavelengths=wavelengths,
    )

    with open(
        os.path.join(output_dir, "endmember_statistics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(endmember_statistics(endmembers.to(device)), file, indent=2)


def parse_stage1_args():
    """Parse stage-specific options, then delegate shared options to config.py."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--unmix_num_endmembers", type=int, default=32)
    parser.add_argument("--unmix_hidden_channels", type=int, default=64)
    parser.add_argument("--unmix_num_blocks", type=int, default=3)
    parser.add_argument("--unmix_init_pixels", type=int, default=50000)
    parser.add_argument("--unmix_diversity_margin", type=float, default=0.98)
    parser.add_argument("--unmix_grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_endmember_div", type=float, default=0.01)
    parser.add_argument("--lambda_abundance_tv", type=float, default=0.001)
    parser.add_argument("--lambda_abundance_entropy", type=float, default=0.001)
    stage_args, remaining = parser.parse_known_args()

    cfg = parse_args(remaining)
    for key, value in vars(stage_args).items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    cfg = parse_stage1_args()
    cfg.stage = "unmix"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    model = Stage1UnmixingNet(
        n_bands=info["n_bands"],
        num_endmembers=cfg.unmix_num_endmembers,
        hidden_channels=cfg.unmix_hidden_channels,
        num_blocks=cfg.unmix_num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * 0.05,
    )
    sam_loss = SAMLoss()

    checkpoint_dir = os.path.join(cfg.checkpoint_root, "stage1_unmix", cfg.dataset)
    output_dir = os.path.join(cfg.output_root, "stage1_unmix", cfg.dataset)
    log_dir = os.path.join(cfg.log_root, "stage1_unmix")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "unmixing_best.pth")
    last_path = os.path.join(checkpoint_dir, "unmixing_last.pth")
    text_log_path = os.path.join(log_dir, f"{cfg.dataset}.log")
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        fieldnames=[
            "epoch",
            "lr",
            "train_total",
            "train_l1",
            "train_sam_deg",
            "val_l1",
            "val_sam_deg",
            "val_sgrad",
            "val_selection",
            "endmember_cosine_mean",
            "endmember_cosine_max",
        ],
    )

    start_epoch = 0
    best_selection = float("inf")
    if cfg.resume:
        start_epoch, best_selection = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        write_log(text_log_path, f"Resumed from {cfg.resume} at epoch {start_epoch}.")
    else:
        pixels = collect_initialization_pixels(
            train_loader,
            max_pixels=cfg.unmix_init_pixels,
        )
        initial_endmembers = farthest_spectral_initialization(
            pixels,
            num_endmembers=cfg.unmix_num_endmembers,
        )
        model.initialize_endmembers(initial_endmembers.to(device))
        write_log(
            text_log_path,
            f"Initialized {cfg.unmix_num_endmembers} endmembers from "
            f"{pixels.size(0)} real LR-HSI pixels.",
        )

    write_log(
        text_log_path,
        f"Stage-1 parameters: {count_parameters(model):.3f} M; "
        f"bands={info['n_bands']}; K={cfg.unmix_num_endmembers}.",
    )

    for epoch in range(start_epoch, cfg.epochs):
        train_result = train_one_epoch(
            model, train_loader, optimizer, sam_loss, cfg, device
        )
        val_result = evaluate(model, test_loader, sam_loss, cfg, device)
        statistics = endmember_statistics(model.get_endmembers().detach())
        scheduler.step()

        train_sam_deg = train_result["sam"] * 180.0 / math.pi
        current_lr = optimizer.param_groups[0]["lr"]
        write_log(
            text_log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"train L1={train_result['l1']:.6f}, SAM={train_sam_deg:.4f} deg | "
            f"val L1={val_result['l1']:.6f}, SAM={val_result['sam_deg']:.4f} deg | "
            f"selection={val_result['selection']:.6f}",
        )
        csv_logger.write(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_total": train_result["total"],
                "train_l1": train_result["l1"],
                "train_sam_deg": train_sam_deg,
                "val_l1": val_result["l1"],
                "val_sam_deg": val_result["sam_deg"],
                "val_sgrad": val_result["sgrad"],
                "val_selection": val_result["selection"],
                **statistics,
            }
        )

        checkpoint_extra = {
            "stage": "unmix",
            "dataset": cfg.dataset,
            "n_bands": info["n_bands"],
            "num_endmembers": cfg.unmix_num_endmembers,
            "hidden_channels": cfg.unmix_hidden_channels,
            "num_blocks": cfg.unmix_num_blocks,
        }
        if val_result["selection"] < best_selection:
            best_selection = val_result["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                best_path,
                extra=checkpoint_extra,
            )
            write_log(text_log_path, f"Saved new best checkpoint: {best_path}")

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            best_selection,
            last_path,
            extra=checkpoint_extra,
        )

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    export_stage1_artifacts(model, test_loader, info, output_dir, device)
    write_log(text_log_path, f"Stage 1 complete. Artifacts: {output_dir}.")


if __name__ == "__main__":
    main()
