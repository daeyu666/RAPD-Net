"""Train the new RAPD-Net Stage 1: LR-HSI spectral basis extraction.

Only LR-HSI is used. The model is initialized by PCA and then refines an affine
orthogonal spectral subspace with reconstruction, SAM, and spectral-shape
objectives. Coefficients are exact signed projections onto the learned basis;
there is no abundance non-negativity or sum-to-one constraint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from models.stage1_spectral_basis import Stage1SpectralBasisNet
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def first_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


@torch.no_grad()
def collect_lr_spectra(
    loader: Iterable,
    max_pixels: int,
) -> torch.Tensor:
    """Collect representative valid LR-HSI pixels for PCA initialization."""
    if max_pixels < 2:
        raise ValueError("max_pixels must be >= 2")

    collected = []
    total = 0
    for batch in loader:
        lr_hsi = batch["lr_hsi"].float()
        pixels = lr_hsi.permute(0, 2, 3, 1).reshape(-1, lr_hsi.size(1))
        valid = torch.isfinite(pixels).all(dim=1) & (pixels.abs().mean(dim=1) > 1e-6)
        pixels = pixels[valid]
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
        raise RuntimeError("No valid LR-HSI pixels were found for PCA initialization.")
    spectra = torch.cat(collected, dim=0)
    if spectra.size(0) < 2:
        raise RuntimeError("At least two LR-HSI spectra are required for PCA.")
    return spectra


@torch.no_grad()
def compute_pca_initialization(
    spectra: torch.Tensor,
    basis_rank: int,
) -> Dict[str, torch.Tensor]:
    """Compute a numerically stable affine PCA initialization."""
    if spectra.ndim != 2:
        raise ValueError(f"Expected spectra [P, B], got {tuple(spectra.shape)}")
    pixels, bands = spectra.shape
    if basis_rank > bands:
        raise ValueError(f"basis_rank={basis_rank} exceeds bands={bands}")

    data = spectra.double()
    mean = data.mean(dim=0)
    centered = data - mean
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(pixels - 1, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]

    basis = eigenvectors[:, :basis_rank]
    retained_eigenvalues = eigenvalues[:basis_rank]
    coefficient_scale = retained_eigenvalues.sqrt().clamp_min(1e-8)
    total_variance = eigenvalues.sum()
    explained_ratio = retained_eigenvalues.sum() / total_variance.clamp_min(1e-12)

    return {
        "mean_spectrum": mean.float(),
        "basis": basis.float(),
        "coefficient_scale": coefficient_scale.float(),
        "eigenvalues": retained_eigenvalues.float(),
        "all_eigenvalues": eigenvalues.float(),
        "total_variance": total_variance.float(),
        "explained_variance_ratio": explained_ratio.float(),
    }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    sam_loss: SAMLoss,
    cfg,
) -> Dict[str, torch.Tensor]:
    reconstruction = outputs["reconstruction"]
    projector = outputs["projector"]
    reference_projector = (
        outputs["basis"].new_tensor(0.0)
    )

    l1 = F.l1_loss(reconstruction, target)
    mse = F.mse_loss(reconstruction, target)
    sam = sam_loss(reconstruction, target)
    sgrad1 = F.l1_loss(
        first_spectral_difference(reconstruction),
        first_spectral_difference(target),
    )
    sgrad2 = F.l1_loss(
        second_spectral_difference(reconstruction),
        second_spectral_difference(target),
    )

    # Projector anchoring constrains the learned subspace rather than individual
    # basis vectors, which are rotation/sign ambiguous mathematical coordinates.
    if cfg.basis_lambda_anchor > 0:
        reference_projector = cfg._basis_reference_projector.to(projector)
        anchor = F.mse_loss(projector, reference_projector)
    else:
        anchor = projector.new_zeros(())

    total = (
        cfg.lambda_l1 * l1
        + cfg.basis_lambda_sam * sam
        + cfg.basis_lambda_sgrad1 * sgrad1
        + cfg.basis_lambda_sgrad2 * sgrad2
        + cfg.basis_lambda_anchor * anchor
    )
    return {
        "total": total,
        "l1": l1,
        "mse": mse,
        "sam": sam,
        "sgrad1": sgrad1,
        "sgrad2": sgrad2,
        "anchor": anchor,
    }


LOSS_NAMES = (
    "total",
    "l1",
    "mse",
    "sam",
    "sgrad1",
    "sgrad2",
    "anchor",
)


def create_meters() -> Dict[str, AverageMeter]:
    return {name: AverageMeter() for name in LOSS_NAMES}


def update_meters(
    meters: Dict[str, AverageMeter],
    losses: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name in LOSS_NAMES:
        meters[name].update(float(losses[name].detach().item()), batch_size)


def train_one_epoch(
    model: Stage1SpectralBasisNet,
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
        if cfg.basis_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.basis_grad_clip
            )
        optimizer.step()
        update_meters(meters, losses, lr_hsi.size(0))

    result = {name: meter.avg for name, meter in meters.items()}
    result["sam_deg"] = result["sam"] * 180.0 / math.pi
    result["psnr"] = -10.0 * math.log10(max(result["mse"], 1e-12))
    return result


@torch.no_grad()
def evaluate(
    model: Stage1SpectralBasisNet,
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
        outputs = model(lr_hsi)
        losses = compute_losses(outputs, lr_hsi, sam_loss, cfg)
        update_meters(meters, losses, lr_hsi.size(0))

    result = {name: meter.avg for name, meter in meters.items()}
    result["sam_deg"] = result["sam"] * 180.0 / math.pi
    result["psnr"] = -10.0 * math.log10(max(result["mse"], 1e-12))
    result["selection"] = (
        cfg.lambda_l1 * result["l1"]
        + cfg.basis_selection_sam_weight * result["sam"]
        + cfg.basis_selection_sgrad1_weight * result["sgrad1"]
        + cfg.basis_selection_sgrad2_weight * result["sgrad2"]
    )

    statistics = model.subspace_statistics()
    for name, value in statistics.items():
        result[name] = float(value.detach().item())
    return result


@torch.no_grad()
def estimate_coefficient_scale(
    model: Stage1SpectralBasisNet,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate final per-coordinate mean/std under the selected basis."""
    model.eval()
    count = 0
    sum_coeff = torch.zeros(model.basis_rank, dtype=torch.float64, device=device)
    sum_square = torch.zeros_like(sum_coeff)

    for batch in loader:
        batch = move_to_device(batch, device)
        coefficients = model(batch["lr_hsi"])["coefficients"].double()
        flattened = coefficients.permute(1, 0, 2, 3).reshape(
            model.basis_rank, -1
        )
        sum_coeff += flattened.sum(dim=1)
        sum_square += flattened.square().sum(dim=1)
        count += flattened.size(1)

    if count == 0:
        raise RuntimeError("Cannot estimate coefficient scale from an empty loader.")
    mean = sum_coeff / count
    variance = (sum_square / count - mean.square()).clamp_min(1e-12)
    return mean.float().cpu(), variance.sqrt().float().cpu()


@torch.no_grad()
def export_stage1_artifacts(
    model: Stage1SpectralBasisNet,
    test_loader,
    info: dict,
    output_dir: str,
    coefficient_mean: torch.Tensor,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    model.eval()

    basis = model.get_basis().detach().cpu()
    mean_spectrum = model.mean_spectrum.detach().cpu()
    coefficient_scale = model.coefficient_scale.detach().cpu()
    projector = basis @ basis.transpose(0, 1)
    wavelengths = info.get("hsi_wavelengths")
    if wavelengths is None:
        wavelengths = np.arange(model.n_bands, dtype=np.float32)
    else:
        wavelengths = np.asarray(wavelengths, dtype=np.float32)

    np.save(os.path.join(output_dir, "spectral_basis.npy"), basis.numpy())
    np.save(os.path.join(output_dir, "mean_spectrum.npy"), mean_spectrum.numpy())
    np.save(
        os.path.join(output_dir, "coefficient_scale.npy"),
        coefficient_scale.numpy(),
    )
    np.save(
        os.path.join(output_dir, "coefficient_mean.npy"),
        coefficient_mean.numpy(),
    )
    np.save(os.path.join(output_dir, "basis_projector.npy"), projector.numpy())
    np.save(
        os.path.join(output_dir, "pca_eigenvalues.npy"),
        model.pca_eigenvalues.detach().cpu().numpy(),
    )

    first_batch = move_to_device(next(iter(test_loader)), device)
    lr_hsi = first_batch["lr_hsi"]
    outputs = model(lr_hsi)
    np.savez_compressed(
        os.path.join(output_dir, "stage1_basis_test_outputs.npz"),
        lr_hsi=lr_hsi.detach().cpu().numpy(),
        reconstruction=outputs["reconstruction"].detach().cpu().numpy(),
        residual=outputs["residual"].detach().cpu().numpy(),
        coefficients=outputs["coefficients"].detach().cpu().numpy(),
        normalized_coefficients=outputs[
            "normalized_coefficients"
        ].detach().cpu().numpy(),
        basis=basis.numpy(),
        mean_spectrum=mean_spectrum.numpy(),
        coefficient_scale=coefficient_scale.numpy(),
        coefficient_mean=coefficient_mean.numpy(),
        projector=projector.numpy(),
        wavelengths=wavelengths,
    )

    statistics = {
        name: float(value.detach().item())
        for name, value in model.subspace_statistics().items()
    }
    statistics.update(
        {
            "n_bands": model.n_bands,
            "basis_rank": model.basis_rank,
            "coefficient_scale_min": float(coefficient_scale.min().item()),
            "coefficient_scale_max": float(coefficient_scale.max().item()),
            "coefficient_scale_mean": float(coefficient_scale.mean().item()),
        }
    )
    with open(
        os.path.join(output_dir, "basis_statistics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(statistics, file, indent=2, ensure_ascii=False)


def parse_stage1_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--basis_rank", type=int, default=32)
    parser.add_argument("--basis_init_pixels", type=int, default=100000)
    parser.add_argument("--basis_grad_clip", type=float, default=1.0)

    parser.add_argument("--basis_lambda_sam", type=float, default=0.5)
    parser.add_argument("--basis_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--basis_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--basis_lambda_anchor", type=float, default=0.001)

    parser.add_argument("--basis_selection_sam_weight", type=float, default=1.0)
    parser.add_argument("--basis_selection_sgrad1_weight", type=float, default=0.2)
    parser.add_argument("--basis_selection_sgrad2_weight", type=float, default=0.1)

    stage_args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(stage_args).items():
        setattr(cfg, key, value)
    return cfg


def checkpoint_extra(
    cfg,
    info: dict,
    result: Dict[str, float],
    initialization: Dict[str, torch.Tensor],
) -> dict:
    return {
        "stage": "spectral_basis",
        "dataset": cfg.dataset,
        "n_bands": int(info["n_bands"]),
        "basis_rank": int(cfg.basis_rank),
        "basis_init_pixels": int(cfg.basis_init_pixels),
        "pca_explained_variance_ratio": float(
            initialization["explained_variance_ratio"].item()
        ),
        "train_sam_weight": cfg.basis_lambda_sam,
        "train_sgrad1_weight": cfg.basis_lambda_sgrad1,
        "train_sgrad2_weight": cfg.basis_lambda_sgrad2,
        "subspace_anchor_weight": cfg.basis_lambda_anchor,
        "selection_sam_weight": cfg.basis_selection_sam_weight,
        "selection_sgrad1_weight": cfg.basis_selection_sgrad1_weight,
        "selection_sgrad2_weight": cfg.basis_selection_sgrad2_weight,
        "validation": result,
    }


def main() -> None:
    cfg = parse_stage1_args()
    cfg.stage = "basis"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    model = Stage1SpectralBasisNet(
        n_bands=info["n_bands"],
        basis_rank=cfg.basis_rank,
    ).to(device)
    sam_loss = SAMLoss()

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root, "stage1_basis", cfg.dataset
    )
    output_dir = os.path.join(cfg.output_root, "stage1_basis", cfg.dataset)
    log_dir = os.path.join(cfg.log_root, "stage1_basis")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    pca_init_path = os.path.join(checkpoint_dir, "basis_pca_init.pth")
    best_path = os.path.join(checkpoint_dir, "basis_best.pth")
    best_sam_path = os.path.join(checkpoint_dir, "basis_best_sam.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "basis_best_psnr.pth")
    last_path = os.path.join(checkpoint_dir, "basis_last.pth")
    deployment_path = os.path.join(checkpoint_dir, "basis_for_stage2.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    initialization = None
    if cfg.resume:
        # Build a placeholder reference projector before loading; checkpoint
        # buffers then restore the actual PCA mean, basis, and statistics.
        cfg._basis_reference_projector = model.get_reference_projector().detach()
    else:
        spectra = collect_lr_spectra(
            train_loader,
            max_pixels=cfg.basis_init_pixels,
        )
        initialization = compute_pca_initialization(
            spectra,
            basis_rank=cfg.basis_rank,
        )
        model.initialize_from_pca(
            mean_spectrum=initialization["mean_spectrum"].to(device),
            basis=initialization["basis"].to(device),
            coefficient_scale=initialization["coefficient_scale"].to(device),
            eigenvalues=initialization["eigenvalues"].to(device),
            total_variance=initialization["total_variance"].to(device),
        )
        cfg._basis_reference_projector = (
            model.get_reference_projector().detach()
        )
        write_log(
            log_path,
            f"PCA initialized from {spectra.size(0)} LR-HSI spectra; "
            f"rank={cfg.basis_rank}; explained variance="
            f"{float(initialization['explained_variance_ratio'].item()):.8f}.",
        )

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

    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        fieldnames=[
            "epoch",
            "lr",
            "train_total",
            "train_l1",
            "train_psnr",
            "train_sam_deg",
            "val_l1",
            "val_psnr",
            "val_sam_deg",
            "val_sgrad1",
            "val_sgrad2",
            "val_selection",
            "orthogonality_error",
            "projector_idempotence_error",
            "projector_drift",
        ],
    )

    start_epoch = 0
    best_selection = float("inf")
    best_sam = float("inf")
    best_psnr = -float("inf")

    if cfg.resume:
        start_epoch, best_selection = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        cfg._basis_reference_projector = (
            model.get_reference_projector().detach()
        )
        initialization = {
            "explained_variance_ratio": model.pca_eigenvalues.sum()
            / model.total_variance.clamp_min(model.eps)
        }
        write_log(log_path, f"Resumed from {cfg.resume} at epoch {start_epoch}.")
    else:
        initial_result = evaluate(model, test_loader, sam_loss, cfg, device)
        initial_extra = checkpoint_extra(cfg, info, initial_result, initialization)
        save_checkpoint(
            model,
            optimizer,
            epoch=0,
            best_metric=initial_result["selection"],
            path=pca_init_path,
            extra=initial_extra,
        )
        save_checkpoint(
            model,
            optimizer,
            epoch=0,
            best_metric=initial_result["selection"],
            path=best_path,
            extra=initial_extra,
        )
        save_checkpoint(
            model,
            optimizer,
            epoch=0,
            best_metric=initial_result["sam"],
            path=best_sam_path,
            extra=initial_extra,
        )
        save_checkpoint(
            model,
            optimizer,
            epoch=0,
            best_metric=initial_result["psnr"],
            path=best_psnr_path,
            extra=initial_extra,
        )
        best_selection = initial_result["selection"]
        best_sam = initial_result["sam"]
        best_psnr = initial_result["psnr"]
        write_log(
            log_path,
            f"PCA baseline | val L1={initial_result['l1']:.8f}, "
            f"PSNR={initial_result['psnr']:.4f}, "
            f"SAM={initial_result['sam_deg']:.4f} deg; saved {pca_init_path}.",
        )

    write_log(
        log_path,
        f"New Stage 1 start | dataset={cfg.dataset}, bands={info['n_bands']}, "
        f"rank={cfg.basis_rank}, trainable={count_parameters(model):.6f} M, "
        f"weights: L1={cfg.lambda_l1}, SAM={cfg.basis_lambda_sam}, "
        f"SGrad1={cfg.basis_lambda_sgrad1}, "
        f"SGrad2={cfg.basis_lambda_sgrad2}, "
        f"anchor={cfg.basis_lambda_anchor}.",
    )

    for epoch in range(start_epoch, cfg.epochs):
        train_result = train_one_epoch(
            model, train_loader, optimizer, sam_loss, cfg, device
        )
        val_result = evaluate(model, test_loader, sam_loss, cfg, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"train PSNR={train_result['psnr']:.4f}, "
            f"SAM={train_result['sam_deg']:.4f} deg | "
            f"val PSNR={val_result['psnr']:.4f}, "
            f"SAM={val_result['sam_deg']:.4f} deg, "
            f"L1={val_result['l1']:.8f} | "
            f"orth={val_result['orthogonality_error']:.3e}, "
            f"drift={val_result['projector_drift']:.3e}.",
        )
        csv_logger.write(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_total": train_result["total"],
                "train_l1": train_result["l1"],
                "train_psnr": train_result["psnr"],
                "train_sam_deg": train_result["sam_deg"],
                "val_l1": val_result["l1"],
                "val_psnr": val_result["psnr"],
                "val_sam_deg": val_result["sam_deg"],
                "val_sgrad1": val_result["sgrad1"],
                "val_sgrad2": val_result["sgrad2"],
                "val_selection": val_result["selection"],
                "orthogonality_error": val_result["orthogonality_error"],
                "projector_idempotence_error": val_result[
                    "projector_idempotence_error"
                ],
                "projector_drift": val_result["projector_drift"],
            }
        )

        extra = checkpoint_extra(cfg, info, val_result, initialization)
        if val_result["selection"] < best_selection:
            best_selection = val_result["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                best_path,
                extra=extra,
            )
            write_log(log_path, f"Saved spectral-first basis: {best_path}")

        if val_result["sam"] < best_sam:
            best_sam = val_result["sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_sam,
                best_sam_path,
                extra=extra,
            )

        if val_result["psnr"] > best_psnr:
            best_psnr = val_result["psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            best_selection,
            last_path,
            extra=extra,
        )

    selected_epoch, selected_metric = load_checkpoint(
        model,
        best_path,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    coefficient_mean, coefficient_scale = estimate_coefficient_scale(
        model,
        train_loader,
        device,
    )
    model.set_coefficient_scale(coefficient_scale.to(device))
    final_result = evaluate(model, test_loader, sam_loss, cfg, device)
    deployment_extra = checkpoint_extra(cfg, info, final_result, initialization)
    deployment_extra.update(
        {
            "source_checkpoint": best_path,
            "source_epoch": selected_epoch,
            "coefficient_mean": coefficient_mean.tolist(),
            "coefficient_scale": coefficient_scale.tolist(),
        }
    )
    save_checkpoint(
        model,
        optimizer=None,
        epoch=selected_epoch,
        best_metric=selected_metric,
        path=deployment_path,
        extra=deployment_extra,
    )
    export_stage1_artifacts(
        model,
        test_loader,
        info,
        output_dir,
        coefficient_mean,
        device,
    )
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_result, file, indent=2, ensure_ascii=False)

    write_log(
        log_path,
        f"New Stage 1 complete | selected epoch={selected_epoch}, "
        f"PSNR={final_result['psnr']:.4f}, "
        f"SAM={final_result['sam_deg']:.4f} deg, "
        f"L1={final_result['l1']:.8f}. "
        f"Stage-2 checkpoint: {deployment_path}",
    )


if __name__ == "__main__":
    main()
