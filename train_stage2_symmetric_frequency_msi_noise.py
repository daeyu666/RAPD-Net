"""Train Stage 2 symmetric-frequency model with synthetic MSI degradation.

This is an ablation script for checking whether the frequency reliability / NSP
path becomes useful when HR-MSI is no longer an almost perfectly clean
reference.  The Stage-2 architecture, warm-start checkpoint, optimizer, losses,
and validation metrics are inherited from ``train_stage2_symmetric_frequency``.
Only the HR-MSI tensor yielded by the data loader is replaced by a noisy copy.

By default, checkpoints/logs are written to separate *_msi_noise directories, so
clean Stage-2 experiments are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_symmetric_frequency import Stage2SymmetricFrequencyNet
from train_stage2_coefficients import (
    MONITOR_NAMES,
    FixedSpatialDegradation,
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from train_stage2_dual_space import (
    DUAL_NAMES,
    evaluate_dual,
    train_one_epoch_dual,
)
from train_stage2_symmetric_frequency import (
    SYMMETRIC_NAMES,
    _has_option,
    evaluate_symmetric,
    export_outputs,
    parse_symmetric_args,
)
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


class MSINoiseInjector:
    """Apply synthetic degradation to an HR-MSI tensor.

    The default additive Gaussian setting is intentionally simple: it mainly
    creates unsupported high-frequency fluctuations, making the reliability map
    and noise-minimization terms non-trivial without changing Stage-2 itself.
    """

    def __init__(
        self,
        noise_type: str = "gaussian",
        std: float = 0.03,
        impulse_prob: float = 0.0,
        speckle_std: float = 0.0,
        clip: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.noise_type = str(noise_type)
        self.std = float(std)
        self.impulse_prob = float(impulse_prob)
        self.speckle_std = float(speckle_std)
        self.clip = bool(clip)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        if self.std < 0:
            raise ValueError("msi_noise_std must be non-negative")
        if self.speckle_std < 0:
            raise ValueError("msi_noise_speckle_std must be non-negative")
        if not 0.0 <= self.impulse_prob < 1.0:
            raise ValueError("msi_noise_impulse_prob must lie in [0, 1)")
        if self.clip and not self.clip_min < self.clip_max:
            raise ValueError("msi_noise_clip_min must be smaller than clip_max")

    def _randn_like(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randn(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )

    def _rand_like(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.rand(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )

    def __call__(
        self,
        clean: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        noisy = clean.clone()

        if self.noise_type in {"gaussian", "gaussian_impulse", "gaussian_speckle"}:
            if self.std > 0:
                noisy = noisy + self.std * self._randn_like(noisy, generator)
        elif self.noise_type == "speckle":
            pass
        else:
            raise ValueError(f"Unsupported msi_noise_type: {self.noise_type}")

        if self.noise_type in {"speckle", "gaussian_speckle"}:
            if self.speckle_std > 0:
                noisy = noisy + clean * self.speckle_std * self._randn_like(
                    noisy,
                    generator,
                )

        if self.noise_type == "gaussian_impulse" or self.impulse_prob > 0:
            if self.impulse_prob > 0:
                mask = self._rand_like(noisy, generator) < self.impulse_prob
                impulse = self._rand_like(noisy, generator)
                impulse = impulse * (self.clip_max - self.clip_min) + self.clip_min
                noisy = torch.where(mask, impulse, noisy)

        if self.clip:
            noisy = noisy.clamp(self.clip_min, self.clip_max)
        return noisy


class NoisyMSILoader:
    """A lightweight loader wrapper that replaces ``batch['hr_msi']``."""

    def __init__(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        injector: MSINoiseInjector,
        seed: int,
        enabled: bool = True,
    ) -> None:
        self.loader = loader
        self.injector = injector
        self.seed = int(seed)
        self.enabled = bool(enabled)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for batch in self.loader:
            if not self.enabled:
                yield batch
                continue
            if "hr_msi" not in batch:
                raise KeyError("Batch does not contain 'hr_msi'")
            clean = batch["hr_msi"]
            noisy = self.injector(clean, generator=generator)
            noisy_batch = dict(batch)
            noisy_batch["clean_hr_msi"] = clean
            noisy_batch["hr_msi"] = noisy
            yield noisy_batch

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


NOISE_STAT_NAMES = [
    "msi_noise_l1",
    "msi_noise_rmse",
    "msi_noise_snr_db",
    "msi_noise_min",
    "msi_noise_max",
]


@torch.no_grad()
def msi_noise_diagnostics(
    loader: Iterable[Dict[str, torch.Tensor]],
    max_batches: int = 10,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in NOISE_STAT_NAMES}
    seen = 0
    for batch in loader:
        if "clean_hr_msi" not in batch:
            continue
        clean = batch["clean_hr_msi"].float()
        noisy = batch["hr_msi"].float()
        diff = noisy - clean
        mse = diff.square().mean()
        signal = clean.square().mean()
        snr = 10.0 * math.log10(
            max(float(signal.item()), 1e-12) / max(float(mse.item()), 1e-12)
        )
        batch_size = int(clean.size(0))
        values = {
            "msi_noise_l1": float(diff.abs().mean().item()),
            "msi_noise_rmse": float(torch.sqrt(mse).item()),
            "msi_noise_snr_db": snr,
            "msi_noise_min": float(noisy.min().item()),
            "msi_noise_max": float(noisy.max().item()),
        }
        for name, value in values.items():
            meters[name].update(value, batch_size)
        seen += 1
        if max_batches > 0 and seen >= max_batches:
            break
    if seen == 0:
        return {name: 0.0 for name in NOISE_STAT_NAMES}
    return {name: meter.avg for name, meter in meters.items()}


def parse_msi_noise_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--msi_noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "speckle", "gaussian_speckle", "gaussian_impulse"],
    )
    parser.add_argument("--msi_noise_std", type=float, default=0.03)
    parser.add_argument("--msi_noise_speckle_std", type=float, default=0.0)
    parser.add_argument("--msi_noise_impulse_prob", type=float, default=0.0)
    parser.add_argument("--msi_noise_clip", action="store_true", default=True)
    parser.add_argument("--msi_noise_no_clip", dest="msi_noise_clip", action="store_false")
    parser.add_argument("--msi_noise_clip_min", type=float, default=0.0)
    parser.add_argument("--msi_noise_clip_max", type=float, default=1.0)
    parser.add_argument(
        "--msi_noise_eval_mode",
        type=str,
        default="both",
        choices=["clean", "noisy", "both"],
        help="Evaluate clean MSI, noisy MSI, or both after every epoch.",
    )
    parser.add_argument(
        "--msi_noise_selection_eval",
        type=str,
        default="noisy",
        choices=["clean", "noisy"],
        help="Validation split used for best checkpoint selection.",
    )
    parser.add_argument(
        "--msi_noise_stats_batches",
        type=int,
        default=10,
        help="Number of batches used to estimate noise diagnostics.",
    )

    noise_args, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_symmetric_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(noise_args).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--checkpoint_root"):
        cfg.checkpoint_root = "./checkpoints_msi_noise"
    if not _has_option(remaining, "--log_root"):
        cfg.log_root = "./logs_msi_noise"
    if not _has_option(remaining, "--output_root"):
        cfg.output_root = "./outputs_msi_noise"

    if cfg.msi_noise_eval_mode == "clean" and cfg.msi_noise_selection_eval == "noisy":
        raise ValueError("Cannot select noisy validation when msi_noise_eval_mode=clean")
    return cfg


def make_injector(cfg) -> MSINoiseInjector:
    return MSINoiseInjector(
        noise_type=cfg.msi_noise_type,
        std=cfg.msi_noise_std,
        impulse_prob=cfg.msi_noise_impulse_prob,
        speckle_std=cfg.msi_noise_speckle_std,
        clip=cfg.msi_noise_clip,
        clip_min=cfg.msi_noise_clip_min,
        clip_max=cfg.msi_noise_clip_max,
    )


def make_noisy_loader(loader, injector: MSINoiseInjector, seed: int) -> NoisyMSILoader:
    return NoisyMSILoader(loader, injector=injector, seed=seed, enabled=True)


VALIDATION_SUMMARY_NAMES = [
    "stage2_psnr",
    "stage2_sam",
    "anchor_psnr",
    "anchor_sam",
    "zero_msi_psnr_drop",
    "stage2_psnr_gain_over_anchor",
    "noise_ratio",
    "reliability_ratio",
    "symmetric_low_share",
    "symmetric_mid_share",
    "symmetric_high_share",
    "symmetric_reliable_high_abs",
]


def add_prefixed_summary(
    row: Dict[str, float],
    prefix: str,
    values: Optional[Dict[str, float]],
) -> None:
    if values is None:
        return
    for name in VALIDATION_SUMMARY_NAMES:
        if name in values:
            row[f"{prefix}_{name}"] = values[name]


def selected_metric_names() -> List[str]:
    return [
        "stage2_psnr",
        "stage2_sam",
        "anchor_psnr",
        "anchor_sam",
        "oracle_psnr",
        "oracle_sam",
        "psnr_gain_over_base",
        "stage2_psnr_gain_over_anchor",
        "zero_msi_psnr_drop",
        "remaining_psnr_to_oracle",
        "recoverable_error_fraction",
        *DUAL_NAMES,
        *SYMMETRIC_NAMES,
        *MONITOR_NAMES,
    ]


@torch.no_grad()
def run_validation(
    model: Stage2SymmetricFrequencyNet,
    clean_loader,
    injector: MSINoiseInjector,
    hsi_degrader: FixedSpatialDegradation,
    coefficient_degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if cfg.msi_noise_eval_mode in {"clean", "both"}:
        results["clean"] = evaluate_symmetric(
            model,
            clean_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
    if cfg.msi_noise_eval_mode in {"noisy", "both"}:
        noisy_loader = make_noisy_loader(clean_loader, injector, seed=seed)
        results["noisy"] = evaluate_symmetric(
            model,
            noisy_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        stats_loader = make_noisy_loader(clean_loader, injector, seed=seed)
        results["noisy"].update(
            msi_noise_diagnostics(
                stats_loader,
                max_batches=cfg.msi_noise_stats_batches,
            )
        )
    return results


def validation_for_selection(results: Dict[str, Dict[str, float]], cfg) -> Dict[str, float]:
    if cfg.msi_noise_selection_eval in results:
        return results[cfg.msi_noise_selection_eval]
    if "clean" in results:
        return results["clean"]
    return results["noisy"]


def main() -> None:
    cfg = parse_msi_noise_args()
    cfg.stage = "symmetric_frequency_msi_noise"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    injector = make_injector(cfg)

    stage1, _ = load_stage1_basis_checkpoint(
        cfg.stage1_basis_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    model = Stage2SymmetricFrequencyNet(
        stage1_model=stage1,
        spectral_response=build_spectral_response(info).to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio,
        anchor_normalized_clip=cfg.anchor_normalized_clip,
        projector_tolerance=cfg.projector_tolerance,
        feature_channels=cfg.stage2_feature_channels,
        encoder_blocks=cfg.stage2_encoder_blocks,
        fusion_channels=cfg.stage2_fusion_channels,
        fusion_blocks=cfg.stage2_fusion_blocks,
        max_normalized_residual=cfg.stage2_max_normalized_residual,
        coefficient_scale_floor=cfg.stage2_coefficient_scale_floor,
        num_frequency_bands=cfg.stage2_num_frequency_bands,
        init_low_boundary=cfg.stage2_init_low_boundary,
        init_high_boundary=cfg.stage2_init_high_boundary,
        boundary_temperature=cfg.stage2_boundary_temperature,
        edge_threshold_mode=cfg.stage2_edge_threshold_mode,
        edge_mask_threshold=cfg.stage2_edge_mask_threshold,
        edge_reference_quantile=cfg.stage2_edge_reference_quantile,
        noise_quantile=cfg.stage2_noise_quantile,
        hard_partition=not cfg.stage2_soft_frequency_partition,
    ).to(device)

    boundary_lr = cfg.lr * cfg.stage2_boundary_lr_multiplier
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.regular_parameters()), "lr": cfg.lr},
            {
                "params": list(model.spectral_boundary_parameters()),
                "lr": boundary_lr,
                "weight_decay": 0.0,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs, 1),
        eta_min=cfg.lr * 0.05,
    )
    sam_loss = SAMLoss()
    hsi_degrader = FixedSpatialDegradation(info["n_bands"]).to(device)
    coefficient_degrader = FixedSpatialDegradation(stage1.basis_rank).to(device)

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage2_symmetric_frequency_msi_noise",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage2_symmetric_frequency_msi_noise",
        cfg.dataset,
    )
    log_dir = os.path.join(cfg.log_root, "stage2_symmetric_frequency_msi_noise")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "symmetric_frequency_msi_noise_best.pth")
    best_psnr_path = os.path.join(
        checkpoint_dir,
        "symmetric_frequency_msi_noise_best_psnr.pth",
    )
    best_sam_path = os.path.join(
        checkpoint_dir,
        "symmetric_frequency_msi_noise_best_sam.pth",
    )
    last_path = os.path.join(checkpoint_dir, "symmetric_frequency_msi_noise_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    start_epoch = 0
    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
    else:
        source_epoch, _ = load_checkpoint(
            model,
            cfg.dual_space_checkpoint,
            optimizer=None,
            strict=True,
            map_location=str(device),
            load_optimizer=False,
        )
        write_log(
            log_path,
            f"Loaded dual-space source {cfg.dual_space_checkpoint} "
            f"at epoch {source_epoch}; state_dict matched strictly.",
        )

    noise_cfg = {
        "type": cfg.msi_noise_type,
        "std": cfg.msi_noise_std,
        "speckle_std": cfg.msi_noise_speckle_std,
        "impulse_prob": cfg.msi_noise_impulse_prob,
        "clip": cfg.msi_noise_clip,
        "clip_min": cfg.msi_noise_clip_min,
        "clip_max": cfg.msi_noise_clip_max,
        "eval_mode": cfg.msi_noise_eval_mode,
        "selection_eval": cfg.msi_noise_selection_eval,
    }
    initial_results = run_validation(
        model,
        test_loader,
        injector,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
        seed=cfg.seed + 100_000,
    )
    initial = validation_for_selection(initial_results, cfg)
    clean_initial = initial_results.get("clean")
    noisy_initial = initial_results.get("noisy")
    write_log(
        log_path,
        "Symmetric SSP MSI-noise start | "
        f"selection={cfg.msi_noise_selection_eval}, "
        f"noise={noise_cfg}, "
        f"selected PSNR={initial['stage2_psnr']:.4f}, "
        f"SAM={initial['stage2_sam']:.4f} deg, "
        f"clean PSNR={(clean_initial or initial)['stage2_psnr']:.4f}, "
        f"noisy PSNR={(noisy_initial or initial)['stage2_psnr']:.4f}, "
        f"noisy reliability={(noisy_initial or initial)['reliability_ratio']:.4f}, "
        f"trainable={count_parameters(model):.3f} M.",
    )

    csv_fields = [
        "epoch",
        "lr",
        "selection_eval",
        "msi_noise_type",
        "msi_noise_std",
        "msi_noise_speckle_std",
        "msi_noise_impulse_prob",
        *NOISE_STAT_NAMES,
        *selected_metric_names(),
    ]
    for prefix in ("clean", "noisy"):
        csv_fields.extend(f"{prefix}_{name}" for name in VALIDATION_SUMMARY_NAMES)
    csv_logger = CSVLogger(os.path.join(log_dir, f"{cfg.dataset}.csv"), csv_fields)

    best_selection = initial["selection"]
    best_psnr = initial["stage2_psnr"]
    best_sam = initial["stage2_sam"]
    initial_extra = {
        "stage": "symmetric_frequency_msi_noise",
        "dataset": cfg.dataset,
        "source_checkpoint": cfg.dual_space_checkpoint,
        "noise": noise_cfg,
        "validation": initial_results,
    }
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_selection,
        best_path,
        extra=initial_extra,
    )
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_psnr,
        best_psnr_path,
        extra=initial_extra,
    )
    save_checkpoint(
        model,
        optimizer,
        start_epoch,
        best_sam,
        best_sam_path,
        extra=initial_extra,
    )

    for epoch in range(start_epoch, cfg.epochs):
        noisy_train_loader = make_noisy_loader(
            train_loader,
            injector,
            seed=cfg.seed + epoch,
        )
        train_one_epoch_dual(
            model,
            noisy_train_loader,
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val_results = run_validation(
            model,
            test_loader,
            injector,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
            seed=cfg.seed + 100_000 + epoch,
        )
        val = validation_for_selection(val_results, cfg)
        scheduler.step()

        clean_val = val_results.get("clean")
        noisy_val = val_results.get("noisy")
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"selected={cfg.msi_noise_selection_eval} | "
            f"PSNR={val['stage2_psnr']:.4f}, SAM={val['stage2_sam']:.4f} deg | "
            f"clean={clean_val['stage2_psnr']:.4f}/{clean_val['stage2_sam']:.4f} "
            if clean_val is not None
            else ""
            + f"noisy={noisy_val['stage2_psnr']:.4f}/{noisy_val['stage2_sam']:.4f} "
            if noisy_val is not None
            else ""
            + f"| reliability={val['reliability_ratio']:.4f}, "
            f"noise_ratio={val['noise_ratio']:.4f}, "
            f"high_share={val['symmetric_high_share']:.3f}.",
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_eval": cfg.msi_noise_selection_eval,
            "msi_noise_type": cfg.msi_noise_type,
            "msi_noise_std": cfg.msi_noise_std,
            "msi_noise_speckle_std": cfg.msi_noise_speckle_std,
            "msi_noise_impulse_prob": cfg.msi_noise_impulse_prob,
        }
        row.update({name: val.get(name, "") for name in selected_metric_names()})
        if noisy_val is not None:
            row.update({name: noisy_val.get(name, "") for name in NOISE_STAT_NAMES})
        add_prefixed_summary(row, "clean", clean_val)
        add_prefixed_summary(row, "noisy", noisy_val)
        csv_logger.write(row)

        extra = {
            "stage": "symmetric_frequency_msi_noise",
            "dataset": cfg.dataset,
            "source_checkpoint": cfg.dual_space_checkpoint,
            "noise": noise_cfg,
            "selection_eval": cfg.msi_noise_selection_eval,
            "validation": val_results,
        }
        if val["selection"] < best_selection:
            best_selection = val["selection"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                best_path,
                extra=extra,
            )
        if val["stage2_psnr"] > best_psnr:
            best_psnr = val["stage2_psnr"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_psnr,
                best_psnr_path,
                extra=extra,
            )
        if val["stage2_sam"] < best_sam:
            best_sam = val["stage2_sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_sam,
                best_sam_path,
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

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    final_results = run_validation(
        model,
        test_loader,
        injector,
        hsi_degrader,
        coefficient_degrader,
        sam_loss,
        cfg,
        device,
        seed=cfg.seed + 200_000,
    )
    final = validation_for_selection(final_results, cfg)

    export_outputs(model, test_loader, os.path.join(output_dir, "clean"), device)
    if cfg.msi_noise_eval_mode in {"noisy", "both"}:
        export_outputs(
            model,
            make_noisy_loader(test_loader, injector, seed=cfg.seed + 200_000),
            os.path.join(output_dir, "noisy"),
            device,
        )
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_results, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Symmetric SSP MSI-noise complete | selected={cfg.msi_noise_selection_eval} | "
        f"PSNR={final['stage2_psnr']:.4f}, SAM={final['stage2_sam']:.4f} deg, "
        f"gain over initial={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
