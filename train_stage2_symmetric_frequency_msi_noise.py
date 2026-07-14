"""Train Stage 2 symmetric-frequency model with synthetic HR-MSI noise.

Purpose:
    Test whether the SFSR-style frequency reliability / NSP path becomes useful
    when HR-MSI is no longer an almost-clean reference.

Important:
    The Stage-2 model, warm-start checkpoint, optimizer, losses, and evaluation
    functions are reused from ``train_stage2_symmetric_frequency.py``.  This
    script only wraps the data loader and replaces ``batch['hr_msi']`` with a
    noisy copy.  Outputs are written to separate *_msi_noise directories, so the
    clean Stage-2 results are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, Optional

import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage2_symmetric_frequency import Stage2SymmetricFrequencyNet
from train_stage2_coefficients import (
    FixedSpatialDegradation,
    build_spectral_response,
    load_stage1_basis_checkpoint,
)
from train_stage2_dual_space import train_one_epoch_dual
from train_stage2_symmetric_frequency import (
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
    save_checkpoint,
    set_seed,
    write_log,
)


class MSINoiseInjector:
    """Synthetic HR-MSI degradation used only by this ablation script."""

    def __init__(
        self,
        noise_type: str = "gaussian",
        std: float = 0.03,
        speckle_std: float = 0.0,
        impulse_prob: float = 0.0,
        clip: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.noise_type = str(noise_type)
        self.std = float(std)
        self.speckle_std = float(speckle_std)
        self.impulse_prob = float(impulse_prob)
        self.clip = bool(clip)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        if self.std < 0.0:
            raise ValueError("msi_noise_std must be non-negative")
        if self.speckle_std < 0.0:
            raise ValueError("msi_noise_speckle_std must be non-negative")
        if not 0.0 <= self.impulse_prob < 1.0:
            raise ValueError("msi_noise_impulse_prob must lie in [0, 1)")
        if self.clip and not self.clip_min < self.clip_max:
            raise ValueError("msi_noise_clip_min must be smaller than msi_noise_clip_max")

    @staticmethod
    def _randn_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )

    @staticmethod
    def _rand_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return torch.rand(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )

    def __call__(self, clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noisy = clean.clone()
        if self.noise_type in {"gaussian", "gaussian_speckle", "gaussian_impulse"}:
            if self.std > 0.0:
                noisy = noisy + self.std * self._randn_like(noisy, generator)
        elif self.noise_type == "speckle":
            pass
        else:
            raise ValueError(f"Unsupported msi_noise_type: {self.noise_type}")

        if self.noise_type in {"speckle", "gaussian_speckle"}:
            if self.speckle_std > 0.0:
                noisy = noisy + clean * self.speckle_std * self._randn_like(
                    noisy,
                    generator,
                )

        if self.noise_type == "gaussian_impulse" or self.impulse_prob > 0.0:
            if self.impulse_prob > 0.0:
                mask = self._rand_like(noisy, generator) < self.impulse_prob
                impulse = self._rand_like(noisy, generator)
                impulse = impulse * (self.clip_max - self.clip_min) + self.clip_min
                noisy = torch.where(mask, impulse, noisy)

        if self.clip:
            noisy = noisy.clamp(self.clip_min, self.clip_max)
        return noisy


class NoisyMSILoader:
    """Wrap a dataloader and replace ``hr_msi`` by a reproducible noisy copy."""

    def __init__(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        injector: MSINoiseInjector,
        seed: int,
    ) -> None:
        self.loader = loader
        self.injector = injector
        self.seed = int(seed)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for batch in self.loader:
            clean = batch["hr_msi"]
            noisy = self.injector(clean, generator)
            out = dict(batch)
            out["clean_hr_msi"] = clean
            out["hr_msi"] = noisy
            yield out

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]


NOISE_STAT_NAMES = ["msi_noise_l1", "msi_noise_rmse", "msi_noise_snr_db"]


@torch.no_grad()
def msi_noise_diagnostics(
    loader: Iterable[Dict[str, torch.Tensor]],
    max_batches: int = 10,
) -> Dict[str, float]:
    meters = {name: AverageMeter() for name in NOISE_STAT_NAMES}
    seen = 0
    for batch in loader:
        clean = batch["clean_hr_msi"].float()
        noisy = batch["hr_msi"].float()
        diff = noisy - clean
        mse = diff.square().mean()
        signal = clean.square().mean()
        snr = 10.0 * math.log10(
            max(float(signal.item()), 1e-12) / max(float(mse.item()), 1e-12)
        )
        batch_size = int(clean.size(0))
        meters["msi_noise_l1"].update(float(diff.abs().mean().item()), batch_size)
        meters["msi_noise_rmse"].update(float(torch.sqrt(mse).item()), batch_size)
        meters["msi_noise_snr_db"].update(snr, batch_size)
        seen += 1
        if max_batches > 0 and seen >= max_batches:
            break
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
    )
    parser.add_argument(
        "--msi_noise_selection_eval",
        type=str,
        default="noisy",
        choices=["clean", "noisy"],
    )
    parser.add_argument("--msi_noise_stats_batches", type=int, default=10)

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
        speckle_std=cfg.msi_noise_speckle_std,
        impulse_prob=cfg.msi_noise_impulse_prob,
        clip=cfg.msi_noise_clip,
        clip_min=cfg.msi_noise_clip_min,
        clip_max=cfg.msi_noise_clip_max,
    )


def noisy_loader(loader, injector: MSINoiseInjector, seed: int) -> NoisyMSILoader:
    return NoisyMSILoader(loader, injector, seed)


@torch.no_grad()
def evaluate_clean_and_noisy(
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
        n_loader = noisy_loader(clean_loader, injector, seed)
        results["noisy"] = evaluate_symmetric(
            model,
            n_loader,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        results["noisy"].update(
            msi_noise_diagnostics(
                noisy_loader(clean_loader, injector, seed),
                max_batches=cfg.msi_noise_stats_batches,
            )
        )
    return results


def select_validation(results: Dict[str, Dict[str, float]], cfg) -> Dict[str, float]:
    if cfg.msi_noise_selection_eval in results:
        return results[cfg.msi_noise_selection_eval]
    if "clean" in results:
        return results["clean"]
    return results["noisy"]


def summary_text(name: str, value: Optional[Dict[str, float]]) -> str:
    if value is None:
        return f"{name}=N/A"
    return (
        f"{name}={value['stage2_psnr']:.4f}/{value['stage2_sam']:.4f}, "
        f"rel={value['reliability_ratio']:.4f}, noise={value['noise_ratio']:.4f}"
    )


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
    initial_results = evaluate_clean_and_noisy(
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
    initial = select_validation(initial_results, cfg)
    write_log(
        log_path,
        "Symmetric SSP MSI-noise start | "
        f"noise={noise_cfg}, selection={cfg.msi_noise_selection_eval}, "
        f"selected={initial['stage2_psnr']:.4f}/{initial['stage2_sam']:.4f}, "
        f"{summary_text('clean', initial_results.get('clean'))}, "
        f"{summary_text('noisy', initial_results.get('noisy'))}, "
        f"trainable={count_parameters(model):.3f} M.",
    )

    csv_fields = [
        "epoch",
        "lr",
        "selection_eval",
        "clean_psnr",
        "clean_sam",
        "clean_reliability_ratio",
        "clean_noise_ratio",
        "noisy_psnr",
        "noisy_sam",
        "noisy_reliability_ratio",
        "noisy_noise_ratio",
        "noisy_high_share",
        *NOISE_STAT_NAMES,
        "selected_psnr",
        "selected_sam",
        "selected_zero_msi_psnr_drop",
        "selected_stage2_psnr_gain_over_anchor",
    ]
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
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, extra=initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, extra=initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, extra=initial_extra)

    for epoch in range(start_epoch, cfg.epochs):
        train_one_epoch_dual(
            model,
            noisy_loader(train_loader, injector, seed=cfg.seed + epoch),
            optimizer,
            hsi_degrader,
            coefficient_degrader,
            sam_loss,
            cfg,
            device,
        )
        val_results = evaluate_clean_and_noisy(
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
        val = select_validation(val_results, cfg)
        scheduler.step()

        clean_val = val_results.get("clean")
        noisy_val = val_results.get("noisy")
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"selected={cfg.msi_noise_selection_eval} "
            f"{val['stage2_psnr']:.4f}/{val['stage2_sam']:.4f} | "
            f"{summary_text('clean', clean_val)} | "
            f"{summary_text('noisy', noisy_val)} | "
            f"high_share={val.get('symmetric_high_share', 0.0):.3f}.",
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_eval": cfg.msi_noise_selection_eval,
            "selected_psnr": val["stage2_psnr"],
            "selected_sam": val["stage2_sam"],
            "selected_zero_msi_psnr_drop": val["zero_msi_psnr_drop"],
            "selected_stage2_psnr_gain_over_anchor": val[
                "stage2_psnr_gain_over_anchor"
            ],
        }
        if clean_val is not None:
            row.update(
                {
                    "clean_psnr": clean_val["stage2_psnr"],
                    "clean_sam": clean_val["stage2_sam"],
                    "clean_reliability_ratio": clean_val["reliability_ratio"],
                    "clean_noise_ratio": clean_val["noise_ratio"],
                }
            )
        if noisy_val is not None:
            row.update(
                {
                    "noisy_psnr": noisy_val["stage2_psnr"],
                    "noisy_sam": noisy_val["stage2_sam"],
                    "noisy_reliability_ratio": noisy_val["reliability_ratio"],
                    "noisy_noise_ratio": noisy_val["noise_ratio"],
                    "noisy_high_share": noisy_val["symmetric_high_share"],
                    **{name: noisy_val[name] for name in NOISE_STAT_NAMES},
                }
            )
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
            save_checkpoint(model, optimizer, epoch + 1, best_selection, best_path, extra=extra)
        if val["stage2_psnr"] > best_psnr:
            best_psnr = val["stage2_psnr"]
            save_checkpoint(model, optimizer, epoch + 1, best_psnr, best_psnr_path, extra=extra)
        if val["stage2_sam"] < best_sam:
            best_sam = val["stage2_sam"]
            save_checkpoint(model, optimizer, epoch + 1, best_sam, best_sam_path, extra=extra)
        save_checkpoint(model, optimizer, epoch + 1, best_selection, last_path, extra=extra)

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    final_results = evaluate_clean_and_noisy(
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
    final = select_validation(final_results, cfg)
    export_outputs(model, test_loader, os.path.join(output_dir, "clean"), device)
    if cfg.msi_noise_eval_mode in {"noisy", "both"}:
        export_outputs(
            model,
            noisy_loader(test_loader, injector, seed=cfg.seed + 200_000),
            os.path.join(output_dir, "noisy"),
            device,
        )
    with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Symmetric SSP MSI-noise complete | selected={cfg.msi_noise_selection_eval} | "
        f"PSNR={final['stage2_psnr']:.4f}, SAM={final['stage2_sam']:.4f} deg, "
        f"gain over initial={final['stage2_psnr'] - initial['stage2_psnr']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
