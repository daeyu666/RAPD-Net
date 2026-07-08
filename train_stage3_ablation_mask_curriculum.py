"""Single-variable experiment for the Stage-3 diffusion training mask.

Controlled quantities:
- the fitted deterministic dual-domain heads are loaded and frozen;
- the original Stage-3 residual scales are loaded unchanged;
- the original noise-prediction diffusion architecture and losses are used;
- DDIM inference is fixed to the empirically best 12 steps;
- the diffusion branches are zero-initialized for every run.

Only the diffusion-training mask changes:
- predicted: original uncertainty mask baseline;
- oracle: residual-error oracle mask upper-bound diagnostic;
- curriculum: oracle-to-predicted linear curriculum.

Inference always uses the predicted uncertainty mask, including oracle mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage3_ablation_mask_curriculum import MaskCurriculumAblationRefiner
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import (
    LOSS_NAMES,
    compute_losses,
    evaluate,
    parse_uncertainty_guided_args,
)
from train_stage3_uncertainty_guided_diffusion_v2 import (
    load_deterministic_initialization,
)
from utils import (
    AverageMeter,
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    set_seed,
    write_log,
)


def parse_ablation_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage3_initial_checkpoint",
        type=str,
        default=(
            "./checkpoints_stage3_extended/"
            "stage3_uncertainty_guided_diffusion/PaviaU/"
            "uncertainty_guided_best_psnr.pth"
        ),
    )
    parser.add_argument(
        "--ablation_mask_mode",
        type=str,
        default="curriculum",
        choices=["predicted", "oracle", "curriculum"],
    )
    parser.add_argument("--ablation_diffusion_epochs", type=int, default=100)
    parser.add_argument("--ablation_diffusion_lr", type=float, default=1e-4)
    parser.add_argument(
        "--ablation_curriculum_fraction",
        type=float,
        default=0.6,
    )
    parser.add_argument("--ablation_grad_clip", type=float, default=1.0)

    specific, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        cfg = parse_uncertainty_guided_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(specific).items():
        setattr(cfg, key, value)

    if cfg.ablation_diffusion_epochs <= 0:
        raise ValueError("ablation_diffusion_epochs must be positive")
    if cfg.ablation_diffusion_lr <= 0:
        raise ValueError("ablation_diffusion_lr must be positive")
    if not 0.0 < cfg.ablation_curriculum_fraction <= 1.0:
        raise ValueError("ablation_curriculum_fraction must be in (0, 1]")
    cfg.epochs = cfg.ablation_diffusion_epochs
    cfg.stage3_inference_steps = 12
    cfg.stage3_initial_noise = "zero"
    return cfg


def build_model(cfg, stage2, device: torch.device):
    return MaskCurriculumAblationRefiner(
        stage2_model=stage2,
        deterministic_hidden_channels=cfg.stage3_det_hidden_channels,
        deterministic_blocks=cfg.stage3_det_blocks,
        diffusion_hidden_channels=cfg.stage3_diff_hidden_channels,
        diffusion_blocks=cfg.stage3_diff_blocks,
        time_channels=cfg.stage3_time_channels,
        diffusion_timesteps=cfg.stage3_diffusion_timesteps,
        beta_start=cfg.stage3_beta_start,
        beta_end=cfg.stage3_beta_end,
        max_normalized_residual=cfg.stage3_max_normalized_residual,
        clean_clip=cfg.stage3_clean_clip,
        log_variance_min=cfg.stage3_log_variance_min,
        log_variance_max=cfg.stage3_log_variance_max,
        basis_mask_threshold=cfg.stage3_basis_mask_threshold,
        orthogonal_mask_threshold=cfg.stage3_orthogonal_mask_threshold,
        mask_temperature=cfg.stage3_mask_temperature,
        mask_spread_floor=cfg.stage3_mask_spread_floor,
        msi_residual_gain=cfg.stage3_msi_residual_gain,
        residual_scale_floor=cfg.stage3_residual_scale_floor,
    ).to(device)


def oracle_mix_for_epoch(epoch: int, cfg) -> float:
    mode = cfg.ablation_mask_mode
    if mode == "predicted":
        return 0.0
    if mode == "oracle":
        return 1.0
    curriculum_epochs = max(
        int(round(
            cfg.ablation_diffusion_epochs
            * cfg.ablation_curriculum_fraction
        )),
        1,
    )
    if epoch >= curriculum_epochs:
        return 0.0
    if curriculum_epochs == 1:
        return 1.0
    return max(1.0 - epoch / float(curriculum_epochs - 1), 0.0)


def train_one_epoch(
    model,
    loader,
    optimizer,
    sam_loss,
    cfg,
    device,
    oracle_mix: float,
) -> Dict[str, float]:
    model.train()
    meters = {name: AverageMeter() for name in LOSS_NAMES}
    extra_names = (
        "train_basis_mask_mean",
        "train_orthogonal_mask_mean",
        "predicted_basis_mask_mean",
        "predicted_orthogonal_mask_mean",
        "basis_mask_gap",
        "orthogonal_mask_gap",
    )
    extras = {name: AverageMeter() for name in extra_names}

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_forward(
            batch["lr_hsi"],
            batch["hr_msi"],
            batch["gt"],
            run_diffusion=True,
            oracle_mask_mix=oracle_mix,
        )
        losses = compute_losses(
            model,
            outputs,
            batch["gt"],
            sam_loss,
            cfg,
            run_diffusion=True,
            phase="diffusion",
        )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(
                f"Non-finite mask-ablation loss at oracle_mix={oracle_mix:.4f}"
            )
        losses["total"].backward()
        if cfg.ablation_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.diffusion_parameters()),
                cfg.ablation_grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite mask-ablation gradient")
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        for name in LOSS_NAMES:
            meters[name].update(float(losses[name].detach().item()), batch_size)
        values = {
            "train_basis_mask_mean": outputs[
                "basis_mask_for_diffusion"
            ].mean(),
            "train_orthogonal_mask_mean": outputs[
                "orthogonal_mask_for_diffusion"
            ].mean(),
            "predicted_basis_mask_mean": outputs[
                "predicted_basis_mask"
            ].mean(),
            "predicted_orthogonal_mask_mean": outputs[
                "predicted_orthogonal_mask"
            ].mean(),
            "basis_mask_gap": (
                outputs["basis_mask_for_diffusion"]
                - outputs["predicted_basis_mask"]
            ).abs().mean(),
            "orthogonal_mask_gap": (
                outputs["orthogonal_mask_for_diffusion"]
                - outputs["predicted_orthogonal_mask"]
            ).abs().mean(),
        }
        for name, value in values.items():
            extras[name].update(float(value.detach().item()), batch_size)

    result = {name: meter.avg for name, meter in meters.items()}
    result.update({name: meter.avg for name, meter in extras.items()})
    return result


def main() -> None:
    cfg = parse_ablation_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_model(cfg, stage2, device)

    source_epoch = load_deterministic_initialization(
        model,
        cfg.stage3_initial_checkpoint,
        device,
    )
    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(False)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.diffusion_parameters(),
        lr=cfg.ablation_diffusion_lr,
        weight_decay=cfg.weight_decay,
    )

    experiment_name = f"stage3_ablation_mask_{cfg.ablation_mask_mode}"
    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        experiment_name,
        cfg.dataset,
    )
    output_dir = os.path.join(cfg.output_root, experiment_name, cfg.dataset)
    log_dir = os.path.join(cfg.log_root, experiment_name)
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=True,
            map_location=str(device),
        )
    else:
        start_epoch = 0

    initial = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"Mask ablation start | mode={cfg.ablation_mask_mode} | "
        f"source_epoch={source_epoch} | scales=original-checkpoint | "
        f"det={initial['deterministic_psnr']:.4f}/"
        f"{initial['deterministic_sam']:.4f} | "
        f"final={initial['final_psnr']:.4f}/"
        f"{initial['final_sam']:.4f} | DDIM=12.",
    )

    metric_fields = [
        "stage2_psnr",
        "stage2_sam",
        "deterministic_psnr",
        "deterministic_sam",
        "final_psnr",
        "final_sam",
        "psnr_gain_over_stage2",
        "sam_gain_over_stage2",
        "diffusion_psnr_gain_over_deterministic",
        "diffusion_sam_gain_over_deterministic",
        "basis_mask_mean",
        "orthogonal_mask_mean",
        "basis_uncertainty_error_correlation",
        "orthogonal_uncertainty_error_correlation",
    ]
    extra_train_fields = [
        "train_basis_mask_mean",
        "train_orthogonal_mask_mean",
        "predicted_basis_mask_mean",
        "predicted_orthogonal_mask_mean",
        "basis_mask_gap",
        "orthogonal_mask_gap",
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        [
            "epoch",
            "mask_mode",
            "oracle_mix",
            *metric_fields,
            *[f"train_{name}" for name in LOSS_NAMES],
            *extra_train_fields,
        ],
    )

    best_selection = initial["selection"]
    best_psnr = initial["final_psnr"]
    best_sam = initial["final_sam"]
    initial_extra = {
        "stage": experiment_name,
        "mask_mode": cfg.ablation_mask_mode,
        "source_checkpoint": cfg.stage3_initial_checkpoint,
        "source_epoch": source_epoch,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, initial_extra)

    sam_loss = SAMLoss()
    for epoch in range(start_epoch, cfg.ablation_diffusion_epochs):
        oracle_mix = oracle_mix_for_epoch(epoch, cfg)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            oracle_mix,
        )
        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={
                    "mask_mode": cfg.ablation_mask_mode,
                    "oracle_mix": oracle_mix,
                },
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.ablation_diffusion_epochs:03d} | "
            f"mode={cfg.ablation_mask_mode} | oracle_mix={oracle_mix:.3f} | "
            f"det={val['deterministic_psnr']:.4f}/"
            f"{val['deterministic_sam']:.4f} | "
            f"final={val['final_psnr']:.4f}/"
            f"{val['final_sam']:.4f} | "
            f"diff=({val['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['diffusion_sam_gain_over_deterministic']:+.4f} deg) | "
            f"mask_gap=({train_result['basis_mask_gap']:.4f}, "
            f"{train_result['orthogonal_mask_gap']:.4f}).",
        )
        row = {
            "epoch": epoch + 1,
            "mask_mode": cfg.ablation_mask_mode,
            "oracle_mix": oracle_mix,
            **{name: val[name] for name in metric_fields},
            **{
                f"train_{name}": train_result[name]
                for name in LOSS_NAMES
            },
            **{
                name: train_result[name]
                for name in extra_train_fields
            },
        }
        csv_logger.write(row)

        extra = {
            "stage": experiment_name,
            "mask_mode": cfg.ablation_mask_mode,
            "oracle_mix": oracle_mix,
            "source_checkpoint": cfg.stage3_initial_checkpoint,
            "validation": val,
            "train": train_result,
        }
        if val["selection"] < best_selection:
            best_selection = val["selection"]
            save_checkpoint(model, optimizer, epoch + 1, best_selection, best_path, extra)
        if val["final_psnr"] > best_psnr:
            best_psnr = val["final_psnr"]
            save_checkpoint(model, optimizer, epoch + 1, best_psnr, best_psnr_path, extra)
        if val["final_sam"] < best_sam:
            best_sam = val["final_sam"]
            save_checkpoint(model, optimizer, epoch + 1, best_sam, best_sam_path, extra)
        save_checkpoint(model, optimizer, epoch + 1, best_selection, last_path, extra)

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )
    final = evaluate(model, test_loader, cfg, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Mask ablation complete | mode={cfg.ablation_mask_mode} | "
        f"PSNR={final['final_psnr']:.4f}, SAM={final['final_sam']:.4f}, "
        f"diffusion gain=({final['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
