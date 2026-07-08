"""Stable training entry for enhanced uncertainty-guided local diffusion.

This entry uses the V2 training logic with the corrected one-time soft-mask
application. It is the recommended launcher for the enhanced diffusion run.
"""

from __future__ import annotations

import json
import os

import torch

from data_loader import build_loaders
from losses import SAMLoss
from models.stage3_uncertainty_guided_diffusion_v2_stable import (
    UncertaintyGuidedDualDomainDiffusionRefinerV2Stable,
)
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import evaluate
from train_stage3_uncertainty_guided_diffusion_v2 import (
    LOSS_NAMES,
    configure_phase,
    estimate_diffusion_scales,
    export_outputs,
    load_deterministic_initialization,
    oracle_mix_for_epoch,
    parse_v2_args,
    train_one_epoch,
)
from utils import (
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_log,
)


def build_stable_model(cfg, stage2, device: torch.device):
    return UncertaintyGuidedDualDomainDiffusionRefinerV2Stable(
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
        direct_x0_weight=cfg.v2_direct_x0_weight,
    ).to(device)


def main() -> None:
    cfg = parse_v2_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_stable_model(cfg, stage2, device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.deterministic_parameters()),
                "lr": cfg.v2_specialization_lr,
                "group_name": "deterministic",
            },
            {
                "params": list(model.diffusion_parameters()),
                "lr": 0.0,
                "group_name": "diffusion",
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage3_uncertainty_guided_diffusion_v2_stable",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage3_uncertainty_guided_diffusion_v2_stable",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage3_uncertainty_guided_diffusion_v2_stable",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "v2_stable_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "v2_stable_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "v2_stable_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "v2_stable_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    if cfg.resume:
        start_epoch, _ = load_checkpoint(
            model,
            cfg.resume,
            optimizer=optimizer,
            strict=True,
            map_location=str(device),
        )
        source_epoch = -1
    else:
        source_epoch = load_deterministic_initialization(
            model,
            cfg.stage3_initial_checkpoint,
            device,
        )
        start_epoch = 0
        scales = estimate_diffusion_scales(
            model,
            train_loader,
            device,
            oracle_weight=cfg.v2_scale_oracle_weight,
            max_batches=cfg.v2_scale_estimation_batches,
        )
        write_log(
            log_path,
            "Initial diffusion scales | coefficient min/median/max="
            f"({scales['coefficient'].min().item():.6e}, "
            f"{scales['coefficient'].median().item():.6e}, "
            f"{scales['coefficient'].max().item():.6e}), "
            f"orthogonal={scales['orthogonal'].item():.6e}.",
        )

    initial = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"Stable V2 start | source_epoch={source_epoch} | "
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
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"),
        [
            "epoch",
            "phase",
            "oracle_mix",
            "det_lr",
            "diff_lr",
            *metric_fields,
            *[f"train_{name}" for name in LOSS_NAMES],
        ],
    )

    best_selection = initial["selection"]
    best_psnr = initial["final_psnr"]
    best_sam = initial["final_sam"]
    initial_extra = {
        "stage": "stage3_uncertainty_guided_diffusion_v2_stable",
        "source_checkpoint": cfg.stage3_initial_checkpoint,
        "source_epoch": source_epoch,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, initial_extra)

    sam_loss = SAMLoss()
    scale_reestimated = start_epoch > cfg.v2_specialization_epochs
    for epoch in range(start_epoch, cfg.epochs):
        if epoch == cfg.v2_specialization_epochs and not scale_reestimated:
            scales = estimate_diffusion_scales(
                model,
                train_loader,
                device,
                oracle_weight=cfg.v2_scale_oracle_weight,
                max_batches=cfg.v2_scale_estimation_batches,
            )
            scale_reestimated = True
            write_log(
                log_path,
                "Post-specialization diffusion scales | coefficient min/median/max="
                f"({scales['coefficient'].min().item():.6e}, "
                f"{scales['coefficient'].median().item():.6e}, "
                f"{scales['coefficient'].max().item():.6e}), "
                f"orthogonal={scales['orthogonal'].item():.6e}.",
            )

        phase, run_diffusion, det_lr, diff_lr = configure_phase(
            model,
            optimizer,
            epoch,
            cfg,
        )
        oracle_mix = oracle_mix_for_epoch(epoch, cfg, phase)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            phase,
            run_diffusion,
            oracle_mix,
        )

        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={"phase": phase, "oracle_mix": oracle_mix},
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | phase={phase} | "
            f"det={val['deterministic_psnr']:.4f}/"
            f"{val['deterministic_sam']:.4f} | "
            f"final={val['final_psnr']:.4f}/"
            f"{val['final_sam']:.4f} | "
            f"diff=({val['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['diffusion_sam_gain_over_deterministic']:+.4f} deg) | "
            f"oracle_mix={oracle_mix:.3f}.",
        )
        row = {
            "epoch": epoch + 1,
            "phase": phase,
            "oracle_mix": oracle_mix,
            "det_lr": det_lr,
            "diff_lr": diff_lr,
            **{name: val[name] for name in metric_fields},
        }
        row.update({f"train_{name}": train_result[name] for name in LOSS_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "stage3_uncertainty_guided_diffusion_v2_stable",
            "phase": phase,
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
    export_outputs(model, test_loader, output_dir, cfg, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Stable V2 complete | PSNR={final['final_psnr']:.4f}, "
        f"SAM={final['final_sam']:.4f} deg, "
        f"diffusion gain=({final['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
