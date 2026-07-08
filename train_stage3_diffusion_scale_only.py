"""Controlled Stage-3 diffusion experiment: change residual scale only.

This launcher deliberately removes the confounded V2 changes that produced
negative diffusion gain. It keeps the successful deterministic dual-domain
reconstruction frozen and trains the original noise-prediction diffusion logic
with exactly one substantive change relative to the earlier positive baseline:
independent RMS normalization of the deterministic remaining residual.

Disabled on purpose:
- deterministic region-specialization fine-tuning;
- oracle/predicted mask curriculum;
- direct x0 prediction contribution;
- joint deterministic/diffusion fine-tuning.

Inference remains deterministic, zero-latent and fixed at 12 DDIM steps.
"""

from __future__ import annotations

import json
import os

import torch

from data_loader import build_loaders
from losses import SAMLoss
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import evaluate
from train_stage3_uncertainty_guided_diffusion_v2 import (
    LOSS_NAMES,
    estimate_diffusion_scales,
    export_outputs,
    load_deterministic_initialization,
    parse_v2_args,
    train_one_epoch,
)
from train_stage3_uncertainty_guided_diffusion_v2_stable import build_stable_model
from utils import (
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_log,
)


def main() -> None:
    cfg = parse_v2_args()

    # Controlled defaults. Command-line epoch/lr values can still be changed,
    # but the architecture and training policy below remain fixed.
    if cfg.v2_diffusion_epochs <= 0:
        cfg.v2_diffusion_epochs = 120
    cfg.v2_specialization_epochs = 0
    cfg.v2_joint_epochs = 0
    cfg.epochs = cfg.v2_diffusion_epochs
    cfg.v2_direct_x0_weight = 0.0
    cfg.v2_scale_oracle_weight = 0.0
    cfg.v2_lambda_direct_x0 = 0.0
    cfg.v2_lambda_hybrid_x0 = 0.2
    cfg.stage3_inference_steps = 12
    cfg.stage3_initial_noise = "zero"

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, stage2_epoch = build_stage2_model(cfg, info, device)
    model = build_stable_model(cfg, stage2, device)
    model.direct_x0_weight = 0.0

    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(False)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        list(model.diffusion_parameters()),
        lr=cfg.v2_diffusion_lr,
        weight_decay=cfg.weight_decay,
    )

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root,
        "stage3_diffusion_scale_only",
        cfg.dataset,
    )
    output_dir = os.path.join(
        cfg.output_root,
        "stage3_diffusion_scale_only",
        cfg.dataset,
    )
    log_dir = os.path.join(
        cfg.log_root,
        "stage3_diffusion_scale_only",
    )
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    best_path = os.path.join(checkpoint_dir, "scale_only_best.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "scale_only_best_psnr.pth")
    best_sam_path = os.path.join(checkpoint_dir, "scale_only_best_sam.pth")
    last_path = os.path.join(checkpoint_dir, "scale_only_last.pth")
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
            oracle_weight=0.0,
            max_batches=cfg.v2_scale_estimation_batches,
        )
        write_log(
            log_path,
            "Scale-only diffusion RMS | coefficient min/median/max="
            f"({scales['coefficient'].min().item():.6e}, "
            f"{scales['coefficient'].median().item():.6e}, "
            f"{scales['coefficient'].max().item():.6e}), "
            f"orthogonal={scales['orthogonal'].item():.6e}.",
        )

    initial = evaluate(model, test_loader, cfg, device)
    write_log(
        log_path,
        f"Scale-only start | source_epoch={source_epoch} | "
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
        "stage": "stage3_diffusion_scale_only",
        "source_checkpoint": cfg.stage3_initial_checkpoint,
        "source_epoch": source_epoch,
        "stage2_epoch": stage2_epoch,
        "validation": initial,
    }
    save_checkpoint(model, optimizer, start_epoch, best_selection, best_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_psnr, best_psnr_path, initial_extra)
    save_checkpoint(model, optimizer, start_epoch, best_sam, best_sam_path, initial_extra)

    sam_loss = SAMLoss()
    for epoch in range(start_epoch, cfg.epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            sam_loss,
            cfg,
            device,
            phase="diffusion",
            run_diffusion=True,
            oracle_mix=0.0,
        )

        if (epoch + 1) % max(cfg.eval_interval, 1) != 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_selection,
                last_path,
                extra={"phase": "diffusion", "oracle_mix": 0.0},
            )
            continue

        val = evaluate(model, test_loader, cfg, device)
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"det={val['deterministic_psnr']:.4f}/"
            f"{val['deterministic_sam']:.4f} | "
            f"final={val['final_psnr']:.4f}/"
            f"{val['final_sam']:.4f} | "
            f"diff=({val['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
            f"{val['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
        )
        row = {
            "epoch": epoch + 1,
            "phase": "diffusion",
            "oracle_mix": 0.0,
            "det_lr": 0.0,
            "diff_lr": cfg.v2_diffusion_lr,
            **{name: val[name] for name in metric_fields},
        }
        row.update({f"train_{name}": train_result[name] for name in LOSS_NAMES})
        csv_logger.write(row)

        extra = {
            "stage": "stage3_diffusion_scale_only",
            "phase": "diffusion",
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
        f"Scale-only complete | PSNR={final['final_psnr']:.4f}, "
        f"SAM={final['final_sam']:.4f} deg, "
        f"diffusion gain=({final['diffusion_psnr_gain_over_deterministic']:+.4f} dB, "
        f"{final['diffusion_sam_gain_over_deterministic']:+.4f} deg).",
    )


if __name__ == "__main__":
    main()
