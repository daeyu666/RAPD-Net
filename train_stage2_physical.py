"""Train Stage 2 of RAPD-Net: reliable abundance and gain correction.

The frozen Stage-1 model provides a normalized material composition map and a
scene endmember bank. Stage 2 uses HR-MSI frequency reliability to predict:

1. a bounded abundance-logit residual;
2. a positive per-pixel illumination/gain map.

The physical reconstruction is

    X_phy = g_hr * E * A_hr,
    A_hr >= 0, sum_k A_hr[k] = 1, g_hr > 0.

This is equivalent to a non-negative cone coefficient model while retaining an
interpretable normalized abundance map.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage1_unmixing import Stage1UnmixingNet
from models.stage2_physical_fusion import Stage2PhysicalFusionNet
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


class FixedSpatialDegradation(nn.Module):
    """Differentiable approximation of the current LR-HSI generation route."""

    def __init__(
        self,
        n_bands: int,
        kernel_size: int = 5,
        sigma: float = 2.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        coordinates = torch.arange(kernel_size, dtype=torch.float32)
        coordinates = coordinates - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        self.register_buffer(
            "kernel",
            kernel.repeat(n_bands, 1, 1, 1),
            persistent=False,
        )
        self.padding = kernel_size // 2
        self.n_bands = int(n_bands)

    def forward(
        self,
        hsi: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        padded = F.pad(
            hsi,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        blurred = F.conv2d(
            padded,
            self.kernel.to(dtype=hsi.dtype),
            groups=self.n_bands,
        )
        return F.interpolate(
            blurred,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )


def first_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def second_spectral_difference(x: torch.Tensor) -> torch.Tensor:
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def spatial_tv(x: torch.Tensor) -> torch.Tensor:
    return (
        (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    )


def build_stage1_from_checkpoint(
    checkpoint_path: str,
    expected_n_bands: int,
    device: torch.device,
) -> Tuple[Stage1UnmixingNet, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Stage-1 checkpoint does not exist: {checkpoint_path}"
        )
    try:
        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    model_state = state.get("model", state)
    extra = state.get("extra", {})
    n_bands = int(extra.get("n_bands", model_state["endmember_logits"].shape[1]))
    if n_bands != expected_n_bands:
        raise ValueError(
            f"Stage-1 checkpoint has {n_bands} bands, current data has "
            f"{expected_n_bands}."
        )

    num_endmembers = int(
        extra.get("num_endmembers", model_state["endmember_logits"].shape[0])
    )
    hidden_channels = int(
        extra.get(
            "hidden_channels",
            model_state["spectral_stem.0.weight"].shape[0],
        )
    )
    block_indices = [
        int(key.split(".")[1])
        for key in model_state
        if key.startswith("spatial_blocks.")
    ]
    num_blocks = int(extra.get("num_blocks", max(block_indices) + 1))

    model = Stage1UnmixingNet(
        n_bands=n_bands,
        num_endmembers=num_endmembers,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state


def build_spectral_response(info: dict) -> torch.Tensor:
    n_bands = int(info["n_bands"])
    n_msi = int(info["n_select_bands"])
    srf = info.get("srf_weights")
    if srf is not None:
        response = torch.from_numpy(np.asarray(srf, dtype=np.float32))
    else:
        indices = np.linspace(0, n_bands - 1, n_msi).round().astype(np.int64)
        response = torch.zeros(n_msi, n_bands, dtype=torch.float32)
        response[torch.arange(n_msi), torch.from_numpy(indices)] = 1.0

    if response.shape != (n_msi, n_bands):
        raise ValueError(
            f"Invalid spectral response {tuple(response.shape)}, expected "
            f"{(n_msi, n_bands)}"
        )
    return response


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
) -> Dict[str, torch.Tensor]:
    gt = batch["gt"]
    lr_hsi = batch["lr_hsi"]
    hr_msi = batch["hr_msi"]
    pred = outputs["physical_hsi"]

    losses: Dict[str, torch.Tensor] = {}
    losses["hsi_l1"] = F.l1_loss(pred, gt)
    losses["sam"] = sam_loss(pred, gt)
    losses["sgrad1"] = F.l1_loss(
        first_spectral_difference(pred),
        first_spectral_difference(gt),
    )
    losses["sgrad2"] = F.l1_loss(
        second_spectral_difference(pred),
        second_spectral_difference(gt),
    )

    degraded_hsi = degrader(pred, lr_hsi.shape[-2:])
    losses["lr_consistency"] = F.l1_loss(degraded_hsi, lr_hsi)
    losses["msi_consistency"] = F.l1_loss(
        outputs["projected_msi"], hr_msi
    )

    abundance_delta = (
        outputs["corrected_abundance"] - outputs["upsampled_abundance"]
    )
    losses["abundance_delta"] = abundance_delta.abs().mean()
    losses["abundance_tv"] = spatial_tv(abundance_delta)
    losses["gain_identity"] = outputs["log_gain_residual"].abs().mean()
    losses["gain_tv"] = spatial_tv(outputs["log_gain_residual"])

    losses["lf_alignment"] = outputs["low_frequency_alignment_loss"]
    losses["noise_minimization"] = outputs["noise_minimization_loss"]
    losses["partition"] = outputs["partition_reconstruction_loss"]

    base_l1 = F.l1_loss(outputs["base_hsi"], gt).detach()
    losses["base_l1"] = base_l1
    losses["improvement"] = F.relu(
        losses["hsi_l1"] - base_l1 + cfg.stage2_improvement_margin
    )

    if "zero_msi_hsi" in outputs:
        zero_l1 = F.l1_loss(outputs["zero_msi_hsi"], gt).detach()
        losses["zero_msi_l1"] = zero_l1
        losses["msi_usage"] = F.relu(
            losses["hsi_l1"] - zero_l1 + cfg.stage2_msi_usage_margin
        )
    else:
        losses["zero_msi_l1"] = losses["hsi_l1"].detach()
        losses["msi_usage"] = losses["hsi_l1"].new_zeros(())

    losses["total"] = (
        cfg.stage2_lambda_l1 * losses["hsi_l1"]
        + cfg.stage2_lambda_sam * losses["sam"]
        + cfg.stage2_lambda_sgrad1 * losses["sgrad1"]
        + cfg.stage2_lambda_sgrad2 * losses["sgrad2"]
        + cfg.stage2_lambda_lr_dc * losses["lr_consistency"]
        + cfg.stage2_lambda_msi_dc * losses["msi_consistency"]
        + cfg.stage2_lambda_abundance_delta * losses["abundance_delta"]
        + cfg.stage2_lambda_abundance_tv * losses["abundance_tv"]
        + cfg.stage2_lambda_gain_identity * losses["gain_identity"]
        + cfg.stage2_lambda_gain_tv * losses["gain_tv"]
        + cfg.stage2_lambda_lf_alignment * losses["lf_alignment"]
        + cfg.stage2_lambda_noise * losses["noise_minimization"]
        + cfg.stage2_lambda_partition * losses["partition"]
        + cfg.stage2_lambda_improvement * losses["improvement"]
        + cfg.stage2_lambda_msi_usage * losses["msi_usage"]
    )
    return losses


LOSS_NAMES = [
    "total",
    "hsi_l1",
    "sam",
    "sgrad1",
    "sgrad2",
    "lr_consistency",
    "msi_consistency",
    "abundance_delta",
    "abundance_tv",
    "gain_identity",
    "gain_tv",
    "lf_alignment",
    "noise_minimization",
    "partition",
    "improvement",
    "msi_usage",
    "base_l1",
    "zero_msi_l1",
]

MONITOR_NAMES = [
    "noise_ratio",
    "reliability_ratio",
    "edge_q10",
    "edge_q50",
    "edge_q90",
    "edge_q99",
    "tau_low_mean",
    "tau_low_min",
    "tau_low_max",
    "tau_high_mean",
    "tau_high_min",
    "tau_high_max",
    "freq_low",
    "freq_mid",
    "freq_high",
    "logit_residual_abs",
    "abundance_change_abs",
    "log_gain_abs",
    "gain_mean",
    "gain_std",
    "gain_min",
    "gain_max",
]


def create_meters(names: Iterable[str]) -> Dict[str, AverageMeter]:
    return {name: AverageMeter() for name in names}


def update_loss_meters(
    meters: Dict[str, AverageMeter],
    losses: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name in LOSS_NAMES:
        meters[name].update(float(losses[name].detach().item()), batch_size)


def monitor_values(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    edge_quantiles = outputs["edge_quantiles"].detach().float().mean(dim=0)
    tau_low = outputs["tau_low"].detach().float()
    tau_high = outputs["tau_high"].detach().float()
    frequency = outputs["frequency_activation_ratio"].detach().float()
    abundance_change = (
        outputs["corrected_abundance"] - outputs["upsampled_abundance"]
    )
    gain = outputs["gain_map"].detach().float()
    return {
        "noise_ratio": float(outputs["noise_ratio"].detach().item()),
        "reliability_ratio": float(outputs["reliability_ratio"].detach().item()),
        "edge_q10": float(edge_quantiles[0].item()),
        "edge_q50": float(edge_quantiles[1].item()),
        "edge_q90": float(edge_quantiles[2].item()),
        "edge_q99": float(edge_quantiles[3].item()),
        "tau_low_mean": float(tau_low.mean().item()),
        "tau_low_min": float(tau_low.min().item()),
        "tau_low_max": float(tau_low.max().item()),
        "tau_high_mean": float(tau_high.mean().item()),
        "tau_high_min": float(tau_high.min().item()),
        "tau_high_max": float(tau_high.max().item()),
        "freq_low": float(frequency[0].item()),
        "freq_mid": float(frequency[1].item()),
        "freq_high": float(frequency[2].item()),
        "logit_residual_abs": float(
            outputs["abundance_logit_residual"].detach().abs().mean().item()
        ),
        "abundance_change_abs": float(
            abundance_change.detach().abs().mean().item()
        ),
        "log_gain_abs": float(
            outputs["log_gain_residual"].detach().abs().mean().item()
        ),
        "gain_mean": float(gain.mean().item()),
        "gain_std": float(gain.std(unbiased=False).item()),
        "gain_min": float(gain.min().item()),
        "gain_max": float(gain.max().item()),
    }


def update_monitor_meters(
    meters: Dict[str, AverageMeter],
    outputs: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in monitor_values(outputs).items():
        meters[name].update(value, batch_size)


def train_one_epoch(
    model: Stage2PhysicalFusionNet,
    loader,
    optimizer: torch.optim.Optimizer,
    degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_meters = create_meters(LOSS_NAMES)
    monitor_meters = create_meters(MONITOR_NAMES)
    use_zero_msi = cfg.stage2_lambda_msi_usage > 0

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=use_zero_msi,
        )
        losses = compute_losses(outputs, batch, degrader, sam_loss, cfg)
        losses["total"].backward()
        if cfg.stage2_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                cfg.stage2_grad_clip,
            )
        optimizer.step()

        batch_size = batch["lr_hsi"].size(0)
        update_loss_meters(loss_meters, losses, batch_size)
        update_monitor_meters(monitor_meters, outputs, batch_size)

    return {
        **{name: meter.avg for name, meter in loss_meters.items()},
        **{name: meter.avg for name, meter in monitor_meters.items()},
    }


@torch.no_grad()
def evaluate(
    model: Stage2PhysicalFusionNet,
    loader,
    degrader: FixedSpatialDegradation,
    sam_loss: SAMLoss,
    cfg,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meters = create_meters(LOSS_NAMES)
    monitor_meters = create_meters(MONITOR_NAMES)
    physical_metrics = MetricAverager()
    base_metrics = MetricAverager()
    zero_metrics = MetricAverager()

    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["lr_hsi"],
            batch["hr_msi"],
            compute_zero_msi=True,
        )
        losses = compute_losses(outputs, batch, degrader, sam_loss, cfg)
        batch_size = batch["lr_hsi"].size(0)
        update_loss_meters(loss_meters, losses, batch_size)
        update_monitor_meters(monitor_meters, outputs, batch_size)

        physical_metrics.update(
            calc_metrics(outputs["physical_hsi"], batch["gt"], cfg.scale_ratio)
        )
        base_metrics.update(
            calc_metrics(outputs["base_hsi"], batch["gt"], cfg.scale_ratio)
        )
        zero_metrics.update(
            calc_metrics(outputs["zero_msi_hsi"], batch["gt"], cfg.scale_ratio)
        )

    result = {
        **{name: meter.avg for name, meter in loss_meters.items()},
        **{name: meter.avg for name, meter in monitor_meters.items()},
    }
    for prefix, metrics in (
        ("physical", physical_metrics.average()),
        ("base", base_metrics.average()),
        ("zero", zero_metrics.average()),
    ):
        for name, value in metrics.items():
            result[f"{prefix}_{name.lower()}"] = value

    result["psnr_gain_over_base"] = (
        result["physical_psnr"] - result["base_psnr"]
    )
    result["sam_gain_over_base"] = (
        result["base_sam"] - result["physical_sam"]
    )
    result["zero_msi_psnr_drop"] = (
        result["physical_psnr"] - result["zero_psnr"]
    )
    result["zero_msi_sam_drop"] = (
        result["zero_sam"] - result["physical_sam"]
    )
    result["selection"] = (
        result["hsi_l1"]
        + cfg.stage2_selection_sam_weight * result["sam"]
        + cfg.stage2_selection_sgrad1_weight * result["sgrad1"]
        + cfg.stage2_selection_sgrad2_weight * result["sgrad2"]
    )
    return result


@torch.no_grad()
def export_stage2_artifacts(
    model: Stage2PhysicalFusionNet,
    loader,
    cfg,
    output_dir: str,
    device: torch.device,
) -> None:
    ensure_dir(output_dir)
    model.eval()
    batch = move_to_device(next(iter(loader)), device)
    outputs = model(
        batch["lr_hsi"],
        batch["hr_msi"],
        compute_zero_msi=True,
    )

    arrays = {
        "lr_hsi": batch["lr_hsi"].detach().cpu().numpy(),
        "hr_msi": batch["hr_msi"].detach().cpu().numpy(),
        "gt": batch["gt"].detach().cpu().numpy(),
        "endmembers": outputs["endmembers"].detach().cpu().numpy(),
        "lr_abundance": outputs["lr_abundance"].detach().cpu().numpy(),
        "upsampled_abundance": outputs["upsampled_abundance"].detach().cpu().numpy(),
        "corrected_abundance": outputs["corrected_abundance"].detach().cpu().numpy(),
        "abundance_logit_residual": outputs[
            "abundance_logit_residual"
        ].detach().cpu().numpy(),
        "log_gain_residual": outputs["log_gain_residual"].detach().cpu().numpy(),
        "gain_map": outputs["gain_map"].detach().cpu().numpy(),
        "cone_coefficients": outputs["cone_coefficients"].detach().cpu().numpy(),
        "base_hsi": outputs["base_hsi"].detach().cpu().numpy(),
        "physical_hsi": outputs["physical_hsi"].detach().cpu().numpy(),
        "zero_msi_hsi": outputs["zero_msi_hsi"].detach().cpu().numpy(),
        "zero_msi_gain_map": outputs["zero_msi_gain_map"].detach().cpu().numpy(),
        "base_msi": outputs["base_msi"].detach().cpu().numpy(),
        "projected_msi": outputs["projected_msi"].detach().cpu().numpy(),
        "msi_residual": outputs["msi_residual"].detach().cpu().numpy(),
        "reliability_map": outputs["reliability_map"].detach().cpu().numpy(),
        "noise_mask": outputs["noise_mask"].detach().cpu().numpy(),
        "edge_magnitude": outputs["edge_magnitude"].detach().cpu().numpy(),
        "edge_score": outputs["edge_score"].detach().cpu().numpy(),
        "mid_feature_mean": outputs["mid_feature"].detach().mean(dim=1).cpu().numpy(),
        "reliable_high_mean": outputs[
            "reliable_high_feature"
        ].detach().mean(dim=1).cpu().numpy(),
        "tau_low": outputs["tau_low"].detach().cpu().numpy(),
        "tau_high": outputs["tau_high"].detach().cpu().numpy(),
        "spectral_response": model.spectral_response.detach().cpu().numpy(),
    }
    np.savez_compressed(
        os.path.join(output_dir, "stage2_test_outputs.npz"),
        **arrays,
    )

    summary = {
        "dataset": cfg.dataset,
        "stage1_checkpoint": cfg.stage1_checkpoint,
        "max_logit_residual": cfg.stage2_max_logit_residual,
        "max_log_gain": cfg.stage2_max_log_gain,
        "edge_threshold_mode": cfg.stage2_edge_threshold_mode,
        "edge_mask_threshold": cfg.stage2_edge_mask_threshold,
        "noise_ratio": float(outputs["noise_ratio"].item()),
        "reliability_ratio": float(outputs["reliability_ratio"].item()),
        "gain_mean": float(outputs["gain_map"].mean().item()),
        "gain_std": float(outputs["gain_map"].std(unbiased=False).item()),
        "gain_min": float(outputs["gain_map"].min().item()),
        "gain_max": float(outputs["gain_map"].max().item()),
        "tau_low_mean": float(outputs["tau_low"].mean().item()),
        "tau_high_mean": float(outputs["tau_high"].mean().item()),
    }
    with open(
        os.path.join(output_dir, "stage2_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def _has_option(arguments: List[str], option: str) -> bool:
    return any(item == option or item.startswith(option + "=") for item in arguments)


def parse_stage2_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="./checkpoints/stage1_unmix/PaviaU/unmixing_best.pth",
    )
    parser.add_argument("--stage2_feature_channels", type=int, default=64)
    parser.add_argument("--stage2_encoder_blocks", type=int, default=3)
    parser.add_argument("--stage2_fusion_channels", type=int, default=96)
    parser.add_argument("--stage2_fusion_blocks", type=int, default=4)
    parser.add_argument("--stage2_max_logit_residual", type=float, default=1.0)
    parser.add_argument("--stage2_max_log_gain", type=float, default=0.7)

    parser.add_argument("--stage2_num_frequency_bands", type=int, default=20)
    parser.add_argument("--stage2_init_low_boundary", type=float, default=5.0)
    parser.add_argument("--stage2_init_high_boundary", type=float, default=18.0)
    parser.add_argument("--stage2_boundary_temperature", type=float, default=0.5)
    parser.add_argument("--stage2_soft_frequency_partition", action="store_true")

    parser.add_argument(
        "--stage2_edge_threshold_mode",
        type=str,
        default="relative",
        choices=["fixed", "relative", "quantile"],
    )
    parser.add_argument("--stage2_edge_mask_threshold", type=float, default=0.1)
    parser.add_argument("--stage2_edge_reference_quantile", type=float, default=0.9)
    parser.add_argument("--stage2_noise_quantile", type=float, default=0.2)

    parser.add_argument("--stage2_boundary_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--stage2_grad_clip", type=float, default=1.0)

    parser.add_argument("--stage2_lambda_l1", type=float, default=1.0)
    parser.add_argument("--stage2_lambda_sam", type=float, default=0.3)
    parser.add_argument("--stage2_lambda_sgrad1", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_sgrad2", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_lr_dc", type=float, default=0.2)
    parser.add_argument("--stage2_lambda_msi_dc", type=float, default=0.2)
    parser.add_argument("--stage2_lambda_abundance_delta", type=float, default=0.001)
    parser.add_argument("--stage2_lambda_abundance_tv", type=float, default=0.001)
    parser.add_argument("--stage2_lambda_gain_identity", type=float, default=0.001)
    parser.add_argument("--stage2_lambda_gain_tv", type=float, default=0.005)
    parser.add_argument("--stage2_lambda_lf_alignment", type=float, default=0.05)
    parser.add_argument("--stage2_lambda_noise", type=float, default=0.01)
    parser.add_argument("--stage2_lambda_partition", type=float, default=0.01)
    parser.add_argument("--stage2_lambda_improvement", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_msi_usage", type=float, default=0.05)
    parser.add_argument("--stage2_improvement_margin", type=float, default=1e-4)
    parser.add_argument("--stage2_msi_usage_margin", type=float, default=1e-4)

    parser.add_argument("--stage2_selection_sam_weight", type=float, default=0.5)
    parser.add_argument("--stage2_selection_sgrad1_weight", type=float, default=0.1)
    parser.add_argument("--stage2_selection_sgrad2_weight", type=float, default=0.05)

    stage_args, remaining = parser.parse_known_args()
    cfg = parse_args(remaining)
    for key, value in vars(stage_args).items():
        setattr(cfg, key, value)

    if not _has_option(remaining, "--msi_mode"):
        cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"):
        cfg.srf_band_set = "wv2_visible6"

    default_stage1 = "./checkpoints/stage1_unmix/PaviaU/unmixing_best.pth"
    if cfg.stage1_checkpoint == default_stage1 and cfg.dataset != "PaviaU":
        cfg.stage1_checkpoint = os.path.join(
            cfg.checkpoint_root,
            "stage1_unmix",
            cfg.dataset,
            "unmixing_best.pth",
        )
    return cfg


def checkpoint_extra(cfg, info: dict, stage1_state: dict, result: dict) -> dict:
    return {
        "stage": "physical_cone",
        "dataset": cfg.dataset,
        "n_bands": int(info["n_bands"]),
        "n_msi_bands": int(info["n_select_bands"]),
        "msi_mode": info["msi_mode"],
        "srf_band_names": info.get("srf_band_names"),
        "stage1_checkpoint": cfg.stage1_checkpoint,
        "stage1_epoch": int(stage1_state.get("epoch", -1)),
        "feature_channels": cfg.stage2_feature_channels,
        "encoder_blocks": cfg.stage2_encoder_blocks,
        "fusion_channels": cfg.stage2_fusion_channels,
        "fusion_blocks": cfg.stage2_fusion_blocks,
        "max_logit_residual": cfg.stage2_max_logit_residual,
        "max_log_gain": cfg.stage2_max_log_gain,
        "edge_threshold_mode": cfg.stage2_edge_threshold_mode,
        "edge_mask_threshold": cfg.stage2_edge_mask_threshold,
        "edge_reference_quantile": cfg.stage2_edge_reference_quantile,
        "noise_quantile": cfg.stage2_noise_quantile,
        "validation": result,
    }


def main() -> None:
    cfg = parse_stage2_args()
    cfg.stage = "physical"
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)

    stage1_model, stage1_state = build_stage1_from_checkpoint(
        cfg.stage1_checkpoint,
        expected_n_bands=info["n_bands"],
        device=device,
    )
    spectral_response = build_spectral_response(info).to(device)
    model = Stage2PhysicalFusionNet(
        stage1_model=stage1_model,
        spectral_response=spectral_response,
        feature_channels=cfg.stage2_feature_channels,
        encoder_blocks=cfg.stage2_encoder_blocks,
        fusion_channels=cfg.stage2_fusion_channels,
        fusion_blocks=cfg.stage2_fusion_blocks,
        max_logit_residual=cfg.stage2_max_logit_residual,
        max_log_gain=cfg.stage2_max_log_gain,
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
    degrader = FixedSpatialDegradation(info["n_bands"]).to(device)

    checkpoint_dir = os.path.join(
        cfg.checkpoint_root, "stage2_physical", cfg.dataset
    )
    output_dir = os.path.join(cfg.output_root, "stage2_physical", cfg.dataset)
    log_dir = os.path.join(cfg.log_root, "stage2_physical")
    ensure_dir(checkpoint_dir)
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    best_path = os.path.join(checkpoint_dir, "physical_best.pth")
    best_sam_path = os.path.join(checkpoint_dir, "physical_best_sam.pth")
    best_psnr_path = os.path.join(checkpoint_dir, "physical_best_psnr.pth")
    last_path = os.path.join(checkpoint_dir, "physical_last.pth")
    log_path = os.path.join(log_dir, f"{cfg.dataset}.log")

    csv_fields = [
        "epoch",
        "lr",
        "boundary_lr",
        "train_total",
        "train_l1",
        "train_sam_deg",
        "val_l1",
        "val_sam_deg",
        "val_psnr",
        "base_psnr",
        "base_sam",
        "psnr_gain_over_base",
        "sam_gain_over_base",
        "zero_psnr",
        "zero_sam",
        "zero_msi_psnr_drop",
        "zero_msi_sam_drop",
        "selection",
        *MONITOR_NAMES,
        "lr_consistency",
        "msi_consistency",
        "gain_identity",
        "gain_tv",
        "lf_alignment",
        "noise_minimization",
        "msi_usage",
    ]
    csv_logger = CSVLogger(
        os.path.join(log_dir, f"{cfg.dataset}.csv"), csv_fields
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
        optimizer.param_groups[0]["lr"] = cfg.lr
        optimizer.param_groups[1]["lr"] = boundary_lr
        write_log(
            log_path,
            f"Resumed from {cfg.resume} at epoch {start_epoch}; "
            f"lr={cfg.lr:.3e}, boundary_lr={boundary_lr:.3e}.",
        )

    write_log(
        log_path,
        f"Stage 2 cone start | dataset={cfg.dataset}, HSI={info['n_bands']}, "
        f"MSI={info['n_select_bands']}, mode={info['msi_mode']}, "
        f"Stage1={cfg.stage1_checkpoint} (epoch {stage1_state.get('epoch', -1)}), "
        f"max_logit={cfg.stage2_max_logit_residual}, "
        f"max_log_gain={cfg.stage2_max_log_gain}, "
        f"trainable={count_parameters(model):.3f} M.",
    )

    for epoch in range(start_epoch, cfg.epochs):
        train_result = train_one_epoch(
            model, train_loader, optimizer, degrader, sam_loss, cfg, device
        )
        val_result = evaluate(
            model, test_loader, degrader, sam_loss, cfg, device
        )
        scheduler.step()

        train_sam_deg = train_result["sam"] * 180.0 / math.pi
        val_sam_deg = val_result["sam"] * 180.0 / math.pi
        write_log(
            log_path,
            f"Epoch {epoch + 1:03d}/{cfg.epochs:03d} | "
            f"train L1={train_result['hsi_l1']:.6f}, "
            f"SAM={train_sam_deg:.4f} deg | "
            f"val PSNR={val_result['physical_psnr']:.4f}, "
            f"SAM={val_result['physical_sam']:.4f} deg, "
            f"L1={val_result['hsi_l1']:.6f} | "
            f"base gain=({val_result['psnr_gain_over_base']:+.4f} dB, "
            f"{val_result['sam_gain_over_base']:+.4f} deg) | "
            f"Zero-MSI drop=({val_result['zero_msi_psnr_drop']:+.4f} dB, "
            f"{val_result['zero_msi_sam_drop']:+.4f} deg) | "
            f"noise={val_result['noise_ratio']:.4f}, "
            f"gain={val_result['gain_mean']:.3f}±{val_result['gain_std']:.3f}, "
            f"tau=({val_result['tau_low_mean']:.3f}, "
            f"{val_result['tau_high_mean']:.3f}).",
        )

        if val_result["noise_ratio"] < 0.005:
            write_log(
                log_path,
                "WARNING: NSP reliability map is nearly all 1.",
            )
        elif val_result["noise_ratio"] > 0.95:
            write_log(
                log_path,
                "WARNING: NSP removes almost all high frequency.",
            )
        if abs(val_result["zero_msi_psnr_drop"]) < 0.01:
            write_log(
                log_path,
                "WARNING: Zero-MSI PSNR drop < 0.01 dB; MSI usage is weak.",
            )
        if val_result["gain_min"] < math.exp(-0.95 * cfg.stage2_max_log_gain):
            write_log(log_path, "WARNING: gain approaches its lower bound.")
        if val_result["gain_max"] > math.exp(0.95 * cfg.stage2_max_log_gain):
            write_log(log_path, "WARNING: gain approaches its upper bound.")

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "boundary_lr": optimizer.param_groups[1]["lr"],
            "train_total": train_result["total"],
            "train_l1": train_result["hsi_l1"],
            "train_sam_deg": train_sam_deg,
            "val_l1": val_result["hsi_l1"],
            "val_sam_deg": val_sam_deg,
            "val_psnr": val_result["physical_psnr"],
            "base_psnr": val_result["base_psnr"],
            "base_sam": val_result["base_sam"],
            "psnr_gain_over_base": val_result["psnr_gain_over_base"],
            "sam_gain_over_base": val_result["sam_gain_over_base"],
            "zero_psnr": val_result["zero_psnr"],
            "zero_sam": val_result["zero_sam"],
            "zero_msi_psnr_drop": val_result["zero_msi_psnr_drop"],
            "zero_msi_sam_drop": val_result["zero_msi_sam_drop"],
            "selection": val_result["selection"],
            "lr_consistency": val_result["lr_consistency"],
            "msi_consistency": val_result["msi_consistency"],
            "gain_identity": val_result["gain_identity"],
            "gain_tv": val_result["gain_tv"],
            "lf_alignment": val_result["lf_alignment"],
            "noise_minimization": val_result["noise_minimization"],
            "msi_usage": val_result["msi_usage"],
        }
        row.update({name: val_result[name] for name in MONITOR_NAMES})
        csv_logger.write(row)

        extra = checkpoint_extra(cfg, info, stage1_state, val_result)
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
            write_log(log_path, f"Saved Stage-2 best: {best_path}")

        if val_result["physical_sam"] < best_sam:
            best_sam = val_result["physical_sam"]
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_sam,
                best_sam_path,
                extra=extra,
            )

        if val_result["physical_psnr"] > best_psnr:
            best_psnr = val_result["physical_psnr"]
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

    load_checkpoint(
        model,
        best_path,
        optimizer=None,
        map_location=str(device),
        load_optimizer=False,
    )
    final_result = evaluate(
        model, test_loader, degrader, sam_loss, cfg, device
    )
    export_stage2_artifacts(model, test_loader, cfg, output_dir, device)
    with open(
        os.path.join(output_dir, "final_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_result, file, indent=2, ensure_ascii=False)
    write_log(
        log_path,
        f"Stage 2 complete | PSNR={final_result['physical_psnr']:.4f}, "
        f"SAM={final_result['physical_sam']:.4f} deg, "
        f"base gain={final_result['psnr_gain_over_base']:+.4f} dB, "
        f"Zero-MSI drop={final_result['zero_msi_psnr_drop']:+.4f} dB.",
    )


if __name__ == "__main__":
    main()
