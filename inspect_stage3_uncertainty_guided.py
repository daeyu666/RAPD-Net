"""Finite-value and identity smoke test for uncertainty-guided Stage 3.

The test performs no optimizer step. It verifies:
- residual scales and input tensors are finite;
- the zero-initialized deterministic and local-diffusion heads preserve Stage 2;
- deterministic-phase forward/backward is finite;
- diffusion-phase forward/backward is finite.
"""

from __future__ import annotations

from typing import Dict

import torch

from data_loader import build_loaders
from losses import SAMLoss
from metrics import calc_metrics
from train_stage3_dual_domain_diffusion import (
    build_stage2_model,
    estimate_residual_scales,
)
from train_stage3_uncertainty_guided_diffusion import (
    LOSS_NAMES,
    build_model,
    compute_losses,
    parse_uncertainty_guided_args,
)
from utils import get_device, move_to_device, set_seed


def finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        detached = tensor.detach().float()
        ratio = float(torch.isfinite(detached).float().mean().item())
        raise FloatingPointError(f"{name} is non-finite; finite ratio={ratio:.6f}")


def check_gradients(model, label: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            finite(f"{label}.gradient.{name}", parameter.grad)


def run_backward(
    model,
    batch: Dict[str, torch.Tensor],
    cfg,
    phase: str,
    run_diffusion: bool,
) -> None:
    model.zero_grad(set_to_none=True)
    outputs = model.training_forward(
        batch["lr_hsi"],
        batch["hr_msi"],
        batch["gt"],
        run_diffusion=run_diffusion,
    )
    for name, value in outputs.items():
        if torch.is_tensor(value):
            finite(f"{phase}.output.{name}", value)
    losses = compute_losses(
        model,
        outputs,
        batch["gt"],
        SAMLoss(),
        cfg,
        run_diffusion,
        phase,
    )
    for name in LOSS_NAMES:
        finite(f"{phase}.loss.{name}", losses[name])
    losses["total"].backward()
    check_gradients(model, phase)
    print(
        f"{phase} backward passed | total={losses['total'].item():.6f}, "
        f"basis_mask={losses['basis_mask_mean'].item():.4f}, "
        f"orth_mask={losses['orthogonal_mask_mean'].item():.4f}"
    )


def main() -> None:
    cfg = parse_uncertainty_guided_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, _ = build_stage2_model(cfg, info, device)
    model = build_model(cfg, stage2, device)
    scales = estimate_residual_scales(
        model,
        train_loader,
        device,
        max_batches=max(1, cfg.stage3_scale_estimation_batches or 4),
    )
    finite("coefficient_scale", scales["coefficient_scale"])
    finite("orthogonal_scale", scales["orthogonal_scale"])

    batch = move_to_device(next(iter(test_loader)), device)
    for name, value in batch.items():
        if torch.is_tensor(value):
            finite(f"batch.{name}", value)

    model.eval()
    with torch.no_grad():
        outputs = model.sample(
            batch["lr_hsi"],
            batch["hr_msi"],
            inference_steps=min(cfg.stage3_inference_steps, 4),
            initial_noise="zero",
        )
        stage2_metrics = calc_metrics(
            outputs["stage2_hsi"],
            batch["gt"],
            cfg.scale_ratio,
        )
        final_metrics = calc_metrics(
            outputs["refined_hsi"],
            batch["gt"],
            cfg.scale_ratio,
        )
        max_identity_error = float(
            (outputs["refined_hsi"] - outputs["stage2_hsi"]).abs().max().item()
        )
    print(
        f"identity check | Stage2={stage2_metrics['PSNR']:.4f} dB/"
        f"{stage2_metrics['SAM']:.4f} deg, "
        f"Stage3={final_metrics['PSNR']:.4f} dB/"
        f"{final_metrics['SAM']:.4f} deg, "
        f"max_abs_delta={max_identity_error:.6e}"
    )
    if max_identity_error > 1e-6:
        raise RuntimeError("Zero-initialized Stage 3 does not preserve Stage 2")

    train_batch = move_to_device(next(iter(train_loader)), device)
    model.train()
    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(True)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(False)
    run_backward(model, train_batch, cfg, "deterministic", False)

    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(False)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(True)
    run_backward(model, train_batch, cfg, "diffusion", True)
    print("Uncertainty-guided Stage-3 smoke test passed.")


if __name__ == "__main__":
    main()
