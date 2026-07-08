"""Finite-value smoke test for enhanced uncertainty-guided Stage 3."""

from __future__ import annotations

import torch

from data_loader import build_loaders
from losses import SAMLoss
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion_v2 import (
    compute_losses,
    estimate_diffusion_scales,
    load_deterministic_initialization,
    parse_v2_args,
)
from train_stage3_uncertainty_guided_diffusion_v2_stable import build_stable_model
from utils import get_device, move_to_device, set_seed


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        ratio = float(torch.isfinite(tensor.detach()).float().mean().item())
        raise FloatingPointError(f"{name} is non-finite; finite ratio={ratio:.6f}")


def check_backward(model, batch, cfg, phase: str, oracle_mix: float) -> None:
    model.zero_grad(set_to_none=True)
    outputs = model.training_forward(
        batch["lr_hsi"],
        batch["hr_msi"],
        batch["gt"],
        run_diffusion=phase != "specialization",
        oracle_mask_mix=oracle_mix,
    )
    for name, value in outputs.items():
        if torch.is_tensor(value):
            assert_finite(f"{phase}.output.{name}", value)
    losses = compute_losses(model, outputs, batch["gt"], SAMLoss(), cfg, phase)
    for name, value in losses.items():
        assert_finite(f"{phase}.loss.{name}", value)
    losses["total"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert_finite(f"{phase}.gradient.{name}", parameter.grad)
    print(
        f"{phase} backward passed | total={losses['total'].item():.6f}, "
        f"direct_x0={losses['diffusion_direct_x0'].item():.6f}, "
        f"hybrid_x0={losses['diffusion_hybrid_x0'].item():.6f}"
    )


def main() -> None:
    cfg = parse_v2_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage2, _, _ = build_stage2_model(cfg, info, device)
    model = build_stable_model(cfg, stage2, device)
    load_deterministic_initialization(
        model,
        cfg.stage3_initial_checkpoint,
        device,
    )
    scales = estimate_diffusion_scales(
        model,
        train_loader,
        device,
        oracle_weight=cfg.v2_scale_oracle_weight,
        max_batches=max(1, cfg.v2_scale_estimation_batches or 4),
    )
    assert_finite("coefficient_diffusion_scale", scales["coefficient"])
    assert_finite("orthogonal_diffusion_scale", scales["orthogonal"])
    print(
        "diffusion scales | coefficient min/median/max="
        f"({scales['coefficient'].min().item():.6e}, "
        f"{scales['coefficient'].median().item():.6e}, "
        f"{scales['coefficient'].max().item():.6e}), "
        f"orthogonal={scales['orthogonal'].item():.6e}"
    )

    batch = move_to_device(next(iter(test_loader)), device)
    model.eval()
    with torch.no_grad():
        outputs = model.sample(
            batch["lr_hsi"],
            batch["hr_msi"],
            inference_steps=12,
            initial_noise="zero",
        )
        identity_error = float(
            (
                outputs["refined_hsi"]
                - outputs["stage3_deterministic_hsi"]
            ).abs().max().item()
        )
    print(f"zero-diffusion identity max_abs_delta={identity_error:.6e}")
    if identity_error > 1e-6:
        raise RuntimeError("Zero-initialized V2 diffusion changed deterministic output")

    train_batch = move_to_device(next(iter(train_loader)), device)
    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(True)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(False)
    check_backward(model, train_batch, cfg, "specialization", 0.0)

    for parameter in model.deterministic_parameters():
        parameter.requires_grad_(False)
    for parameter in model.diffusion_parameters():
        parameter.requires_grad_(True)
    check_backward(model, train_batch, cfg, "diffusion", 1.0)
    print("Enhanced Stage-3 smoke test passed.")


if __name__ == "__main__":
    main()
