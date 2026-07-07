"""Single-batch finite-value diagnostic for Stage-3 diffusion training.

The script reconstructs frozen Stage 2, estimates residual scales on a limited
number of batches, runs one Stage-3 forward/backward pass, and reports the first
non-finite tensor or gradient. It does not update any parameter.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from data_loader import build_loaders
from losses import SAMLoss
from train_stage3_dual_domain_diffusion import (
    STAGE3_LOSS_NAMES,
    build_stage2_model,
    build_stage3_model,
    compute_stage3_losses,
    estimate_residual_scales,
    parse_stage3_args,
)
from utils import get_device, move_to_device, set_seed


def tensor_summary(name: str, tensor: torch.Tensor) -> str:
    detached = tensor.detach().float()
    finite = torch.isfinite(detached)
    finite_ratio = float(finite.float().mean().item())
    if finite.any():
        values = detached[finite]
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        mean = float(values.mean().item())
    else:
        minimum = maximum = mean = float("nan")
    return (
        f"{name}: shape={tuple(tensor.shape)}, finite={finite_ratio:.6f}, "
        f"min={minimum:.6e}, max={maximum:.6e}, mean={mean:.6e}"
    )


def walk_tensors(prefix: str, value: Any):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk_tensors(f"{prefix}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk_tensors(f"{prefix}[{index}]", item)


def main() -> None:
    cfg = parse_stage3_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, _, info = build_loaders(cfg)
    stage2, _, _ = build_stage2_model(cfg, info, device)
    model = build_stage3_model(cfg, stage2, device)

    scales = estimate_residual_scales(
        model,
        train_loader,
        device,
        max_batches=max(1, cfg.stage3_scale_estimation_batches or 4),
    )
    print(tensor_summary("coefficient_scale", scales["coefficient_scale"]))
    print(tensor_summary("orthogonal_scale", scales["orthogonal_scale"]))

    batch = move_to_device(next(iter(train_loader)), device)
    for name, tensor in walk_tensors("batch", batch):
        print(tensor_summary(name, tensor))
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite input detected in {name}")

    model.train()
    outputs = model.training_forward(
        batch["lr_hsi"],
        batch["hr_msi"],
        batch["gt"],
    )
    for name, tensor in walk_tensors("outputs", outputs):
        if not torch.isfinite(tensor).all():
            print(tensor_summary(name, tensor))
            raise RuntimeError(f"Non-finite Stage-3 output detected in {name}")

    losses = compute_stage3_losses(
        model,
        outputs,
        batch["gt"],
        SAMLoss(),
        cfg,
        reconstruction_weight=0.0,
    )
    for name in STAGE3_LOSS_NAMES:
        value = losses[name]
        print(tensor_summary(f"loss.{name}", value.reshape(1)))
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite Stage-3 loss detected in {name}")

    losses["total"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                print(tensor_summary(f"gradient.{name}", parameter.grad))
                raise RuntimeError(f"Non-finite gradient detected in {name}")

    print("Stage-3 finite-value diagnostic passed.")


if __name__ == "__main__":
    main()
