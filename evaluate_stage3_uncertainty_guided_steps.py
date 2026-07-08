"""Evaluate an uncertainty-guided Stage-3 checkpoint at several DDIM step counts.

This script changes only deterministic DDIM inference steps. It does not train
or alter the checkpoint, and reports whether additional sampling iterations
actually improve PSNR/SAM before a longer experiment is launched.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from data_loader import build_loaders
from train_stage3_dual_domain_diffusion import build_stage2_model
from train_stage3_uncertainty_guided_diffusion import (
    build_model,
    evaluate,
    parse_uncertainty_guided_args,
)
from utils import get_device, load_checkpoint, set_seed


def extract_custom_option(
    arguments: List[str],
    option: str,
    default: str,
) -> Tuple[str, List[str]]:
    remaining: List[str] = []
    value = default
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith(option + "="):
            value = argument.split("=", 1)[1]
        elif argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"Missing value after {option}")
            value = arguments[index + 1]
            index += 1
        else:
            remaining.append(argument)
        index += 1
    return value, remaining


def main() -> None:
    checkpoint, remaining = extract_custom_option(
        sys.argv[1:],
        "--stage3_model_checkpoint",
        (
            "./checkpoints_stage3_extended/"
            "stage3_uncertainty_guided_diffusion/PaviaU/"
            "uncertainty_guided_best_psnr.pth"
        ),
    )
    raw_steps, remaining = extract_custom_option(
        remaining,
        "--sampling_steps",
        "12,24,32,48",
    )
    sampling_steps = [
        int(item.strip())
        for item in raw_steps.split(",")
        if item.strip()
    ]
    if not sampling_steps or any(step <= 0 for step in sampling_steps):
        raise ValueError("sampling_steps must contain positive integers")

    sys.argv = [sys.argv[0], *remaining]
    cfg = parse_uncertainty_guided_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    _, test_loader, info = build_loaders(cfg)
    stage2, _, _ = build_stage2_model(cfg, info, device)
    model = build_model(cfg, stage2, device)
    load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        strict=True,
        map_location=str(device),
        load_optimizer=False,
    )

    print(f"Loaded checkpoint: {checkpoint}")
    print("steps\tPSNR\tSAM\tDiff-PSNR\tDiff-SAM")
    for steps in sampling_steps:
        cfg.stage3_inference_steps = steps
        result = evaluate(model, test_loader, cfg, device)
        print(
            f"{steps}\t{result['final_psnr']:.4f}\t"
            f"{result['final_sam']:.4f}\t"
            f"{result['diffusion_psnr_gain_over_deterministic']:+.4f}\t"
            f"{result['diffusion_sam_gain_over_deterministic']:+.4f}"
        )


if __name__ == "__main__":
    main()
