"""Extended training profile for uncertainty-guided Stage 3.

This launcher preserves the original 100-epoch profile and injects a longer,
clean schedule unless the same options are explicitly supplied by the user:

- deterministic residual/uncertainty warm-up: epochs 1-50;
- frozen-deterministic local diffusion: epochs 51-150;
- low-rate joint fine-tuning: epochs 151-200;
- deterministic DDIM validation/sampling: 32 steps instead of 12.

Outputs use separate roots by default so the previous 45.19 dB experiment is
not overwritten. Every injected value remains overrideable from the command
line.
"""

from __future__ import annotations

import sys
from typing import Dict, List


DEFAULT_ARGUMENTS: Dict[str, str] = {
    "--epochs": "200",
    "--stage3_det_warmup_epochs": "50",
    "--stage3_joint_start_epoch": "150",
    "--stage3_inference_steps": "32",
    "--eval_interval": "5",
    "--checkpoint_root": "./checkpoints_stage3_extended",
    "--output_root": "./outputs_stage3_extended",
    "--log_root": "./logs_stage3_extended",
}


def has_option(arguments: List[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
    )


def inject_defaults(arguments: List[str]) -> List[str]:
    resolved = list(arguments)
    for option, value in DEFAULT_ARGUMENTS.items():
        if not has_option(resolved, option):
            resolved.extend([option, value])
    return resolved


def option_value(arguments: List[str], option: str) -> str:
    for index, argument in enumerate(arguments):
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
    raise KeyError(option)


def main() -> None:
    resolved = inject_defaults(sys.argv[1:])
    sys.argv = [sys.argv[0], *resolved]

    epochs = int(option_value(resolved, "--epochs"))
    deterministic_end = int(
        option_value(resolved, "--stage3_det_warmup_epochs")
    )
    diffusion_end = int(option_value(resolved, "--stage3_joint_start_epoch"))
    inference_steps = int(option_value(resolved, "--stage3_inference_steps"))
    if not 0 <= deterministic_end <= diffusion_end <= epochs:
        raise ValueError(
            "Require 0 <= deterministic warm-up <= joint start <= epochs, got "
            f"{deterministic_end}, {diffusion_end}, {epochs}"
        )

    print(
        "Extended Stage-3 schedule | "
        f"deterministic=1-{deterministic_end}, "
        f"diffusion={deterministic_end + 1}-{diffusion_end}, "
        f"joint={diffusion_end + 1}-{epochs}, "
        f"DDIM steps={inference_steps}."
    )
    print(
        "Separate output roots | "
        f"checkpoint={option_value(resolved, '--checkpoint_root')}, "
        f"output={option_value(resolved, '--output_root')}, "
        f"log={option_value(resolved, '--log_root')}."
    )

    from train_stage3_uncertainty_guided_diffusion import main as train_main

    train_main()


if __name__ == "__main__":
    main()
