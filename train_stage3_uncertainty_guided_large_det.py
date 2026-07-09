"""Large deterministic-capacity profile for corrected Stage 3.

This launcher keeps the uncertainty-guided Stage-3 formulation unchanged, but
uses a stronger deterministic residual/uncertainty predictor because the
current best run is dominated by deterministic gain while local diffusion adds
only a very small correction.

Default schedule:

- deterministic residual/uncertainty warm-up: epochs 1-240;
- frozen-deterministic local diffusion: epochs 241-360;
- low-rate joint fine-tuning: epochs 361-480;
- deterministic DDIM validation/sampling: 32 steps.

Default capacity change:

- deterministic hidden channels: 96 -> 128;
- deterministic residual blocks: 5 -> 7;
- diffusion capacity is kept unchanged by default to isolate the deterministic
  capacity effect.

All defaults are injected only when the corresponding command-line option is
not supplied, so every value can still be overridden from the shell.
"""

from __future__ import annotations

import sys
from typing import Dict, List


DEFAULT_ARGUMENTS: Dict[str, str] = {
    "--epochs": "480",
    "--stage3_det_warmup_epochs": "240",
    "--stage3_joint_start_epoch": "360",
    "--stage3_det_hidden_channels": "128",
    "--stage3_det_blocks": "7",
    "--stage3_inference_steps": "32",
    "--eval_interval": "10",
    "--checkpoint_root": "./checkpoints_stage3_large_det",
    "--output_root": "./outputs_stage3_large_det",
    "--log_root": "./logs_stage3_large_det",
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
    det_hidden = int(option_value(resolved, "--stage3_det_hidden_channels"))
    det_blocks = int(option_value(resolved, "--stage3_det_blocks"))
    if not 0 <= deterministic_end <= diffusion_end <= epochs:
        raise ValueError(
            "Require 0 <= deterministic warm-up <= joint start <= epochs, got "
            f"{deterministic_end}, {diffusion_end}, {epochs}"
        )

    print(
        "Large-det Stage-3 schedule | "
        f"deterministic=1-{deterministic_end}, "
        f"diffusion={deterministic_end + 1}-{diffusion_end}, "
        f"joint={diffusion_end + 1}-{epochs}, "
        f"DDIM steps={inference_steps}."
    )
    print(
        "Large-det capacity | "
        f"det_hidden={det_hidden}, det_blocks={det_blocks}."
    )
    print(
        "Separate output roots | "
        f"checkpoint={option_value(resolved, '--checkpoint_root')}, "
        f"output={option_value(resolved, '--output_root')}, "
        f"log={option_value(resolved, '--log_root')}.")

    from train_stage3_uncertainty_guided_diffusion import main as train_main

    train_main()


if __name__ == "__main__":
    main()
