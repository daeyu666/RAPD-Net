"""Numerically faithful V2 hybrid diffusion without repeated soft-mask scaling."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from .stage3_uncertainty_guided_diffusion_v2 import (
    UncertaintyGuidedDualDomainDiffusionRefinerV2,
)


class UncertaintyGuidedDualDomainDiffusionRefinerV2Stable(
    UncertaintyGuidedDualDomainDiffusionRefinerV2
):
    """Apply each soft uncertainty mask exactly once per predicted quantity."""

    def _hybrid_clean(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = prediction["noise"] * mask
        direct_x0 = prediction["x0"] * mask
        if project_fn is not None:
            predicted_noise = project_fn(predicted_noise)
            direct_x0 = project_fn(direct_x0)
        noise_x0 = self.diffusion.predict_clean_from_noise(
            noisy,
            timesteps,
            predicted_noise,
        ).clamp(-self.clean_clip, self.clean_clip)
        if project_fn is not None:
            noise_x0 = project_fn(noise_x0)
        clean = (
            self.direct_x0_weight * direct_x0
            + (1.0 - self.direct_x0_weight) * noise_x0
        )
        if project_fn is not None:
            clean = project_fn(clean)
        return predicted_noise, direct_x0, clean
