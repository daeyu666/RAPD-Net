"""Direct x0 residual ablation for Stage 3.

This is the last control experiment after residual-scale and mask-curriculum
ablations. It removes the diffusion denoising chain entirely and keeps only a
conditioned direct clean-residual predictor on top of the fitted deterministic
Stage-3 branches.

Interpretation:
- if this direct x0 branch improves clearly while diffusion does not, the
  bottleneck is the diffusion denoising / sampling formulation;
- if this branch also gives only tiny gains, the remaining residual is not
  learnable from the current conditions and masks.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import CoefficientResidualBlock, _group_count
from .stage3_uncertainty_guided_diffusion import (
    UncertaintyGuidedDualDomainDiffusionRefiner,
)


class DirectX0ResidualHead(nn.Module):
    """Predict a bounded normalized clean residual without diffusion noise."""

    def __init__(
        self,
        condition_channels: int,
        output_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        clean_clip: float = 8.0,
    ):
        super().__init__()
        if condition_channels <= 0 or output_channels <= 0:
            raise ValueError("Direct x0 channel counts must be positive")
        if num_blocks <= 0 or clean_clip <= 0:
            raise ValueError("num_blocks and clean_clip must be positive")
        self.output_channels = int(output_channels)
        self.clean_clip = float(clean_clip)
        groups = _group_count(hidden_channels)
        self.input_projection = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            *[CoefficientResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.x0_head = nn.Conv2d(hidden_channels, output_channels, 3, padding=1)
        nn.init.zeros_(self.x0_head.weight)
        nn.init.zeros_(self.x0_head.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(self.input_projection(condition))
        return self.clean_clip * torch.tanh(self.x0_head(hidden))


class DirectX0Stage3AblationRefiner(UncertaintyGuidedDualDomainDiffusionRefiner):
    """Stage-3 deterministic heads plus direct residual prediction only."""

    def __init__(
        self,
        *args,
        direct_hidden_channels: int = 96,
        direct_blocks: int = 6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        coefficient_condition_channels = (
            self.basis_rank + self.msi_channels * 2 + 1 + self.basis_rank + 1
        )
        orthogonal_condition_channels = (
            self.n_bands + self.msi_channels * 2 + 1 + self.n_bands + 1
        )
        self.coefficient_direct_x0 = DirectX0ResidualHead(
            condition_channels=coefficient_condition_channels,
            output_channels=self.basis_rank,
            hidden_channels=direct_hidden_channels,
            num_blocks=direct_blocks,
            clean_clip=self.clean_clip,
        )
        self.orthogonal_direct_x0 = DirectX0ResidualHead(
            condition_channels=orthogonal_condition_channels,
            output_channels=self.n_bands,
            hidden_channels=direct_hidden_channels,
            num_blocks=direct_blocks,
            clean_clip=self.clean_clip,
        )

    def direct_parameters(self):
        yield from self.coefficient_direct_x0.parameters()
        yield from self.orthogonal_direct_x0.parameters()

    def diffusion_parameters(self):
        # Preserve compatibility with older helper code.
        yield from self.direct_parameters()

    def choose_mask(
        self,
        deterministic: Dict[str, torch.Tensor],
        basis_oracle_mask: torch.Tensor,
        orthogonal_oracle_mask: torch.Tensor,
        mask_mode: str,
    ):
        if mask_mode == "predicted":
            return deterministic["basis_mask"].detach(), deterministic[
                "orthogonal_mask"
            ].detach()
        if mask_mode == "oracle":
            return basis_oracle_mask.detach(), orthogonal_oracle_mask.detach()
        if mask_mode == "none":
            return torch.ones_like(basis_oracle_mask), torch.ones_like(
                orthogonal_oracle_mask
            )
        raise ValueError("mask_mode must be one of: predicted, oracle, none")

    def _forward_from_stage2(
        self,
        stage2_outputs: Dict[str, torch.Tensor],
        hr_msi: torch.Tensor,
        gt_hsi: torch.Tensor | None = None,
        mask_mode: str = "predicted",
    ) -> Dict[str, torch.Tensor]:
        basis = stage2_outputs["basis"]
        stage2_hsi = stage2_outputs["reconstructed_hsi"]
        deterministic = self.deterministic_forward_from_stage2(stage2_outputs, hr_msi)
        coefficient_scale = self.coefficient_residual_scale.view(1, -1, 1, 1)
        orthogonal_scale = self.orthogonal_residual_scale.view(1, 1, 1, 1)

        basis_oracle_mask = torch.zeros(
            stage2_hsi.size(0),
            1,
            stage2_hsi.size(2),
            stage2_hsi.size(3),
            device=stage2_hsi.device,
            dtype=stage2_hsi.dtype,
        )
        orthogonal_oracle_mask = torch.zeros_like(basis_oracle_mask)
        target_coefficient = torch.zeros_like(stage2_outputs["corrected_coefficients"])
        target_orthogonal = torch.zeros_like(stage2_hsi)
        remaining_coefficient_normalized = torch.zeros_like(target_coefficient)
        remaining_orthogonal_normalized = torch.zeros_like(target_orthogonal)
        target_residual = torch.zeros_like(stage2_hsi)

        if gt_hsi is not None:
            target_residual = gt_hsi - stage2_hsi
            target_coefficient, _, target_orthogonal = self.decompose_residual(
                target_residual,
                basis,
            )
            remaining_coefficient = (
                target_coefficient
                - deterministic["deterministic_coefficient_residual"]
            )
            remaining_orthogonal = self.project_orthogonal(
                target_orthogonal
                - deterministic["deterministic_orthogonal_residual"],
                basis,
            )
            remaining_coefficient_normalized = remaining_coefficient / coefficient_scale
            remaining_orthogonal_normalized = self.project_orthogonal(
                remaining_orthogonal / orthogonal_scale,
                basis,
            )
            basis_oracle_mask = self.error_to_oracle_mask(
                remaining_coefficient_normalized,
                self.basis_mask_threshold,
            )
            orthogonal_oracle_mask = self.error_to_oracle_mask(
                remaining_orthogonal_normalized,
                self.orthogonal_mask_threshold,
            )

        basis_mask, orthogonal_mask = self.choose_mask(
            deterministic,
            basis_oracle_mask,
            orthogonal_oracle_mask,
            mask_mode,
        )
        coefficient_condition = torch.cat(
            [
                deterministic["coefficient_condition"],
                deterministic["deterministic_coefficient_normalized"].detach(),
                basis_mask,
            ],
            dim=1,
        )
        orthogonal_condition = torch.cat(
            [
                deterministic["orthogonal_condition"],
                deterministic["deterministic_orthogonal_normalized"].detach(),
                orthogonal_mask,
            ],
            dim=1,
        )
        predicted_coefficient_clean = self.coefficient_direct_x0(
            coefficient_condition
        ) * basis_mask
        predicted_orthogonal_clean = self.orthogonal_direct_x0(
            orthogonal_condition
        ) * orthogonal_mask
        predicted_orthogonal_clean = self.project_orthogonal(
            predicted_orthogonal_clean,
            basis,
        )
        direct_coefficient_residual = predicted_coefficient_clean * coefficient_scale
        direct_parallel_residual = self.decode_basis_coefficients(
            direct_coefficient_residual,
            basis,
        )
        direct_orthogonal_residual = predicted_orthogonal_clean * orthogonal_scale
        direct_orthogonal_residual = self.project_orthogonal(
            direct_orthogonal_residual,
            basis,
        )
        refined_hsi = (
            deterministic["deterministic_hsi"]
            + direct_parallel_residual
            + direct_orthogonal_residual
        )
        return {
            "stage2_outputs": stage2_outputs,
            "basis": basis,
            "stage2_hsi": stage2_hsi,
            "target_residual": target_residual,
            "target_coefficient_residual": target_coefficient,
            "target_orthogonal_residual": target_orthogonal,
            "remaining_coefficient_normalized": remaining_coefficient_normalized,
            "remaining_orthogonal_normalized": remaining_orthogonal_normalized,
            "basis_oracle_mask": basis_oracle_mask,
            "orthogonal_oracle_mask": orthogonal_oracle_mask,
            "basis_train_mask": basis_mask,
            "orthogonal_train_mask": orthogonal_mask,
            "predicted_coefficient_clean": predicted_coefficient_clean,
            "predicted_orthogonal_clean": predicted_orthogonal_clean,
            "diffusion_coefficient_residual": direct_coefficient_residual,
            "diffusion_parallel_residual": direct_parallel_residual,
            "diffusion_orthogonal_residual": direct_orthogonal_residual,
            "refined_hsi": refined_hsi,
            **deterministic,
        }

    def training_forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        gt_hsi: torch.Tensor,
        mask_mode: str = "predicted",
    ) -> Dict[str, torch.Tensor]:
        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        return self._forward_from_stage2(
            stage2_outputs,
            hr_msi,
            gt_hsi=gt_hsi,
            mask_mode=mask_mode,
        )

    @torch.no_grad()
    def sample(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        inference_steps: int = 1,
        initial_noise: str = "zero",
        mask_mode: str = "predicted",
    ) -> Dict[str, torch.Tensor]:
        if initial_noise != "zero":
            raise ValueError("Direct x0 ablation is deterministic and uses no noise")
        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        outputs = self._forward_from_stage2(
            stage2_outputs,
            hr_msi,
            gt_hsi=None,
            mask_mode=mask_mode,
        )
        return {
            **stage2_outputs,
            **outputs,
            "stage2_hsi": stage2_outputs["reconstructed_hsi"],
            "stage3_deterministic_hsi": outputs["deterministic_hsi"],
            "stage3_diffusion_coefficient_residual": outputs[
                "diffusion_coefficient_residual"
            ],
            "stage3_diffusion_parallel_residual": outputs[
                "diffusion_parallel_residual"
            ],
            "stage3_diffusion_orthogonal_residual": outputs[
                "diffusion_orthogonal_residual"
            ],
            "refined_hsi": outputs["refined_hsi"],
        }

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        inference_steps: int = 1,
        initial_noise: str = "zero",
    ) -> Dict[str, torch.Tensor]:
        return self.sample(
            lr_hsi,
            hr_msi,
            inference_steps=inference_steps,
            initial_noise=initial_noise,
            mask_mode="predicted",
        )
