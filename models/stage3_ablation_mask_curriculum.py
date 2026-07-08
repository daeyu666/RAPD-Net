"""Single-variable Stage-3 ablation: diffusion training mask only.

Everything is identical to the original uncertainty-guided Stage 3 except the
mask used during diffusion training. The experiment can use:

- predicted: the original predicted uncertainty mask;
- oracle: the residual-error oracle mask;
- curriculum: a convex mixture of oracle and predicted masks.

Inference always uses the predicted uncertainty mask. Residual scales, noise-
prediction architecture, losses, deterministic heads, DDIM sampler, and the
12-step inference setting are unchanged.
"""

from __future__ import annotations

from typing import Dict

import torch

from .stage3_uncertainty_guided_diffusion import (
    UncertaintyGuidedDualDomainDiffusionRefiner,
)


class MaskCurriculumAblationRefiner(UncertaintyGuidedDualDomainDiffusionRefiner):
    """Original Stage 3 with a controllable diffusion-training mask."""

    def training_forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        gt_hsi: torch.Tensor,
        run_diffusion: bool = True,
        oracle_mask_mix: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if not 0.0 <= oracle_mask_mix <= 1.0:
            raise ValueError("oracle_mask_mix must be in [0, 1]")

        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        basis = stage2_outputs["basis"]
        stage2_hsi = stage2_outputs["reconstructed_hsi"]
        target_residual = gt_hsi - stage2_hsi
        target_coefficient, target_parallel, target_orthogonal = (
            self.decompose_residual(target_residual, basis)
        )
        deterministic = self.deterministic_forward_from_stage2(
            stage2_outputs,
            hr_msi,
        )

        coefficient_scale = self.coefficient_residual_scale.view(1, -1, 1, 1)
        orthogonal_scale = self.orthogonal_residual_scale.view(1, 1, 1, 1)
        target_coefficient_normalized = target_coefficient / coefficient_scale
        target_orthogonal_normalized = target_orthogonal / orthogonal_scale
        coefficient_error_normalized = (
            target_coefficient_normalized
            - deterministic["deterministic_coefficient_normalized"]
        )
        orthogonal_error_normalized = self.project_orthogonal(
            target_orthogonal_normalized
            - deterministic["deterministic_orthogonal_normalized"],
            basis,
        )
        basis_oracle_mask = self.error_to_oracle_mask(
            coefficient_error_normalized,
            self.basis_mask_threshold,
        )
        orthogonal_oracle_mask = self.error_to_oracle_mask(
            orthogonal_error_normalized,
            self.orthogonal_mask_threshold,
        )
        predicted_basis_mask = deterministic["basis_mask"].detach()
        predicted_orthogonal_mask = deterministic["orthogonal_mask"].detach()
        basis_mask_for_diffusion = (
            oracle_mask_mix * basis_oracle_mask
            + (1.0 - oracle_mask_mix) * predicted_basis_mask
        ).detach()
        orthogonal_mask_for_diffusion = (
            oracle_mask_mix * orthogonal_oracle_mask
            + (1.0 - oracle_mask_mix) * predicted_orthogonal_mask
        ).detach()

        coefficient_diffusion_condition = torch.cat(
            [
                deterministic["coefficient_condition"],
                deterministic["deterministic_coefficient_normalized"].detach(),
                basis_mask_for_diffusion,
            ],
            dim=1,
        )
        orthogonal_diffusion_condition = torch.cat(
            [
                deterministic["orthogonal_condition"],
                deterministic["deterministic_orthogonal_normalized"].detach(),
                orthogonal_mask_for_diffusion,
            ],
            dim=1,
        )

        batch_size = lr_hsi.size(0)
        zero_coefficient = torch.zeros_like(coefficient_error_normalized)
        zero_orthogonal = torch.zeros_like(orthogonal_error_normalized)
        coefficient_t = torch.zeros(
            batch_size,
            device=lr_hsi.device,
            dtype=torch.long,
        )
        orthogonal_t = coefficient_t.clone()
        coefficient_noise = zero_coefficient
        orthogonal_noise = zero_orthogonal
        noisy_coefficient = zero_coefficient
        noisy_orthogonal = zero_orthogonal
        predicted_coefficient_noise = zero_coefficient
        predicted_orthogonal_noise = zero_orthogonal
        predicted_coefficient_clean = zero_coefficient
        predicted_orthogonal_clean = zero_orthogonal

        if run_diffusion:
            coefficient_t = torch.randint(
                0,
                self.diffusion.timesteps,
                (batch_size,),
                device=lr_hsi.device,
            )
            orthogonal_t = torch.randint(
                0,
                self.diffusion.timesteps,
                (batch_size,),
                device=lr_hsi.device,
            )
            coefficient_noise = torch.randn_like(coefficient_error_normalized)
            orthogonal_noise = self.project_orthogonal(
                torch.randn_like(orthogonal_error_normalized),
                basis,
            )
            noisy_coefficient = self._masked_q_sample(
                coefficient_error_normalized,
                coefficient_t,
                coefficient_noise,
                basis_mask_for_diffusion,
            )
            noisy_orthogonal = self._masked_q_sample(
                orthogonal_error_normalized,
                orthogonal_t,
                orthogonal_noise,
                orthogonal_mask_for_diffusion,
                project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
            )
            predicted_coefficient_noise = self.coefficient_diffusion(
                noisy_coefficient,
                coefficient_t,
                coefficient_diffusion_condition,
            ) * basis_mask_for_diffusion
            predicted_orthogonal_noise = self.orthogonal_diffusion(
                noisy_orthogonal,
                orthogonal_t,
                orthogonal_diffusion_condition,
            ) * orthogonal_mask_for_diffusion
            predicted_orthogonal_noise = self.project_orthogonal(
                predicted_orthogonal_noise,
                basis,
            )
            predicted_coefficient_clean = self.diffusion.predict_clean_from_noise(
                noisy_coefficient,
                coefficient_t,
                predicted_coefficient_noise,
            ).clamp(-self.clean_clip, self.clean_clip)
            predicted_orthogonal_clean = self.diffusion.predict_clean_from_noise(
                noisy_orthogonal,
                orthogonal_t,
                predicted_orthogonal_noise,
            ).clamp(-self.clean_clip, self.clean_clip)
            predicted_orthogonal_clean = self.project_orthogonal(
                predicted_orthogonal_clean,
                basis,
            )

        diffusion_coefficient_residual = (
            predicted_coefficient_clean * coefficient_scale
        )
        diffusion_parallel_residual = self.decode_basis_coefficients(
            diffusion_coefficient_residual,
            basis,
        )
        diffusion_orthogonal_residual = (
            predicted_orthogonal_clean * orthogonal_scale
        )
        refined_hsi = (
            deterministic["deterministic_hsi"]
            + diffusion_parallel_residual
            + diffusion_orthogonal_residual
        )
        return {
            "stage2_outputs": stage2_outputs,
            "basis": basis,
            "stage2_hsi": stage2_hsi,
            "target_residual": target_residual,
            "target_coefficient_residual": target_coefficient,
            "target_parallel_residual": target_parallel,
            "target_orthogonal_residual": target_orthogonal,
            "target_coefficient_normalized": target_coefficient_normalized,
            "target_orthogonal_normalized": target_orthogonal_normalized,
            "remaining_coefficient_normalized": coefficient_error_normalized,
            "remaining_orthogonal_normalized": orthogonal_error_normalized,
            "basis_oracle_mask": basis_oracle_mask,
            "orthogonal_oracle_mask": orthogonal_oracle_mask,
            "predicted_basis_mask": predicted_basis_mask,
            "predicted_orthogonal_mask": predicted_orthogonal_mask,
            "basis_mask_for_diffusion": basis_mask_for_diffusion,
            "orthogonal_mask_for_diffusion": orthogonal_mask_for_diffusion,
            "coefficient_noise": coefficient_noise,
            "orthogonal_noise": orthogonal_noise,
            "noisy_coefficient": noisy_coefficient,
            "noisy_orthogonal": noisy_orthogonal,
            "predicted_coefficient_noise": predicted_coefficient_noise,
            "predicted_orthogonal_noise": predicted_orthogonal_noise,
            "predicted_coefficient_clean": predicted_coefficient_clean,
            "predicted_orthogonal_clean": predicted_orthogonal_clean,
            "diffusion_coefficient_residual": diffusion_coefficient_residual,
            "diffusion_parallel_residual": diffusion_parallel_residual,
            "diffusion_orthogonal_residual": diffusion_orthogonal_residual,
            "coefficient_t": coefficient_t,
            "orthogonal_t": orthogonal_t,
            "oracle_mask_mix": gt_hsi.new_tensor(float(oracle_mask_mix)),
            "refined_hsi": refined_hsi,
            **deterministic,
        }
