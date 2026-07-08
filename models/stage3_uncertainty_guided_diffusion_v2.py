"""Enhanced uncertainty-guided Stage-3 local diffusion.

This revision keeps the successful deterministic dual-domain reconstruction and
strengthens only the diffusion refinement through four changes:

1. diffusion-specific residual RMS scales estimated after deterministic fitting;
2. oracle/predicted uncertainty-mask curriculum during diffusion training;
3. an independent clean-residual (x0) head beside the noise head;
4. fixed short deterministic DDIM sampling, intended for the empirically best
   12-step setting.

The previous Stage-3 implementation remains unchanged for ablation studies.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count
from .stage3_dual_domain_diffusion import (
    TimeConditionedResidualBlock,
    sinusoidal_timestep_embedding,
)
from .stage3_uncertainty_guided_diffusion import (
    UncertaintyGuidedDualDomainDiffusionRefiner,
)


class LocalConditionalHybridDenoiser(nn.Module):
    """Predict both diffusion noise and the clean masked residual directly."""

    def __init__(
        self,
        sample_channels: int,
        condition_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        time_channels: int = 192,
        clean_clip: float = 8.0,
    ):
        super().__init__()
        if sample_channels <= 0 or condition_channels <= 0:
            raise ValueError("Hybrid denoiser channel counts must be positive")
        if num_blocks <= 0 or clean_clip <= 0:
            raise ValueError("num_blocks and clean_clip must be positive")

        self.sample_channels = int(sample_channels)
        self.condition_channels = int(condition_channels)
        self.time_channels = int(time_channels)
        self.clean_clip = float(clean_clip)
        groups = _group_count(hidden_channels)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input_projection = nn.Conv2d(
            sample_channels + condition_channels,
            hidden_channels,
            3,
            padding=1,
        )
        dilations = (1, 2, 4)
        self.blocks = nn.ModuleList(
            [
                TimeConditionedResidualBlock(
                    hidden_channels,
                    time_channels,
                    dilation=dilations[index % len(dilations)],
                )
                for index in range(num_blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(groups, hidden_channels)
        self.noise_head = nn.Conv2d(
            hidden_channels,
            sample_channels,
            3,
            padding=1,
        )
        self.x0_head = nn.Conv2d(
            hidden_channels,
            sample_channels,
            3,
            padding=1,
        )
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)
        nn.init.zeros_(self.x0_head.weight)
        nn.init.zeros_(self.x0_head.bias)

    def forward(
        self,
        noisy_sample: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if noisy_sample.ndim != 4 or noisy_sample.size(1) != self.sample_channels:
            raise ValueError(
                f"Expected noisy sample [N, {self.sample_channels}, H, W], "
                f"got {tuple(noisy_sample.shape)}"
            )
        if condition.ndim != 4 or condition.size(1) != self.condition_channels:
            raise ValueError(
                f"Expected condition [N, {self.condition_channels}, H, W], "
                f"got {tuple(condition.shape)}"
            )
        if noisy_sample.shape[0] != condition.shape[0] or noisy_sample.shape[-2:] != condition.shape[-2:]:
            raise ValueError("Noisy sample and condition must share shape")

        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.time_channels,
        )
        time_embedding = self.time_mlp(time_embedding)
        hidden = self.input_projection(torch.cat([noisy_sample, condition], dim=1))
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        hidden = F.silu(self.output_norm(hidden))
        return {
            "noise": self.noise_head(hidden),
            "x0": self.clean_clip * torch.tanh(self.x0_head(hidden)),
        }


class UncertaintyGuidedDualDomainDiffusionRefinerV2(
    UncertaintyGuidedDualDomainDiffusionRefiner
):
    """Stage 3 with local residual re-scaling and hybrid x0/noise diffusion."""

    def __init__(
        self,
        *args,
        direct_x0_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= direct_x0_weight <= 1.0:
            raise ValueError("direct_x0_weight must be in [0, 1]")
        self.direct_x0_weight = float(direct_x0_weight)

        coefficient_condition_channels = (
            self.basis_rank + self.msi_channels * 2 + 1 + self.basis_rank + 1
        )
        orthogonal_condition_channels = (
            self.n_bands + self.msi_channels * 2 + 1 + self.n_bands + 1
        )
        old_coefficient = self.coefficient_diffusion
        old_orthogonal = self.orthogonal_diffusion
        hidden_channels = old_coefficient.input_projection.out_channels
        num_blocks = len(old_coefficient.blocks)
        time_channels = old_coefficient.time_channels
        self.coefficient_diffusion = LocalConditionalHybridDenoiser(
            sample_channels=self.basis_rank,
            condition_channels=coefficient_condition_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            time_channels=time_channels,
            clean_clip=self.clean_clip,
        )
        self.orthogonal_diffusion = LocalConditionalHybridDenoiser(
            sample_channels=self.n_bands,
            condition_channels=orthogonal_condition_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            time_channels=time_channels,
            clean_clip=self.clean_clip,
        )
        self.register_buffer(
            "coefficient_diffusion_scale",
            torch.ones(self.basis_rank, dtype=torch.float32),
        )
        self.register_buffer(
            "orthogonal_diffusion_scale",
            torch.ones(1, dtype=torch.float32),
        )

    @torch.no_grad()
    def set_diffusion_scales(
        self,
        coefficient_scale: torch.Tensor,
        orthogonal_scale: torch.Tensor | float,
    ) -> None:
        coefficient_scale = torch.as_tensor(
            coefficient_scale,
            device=self.coefficient_diffusion_scale.device,
            dtype=self.coefficient_diffusion_scale.dtype,
        ).flatten()
        if coefficient_scale.numel() != self.basis_rank:
            raise ValueError(
                f"Expected {self.basis_rank} diffusion coefficient scales, "
                f"got {coefficient_scale.numel()}"
            )
        orthogonal_scale = torch.as_tensor(
            orthogonal_scale,
            device=self.orthogonal_diffusion_scale.device,
            dtype=self.orthogonal_diffusion_scale.dtype,
        ).reshape(1)
        self.coefficient_diffusion_scale.copy_(
            coefficient_scale.clamp_min(self.residual_scale_floor)
        )
        self.orthogonal_diffusion_scale.copy_(
            orthogonal_scale.clamp_min(self.residual_scale_floor)
        )

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
        clean = clean * mask
        if project_fn is not None:
            clean = project_fn(clean)
        return predicted_noise, direct_x0, clean

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

        base_coefficient_scale = self.coefficient_residual_scale.view(1, -1, 1, 1)
        base_orthogonal_scale = self.orthogonal_residual_scale.view(1, 1, 1, 1)
        target_coefficient_normalized = target_coefficient / base_coefficient_scale
        target_orthogonal_normalized = target_orthogonal / base_orthogonal_scale
        deterministic_coefficient = deterministic[
            "deterministic_coefficient_residual"
        ]
        deterministic_orthogonal = deterministic[
            "deterministic_orthogonal_residual"
        ]
        remaining_coefficient = target_coefficient - deterministic_coefficient
        remaining_orthogonal = self.project_orthogonal(
            target_orthogonal - deterministic_orthogonal,
            basis,
        )

        diffusion_coefficient_scale = self.coefficient_diffusion_scale.view(
            1,
            -1,
            1,
            1,
        )
        diffusion_orthogonal_scale = self.orthogonal_diffusion_scale.view(
            1,
            1,
            1,
            1,
        )
        remaining_coefficient_normalized = (
            remaining_coefficient / diffusion_coefficient_scale
        )
        remaining_orthogonal_normalized = self.project_orthogonal(
            remaining_orthogonal / diffusion_orthogonal_scale,
            basis,
        )

        basis_oracle_mask = self.error_to_oracle_mask(
            remaining_coefficient / base_coefficient_scale,
            self.basis_mask_threshold,
        )
        orthogonal_oracle_mask = self.error_to_oracle_mask(
            remaining_orthogonal / base_orthogonal_scale,
            self.orthogonal_mask_threshold,
        )
        predicted_basis_mask = deterministic["basis_mask"].detach()
        predicted_orthogonal_mask = deterministic["orthogonal_mask"].detach()
        basis_train_mask = (
            oracle_mask_mix * basis_oracle_mask
            + (1.0 - oracle_mask_mix) * predicted_basis_mask
        ).detach()
        orthogonal_train_mask = (
            oracle_mask_mix * orthogonal_oracle_mask
            + (1.0 - oracle_mask_mix) * predicted_orthogonal_mask
        ).detach()

        coefficient_condition = torch.cat(
            [
                deterministic["coefficient_condition"],
                deterministic["deterministic_coefficient_normalized"].detach(),
                basis_train_mask,
            ],
            dim=1,
        )
        orthogonal_condition = torch.cat(
            [
                deterministic["orthogonal_condition"],
                deterministic["deterministic_orthogonal_normalized"].detach(),
                orthogonal_train_mask,
            ],
            dim=1,
        )

        batch_size = lr_hsi.size(0)
        coefficient_t = torch.zeros(batch_size, device=lr_hsi.device, dtype=torch.long)
        orthogonal_t = coefficient_t.clone()
        coefficient_noise = torch.zeros_like(remaining_coefficient_normalized)
        orthogonal_noise = torch.zeros_like(remaining_orthogonal_normalized)
        noisy_coefficient = torch.zeros_like(remaining_coefficient_normalized)
        noisy_orthogonal = torch.zeros_like(remaining_orthogonal_normalized)
        predicted_coefficient_noise = torch.zeros_like(remaining_coefficient_normalized)
        predicted_orthogonal_noise = torch.zeros_like(remaining_orthogonal_normalized)
        direct_coefficient_x0 = torch.zeros_like(remaining_coefficient_normalized)
        direct_orthogonal_x0 = torch.zeros_like(remaining_orthogonal_normalized)
        predicted_coefficient_clean = torch.zeros_like(remaining_coefficient_normalized)
        predicted_orthogonal_clean = torch.zeros_like(remaining_orthogonal_normalized)

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
            coefficient_noise = torch.randn_like(remaining_coefficient_normalized)
            orthogonal_noise = self.project_orthogonal(
                torch.randn_like(remaining_orthogonal_normalized),
                basis,
            )
            noisy_coefficient = self._masked_q_sample(
                remaining_coefficient_normalized,
                coefficient_t,
                coefficient_noise,
                basis_train_mask,
            )
            noisy_orthogonal = self._masked_q_sample(
                remaining_orthogonal_normalized,
                orthogonal_t,
                orthogonal_noise,
                orthogonal_train_mask,
                project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
            )
            coefficient_prediction = self.coefficient_diffusion(
                noisy_coefficient,
                coefficient_t,
                coefficient_condition,
            )
            orthogonal_prediction = self.orthogonal_diffusion(
                noisy_orthogonal,
                orthogonal_t,
                orthogonal_condition,
            )
            (
                predicted_coefficient_noise,
                direct_coefficient_x0,
                predicted_coefficient_clean,
            ) = self._hybrid_clean(
                noisy_coefficient,
                coefficient_t,
                coefficient_prediction,
                basis_train_mask,
            )
            (
                predicted_orthogonal_noise,
                direct_orthogonal_x0,
                predicted_orthogonal_clean,
            ) = self._hybrid_clean(
                noisy_orthogonal,
                orthogonal_t,
                orthogonal_prediction,
                orthogonal_train_mask,
                project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
            )

        diffusion_coefficient_residual = (
            predicted_coefficient_clean * diffusion_coefficient_scale
        )
        diffusion_parallel_residual = self.decode_basis_coefficients(
            diffusion_coefficient_residual,
            basis,
        )
        diffusion_orthogonal_residual = (
            predicted_orthogonal_clean * diffusion_orthogonal_scale
        )
        diffusion_orthogonal_residual = self.project_orthogonal(
            diffusion_orthogonal_residual,
            basis,
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
            "remaining_coefficient_residual": remaining_coefficient,
            "remaining_orthogonal_residual": remaining_orthogonal,
            "remaining_coefficient_normalized": remaining_coefficient_normalized,
            "remaining_orthogonal_normalized": remaining_orthogonal_normalized,
            "basis_oracle_mask": basis_oracle_mask,
            "orthogonal_oracle_mask": orthogonal_oracle_mask,
            "basis_train_mask": basis_train_mask,
            "orthogonal_train_mask": orthogonal_train_mask,
            "basis_mask_for_diffusion": predicted_basis_mask,
            "orthogonal_mask_for_diffusion": predicted_orthogonal_mask,
            "coefficient_noise": coefficient_noise,
            "orthogonal_noise": orthogonal_noise,
            "noisy_coefficient": noisy_coefficient,
            "noisy_orthogonal": noisy_orthogonal,
            "predicted_coefficient_noise": predicted_coefficient_noise,
            "predicted_orthogonal_noise": predicted_orthogonal_noise,
            "direct_coefficient_x0": direct_coefficient_x0,
            "direct_orthogonal_x0": direct_orthogonal_x0,
            "predicted_coefficient_clean": predicted_coefficient_clean,
            "predicted_orthogonal_clean": predicted_orthogonal_clean,
            "diffusion_coefficient_residual": diffusion_coefficient_residual,
            "diffusion_parallel_residual": diffusion_parallel_residual,
            "diffusion_orthogonal_residual": diffusion_orthogonal_residual,
            "coefficient_t": coefficient_t,
            "orthogonal_t": orthogonal_t,
            "refined_hsi": refined_hsi,
            **deterministic,
        }

    @torch.no_grad()
    def _hybrid_ddim_sample(
        self,
        denoiser: LocalConditionalHybridDenoiser,
        condition: torch.Tensor,
        sample_shape: Tuple[int, int, int, int],
        mask: torch.Tensor,
        inference_steps: int,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        current = condition.new_zeros(sample_shape)
        if project_fn is not None:
            current = project_fn(current)
        times = torch.unique_consecutive(
            torch.linspace(
                self.diffusion.timesteps - 1,
                0,
                min(inference_steps, self.diffusion.timesteps),
                device=condition.device,
            ).round().long()
        )
        for index, timestep_value in enumerate(times):
            timestep = torch.full(
                (sample_shape[0],),
                int(timestep_value.item()),
                device=condition.device,
                dtype=torch.long,
            )
            prediction = denoiser(current, timestep, condition)
            predicted_noise, _, clean = self._hybrid_clean(
                current,
                timestep,
                prediction,
                mask,
                project_fn=project_fn,
            )
            if index == len(times) - 1:
                current = clean
                break
            next_timestep = times[index + 1]
            alpha_next = self.diffusion.alpha_bars[next_timestep].to(current)
            current = (
                torch.sqrt(alpha_next) * clean
                + torch.sqrt(1.0 - alpha_next) * predicted_noise
            )
            if project_fn is not None:
                current = project_fn(current)
        return current

    @torch.no_grad()
    def sample(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        inference_steps: int = 12,
        initial_noise: str = "zero",
    ) -> Dict[str, torch.Tensor]:
        if initial_noise != "zero":
            raise ValueError("V2 uses deterministic zero-latent sampling only")
        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        basis = stage2_outputs["basis"]
        deterministic = self.deterministic_forward_from_stage2(
            stage2_outputs,
            hr_msi,
        )
        basis_mask = deterministic["basis_mask"]
        orthogonal_mask = deterministic["orthogonal_mask"]
        coefficient_condition = torch.cat(
            [
                deterministic["coefficient_condition"],
                deterministic["deterministic_coefficient_normalized"],
                basis_mask,
            ],
            dim=1,
        )
        orthogonal_condition = torch.cat(
            [
                deterministic["orthogonal_condition"],
                deterministic["deterministic_orthogonal_normalized"],
                orthogonal_mask,
            ],
            dim=1,
        )
        coefficient_clean = self._hybrid_ddim_sample(
            self.coefficient_diffusion,
            coefficient_condition,
            tuple(stage2_outputs["corrected_coefficients"].shape),
            basis_mask,
            inference_steps,
        )
        orthogonal_clean = self._hybrid_ddim_sample(
            self.orthogonal_diffusion,
            orthogonal_condition,
            tuple(stage2_outputs["reconstructed_hsi"].shape),
            orthogonal_mask,
            inference_steps,
            project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
        )
        diffusion_coefficient = (
            coefficient_clean
            * self.coefficient_diffusion_scale.view(1, -1, 1, 1)
        )
        diffusion_parallel = self.decode_basis_coefficients(
            diffusion_coefficient,
            basis,
        )
        diffusion_orthogonal = (
            orthogonal_clean
            * self.orthogonal_diffusion_scale.view(1, 1, 1, 1)
        )
        diffusion_orthogonal = self.project_orthogonal(
            diffusion_orthogonal,
            basis,
        )
        refined_hsi = (
            deterministic["deterministic_hsi"]
            + diffusion_parallel
            + diffusion_orthogonal
        )
        return {
            **stage2_outputs,
            **deterministic,
            "stage2_hsi": stage2_outputs["reconstructed_hsi"],
            "stage3_deterministic_hsi": deterministic["deterministic_hsi"],
            "stage3_diffusion_coefficient_residual": diffusion_coefficient,
            "stage3_diffusion_parallel_residual": diffusion_parallel,
            "stage3_diffusion_orthogonal_residual": diffusion_orthogonal,
            "refined_hsi": refined_hsi,
        }
