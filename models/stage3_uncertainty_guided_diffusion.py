"""Uncertainty-guided local dual-domain diffusion refinement for Stage 3.

The frozen Stage-2 reconstruction is refined in the originally intended order:

1. deterministic basis-coefficient and orthogonal-complement residual heads;
2. heteroscedastic uncertainty estimation for both deterministic heads;
3. soft high-uncertainty masks derived from the calibrated uncertainty maps;
4. masked diffusion of only the deterministic branches' remaining error.

The final reconstruction is

    X3 = X2 + U dC_det + R_perp_det
            + U (M_basis * dC_diff)
            + M_perp * R_perp_diff.

The masks are detached on the diffusion path. Therefore the uncertainty heads
cannot collapse their masks merely to avoid the diffusion objective; they are
trained by heteroscedastic residual likelihood and explicit error-mask
calibration instead.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import CoefficientResidualBlock, _group_count
from .stage2_multiscale_pyramid import Stage2MultiScalePyramidNet
from .stage3_dual_domain_diffusion import (
    GaussianDiffusionSchedule,
    TimeConditionedResidualBlock,
    sinusoidal_timestep_embedding,
)


class DeterministicUncertaintyPredictor(nn.Module):
    """Predict a bounded deterministic residual and per-channel log variance."""

    def __init__(
        self,
        condition_channels: int,
        output_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 5,
        max_normalized_residual: float = 6.0,
        log_variance_min: float = -6.0,
        log_variance_max: float = 3.0,
    ):
        super().__init__()
        if condition_channels <= 0 or output_channels <= 0:
            raise ValueError("Deterministic predictor channel counts must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if max_normalized_residual <= 0:
            raise ValueError("max_normalized_residual must be positive")
        if log_variance_min >= 0.0 or log_variance_max <= 0.0:
            raise ValueError("Log-variance range must contain zero")

        self.output_channels = int(output_channels)
        self.max_normalized_residual = float(max_normalized_residual)
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)

        groups = _group_count(hidden_channels)
        self.input_projection = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            *[CoefficientResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.mean_head = nn.Conv2d(
            hidden_channels,
            output_channels,
            3,
            padding=1,
        )
        self.log_variance_head = nn.Conv2d(
            hidden_channels,
            output_channels,
            3,
            padding=1,
        )
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

        # sigmoid(offset) maps a zero raw head exactly to log variance 0.
        zero_fraction = (0.0 - self.log_variance_min) / (
            self.log_variance_max - self.log_variance_min
        )
        self.log_variance_offset = float(
            math.log(zero_fraction / (1.0 - zero_fraction))
        )

    def forward(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(self.input_projection(condition))
        raw_mean = self.mean_head(hidden)
        normalized_mean = self.max_normalized_residual * torch.tanh(raw_mean)
        raw_log_variance = self.log_variance_head(hidden)
        log_variance = self.log_variance_min + (
            self.log_variance_max - self.log_variance_min
        ) * torch.sigmoid(raw_log_variance + self.log_variance_offset)
        return {
            "raw_mean": raw_mean,
            "normalized_mean": normalized_mean,
            "raw_log_variance": raw_log_variance,
            "log_variance": log_variance,
        }


class LocalConditionalNoiseDenoiser(nn.Module):
    """Timestep-conditioned local diffusion noise predictor."""

    def __init__(
        self,
        sample_channels: int,
        condition_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        time_channels: int = 192,
    ):
        super().__init__()
        if sample_channels <= 0 or condition_channels <= 0:
            raise ValueError("Local diffusion channel counts must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.sample_channels = int(sample_channels)
        self.condition_channels = int(condition_channels)
        self.time_channels = int(time_channels)
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
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

    def forward(
        self,
        noisy_sample: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
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
            raise ValueError("Noisy sample and condition must share batch/spatial shape")

        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.time_channels,
        )
        time_embedding = self.time_mlp(time_embedding)
        hidden = self.input_projection(torch.cat([noisy_sample, condition], dim=1))
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        hidden = F.silu(self.output_norm(hidden))
        return self.noise_head(hidden)


class UncertaintyGuidedDualDomainDiffusionRefiner(nn.Module):
    """Frozen Stage 2 followed by deterministic and local diffusion refinement."""

    def __init__(
        self,
        stage2_model: Stage2MultiScalePyramidNet,
        deterministic_hidden_channels: int = 96,
        deterministic_blocks: int = 5,
        diffusion_hidden_channels: int = 96,
        diffusion_blocks: int = 6,
        time_channels: int = 192,
        diffusion_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        max_normalized_residual: float = 6.0,
        clean_clip: float = 8.0,
        log_variance_min: float = -6.0,
        log_variance_max: float = 3.0,
        basis_mask_threshold: float = 0.5,
        orthogonal_mask_threshold: float = 0.5,
        mask_temperature: float = 0.25,
        mask_spread_floor: float = 1e-4,
        msi_residual_gain: float = 10.0,
        residual_scale_floor: float = 1e-5,
    ):
        super().__init__()
        if clean_clip <= 0:
            raise ValueError("clean_clip must be positive")
        if mask_temperature <= 0 or mask_spread_floor <= 0:
            raise ValueError("Mask temperature and spread floor must be positive")
        if msi_residual_gain <= 0 or residual_scale_floor <= 0:
            raise ValueError("Residual gains and scale floor must be positive")

        self.stage2 = stage2_model
        self.n_bands = int(stage2_model.n_bands)
        self.basis_rank = int(stage2_model.basis_rank)
        self.msi_channels = int(stage2_model.msi_channels)
        self.clean_clip = float(clean_clip)
        self.basis_mask_threshold = float(basis_mask_threshold)
        self.orthogonal_mask_threshold = float(orthogonal_mask_threshold)
        self.mask_temperature = float(mask_temperature)
        self.mask_spread_floor = float(mask_spread_floor)
        self.msi_residual_gain = float(msi_residual_gain)
        self.residual_scale_floor = float(residual_scale_floor)
        self._freeze_stage2()

        coefficient_condition_channels = self.basis_rank + self.msi_channels * 2 + 1
        orthogonal_condition_channels = self.n_bands + self.msi_channels * 2 + 1
        coefficient_diffusion_condition_channels = (
            coefficient_condition_channels + self.basis_rank + 1
        )
        orthogonal_diffusion_condition_channels = (
            orthogonal_condition_channels + self.n_bands + 1
        )

        self.coefficient_deterministic = DeterministicUncertaintyPredictor(
            condition_channels=coefficient_condition_channels,
            output_channels=self.basis_rank,
            hidden_channels=deterministic_hidden_channels,
            num_blocks=deterministic_blocks,
            max_normalized_residual=max_normalized_residual,
            log_variance_min=log_variance_min,
            log_variance_max=log_variance_max,
        )
        self.orthogonal_deterministic = DeterministicUncertaintyPredictor(
            condition_channels=orthogonal_condition_channels,
            output_channels=self.n_bands,
            hidden_channels=deterministic_hidden_channels,
            num_blocks=deterministic_blocks,
            max_normalized_residual=max_normalized_residual,
            log_variance_min=log_variance_min,
            log_variance_max=log_variance_max,
        )
        self.coefficient_diffusion = LocalConditionalNoiseDenoiser(
            sample_channels=self.basis_rank,
            condition_channels=coefficient_diffusion_condition_channels,
            hidden_channels=diffusion_hidden_channels,
            num_blocks=diffusion_blocks,
            time_channels=time_channels,
        )
        self.orthogonal_diffusion = LocalConditionalNoiseDenoiser(
            sample_channels=self.n_bands,
            condition_channels=orthogonal_diffusion_condition_channels,
            hidden_channels=diffusion_hidden_channels,
            num_blocks=diffusion_blocks,
            time_channels=time_channels,
        )
        self.diffusion = GaussianDiffusionSchedule(
            timesteps=diffusion_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.register_buffer(
            "coefficient_residual_scale",
            torch.ones(self.basis_rank, dtype=torch.float32),
        )
        self.register_buffer(
            "orthogonal_residual_scale",
            torch.ones(1, dtype=torch.float32),
        )

    def _freeze_stage2(self) -> None:
        self.stage2.eval()
        for parameter in self.stage2.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage2.eval()
        return self

    def deterministic_parameters(self):
        yield from self.coefficient_deterministic.parameters()
        yield from self.orthogonal_deterministic.parameters()

    def diffusion_parameters(self):
        yield from self.coefficient_diffusion.parameters()
        yield from self.orthogonal_diffusion.parameters()

    @torch.no_grad()
    def set_residual_scales(
        self,
        coefficient_scale: torch.Tensor,
        orthogonal_scale: torch.Tensor | float,
    ) -> None:
        coefficient_scale = torch.as_tensor(
            coefficient_scale,
            device=self.coefficient_residual_scale.device,
            dtype=self.coefficient_residual_scale.dtype,
        ).flatten()
        if coefficient_scale.numel() != self.basis_rank:
            raise ValueError(
                f"Expected {self.basis_rank} coefficient scales, "
                f"got {coefficient_scale.numel()}"
            )
        orthogonal_scale = torch.as_tensor(
            orthogonal_scale,
            device=self.orthogonal_residual_scale.device,
            dtype=self.orthogonal_residual_scale.dtype,
        ).reshape(1)
        self.coefficient_residual_scale.copy_(
            coefficient_scale.clamp_min(self.residual_scale_floor)
        )
        self.orthogonal_residual_scale.copy_(
            orthogonal_scale.clamp_min(self.residual_scale_floor)
        )

    @staticmethod
    def project_to_basis_coefficients(
        spectral_tensor: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("br,nbhw->nrhw", basis, spectral_tensor)

    @staticmethod
    def decode_basis_coefficients(
        coefficients: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("br,nrhw->nbhw", basis, coefficients)

    def project_orthogonal(
        self,
        spectral_tensor: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.project_to_basis_coefficients(spectral_tensor, basis)
        parallel = self.decode_basis_coefficients(coefficients, basis)
        return spectral_tensor - parallel

    def decompose_residual(
        self,
        residual: torch.Tensor,
        basis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficient = self.project_to_basis_coefficients(residual, basis)
        parallel = self.decode_basis_coefficients(coefficient, basis)
        orthogonal = residual - parallel
        return coefficient, parallel, orthogonal

    @torch.no_grad()
    def stage2_forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.stage2(
            lr_hsi,
            hr_msi,
            compute_zero_msi=False,
        )
        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in outputs.items()
        }

    def build_base_conditions(
        self,
        stage2_outputs: Dict[str, torch.Tensor],
        hr_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage2_coefficient_scale = stage2_outputs["coefficient_scale"].view(
            1,
            -1,
            1,
            1,
        )
        normalized_coefficients = (
            stage2_outputs["corrected_coefficients"] / stage2_coefficient_scale
        )
        msi_residual = (
            hr_msi - stage2_outputs["projected_msi"]
        ) * self.msi_residual_gain
        reliability = stage2_outputs["reliability_map"]
        coefficient_condition = torch.cat(
            [normalized_coefficients, hr_msi, msi_residual, reliability],
            dim=1,
        )
        orthogonal_condition = torch.cat(
            [
                stage2_outputs["reconstructed_hsi"],
                hr_msi,
                msi_residual,
                reliability,
            ],
            dim=1,
        )
        return coefficient_condition, orthogonal_condition

    def uncertainty_to_mask(
        self,
        log_variance: torch.Tensor,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = torch.exp(0.5 * log_variance).mean(dim=1, keepdim=True)
        center = uncertainty.mean(dim=(2, 3), keepdim=True).detach()
        spread = uncertainty.std(
            dim=(2, 3),
            unbiased=False,
            keepdim=True,
        ).detach().clamp_min(self.mask_spread_floor)
        standardized = (uncertainty - center) / spread
        mask = torch.sigmoid(
            (standardized - threshold) / self.mask_temperature
        )
        return uncertainty, standardized, mask

    def error_to_oracle_mask(
        self,
        normalized_error: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        error_map = normalized_error.detach().abs().mean(dim=1, keepdim=True)
        center = error_map.mean(dim=(2, 3), keepdim=True)
        spread = error_map.std(
            dim=(2, 3),
            unbiased=False,
            keepdim=True,
        ).clamp_min(self.mask_spread_floor)
        standardized = (error_map - center) / spread
        return torch.sigmoid(
            (standardized - threshold) / self.mask_temperature
        )

    def deterministic_forward_from_stage2(
        self,
        stage2_outputs: Dict[str, torch.Tensor],
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        basis = stage2_outputs["basis"]
        coefficient_condition, orthogonal_condition = self.build_base_conditions(
            stage2_outputs,
            hr_msi,
        )
        coefficient_prediction = self.coefficient_deterministic(
            coefficient_condition
        )
        orthogonal_prediction = self.orthogonal_deterministic(
            orthogonal_condition
        )

        coefficient_scale = self.coefficient_residual_scale.view(1, -1, 1, 1)
        orthogonal_scale = self.orthogonal_residual_scale.view(1, 1, 1, 1)
        deterministic_coefficient = (
            coefficient_prediction["normalized_mean"] * coefficient_scale
        )
        deterministic_parallel = self.decode_basis_coefficients(
            deterministic_coefficient,
            basis,
        )
        deterministic_orthogonal_normalized = self.project_orthogonal(
            orthogonal_prediction["normalized_mean"],
            basis,
        )
        deterministic_orthogonal = (
            deterministic_orthogonal_normalized * orthogonal_scale
        )
        deterministic_hsi = (
            stage2_outputs["reconstructed_hsi"]
            + deterministic_parallel
            + deterministic_orthogonal
        )

        basis_uncertainty, basis_uncertainty_z, basis_mask = (
            self.uncertainty_to_mask(
                coefficient_prediction["log_variance"],
                self.basis_mask_threshold,
            )
        )
        orthogonal_uncertainty, orthogonal_uncertainty_z, orthogonal_mask = (
            self.uncertainty_to_mask(
                orthogonal_prediction["log_variance"],
                self.orthogonal_mask_threshold,
            )
        )
        coefficient_diffusion_condition = torch.cat(
            [
                coefficient_condition,
                coefficient_prediction["normalized_mean"].detach(),
                basis_mask.detach(),
            ],
            dim=1,
        )
        orthogonal_diffusion_condition = torch.cat(
            [
                orthogonal_condition,
                deterministic_orthogonal_normalized.detach(),
                orthogonal_mask.detach(),
            ],
            dim=1,
        )
        return {
            "coefficient_condition": coefficient_condition,
            "orthogonal_condition": orthogonal_condition,
            "coefficient_diffusion_condition": coefficient_diffusion_condition,
            "orthogonal_diffusion_condition": orthogonal_diffusion_condition,
            "deterministic_coefficient_normalized": coefficient_prediction[
                "normalized_mean"
            ],
            "deterministic_coefficient_log_variance": coefficient_prediction[
                "log_variance"
            ],
            "deterministic_coefficient_residual": deterministic_coefficient,
            "deterministic_parallel_residual": deterministic_parallel,
            "deterministic_orthogonal_normalized": deterministic_orthogonal_normalized,
            "deterministic_orthogonal_log_variance": orthogonal_prediction[
                "log_variance"
            ],
            "deterministic_orthogonal_residual": deterministic_orthogonal,
            "deterministic_hsi": deterministic_hsi,
            "basis_uncertainty": basis_uncertainty,
            "basis_uncertainty_z": basis_uncertainty_z,
            "basis_mask": basis_mask,
            "orthogonal_uncertainty": orthogonal_uncertainty,
            "orthogonal_uncertainty_z": orthogonal_uncertainty_z,
            "orthogonal_mask": orthogonal_mask,
        }

    def _masked_q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        mask: torch.Tensor,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Apply the soft mask exactly once. Both clean and noise already carry
        # the same local support, so their Gaussian mixture remains masked.
        clean = clean * mask
        noise = noise * mask
        if project_fn is not None:
            clean = project_fn(clean)
            noise = project_fn(noise)
        noisy = self.diffusion.q_sample(clean, timesteps, noise)
        if project_fn is not None:
            noisy = project_fn(noisy)
        return noisy

    def training_forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        gt_hsi: torch.Tensor,
        run_diffusion: bool = True,
    ) -> Dict[str, torch.Tensor]:
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

        basis_mask_for_diffusion = deterministic["basis_mask"].detach()
        orthogonal_mask_for_diffusion = deterministic[
            "orthogonal_mask"
        ].detach()

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
                deterministic["coefficient_diffusion_condition"],
            ) * basis_mask_for_diffusion
            predicted_orthogonal_noise = self.orthogonal_diffusion(
                noisy_orthogonal,
                orthogonal_t,
                deterministic["orthogonal_diffusion_condition"],
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
            "refined_hsi": refined_hsi,
            **deterministic,
        }

    @torch.no_grad()
    def _masked_ddim_sample(
        self,
        denoiser: LocalConditionalNoiseDenoiser,
        condition: torch.Tensor,
        sample_shape: Tuple[int, int, int, int],
        mask: torch.Tensor,
        inference_steps: int,
        initial_noise: str,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if initial_noise not in {"zero", "random"}:
            raise ValueError("initial_noise must be 'zero' or 'random'")
        if initial_noise == "zero":
            current = condition.new_zeros(sample_shape)
        else:
            current = torch.randn(
                sample_shape,
                device=condition.device,
                dtype=condition.dtype,
            ) * mask
        if project_fn is not None:
            current = project_fn(current)

        raw_times = torch.linspace(
            self.diffusion.timesteps - 1,
            0,
            min(inference_steps, self.diffusion.timesteps),
            device=condition.device,
        ).round().long()
        times = torch.unique_consecutive(raw_times)

        for index, timestep_value in enumerate(times):
            timestep = torch.full(
                (sample_shape[0],),
                int(timestep_value.item()),
                device=condition.device,
                dtype=torch.long,
            )
            predicted_noise = denoiser(current, timestep, condition) * mask
            if project_fn is not None:
                predicted_noise = project_fn(predicted_noise)
            clean = self.diffusion.predict_clean_from_noise(
                current,
                timestep,
                predicted_noise,
            ).clamp(-self.clean_clip, self.clean_clip)
            if project_fn is not None:
                clean = project_fn(clean)

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
        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        basis = stage2_outputs["basis"]
        deterministic = self.deterministic_forward_from_stage2(
            stage2_outputs,
            hr_msi,
        )
        basis_mask = deterministic["basis_mask"]
        orthogonal_mask = deterministic["orthogonal_mask"]
        coefficient_shape = tuple(stage2_outputs["corrected_coefficients"].shape)
        orthogonal_shape = tuple(stage2_outputs["reconstructed_hsi"].shape)

        coefficient_clean = self._masked_ddim_sample(
            self.coefficient_diffusion,
            deterministic["coefficient_diffusion_condition"],
            coefficient_shape,
            basis_mask,
            inference_steps,
            initial_noise,
        )
        orthogonal_clean = self._masked_ddim_sample(
            self.orthogonal_diffusion,
            deterministic["orthogonal_diffusion_condition"],
            orthogonal_shape,
            orthogonal_mask,
            inference_steps,
            initial_noise,
            project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
        )

        diffusion_coefficient = (
            coefficient_clean
            * self.coefficient_residual_scale.view(1, -1, 1, 1)
        )
        diffusion_parallel = self.decode_basis_coefficients(
            diffusion_coefficient,
            basis,
        )
        diffusion_orthogonal = (
            orthogonal_clean
            * self.orthogonal_residual_scale.view(1, 1, 1, 1)
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

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        inference_steps: int = 12,
        initial_noise: str = "zero",
    ) -> Dict[str, torch.Tensor]:
        return self.sample(
            lr_hsi,
            hr_msi,
            inference_steps=inference_steps,
            initial_noise=initial_noise,
        )
