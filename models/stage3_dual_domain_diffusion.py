"""Stage-3 dual-domain uncertainty-aware residual diffusion refiner.

The frozen Stage-2 reconstruction ``X2`` is refined in two orthogonal domains:

    R_parallel = U_r Delta C_res
    R_perp     = (I - U_r U_r^T) R

A coefficient-domain diffusion branch models ``Delta C_res`` and a spectral
orthogonal-complement branch models ``R_perp``. Both denoisers predict noise and
bounded log variance. Uncertainty is trained and reported but is deliberately
not used to gate the reconstructed output in Stage 3.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count
from .stage2_multiscale_pyramid import Stage2MultiScalePyramidNet


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dimension: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Standard sinusoidal diffusion timestep embedding."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeConditionedResidualBlock(nn.Module):
    """Resolution-preserving residual block with timestep modulation."""

    def __init__(
        self,
        channels: int,
        time_channels: int,
        dilation: int = 1,
    ):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.time_projection = nn.Linear(time_channels, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        feature: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(feature)))
        modulation = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        scale, shift = modulation.chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.conv2(F.silu(hidden))
        return feature + hidden


class ConditionalResidualDenoiser(nn.Module):
    """Compact conditional denoiser with noise and log-variance heads."""

    def __init__(
        self,
        sample_channels: int,
        condition_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        time_channels: int = 192,
        log_variance_min: float = -6.0,
        log_variance_max: float = 3.0,
    ):
        super().__init__()
        if sample_channels <= 0 or condition_channels <= 0:
            raise ValueError("Diffusion channel counts must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if log_variance_min >= log_variance_max:
            raise ValueError("Invalid log-variance range")

        self.sample_channels = int(sample_channels)
        self.condition_channels = int(condition_channels)
        self.hidden_channels = int(hidden_channels)
        self.time_channels = int(time_channels)
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)

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
        self.log_variance_head = nn.Conv2d(
            hidden_channels,
            sample_channels,
            3,
            padding=1,
        )

        # With zero initial latent, zero noise prediction keeps Stage 3 exactly
        # equal to the frozen Stage-2 reconstruction before training.
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

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
        predicted_noise = self.noise_head(hidden)
        log_variance = self.log_variance_head(hidden).clamp(
            self.log_variance_min,
            self.log_variance_max,
        )
        return {
            "noise": predicted_noise,
            "log_variance": log_variance,
        }


class GaussianDiffusionSchedule(nn.Module):
    """Linear Gaussian schedule with deterministic DDIM inference."""

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Invalid beta schedule")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.view(reference.size(0), 1, 1, 1).to(reference)

    def q_sample(
        self,
        clean_sample: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean_sample)
            * clean_sample
            + self._extract(
                self.sqrt_one_minus_alpha_bars,
                timesteps,
                clean_sample,
            )
            * noise
        )

    def predict_clean_from_noise(
        self,
        noisy_sample: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(
            self.sqrt_alpha_bars,
            timesteps,
            noisy_sample,
        ).clamp_min(1e-8)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            noisy_sample,
        )
        return (noisy_sample - sqrt_one_minus * predicted_noise) / sqrt_alpha

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser: ConditionalResidualDenoiser,
        condition: torch.Tensor,
        sample_shape: Tuple[int, int, int, int],
        inference_steps: int = 12,
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        clean_clip: float = 8.0,
        initial_noise: str = "zero",
    ) -> Dict[str, torch.Tensor]:
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if initial_noise not in {"zero", "random"}:
            raise ValueError("initial_noise must be 'zero' or 'random'")

        if initial_noise == "zero":
            current = condition.new_zeros(sample_shape)
        else:
            current = torch.randn(sample_shape, device=condition.device, dtype=condition.dtype)
        if project_fn is not None:
            current = project_fn(current)

        raw_times = torch.linspace(
            self.timesteps - 1,
            0,
            min(inference_steps, self.timesteps),
            device=condition.device,
        ).round().long()
        times = torch.unique_consecutive(raw_times)
        final_log_variance = current.new_zeros(current.shape)

        for index, timestep_value in enumerate(times):
            timestep = torch.full(
                (sample_shape[0],),
                int(timestep_value.item()),
                device=condition.device,
                dtype=torch.long,
            )
            prediction = denoiser(current, timestep, condition)
            predicted_noise = prediction["noise"]
            if project_fn is not None:
                predicted_noise = project_fn(predicted_noise)
            clean = self.predict_clean_from_noise(
                current,
                timestep,
                predicted_noise,
            ).clamp(-clean_clip, clean_clip)
            if project_fn is not None:
                clean = project_fn(clean)
            final_log_variance = prediction["log_variance"]

            if index == len(times) - 1:
                current = clean
                break

            next_timestep = times[index + 1]
            alpha_next = self.alpha_bars[next_timestep].to(current)
            current = (
                torch.sqrt(alpha_next) * clean
                + torch.sqrt(1.0 - alpha_next) * predicted_noise
            )
            if project_fn is not None:
                current = project_fn(current)

        return {
            "clean": current,
            "log_variance": final_log_variance,
        }


class BasisOrthogonalResidualDiffusionRefiner(nn.Module):
    """Frozen Stage 2 plus basis/orthogonal residual diffusion branches."""

    def __init__(
        self,
        stage2_model: Stage2MultiScalePyramidNet,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        time_channels: int = 192,
        diffusion_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        log_variance_min: float = -6.0,
        log_variance_max: float = 3.0,
        clean_clip: float = 8.0,
        msi_residual_gain: float = 10.0,
        residual_scale_floor: float = 1e-5,
    ):
        super().__init__()
        if clean_clip <= 0:
            raise ValueError("clean_clip must be positive")
        if msi_residual_gain <= 0:
            raise ValueError("msi_residual_gain must be positive")
        if residual_scale_floor <= 0:
            raise ValueError("residual_scale_floor must be positive")

        self.stage2 = stage2_model
        self.n_bands = int(stage2_model.n_bands)
        self.basis_rank = int(stage2_model.basis_rank)
        self.msi_channels = int(stage2_model.msi_channels)
        self.clean_clip = float(clean_clip)
        self.msi_residual_gain = float(msi_residual_gain)
        self.residual_scale_floor = float(residual_scale_floor)
        self._freeze_stage2()

        coefficient_condition_channels = self.basis_rank + self.msi_channels + 1
        orthogonal_condition_channels = (
            self.n_bands + self.msi_channels * 2 + 1
        )
        self.coefficient_denoiser = ConditionalResidualDenoiser(
            sample_channels=self.basis_rank,
            condition_channels=coefficient_condition_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            time_channels=time_channels,
            log_variance_min=log_variance_min,
            log_variance_max=log_variance_max,
        )
        self.orthogonal_denoiser = ConditionalResidualDenoiser(
            sample_channels=self.n_bands,
            condition_channels=orthogonal_condition_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            time_channels=time_channels,
            log_variance_min=log_variance_min,
            log_variance_max=log_variance_max,
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

    def get_basis(self) -> torch.Tensor:
        return self.stage2.stage1.get_basis().detach()

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

    def build_conditions(
        self,
        stage2_outputs: Dict[str, torch.Tensor],
        hr_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage2_scale = stage2_outputs["coefficient_scale"].view(1, -1, 1, 1)
        normalized_coefficients = (
            stage2_outputs["corrected_coefficients"] / stage2_scale
        )
        msi_residual = (
            hr_msi - stage2_outputs["projected_msi"]
        ) * self.msi_residual_gain
        reliability = stage2_outputs["reliability_map"]
        coefficient_condition = torch.cat(
            [normalized_coefficients, msi_residual, reliability],
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

    def decompose_residual(
        self,
        residual: torch.Tensor,
        basis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficient_residual = self.project_to_basis_coefficients(residual, basis)
        parallel_residual = self.decode_basis_coefficients(
            coefficient_residual,
            basis,
        )
        orthogonal_residual = residual - parallel_residual
        return coefficient_residual, parallel_residual, orthogonal_residual

    def training_forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        gt_hsi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        stage2_outputs = self.stage2_forward(lr_hsi, hr_msi)
        basis = stage2_outputs["basis"]
        stage2_hsi = stage2_outputs["reconstructed_hsi"]
        residual = gt_hsi - stage2_hsi
        target_coefficients, target_parallel, target_orthogonal = (
            self.decompose_residual(residual, basis)
        )

        coefficient_scale = self.coefficient_residual_scale.view(1, -1, 1, 1)
        orthogonal_scale = self.orthogonal_residual_scale.view(1, 1, 1, 1)
        coefficient_clean = target_coefficients / coefficient_scale
        orthogonal_clean = target_orthogonal / orthogonal_scale
        coefficient_condition, orthogonal_condition = self.build_conditions(
            stage2_outputs,
            hr_msi,
        )

        batch_size = lr_hsi.size(0)
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
        coefficient_noise = torch.randn_like(coefficient_clean)
        orthogonal_noise = self.project_orthogonal(
            torch.randn_like(orthogonal_clean),
            basis,
        )
        noisy_coefficients = self.diffusion.q_sample(
            coefficient_clean,
            coefficient_t,
            coefficient_noise,
        )
        noisy_orthogonal = self.diffusion.q_sample(
            orthogonal_clean,
            orthogonal_t,
            orthogonal_noise,
        )

        coefficient_prediction = self.coefficient_denoiser(
            noisy_coefficients,
            coefficient_t,
            coefficient_condition,
        )
        orthogonal_prediction = self.orthogonal_denoiser(
            noisy_orthogonal,
            orthogonal_t,
            orthogonal_condition,
        )
        predicted_coefficient_noise = coefficient_prediction["noise"]
        predicted_orthogonal_noise = self.project_orthogonal(
            orthogonal_prediction["noise"],
            basis,
        )
        predicted_coefficient_clean = self.diffusion.predict_clean_from_noise(
            noisy_coefficients,
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

        predicted_coefficient_residual = (
            predicted_coefficient_clean * coefficient_scale
        )
        predicted_parallel_residual = self.decode_basis_coefficients(
            predicted_coefficient_residual,
            basis,
        )
        predicted_orthogonal_residual = (
            predicted_orthogonal_clean * orthogonal_scale
        )
        refined_hsi = (
            stage2_hsi
            + predicted_parallel_residual
            + predicted_orthogonal_residual
        )

        return {
            "stage2_outputs": stage2_outputs,
            "basis": basis,
            "stage2_hsi": stage2_hsi,
            "refined_hsi": refined_hsi,
            "target_residual": residual,
            "target_coefficient_residual": target_coefficients,
            "target_parallel_residual": target_parallel,
            "target_orthogonal_residual": target_orthogonal,
            "coefficient_clean": coefficient_clean,
            "orthogonal_clean": orthogonal_clean,
            "coefficient_noise": coefficient_noise,
            "orthogonal_noise": orthogonal_noise,
            "predicted_coefficient_noise": predicted_coefficient_noise,
            "predicted_orthogonal_noise": predicted_orthogonal_noise,
            "predicted_coefficient_clean": predicted_coefficient_clean,
            "predicted_orthogonal_clean": predicted_orthogonal_clean,
            "predicted_coefficient_residual": predicted_coefficient_residual,
            "predicted_parallel_residual": predicted_parallel_residual,
            "predicted_orthogonal_residual": predicted_orthogonal_residual,
            "coefficient_log_variance": coefficient_prediction["log_variance"],
            "orthogonal_log_variance": orthogonal_prediction["log_variance"],
            "coefficient_t": coefficient_t,
            "orthogonal_t": orthogonal_t,
        }

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
        stage2_hsi = stage2_outputs["reconstructed_hsi"]
        coefficient_condition, orthogonal_condition = self.build_conditions(
            stage2_outputs,
            hr_msi,
        )
        coefficient_shape = tuple(stage2_outputs["corrected_coefficients"].shape)
        orthogonal_shape = tuple(stage2_hsi.shape)

        coefficient_sample = self.diffusion.ddim_sample(
            self.coefficient_denoiser,
            coefficient_condition,
            coefficient_shape,
            inference_steps=inference_steps,
            clean_clip=self.clean_clip,
            initial_noise=initial_noise,
        )
        orthogonal_sample = self.diffusion.ddim_sample(
            self.orthogonal_denoiser,
            orthogonal_condition,
            orthogonal_shape,
            inference_steps=inference_steps,
            project_fn=lambda tensor: self.project_orthogonal(tensor, basis),
            clean_clip=self.clean_clip,
            initial_noise=initial_noise,
        )

        coefficient_residual = (
            coefficient_sample["clean"]
            * self.coefficient_residual_scale.view(1, -1, 1, 1)
        )
        parallel_residual = self.decode_basis_coefficients(
            coefficient_residual,
            basis,
        )
        orthogonal_residual = (
            orthogonal_sample["clean"]
            * self.orthogonal_residual_scale.view(1, 1, 1, 1)
        )
        orthogonal_residual = self.project_orthogonal(
            orthogonal_residual,
            basis,
        )
        refined_hsi = stage2_hsi + parallel_residual + orthogonal_residual

        coefficient_uncertainty = torch.exp(
            0.5 * coefficient_sample["log_variance"]
        )
        orthogonal_uncertainty = torch.exp(
            0.5 * orthogonal_sample["log_variance"]
        )
        return {
            **stage2_outputs,
            "stage2_hsi": stage2_hsi,
            "refined_hsi": refined_hsi,
            "stage3_coefficient_residual": coefficient_residual,
            "stage3_parallel_residual": parallel_residual,
            "stage3_orthogonal_residual": orthogonal_residual,
            "stage3_coefficient_uncertainty": coefficient_uncertainty,
            "stage3_orthogonal_uncertainty": orthogonal_uncertainty,
            "stage3_coefficient_uncertainty_map": coefficient_uncertainty.mean(
                dim=1,
                keepdim=True,
            ),
            "stage3_orthogonal_uncertainty_map": orthogonal_uncertainty.mean(
                dim=1,
                keepdim=True,
            ),
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
