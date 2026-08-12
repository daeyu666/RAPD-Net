"""Adaptive frequency routing for Stage-2 symmetric MSI-HSI differences.

This controlled variant keeps the mature Stage2 symmetric-frequency pipeline:

    LR-HSI -> spectral basis / SRF analytical anchor
    physical MSI = R X_base
    observed MSI = Z
    shared MSI encoder
    symmetric same-band differences
    observable/null-space coefficient residual heads

The only representation change is that the low/mid/high frequency boundaries
are no longer sample-invariant. A lightweight cross-observation predictor uses
statistics from physical/reference features and their absolute discrepancy to
produce per-sample, per-channel boundary offsets. The predictor is zero
initialized so a warm-started symmetric-frequency checkpoint begins exactly
from its learned static boundaries.

NSP/reliability screening is intentionally bypassed. No MSI discrepancy is
suppressed: all high-frequency difference features are routed to the coefficient
predictor. This isolates the hypothesis that the useful improvement comes from
how MSI-HSI discrepancies are decomposed/routed across frequency scales, not
from deciding which discrepancies should be discarded.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_symmetric_frequency import (
    Stage2SymmetricFrequencyNet,
    SymmetricFrequencyReliabilityScreen,
)


class CrossObservationBoundaryPredictor(nn.Module):
    """Predict bounded per-sample frequency-boundary offsets."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        max_shift: float = 2.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")

        self.channels = int(channels)
        self.max_shift = float(max_shift)
        self.net = nn.Sequential(
            nn.Linear(3 * channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        physical_feature: torch.Tensor,
        reference_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if physical_feature.shape != reference_feature.shape:
            raise ValueError(
                "physical_feature and reference_feature must have the same shape"
            )
        if physical_feature.ndim != 4 or physical_feature.size(1) != self.channels:
            raise ValueError(
                f"Expected [N, {self.channels}, H, W], "
                f"got {tuple(physical_feature.shape)}"
            )

        discrepancy = reference_feature - physical_feature
        physical_stat = physical_feature.abs().mean(dim=(-2, -1))
        reference_stat = reference_feature.abs().mean(dim=(-2, -1))
        discrepancy_stat = discrepancy.abs().mean(dim=(-2, -1))
        condition = torch.cat(
            [physical_stat, reference_stat, discrepancy_stat],
            dim=1,
        )

        raw = self.net(condition)
        raw_low, raw_high = raw.chunk(2, dim=1)
        low_shift = self.max_shift * torch.tanh(raw_low)
        high_shift = self.max_shift * torch.tanh(raw_high)
        return {
            "boundary_condition": condition,
            "boundary_low_shift": low_shift,
            "boundary_high_shift": high_shift,
            "boundary_raw_low_shift": raw_low,
            "boundary_raw_high_shift": raw_high,
        }


class AdaptiveSymmetricFrequencyRouter(SymmetricFrequencyReliabilityScreen):
    """Symmetric SSP with cross-observation-conditioned dynamic boundaries."""

    def __init__(
        self,
        *args,
        adaptive_boundary_hidden_channels: int = 64,
        adaptive_boundary_max_shift: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.boundary_predictor = CrossObservationBoundaryPredictor(
            channels=self.feature_channels,
            hidden_channels=adaptive_boundary_hidden_channels,
            max_shift=adaptive_boundary_max_shift,
        )

    def _adaptive_boundaries(
        self,
        physical_full: torch.Tensor,
        reference_full: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_low, base_high = self.spectral_splitter.boundaries()
        predicted = self.boundary_predictor(physical_full, reference_full)

        max_low = float(self.spectral_splitter.num_frequency_bands - 2)
        max_high = float(self.spectral_splitter.num_frequency_bands - 1)

        tau_low = (
            base_low.unsqueeze(0) + predicted["boundary_low_shift"]
        ).clamp(0.0, max_low)
        candidate_high = (
            base_high.unsqueeze(0) + predicted["boundary_high_shift"]
        ).clamp(1.0, max_high)
        tau_high = torch.maximum(candidate_high, tau_low + 1.0).clamp(
            max=max_high
        )

        return {
            **predicted,
            "base_tau_low": base_low,
            "base_tau_high": base_high,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "adaptive_boundary_gap": tau_high - tau_low,
        }

    def _partition_masks(
        self,
        band_index: torch.Tensor,
        tau_low: torch.Tensor,
        tau_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-aware version of the original SSP partition."""
        tau_low_4d = tau_low.unsqueeze(-1).unsqueeze(-1)
        tau_high_4d = tau_high.unsqueeze(-1).unsqueeze(-1)
        temperature = self.spectral_splitter.boundary_temperature

        soft_low = torch.sigmoid((tau_low_4d - band_index) / temperature)
        soft_below_high = torch.sigmoid(
            (tau_high_4d - band_index) / temperature
        )
        soft_mid = (soft_below_high - soft_low).clamp(0.0, 1.0)
        soft_high = 1.0 - soft_below_high

        if not self.spectral_splitter.hard_partition:
            return soft_low, soft_mid, soft_high

        hard_low = (band_index < tau_low_4d).to(dtype=band_index.dtype)
        hard_high = (band_index > tau_high_4d).to(dtype=band_index.dtype)
        hard_mid = 1.0 - hard_low - hard_high

        low = hard_low + soft_low - soft_low.detach()
        mid = hard_mid + soft_mid - soft_mid.detach()
        high = hard_high + soft_high - soft_high.detach()
        return low, mid, high

    def _split_with_boundaries(
        self,
        feature: torch.Tensor,
        tau_low: torch.Tensor,
        tau_high: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(feature, norm="ortho"),
            dim=(-2, -1),
        )
        band_index = self.spectral_splitter._radial_band_index(feature)
        low_mask, mid_mask, high_mask = self._partition_masks(
            band_index,
            tau_low,
            tau_high,
        )

        def inverse(mask: torch.Tensor) -> torch.Tensor:
            component = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)),
                norm="ortho",
            )
            return component.real

        low = inverse(low_mask)
        mid = inverse(mid_mask)
        high = inverse(high_mask)
        return {
            "low": low,
            "mid": mid,
            "high": high,
            "low_mask": low_mask,
            "mid_mask": mid_mask,
            "high_mask": high_mask,
            "partition_reconstruction": low + mid + high,
        }

    def forward(
        self,
        physical_msi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if physical_msi.shape != hr_msi.shape:
            raise ValueError(
                "physical_msi and hr_msi must share [N, M, H, W], got "
                f"{tuple(physical_msi.shape)} and {tuple(hr_msi.shape)}"
            )
        if hr_msi.ndim != 4 or hr_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected MSI [N, {self.msi_channels}, H, W], "
                f"got {tuple(hr_msi.shape)}"
            )

        physical_full = self.shared_encoder(physical_msi)
        reference_full = self.shared_encoder(hr_msi)
        boundary = self._adaptive_boundaries(physical_full, reference_full)

        physical = self._split_with_boundaries(
            physical_full,
            boundary["tau_low"],
            boundary["tau_high"],
        )
        reference = self._split_with_boundaries(
            reference_full,
            boundary["tau_low"],
            boundary["tau_high"],
        )

        low_difference = reference["low"] - physical["low"]
        mid_difference = reference["mid"] - physical["mid"]
        high_difference = reference["high"] - physical["high"]

        reference_partition_loss = F.l1_loss(
            reference["partition_reconstruction"],
            reference_full,
        )
        physical_partition_loss = F.l1_loss(
            physical["partition_reconstruction"],
            physical_full,
        )
        partition_reconstruction_loss = 0.5 * (
            reference_partition_loss + physical_partition_loss
        )
        low_frequency_alignment_loss = F.mse_loss(
            reference["low"],
            physical["low"],
        )

        n, _, h, w = high_difference.shape
        reliability_map = high_difference.new_ones((n, 1, h, w))
        noise_mask = high_difference.new_zeros((n, 1, h, w))
        noise_feature = torch.zeros_like(high_difference)
        reliable_high = high_difference
        zero_edge = high_difference.new_zeros((n, 1, h, w))
        edge_quantiles = high_difference.new_zeros((n, 4))

        reliable_detail = torch.cat(
            [mid_difference, reliable_high],
            dim=1,
        )
        symmetric_difference = torch.cat(
            [low_difference, mid_difference, reliable_high],
            dim=1,
        )

        physical_activation = self.frequency_activation_ratios(
            physical["low"],
            physical["mid"],
            physical["high"],
        )
        reference_activation = self.frequency_activation_ratios(
            reference["low"],
            reference["mid"],
            reference["high"],
        )
        difference_activation = self.frequency_activation_ratios(
            low_difference,
            mid_difference,
            high_difference,
        )

        reliability_ratio_per_sample = reliability_map.mean(dim=(1, 2, 3))
        noise_ratio_per_sample = noise_mask.mean(dim=(1, 2, 3))

        return {
            "physical_feature": physical["low"],
            "reference_feature": reference_full,
            "physical_full_feature": physical_full,
            "reference_full_feature": reference_full,
            "low_feature": reference["low"],
            "mid_feature": mid_difference,
            "high_feature": high_difference,
            "physical_low_feature": physical["low"],
            "physical_mid_feature": physical["mid"],
            "physical_high_feature": physical["high"],
            "reference_low_feature": reference["low"],
            "reference_mid_feature": reference["mid"],
            "reference_high_feature": reference["high"],
            "low_difference_feature": low_difference,
            "mid_difference_feature": mid_difference,
            "high_difference_feature": high_difference,
            "low_mid_feature": low_difference + mid_difference,
            "edge_magnitude": zero_edge,
            "edge_score": zero_edge,
            "effective_threshold": high_difference.new_zeros((n, 1, 1, 1)),
            "edge_reference_scale": high_difference.new_ones((n, 1, 1, 1)),
            "edge_quantiles": edge_quantiles,
            "noise_mask": noise_mask,
            "reliability_mask_channel": reliability_map.expand_as(
                high_difference
            ),
            "reliability_map": reliability_map,
            "noise_feature": noise_feature,
            "reliable_high_feature": reliable_high,
            "reliable_high_difference_feature": reliable_high,
            "reliable_detail_feature": reliable_detail,
            "refined_reference_feature": symmetric_difference,
            "symmetric_difference_feature": symmetric_difference,
            "noise_ratio_per_sample": noise_ratio_per_sample,
            "reliability_ratio_per_sample": reliability_ratio_per_sample,
            "noise_ratio": noise_ratio_per_sample.mean(),
            "reliability_ratio": reliability_ratio_per_sample.mean(),
            "tau_low": boundary["tau_low"],
            "tau_high": boundary["tau_high"],
            "base_tau_low": boundary["base_tau_low"],
            "base_tau_high": boundary["base_tau_high"],
            "boundary_low_shift": boundary["boundary_low_shift"],
            "boundary_high_shift": boundary["boundary_high_shift"],
            "boundary_raw_low_shift": boundary["boundary_raw_low_shift"],
            "boundary_raw_high_shift": boundary["boundary_raw_high_shift"],
            "adaptive_boundary_gap": boundary["adaptive_boundary_gap"],
            "low_mask": reference["low_mask"],
            "mid_mask": reference["mid_mask"],
            "high_mask": reference["high_mask"],
            "physical_frequency_activation_ratio": physical_activation,
            "reference_frequency_activation_ratio": reference_activation,
            "difference_frequency_activation_ratio": difference_activation,
            "frequency_activation_ratio": difference_activation,
            "low_frequency_alignment_loss": low_frequency_alignment_loss,
            "noise_minimization_loss": high_difference.new_zeros(()),
            "partition_reconstruction_loss": partition_reconstruction_loss,
            "physical_partition_reconstruction_loss": physical_partition_loss,
            "reference_partition_reconstruction_loss": reference_partition_loss,
        }


class Stage2AdaptiveFrequencyRoutingNet(Stage2SymmetricFrequencyNet):
    """Symmetric-frequency Stage 2 with sample-conditioned frequency routing."""

    def __init__(
        self,
        *args,
        adaptive_boundary_hidden_channels: int = 64,
        adaptive_boundary_max_shift: float = 2.0,
        **kwargs,
    ):
        feature_channels = int(kwargs.get("feature_channels", 64))
        encoder_blocks = int(kwargs.get("encoder_blocks", 3))
        num_frequency_bands = int(kwargs.get("num_frequency_bands", 20))
        init_low_boundary = float(kwargs.get("init_low_boundary", 5.0))
        init_high_boundary = float(kwargs.get("init_high_boundary", 18.0))
        boundary_temperature = float(kwargs.get("boundary_temperature", 0.5))
        edge_threshold_mode = str(
            kwargs.get("edge_threshold_mode", "relative")
        )
        edge_mask_threshold = float(kwargs.get("edge_mask_threshold", 0.1))
        edge_reference_quantile = float(
            kwargs.get("edge_reference_quantile", 0.9)
        )
        noise_quantile = float(kwargs.get("noise_quantile", 0.2))
        hard_partition = bool(kwargs.get("hard_partition", True))

        super().__init__(*args, **kwargs)
        self.reliability = AdaptiveSymmetricFrequencyRouter(
            msi_channels=self.msi_channels,
            feature_channels=feature_channels,
            encoder_blocks=encoder_blocks,
            num_frequency_bands=num_frequency_bands,
            init_low_boundary=init_low_boundary,
            init_high_boundary=init_high_boundary,
            boundary_temperature=boundary_temperature,
            edge_threshold_mode=edge_threshold_mode,
            edge_mask_threshold=edge_mask_threshold,
            edge_reference_quantile=edge_reference_quantile,
            noise_quantile=noise_quantile,
            hard_partition=hard_partition,
            adaptive_boundary_hidden_channels=adaptive_boundary_hidden_channels,
            adaptive_boundary_max_shift=adaptive_boundary_max_shift,
        )
