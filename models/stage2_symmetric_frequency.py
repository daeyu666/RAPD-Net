"""Dual-source symmetric SSP frequency differencing for Stage 2.

Both the physical MSI projection and observed HR-MSI are encoded by the same
encoder and decomposed by the same channel-wise SSP:

    (F_p^L, F_p^M, F_p^H) = SSP(E(Z_base))
    (F_r^L, F_r^M, F_r^H) = SSP(E(Z_hr))

The coefficient network receives same-band differences

    Delta F^L = F_r^L - F_p^L
    Delta F^M = F_r^M - F_p^M
    Delta F^H = Q * (F_r^H - F_p^H),

where Q is produced by the existing NSP from the low/mid/high difference
branches. The SRF anchor, observable/null-space heads, fusion trunk, and losses
remain otherwise unchanged.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .stage2_dual_space import Stage2DualSpaceNet
from .stage2_frequency_reliability import FrequencyReliabilityScreen


class SymmetricFrequencyReliabilityScreen(FrequencyReliabilityScreen):
    """Apply the shared SSP to both sources and expose same-band differences."""

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
        physical = self.spectral_splitter(physical_full)
        reference = self.spectral_splitter(reference_full)

        low_difference = reference["low"] - physical["low"]
        mid_difference = reference["mid"] - physical["mid"]
        high_difference = reference["high"] - physical["high"]

        # NSP now judges whether high-frequency cross-source discrepancy is
        # supported by lower-frequency cross-source discrepancy.
        noise = self.noise_splitter(
            low_difference,
            mid_difference,
            high_difference,
        )

        reference_partition_loss = F.l1_loss(
            reference["partition_reconstruction"],
            reference_full,
        )
        physical_partition_loss = F.l1_loss(
            physical["partition_reconstruction"],
            physical_full,
        )
        low_frequency_alignment_loss = F.mse_loss(
            reference["low"],
            physical["low"],
        )
        noise_minimization_loss = noise["noise_feature"].square().mean()
        partition_reconstruction_loss = 0.5 * (
            reference_partition_loss + physical_partition_loss
        )

        reliable_detail = torch.cat(
            [mid_difference, noise["reliable_high"]],
            dim=1,
        )
        symmetric_difference = torch.cat(
            [low_difference, mid_difference, noise["reliable_high"]],
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
            noise["reliable_high"],
        )

        return {
            # The inherited Stage-2 forward subtracts low_feature -
            # physical_feature. Returning the two low SSP branches here makes
            # that operation exactly Delta F^L.
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
            "low_mid_feature": noise["low_mid"],
            "edge_magnitude": noise["edge_magnitude"],
            "edge_score": noise["edge_score"],
            "effective_threshold": noise["effective_threshold"],
            "edge_reference_scale": noise["edge_reference_scale"],
            "edge_quantiles": noise["edge_quantiles"],
            "noise_mask": noise["noise_mask"],
            "reliability_mask_channel": noise["reliability_mask_channel"],
            "reliability_map": noise["reliability_map"],
            "noise_feature": noise["noise_feature"],
            "reliable_high_feature": noise["reliable_high"],
            "reliable_high_difference_feature": noise["reliable_high"],
            "reliable_detail_feature": reliable_detail,
            "refined_reference_feature": symmetric_difference,
            "symmetric_difference_feature": symmetric_difference,
            "noise_ratio_per_sample": noise["noise_ratio_per_sample"],
            "reliability_ratio_per_sample": noise[
                "reliability_ratio_per_sample"
            ],
            "noise_ratio": noise["noise_ratio"],
            "reliability_ratio": noise["reliability_ratio"],
            "tau_low": reference["tau_low"],
            "tau_high": reference["tau_high"],
            "low_mask": reference["low_mask"],
            "mid_mask": reference["mid_mask"],
            "high_mask": reference["high_mask"],
            "physical_frequency_activation_ratio": physical_activation,
            "reference_frequency_activation_ratio": reference_activation,
            "difference_frequency_activation_ratio": difference_activation,
            # Existing monitor_values reads this key.
            "frequency_activation_ratio": difference_activation,
            "low_frequency_alignment_loss": low_frequency_alignment_loss,
            "noise_minimization_loss": noise_minimization_loss,
            "partition_reconstruction_loss": partition_reconstruction_loss,
            "physical_partition_reconstruction_loss": physical_partition_loss,
            "reference_partition_reconstruction_loss": reference_partition_loss,
        }


class Stage2SymmetricFrequencyNet(Stage2DualSpaceNet):
    """Dual-space Stage 2 with dual-source symmetric SSP differences."""

    def __init__(self, *args, **kwargs):
        feature_channels = int(kwargs.get("feature_channels", 64))
        encoder_blocks = int(kwargs.get("encoder_blocks", 3))
        num_frequency_bands = int(kwargs.get("num_frequency_bands", 20))
        init_low_boundary = float(kwargs.get("init_low_boundary", 5.0))
        init_high_boundary = float(kwargs.get("init_high_boundary", 18.0))
        boundary_temperature = float(kwargs.get("boundary_temperature", 0.5))
        edge_threshold_mode = str(kwargs.get("edge_threshold_mode", "relative"))
        edge_mask_threshold = float(kwargs.get("edge_mask_threshold", 0.1))
        edge_reference_quantile = float(
            kwargs.get("edge_reference_quantile", 0.9)
        )
        noise_quantile = float(kwargs.get("noise_quantile", 0.2))
        hard_partition = bool(kwargs.get("hard_partition", True))

        super().__init__(*args, **kwargs)
        # The submodule names and parameter shapes are identical to the original
        # FrequencyReliabilityScreen, so a dual-space checkpoint loads strictly.
        self.reliability = SymmetricFrequencyReliabilityScreen(
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
        )
