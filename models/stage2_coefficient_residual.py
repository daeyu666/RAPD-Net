"""New Stage-2 coefficient residual injection for RAPD-Net.

The model consumes the frozen Stage-1 affine spectral basis and reconstructs
high-resolution HSI as

    X_2 = mu + U_r (C_up + Delta C_rel).

``Delta C_rel`` is a signed coefficient residual predicted from:

* the upsampled LR spectral coefficients;
* the physical MSI projection of the Stage-1 base reconstruction;
* SFSR-style low/mid/high frequency features from the observed HR-MSI;
* the NSP reliability map.

The raw HR-MSI or its unfiltered high-frequency residual is deliberately not
fed through a bypass branch. MSI spatial information must pass through the
frequency reliability structure before changing spectral coefficients.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_spectral_basis import Stage1SpectralBasisNet
from .stage2_frequency_reliability import FrequencyReliabilityScreen


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CoefficientResidualBlock(nn.Module):
    """Resolution-preserving residual block used by the coefficient head."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class Stage2CoefficientResidualNet(nn.Module):
    """Frequency-reliable signed spectral coefficient residual injection."""

    def __init__(
        self,
        stage1_model: Stage1SpectralBasisNet,
        spectral_response: torch.Tensor,
        feature_channels: int = 64,
        encoder_blocks: int = 3,
        fusion_channels: int = 96,
        fusion_blocks: int = 4,
        max_normalized_residual: float = 6.0,
        coefficient_scale_floor: float = 1e-4,
        num_frequency_bands: int = 20,
        init_low_boundary: float = 5.0,
        init_high_boundary: float = 18.0,
        boundary_temperature: float = 0.5,
        edge_threshold_mode: str = "relative",
        edge_mask_threshold: float = 0.1,
        edge_reference_quantile: float = 0.9,
        noise_quantile: float = 0.2,
        hard_partition: bool = True,
    ):
        super().__init__()
        if spectral_response.ndim != 2:
            raise ValueError(
                "spectral_response must be [M, B], got "
                f"{tuple(spectral_response.shape)}"
            )
        if spectral_response.size(1) != stage1_model.n_bands:
            raise ValueError(
                f"SRF has {spectral_response.size(1)} HSI bands, but Stage 1 "
                f"uses {stage1_model.n_bands}"
            )
        if max_normalized_residual <= 0:
            raise ValueError("max_normalized_residual must be positive")
        if coefficient_scale_floor <= 0:
            raise ValueError("coefficient_scale_floor must be positive")

        self.stage1 = stage1_model
        self.n_bands = int(stage1_model.n_bands)
        self.basis_rank = int(stage1_model.basis_rank)
        self.msi_channels = int(spectral_response.size(0))
        self.feature_channels = int(feature_channels)
        self.max_normalized_residual = float(max_normalized_residual)
        self.coefficient_scale_floor = float(coefficient_scale_floor)

        self.register_buffer(
            "spectral_response",
            spectral_response.detach().float().contiguous(),
        )
        self._freeze_stage1()

        self.reliability = FrequencyReliabilityScreen(
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

        feature_groups = _group_count(feature_channels)
        self.coefficient_context = nn.Sequential(
            nn.Conv2d(
                self.basis_rank,
                feature_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(feature_groups, feature_channels),
            nn.GELU(),
            CoefficientResidualBlock(feature_channels),
        )
        self.physical_context_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(feature_groups, feature_channels),
            nn.GELU(),
        )
        self.low_discrepancy_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(feature_groups, feature_channels),
            nn.GELU(),
        )
        self.mid_detail_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(feature_groups, feature_channels),
            nn.GELU(),
        )
        self.high_detail_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(feature_groups, feature_channels),
            nn.GELU(),
        )

        fusion_in_channels = feature_channels * 5 + 1
        fusion_groups = _group_count(fusion_channels)
        self.fusion_trunk = nn.Sequential(
            nn.Conv2d(
                fusion_in_channels,
                fusion_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(fusion_groups, fusion_channels),
            nn.GELU(),
            *[
                CoefficientResidualBlock(fusion_channels)
                for _ in range(fusion_blocks)
            ],
        )
        self.normalized_residual_head = nn.Conv2d(
            fusion_channels,
            self.basis_rank,
            3,
            padding=1,
        )
        # The model starts exactly from bicubic coefficient upsampling.
        nn.init.zeros_(self.normalized_residual_head.weight)
        nn.init.zeros_(self.normalized_residual_head.bias)

    def _freeze_stage1(self) -> None:
        self.stage1.eval()
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.eval()
        return self

    def coefficient_scale(self) -> torch.Tensor:
        """Per-coordinate scale with a floor for numerically tiny PCA axes."""
        return self.stage1.coefficient_scale.detach().clamp_min(
            self.coefficient_scale_floor
        )

    def project_hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        if hsi.ndim != 4 or hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected HSI [N, {self.n_bands}, H, W], got {tuple(hsi.shape)}"
            )
        return torch.einsum("mb,nbhw->nmhw", self.spectral_response, hsi)

    def _predict_normalized_residual(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coefficient_feature = self.coefficient_context(
            normalized_upsampled_coefficients
        )
        physical_feature = self.physical_context_adapter(physical_feature)
        low_feature = self.low_discrepancy_adapter(low_discrepancy_feature)
        mid_feature = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(reliable_high_feature)

        fused = torch.cat(
            [
                coefficient_feature,
                physical_feature,
                low_feature,
                mid_feature,
                high_feature,
                reliability_map,
            ],
            dim=1,
        )
        hidden = self.fusion_trunk(fused)
        raw = self.normalized_residual_head(hidden)
        normalized_residual = self.max_normalized_residual * torch.tanh(raw)
        scale = self.coefficient_scale().view(1, -1, 1, 1)
        coefficient_residual = normalized_residual * scale
        return {
            "raw_normalized_coefficient_residual": raw,
            "normalized_coefficient_residual": normalized_residual,
            "coefficient_residual": coefficient_residual,
        }

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        compute_zero_msi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if lr_hsi.ndim != 4 or lr_hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected LR-HSI [N, {self.n_bands}, h, w], got "
                f"{tuple(lr_hsi.shape)}"
            )
        if hr_msi.ndim != 4 or hr_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected HR-MSI [N, {self.msi_channels}, H, W], got "
                f"{tuple(hr_msi.shape)}"
            )

        with torch.no_grad():
            basis = self.stage1.get_basis().detach()
            mean_spectrum = self.stage1.mean_spectrum.detach()
            lr_coefficients = self.stage1.encode(
                lr_hsi,
                basis=basis,
            ).detach()
            lr_reconstruction = self.stage1.decode(
                lr_coefficients,
                basis=basis,
            ).detach()

        target_size = hr_msi.shape[-2:]
        upsampled_coefficients = F.interpolate(
            lr_coefficients,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        scale = self.coefficient_scale().view(1, -1, 1, 1)
        normalized_upsampled = upsampled_coefficients / scale
        base_hsi = self.stage1.decode(
            upsampled_coefficients,
            basis=basis,
        )
        base_msi = self.project_hsi_to_msi(base_hsi)

        reliability = self.reliability(base_msi, hr_msi)
        low_discrepancy = (
            reliability["low_feature"] - reliability["physical_feature"]
        )
        correction = self._predict_normalized_residual(
            normalized_upsampled_coefficients=normalized_upsampled,
            physical_feature=reliability["physical_feature"],
            low_discrepancy_feature=low_discrepancy,
            mid_feature=reliability["mid_feature"],
            reliable_high_feature=reliability["reliable_high_feature"],
            reliability_map=reliability["reliability_map"],
        )

        corrected_coefficients = (
            upsampled_coefficients + correction["coefficient_residual"]
        )
        reconstructed_hsi = self.stage1.decode(
            corrected_coefficients,
            basis=basis,
        )
        projected_msi = self.project_hsi_to_msi(reconstructed_hsi)

        output = {
            "basis": basis,
            "mean_spectrum": mean_spectrum,
            "coefficient_scale": self.coefficient_scale(),
            "lr_coefficients": lr_coefficients,
            "lr_reconstruction": lr_reconstruction,
            "upsampled_coefficients": upsampled_coefficients,
            "normalized_upsampled_coefficients": normalized_upsampled,
            "base_hsi": base_hsi,
            "base_msi": base_msi,
            "low_discrepancy_feature": low_discrepancy,
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": projected_msi,
            **reliability,
            **correction,
        }

        if compute_zero_msi:
            zero_feature = torch.zeros_like(reliability["mid_feature"])
            zero_map = torch.zeros_like(reliability["reliability_map"])
            zero_correction = self._predict_normalized_residual(
                normalized_upsampled_coefficients=normalized_upsampled,
                physical_feature=reliability["physical_feature"],
                low_discrepancy_feature=zero_feature,
                mid_feature=zero_feature,
                reliable_high_feature=zero_feature,
                reliability_map=zero_map,
            )
            zero_coefficients = (
                upsampled_coefficients
                + zero_correction["coefficient_residual"]
            )
            zero_hsi = self.stage1.decode(
                zero_coefficients,
                basis=basis,
            )
            output.update(
                {
                    "zero_msi_normalized_coefficient_residual": zero_correction[
                        "normalized_coefficient_residual"
                    ],
                    "zero_msi_coefficient_residual": zero_correction[
                        "coefficient_residual"
                    ],
                    "zero_msi_coefficients": zero_coefficients,
                    "zero_msi_hsi": zero_hsi,
                    "zero_msi_projected_msi": self.project_hsi_to_msi(zero_hsi),
                }
            )

        return output

    def regular_parameters(self) -> Iterable[nn.Parameter]:
        boundary_ids = {
            id(parameter)
            for parameter in self.reliability.spectral_boundary_parameters()
        }
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in boundary_ids:
                yield parameter

    def spectral_boundary_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.reliability.spectral_boundary_parameters()
