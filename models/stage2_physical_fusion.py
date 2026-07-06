"""Complete Stage-2 physical fusion model for RAPD-Net.

Stage 2 freezes the Stage-1 LR-HSI unmixing model and learns only:

1. SFSR-style frequency reliability screening of HR-MSI;
2. bounded abundance-logit residual injection;
3. a positive per-pixel illumination/gain field;
4. high-resolution physical reconstruction with the frozen endmember bank.

The gain field relaxes the overly restrictive abundance-sum-to-one brightness
assumption while preserving a normalized material-composition abundance map:

    X_phy = g_hr * E * A_hr,
    A_hr >= 0, sum_k A_hr[k] = 1, g_hr > 0.

Equivalently, C_hr = g_hr * A_hr is a non-negative cone coefficient map. This
matches the representation-ceiling diagnosis without turning the abundance map
into unconstrained signed coefficients.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_unmixing import Stage1UnmixingNet
from .stage2_frequency_reliability import FrequencyReliabilityScreen


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualFusionBlock(nn.Module):
    """Compact residual block for abundance and gain correction."""

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


class Stage2PhysicalFusionNet(nn.Module):
    """Frozen physical unmixing + reliable MSI abundance/gain correction."""

    def __init__(
        self,
        stage1_model: Stage1UnmixingNet,
        spectral_response: torch.Tensor,
        feature_channels: int = 64,
        encoder_blocks: int = 3,
        fusion_channels: int = 96,
        fusion_blocks: int = 4,
        max_logit_residual: float = 1.0,
        max_log_gain: float = 0.7,
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
                f"SRF expects {spectral_response.size(1)} HSI bands, but "
                f"Stage-1 uses {stage1_model.n_bands}"
            )
        if max_logit_residual <= 0:
            raise ValueError("max_logit_residual must be positive")
        if max_log_gain <= 0:
            raise ValueError("max_log_gain must be positive")

        self.stage1 = stage1_model
        self.n_bands = int(stage1_model.n_bands)
        self.num_endmembers = int(stage1_model.num_endmembers)
        self.msi_channels = int(spectral_response.size(0))
        self.feature_channels = int(feature_channels)
        self.max_logit_residual = float(max_logit_residual)
        self.max_log_gain = float(max_log_gain)

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

        groups = _group_count(feature_channels)
        self.abundance_context = nn.Sequential(
            nn.Conv2d(
                self.num_endmembers,
                feature_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
            ResidualFusionBlock(feature_channels),
        )
        self.mid_detail_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.high_detail_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.physical_context_adapter = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.msi_residual_adapter = nn.Sequential(
            nn.Conv2d(
                self.msi_channels,
                feature_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )

        fusion_in_channels = feature_channels * 5 + 1
        fusion_groups = _group_count(fusion_channels)
        self.physical_correction_trunk = nn.Sequential(
            nn.Conv2d(
                fusion_in_channels,
                fusion_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(fusion_groups, fusion_channels),
            nn.GELU(),
            *[ResidualFusionBlock(fusion_channels) for _ in range(fusion_blocks)],
        )
        self.abundance_residual_head = nn.Conv2d(
            fusion_channels,
            self.num_endmembers,
            3,
            padding=1,
        )
        self.log_gain_head = nn.Conv2d(
            fusion_channels,
            1,
            3,
            padding=1,
        )

        # Stage 2 starts exactly from the frozen Stage-1 physical baseline:
        # abundance residual = 0 and gain = exp(0) = 1.
        nn.init.zeros_(self.abundance_residual_head.weight)
        nn.init.zeros_(self.abundance_residual_head.bias)
        nn.init.zeros_(self.log_gain_head.weight)
        nn.init.zeros_(self.log_gain_head.bias)

    def _freeze_stage1(self) -> None:
        self.stage1.eval()
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.eval()
        return self

    def project_hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        """Apply the fixed SRF/band-selection matrix: [B,H,W] -> [M,H,W]."""
        if hsi.ndim != 4 or hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected HSI [N, {self.n_bands}, H, W], got {tuple(hsi.shape)}"
            )
        return torch.einsum("mb,nbhw->nmhw", self.spectral_response, hsi)

    @staticmethod
    def reconstruct_hsi(
        endmembers: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bk,nkhw->nbhw", endmembers, coefficients)

    def _predict_physical_correction(
        self,
        upsampled_logits: torch.Tensor,
        physical_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
        msi_residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        abundance_feature = self.abundance_context(upsampled_logits)
        mid_feature = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(reliable_high_feature)
        physical_feature = self.physical_context_adapter(physical_feature)
        residual_feature = self.msi_residual_adapter(msi_residual)

        fused = torch.cat(
            [
                abundance_feature,
                physical_feature,
                mid_feature,
                high_feature,
                residual_feature,
                reliability_map,
            ],
            dim=1,
        )
        hidden = self.physical_correction_trunk(fused)

        raw_abundance_residual = self.abundance_residual_head(hidden)
        abundance_logit_residual = self.max_logit_residual * torch.tanh(
            raw_abundance_residual
        )
        corrected_logits = upsampled_logits + abundance_logit_residual
        corrected_abundance = torch.softmax(corrected_logits, dim=1)

        raw_log_gain = self.log_gain_head(hidden)
        log_gain_residual = self.max_log_gain * torch.tanh(raw_log_gain)
        gain_map = torch.exp(log_gain_residual)
        cone_coefficients = gain_map * corrected_abundance

        return {
            "raw_abundance_logit_residual": raw_abundance_residual,
            "abundance_logit_residual": abundance_logit_residual,
            "corrected_abundance_logits": corrected_logits,
            "corrected_abundance": corrected_abundance,
            "raw_log_gain": raw_log_gain,
            "log_gain_residual": log_gain_residual,
            "gain_map": gain_map,
            "cone_coefficients": cone_coefficients,
        }

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        compute_zero_msi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if lr_hsi.ndim != 4 or lr_hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected LR-HSI [N, {self.n_bands}, h, w], "
                f"got {tuple(lr_hsi.shape)}"
            )
        if hr_msi.ndim != 4 or hr_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected HR-MSI [N, {self.msi_channels}, H, W], "
                f"got {tuple(hr_msi.shape)}"
            )

        # Native-resolution unmixing is executed once. Bicubic interpolation is
        # applied only to abundance logits, never to LR-HSI for re-unmixing.
        with torch.no_grad():
            stage1_output = self.stage1(lr_hsi)
            endmembers = stage1_output["endmembers"].detach()
            lr_abundance_logits = stage1_output["abundance_logits"].detach()
            lr_abundance = stage1_output["abundance"].detach()
            lr_reconstruction = stage1_output["reconstruction"].detach()

        target_size = hr_msi.shape[-2:]
        upsampled_logits = F.interpolate(
            lr_abundance_logits,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        upsampled_abundance = torch.softmax(upsampled_logits, dim=1)
        base_hsi = self.reconstruct_hsi(endmembers, upsampled_abundance)
        base_msi = self.project_hsi_to_msi(base_hsi)
        msi_residual = hr_msi - base_msi

        reliability = self.reliability(base_msi, hr_msi)
        physical_correction = self._predict_physical_correction(
            upsampled_logits=upsampled_logits,
            physical_feature=reliability["physical_feature"],
            mid_feature=reliability["mid_feature"],
            reliable_high_feature=reliability["reliable_high_feature"],
            reliability_map=reliability["reliability_map"],
            msi_residual=msi_residual,
        )

        physical_hsi = self.reconstruct_hsi(
            endmembers,
            physical_correction["cone_coefficients"],
        )
        projected_msi = self.project_hsi_to_msi(physical_hsi)

        output = {
            "endmembers": endmembers,
            "lr_abundance_logits": lr_abundance_logits,
            "lr_abundance": lr_abundance,
            "lr_reconstruction": lr_reconstruction,
            "upsampled_abundance_logits": upsampled_logits,
            "upsampled_abundance": upsampled_abundance,
            "base_hsi": base_hsi,
            "base_msi": base_msi,
            "msi_residual": msi_residual,
            "physical_hsi": physical_hsi,
            "projected_msi": projected_msi,
            **reliability,
            **physical_correction,
        }

        if compute_zero_msi:
            zeros_feature = torch.zeros_like(reliability["mid_feature"])
            zeros_map = torch.zeros_like(reliability["reliability_map"])
            zeros_msi = torch.zeros_like(msi_residual)
            zero_correction = self._predict_physical_correction(
                upsampled_logits=upsampled_logits,
                physical_feature=reliability["physical_feature"],
                mid_feature=zeros_feature,
                reliable_high_feature=zeros_feature,
                reliability_map=zeros_map,
                msi_residual=zeros_msi,
            )
            zero_hsi = self.reconstruct_hsi(
                endmembers,
                zero_correction["cone_coefficients"],
            )
            output.update(
                {
                    "zero_msi_abundance_logit_residual": zero_correction[
                        "abundance_logit_residual"
                    ],
                    "zero_msi_abundance": zero_correction[
                        "corrected_abundance"
                    ],
                    "zero_msi_log_gain_residual": zero_correction[
                        "log_gain_residual"
                    ],
                    "zero_msi_gain_map": zero_correction["gain_map"],
                    "zero_msi_cone_coefficients": zero_correction[
                        "cone_coefficients"
                    ],
                    "zero_msi_hsi": zero_hsi,
                    "zero_msi_projected_msi": self.project_hsi_to_msi(zero_hsi),
                }
            )

        return output

    def regular_parameters(self):
        """All trainable parameters except SSP boundary parameters."""
        boundary_ids = {
            id(parameter)
            for parameter in self.reliability.spectral_boundary_parameters()
        }
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in boundary_ids:
                yield parameter

    def spectral_boundary_parameters(self):
        yield from self.reliability.spectral_boundary_parameters()
