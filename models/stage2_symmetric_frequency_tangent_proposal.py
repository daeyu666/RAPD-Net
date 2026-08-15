"""Symmetric-frequency guided tangent-projected proposal for RAPD-Net Stage 2.

The analytical SRF anchor handles the directly MSI-observable coefficient
component. The remaining null-space correction is still constrained by the
LR-HSI-derived local tangent projector, but the proposal predictor no longer
uses raw ``[HR-MSI, base-MSI, residual]`` channels as its spatial cue. Instead,
a shared MSI encoder and a shared channel-wise spectral splitter build
low/mid/high same-band differences between the bicubic physical MSI and the
observed HR-MSI:

    F_p = E(Z_base), F_r = E(Z_hr)
    (F_p^L,F_p^M,F_p^H) = SSP(F_p)
    (F_r^L,F_r^M,F_r^H) = SSP(F_r)
    D^b = F_r^b - F_p^b, b in {L,M,H}

No NSP, reliability mask, adaptive input-conditioned boundary, or
observable-to-null routing is used. The three symmetric difference branches are
projected back to three MSI-sized guidance blocks and replace the three raw MSI
blocks in the previous tangent-projected proposal model. The final correction
remains

    Delta C_null(p) = T_p T_p^T r_tilde(p),

so HR-MSI supplies spatial discrepancy evidence while LR-HSI controls the
admissible spectral directions.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_frequency_reliability import (
    ChannelWiseSpectralSplitter,
    SharedMSIFeatureEncoder,
)
from .stage2_null_tangent_manifold import build_local_tangent_field
from .stage2_tangent_projected_proposal import Stage2TangentProjectedProposalNet


class SymmetricFrequencyDifferenceGuidance(nn.Module):
    """Shared-encoder/shared-SSP low/mid/high cross-observation differences."""

    def __init__(
        self,
        msi_channels: int,
        feature_channels: int = 64,
        encoder_blocks: int = 3,
        num_frequency_bands: int = 20,
        init_low_boundary: float = 5.0,
        init_high_boundary: float = 18.0,
        boundary_temperature: float = 0.5,
        hard_partition: bool = True,
    ) -> None:
        super().__init__()
        self.msi_channels = int(msi_channels)
        self.feature_channels = int(feature_channels)
        self.shared_encoder = SharedMSIFeatureEncoder(
            in_channels=self.msi_channels,
            feature_channels=self.feature_channels,
            num_blocks=encoder_blocks,
        )
        self.spectral_splitter = ChannelWiseSpectralSplitter(
            channels=self.feature_channels,
            num_frequency_bands=num_frequency_bands,
            init_low_boundary=init_low_boundary,
            init_high_boundary=init_high_boundary,
            boundary_temperature=boundary_temperature,
            hard_partition=hard_partition,
        )

        # Use one shared adapter for L/M/H so only the frequency content changes;
        # the three output blocks have exactly the same channel count as the
        # [HR-MSI, base-MSI, MSI-residual] blocks they replace.
        self.band_adapter = nn.Conv2d(
            self.feature_channels,
            self.msi_channels,
            kernel_size=1,
            bias=True,
        )

    @staticmethod
    def _activation_share(
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.stack(
            [
                low.detach().abs().mean(),
                mid.detach().abs().mean(),
                high.detach().abs().mean(),
            ]
        )
        return values / values.sum().clamp_min(1e-12)

    def forward(
        self,
        physical_msi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if physical_msi.shape != hr_msi.shape:
            raise ValueError(
                "physical_msi and hr_msi must have identical shape, got "
                f"{tuple(physical_msi.shape)} and {tuple(hr_msi.shape)}"
            )
        if physical_msi.ndim != 4 or physical_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected MSI [N,{self.msi_channels},H,W], got "
                f"{tuple(physical_msi.shape)}"
            )

        physical_full = self.shared_encoder(physical_msi)
        reference_full = self.shared_encoder(hr_msi)
        physical = self.spectral_splitter(physical_full)
        reference = self.spectral_splitter(reference_full)

        low_difference = reference["low"] - physical["low"]
        mid_difference = reference["mid"] - physical["mid"]
        high_difference = reference["high"] - physical["high"]

        low_guidance = self.band_adapter(low_difference)
        mid_guidance = self.band_adapter(mid_difference)
        high_guidance = self.band_adapter(high_difference)

        physical_partition_loss = F.l1_loss(
            physical["partition_reconstruction"], physical_full
        )
        reference_partition_loss = F.l1_loss(
            reference["partition_reconstruction"], reference_full
        )

        return {
            "physical_frequency_feature": physical_full,
            "reference_frequency_feature": reference_full,
            "physical_low_feature": physical["low"],
            "physical_mid_feature": physical["mid"],
            "physical_high_feature": physical["high"],
            "reference_low_feature": reference["low"],
            "reference_mid_feature": reference["mid"],
            "reference_high_feature": reference["high"],
            "low_difference_feature": low_difference,
            "mid_difference_feature": mid_difference,
            "high_difference_feature": high_difference,
            "low_frequency_guidance": low_guidance,
            "mid_frequency_guidance": mid_guidance,
            "high_frequency_guidance": high_guidance,
            "difference_activation_share": self._activation_share(
                low_difference, mid_difference, high_difference
            ),
            "physical_activation_share": self._activation_share(
                physical["low"], physical["mid"], physical["high"]
            ),
            "reference_activation_share": self._activation_share(
                reference["low"], reference["mid"], reference["high"]
            ),
            "tau_low": reference["tau_low"],
            "tau_high": reference["tau_high"],
            "physical_partition_reconstruction_loss": physical_partition_loss,
            "reference_partition_reconstruction_loss": reference_partition_loss,
        }

    def boundary_parameters(self):
        yield self.spectral_splitter.low_boundary_raw
        yield self.spectral_splitter.high_gap_raw

    def regular_parameters(self):
        boundary_ids = {id(parameter) for parameter in self.boundary_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in boundary_ids:
                yield parameter


class Stage2SymmetricFrequencyTangentProposalNet(Stage2TangentProjectedProposalNet):
    """Symmetric-frequency spatial evidence plus LR-null tangent projection."""

    def __init__(
        self,
        *args,
        frequency_feature_channels: int = 64,
        frequency_encoder_blocks: int = 3,
        num_frequency_bands: int = 20,
        init_low_boundary: float = 5.0,
        init_high_boundary: float = 18.0,
        boundary_temperature: float = 0.5,
        hard_frequency_partition: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frequency_guidance = SymmetricFrequencyDifferenceGuidance(
            msi_channels=self.msi_channels,
            feature_channels=frequency_feature_channels,
            encoder_blocks=frequency_encoder_blocks,
            num_frequency_bands=num_frequency_bands,
            init_low_boundary=init_low_boundary,
            init_high_boundary=init_high_boundary,
            boundary_temperature=boundary_temperature,
            hard_partition=hard_frequency_partition,
        )

    def regular_trainable_parameters(self):
        # Proposal predictor plus non-boundary frequency parameters.
        yield from self.proposal_predictor.parameters()
        yield from self.frequency_guidance.regular_parameters()

    def frequency_boundary_parameters(self):
        yield from self.frequency_guidance.boundary_parameters()

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if lr_hsi.ndim != 4 or lr_hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected LR-HSI [N,{self.n_bands},h,w], got {tuple(lr_hsi.shape)}"
            )
        if hr_msi.ndim != 4 or hr_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected HR-MSI [N,{self.msi_channels},H,W], got {tuple(hr_msi.shape)}"
            )

        with torch.no_grad():
            basis = self.stage1.get_basis().detach()
            mean_spectrum = self.stage1.mean_spectrum.detach()
            lr_coefficients = self.stage1.encode(lr_hsi, basis=basis).detach()

        target_size = hr_msi.shape[-2:]
        bicubic_coefficients = F.interpolate(
            lr_coefficients,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        base_hsi = self.stage1.decode(bicubic_coefficients, basis=basis)
        base_msi = self.project_hsi_to_msi(base_hsi)
        msi_residual = hr_msi - base_msi

        analytic_residual = self.analytical_coefficient_residual(msi_residual)
        anchor_coefficients = bicubic_coefficients + analytic_residual
        anchor_hsi = self.stage1.decode(anchor_coefficients, basis=basis)

        null_seed = self._project(
            self.exact_null_projector.to(bicubic_coefficients),
            bicubic_coefficients,
        )
        tangent_basis, tangent_scale, tangent_singular_values = build_local_tangent_field(
            null_seed=null_seed,
            dimension=self.tangent_dimension,
            kernel_size=self.tangent_kernel_size,
            dilation=self.tangent_dilation,
            chunk_pixels=self.tangent_chunk_pixels,
        )

        coefficient_scale = self.coefficient_scale()
        normalized_null_seed = null_seed / coefficient_scale.view(1, -1, 1, 1)
        global_scale = coefficient_scale.mean().clamp_min(1e-8)
        normalized_tangent_scale = tangent_scale / global_scale
        tangent_projector_diagonal = tangent_basis.square().sum(dim=2)

        frequency = self.frequency_guidance(base_msi, hr_msi)
        predictor_input = torch.cat(
            [
                frequency["low_frequency_guidance"],
                frequency["mid_frequency_guidance"],
                frequency["high_frequency_guidance"],
                normalized_null_seed,
                tangent_projector_diagonal,
                normalized_tangent_scale,
            ],
            dim=1,
        )

        raw_proposal = self.proposal_predictor(predictor_input)
        normalized_proposal = torch.tanh(raw_proposal)
        proposal_limit = (
            self.proposal_amplitude_multiplier
            * coefficient_scale.view(1, -1, 1, 1)
        )
        coefficient_proposal = normalized_proposal * proposal_limit

        tangent_projected, tangent_coordinates = self.tangent_project(
            tangent_basis,
            coefficient_proposal,
        )
        tangent_residual = self._project(
            self.exact_null_projector.to(tangent_projected),
            tangent_projected,
        )
        off_tangent_proposal = coefficient_proposal - tangent_projected

        proposal_energy = coefficient_proposal.double().square().sum()
        tangent_energy = tangent_projected.double().square().sum()
        off_tangent_energy = off_tangent_proposal.double().square().sum()
        tangent_projection_energy_ratio = (
            tangent_energy / proposal_energy.clamp_min(1e-30)
        ).to(coefficient_proposal.dtype)
        off_tangent_energy_ratio = (
            off_tangent_energy / proposal_energy.clamp_min(1e-30)
        ).to(coefficient_proposal.dtype)
        proposal_saturation_ratio = (
            normalized_proposal.detach().abs() > 0.98
        ).float().mean()

        corrected_coefficients = anchor_coefficients + tangent_residual
        reconstructed_hsi = self.stage1.decode(corrected_coefficients, basis=basis)
        projected_msi = self.project_hsi_to_msi(reconstructed_hsi)

        return {
            "basis": basis,
            "mean_spectrum": mean_spectrum,
            "coefficient_scale": coefficient_scale,
            "lr_coefficients": lr_coefficients,
            "bicubic_coefficients": bicubic_coefficients,
            "base_hsi": base_hsi,
            "base_msi": base_msi,
            "msi_residual": msi_residual,
            "analytic_coefficient_residual": analytic_residual,
            "anchor_coefficients": anchor_coefficients,
            "anchor_hsi": anchor_hsi,
            "null_seed_coefficients": null_seed,
            "tangent_basis": tangent_basis,
            "tangent_scale": tangent_scale,
            "tangent_singular_values": tangent_singular_values,
            "tangent_projector_diagonal": tangent_projector_diagonal,
            "raw_global_coefficient_proposal": raw_proposal,
            "normalized_global_coefficient_proposal": normalized_proposal,
            "global_coefficient_proposal": coefficient_proposal,
            "proposal_limit": proposal_limit,
            "proposal_tangent_coordinates": tangent_coordinates,
            "tangent_projected_proposal": tangent_projected,
            "off_tangent_proposal": off_tangent_proposal,
            "tangent_residual": tangent_residual,
            "tangent_projection_energy_ratio": tangent_projection_energy_ratio,
            "off_tangent_energy_ratio": off_tangent_energy_ratio,
            "proposal_saturation_ratio": proposal_saturation_ratio,
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": projected_msi,
            "observable_rank": self.observable_rank.to(hr_msi.device),
            "actual_anchor_ridge": self.actual_anchor_ridge,
            **frequency,
            **self.projector_statistics(),
        }
