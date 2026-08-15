"""Basis-invariant LR-null tangent projection for RAPD-Net Stage 2.

The frozen LR-HSI null field still defines a local tangent subspace at each HR
pixel, but the trainable network no longer predicts coordinates in the local
SVD basis. Instead it predicts one coefficient residual proposal in the fixed
global Stage-1 coefficient coordinates. Only the component that lies inside the
LR-derived tangent subspace is allowed to affect reconstruction:

    C_null^0 = P_null up(C_lr)
    T_p = Tangent(C_null^0)
    r_tilde(p) = G(Z_hr, C_null^0, tangent descriptors)
    Delta C_null(p) = T_p T_p^T r_tilde(p)

The projector T_p T_p^T is invariant to sign flips, permutations, and arbitrary
orthogonal rotations of the local tangent basis. The model therefore removes the
local coordinate-gauge ambiguity while preserving the same LR-HSI spectral
constraint. HR-MSI can propose a change, but only LR-HSI-approved tangent
components survive.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .stage2_null_tangent_manifold import (
    Stage2NullTangentManifoldNet,
    TangentPredictorBlock,
    build_local_tangent_field,
)


class GlobalCoefficientProposalPredictor(nn.Module):
    """Predict a residual proposal in the fixed global coefficient coordinates."""

    def __init__(
        self,
        input_channels: int,
        coefficient_channels: int,
        hidden_channels: int = 96,
        blocks: int = 4,
    ):
        super().__init__()
        if hidden_channels < 16:
            raise ValueError("hidden_channels must be >= 16")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[TangentPredictorBlock(hidden_channels) for _ in range(blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, coefficient_channels, 3, padding=1)
        # Exact analytical-anchor initialization.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class Stage2TangentProjectedProposalNet(Stage2NullTangentManifoldNet):
    """Global coefficient proposal followed by a basis-invariant tangent projector."""

    def __init__(
        self,
        *args,
        proposal_amplitude_multiplier: float = 8.0,
        predictor_hidden_channels: int = 96,
        predictor_blocks: int = 4,
        **kwargs,
    ):
        if proposal_amplitude_multiplier <= 0:
            raise ValueError("proposal_amplitude_multiplier must be positive")

        # Build all analytical-anchor/null/tangent geometry from the validated
        # tangent model, then replace only the coordinate predictor.
        super().__init__(
            *args,
            tangent_amplitude_multiplier=1.0,
            predictor_hidden_channels=predictor_hidden_channels,
            predictor_blocks=predictor_blocks,
            **kwargs,
        )
        del self.coordinate_predictor
        self.proposal_amplitude_multiplier = float(proposal_amplitude_multiplier)

        # Gauge-invariant tangent descriptors:
        # - diagonal(T T^T), one value per global coefficient direction;
        # - local singular scales, ordered but independent of tangent signs.
        predictor_input_channels = (
            3 * self.msi_channels
            + self.basis_rank
            + self.basis_rank
            + self.tangent_dimension
        )
        self.proposal_predictor = GlobalCoefficientProposalPredictor(
            input_channels=predictor_input_channels,
            coefficient_channels=self.basis_rank,
            hidden_channels=predictor_hidden_channels,
            blocks=predictor_blocks,
        )

    @staticmethod
    def tangent_project(
        tangent_basis: torch.Tensor,
        proposal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a global coefficient proposal into the local tangent subspace."""
        coordinates = torch.einsum(
            "nrdhw,nrhw->ndhw",
            tangent_basis,
            proposal,
        )
        projected = torch.einsum(
            "nrdhw,ndhw->nrhw",
            tangent_basis,
            coordinates,
        )
        return projected, coordinates

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        bicubic_coefficients = torch.nn.functional.interpolate(
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

        predictor_input = torch.cat(
            [
                hr_msi,
                base_msi,
                msi_residual,
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
            **self.projector_statistics(),
        }
