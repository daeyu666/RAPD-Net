"""Stage-2 SRF analytical coefficient anchor variant.

This controlled variant keeps the existing frequency-reliability network and
coefficient residual head unchanged, but changes the deterministic coefficient
starting point from bicubic ``C_up`` to

    C_anchor = C_up + S^T (S S^T + lambda I)^(-1) (Z - R X_base),
    S = R U_r.

The frequency screen still compares the original bicubic physical MSI with the
observed HR-MSI. This is intentional: replacing the physical MSI with the
analytical anchor MSI would make the MSI discrepancy nearly zero and would
starve the learned branch of useful cross-source evidence.

The zero-MSI branch does not receive the analytical anchor, so it still falls
back to the bicubic coefficient baseline. The difference between the full and
zero-MSI predictions therefore measures the complete MSI contribution.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .stage1_spectral_basis import Stage1SpectralBasisNet
from .stage2_coefficient_residual import Stage2CoefficientResidualNet


class Stage2SRFAnchorNet(Stage2CoefficientResidualNet):
    """Existing Stage-2 network with a deterministic SRF coefficient anchor."""

    def __init__(
        self,
        stage1_model: Stage1SpectralBasisNet,
        spectral_response: torch.Tensor,
        anchor_ridge_ratio: float = 1e-3,
        anchor_normalized_clip: float = 0.0,
        **kwargs,
    ):
        if anchor_ridge_ratio <= 0:
            raise ValueError("anchor_ridge_ratio must be positive")
        if anchor_normalized_clip < 0:
            raise ValueError("anchor_normalized_clip must be non-negative")

        super().__init__(
            stage1_model=stage1_model,
            spectral_response=spectral_response,
            **kwargs,
        )
        self.anchor_ridge_ratio = float(anchor_ridge_ratio)
        self.anchor_normalized_clip = float(anchor_normalized_clip)

        with torch.no_grad():
            basis = self.stage1.get_basis().detach().float()
            reduced_response = self.spectral_response.float() @ basis
            gram = reduced_response @ reduced_response.transpose(0, 1)
            gram_scale = torch.trace(gram) / max(self.msi_channels, 1)
            actual_ridge = self.anchor_ridge_ratio * gram_scale
            regularized = gram + actual_ridge * torch.eye(
                self.msi_channels,
                dtype=gram.dtype,
                device=gram.device,
            )
            inverse = torch.linalg.solve(
                regularized,
                torch.eye(
                    self.msi_channels,
                    dtype=gram.dtype,
                    device=gram.device,
                ),
            )
            backprojector = reduced_response.transpose(0, 1) @ inverse
            observable_projector = backprojector @ reduced_response

        self.register_buffer(
            "reduced_response",
            reduced_response.detach().contiguous(),
        )
        self.register_buffer(
            "coefficient_backprojector",
            backprojector.detach().contiguous(),
        )
        self.register_buffer(
            "observable_projector",
            observable_projector.detach().contiguous(),
        )
        self.register_buffer(
            "actual_anchor_ridge",
            actual_ridge.detach().reshape(()),
        )

    def analytical_coefficient_anchor(
        self,
        msi_residual: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if msi_residual.ndim != 4 or msi_residual.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected MSI residual [N, {self.msi_channels}, H, W], got "
                f"{tuple(msi_residual.shape)}"
            )
        coefficient_residual = torch.einsum(
            "rm,nmhw->nrhw",
            self.coefficient_backprojector.to(msi_residual),
            msi_residual,
        )
        scale = self.coefficient_scale().view(1, -1, 1, 1)
        normalized = coefficient_residual / scale
        if self.anchor_normalized_clip > 0:
            normalized = normalized.clamp(
                -self.anchor_normalized_clip,
                self.anchor_normalized_clip,
            )
            coefficient_residual = normalized * scale
        projected_msi_residual = torch.einsum(
            "mr,nrhw->nmhw",
            self.reduced_response.to(msi_residual),
            coefficient_residual,
        )
        return {
            "analytic_coefficient_residual": coefficient_residual,
            "normalized_analytic_coefficient_residual": normalized,
            "analytic_projected_msi_residual": projected_msi_residual,
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
        bicubic_coefficients = F.interpolate(
            lr_coefficients,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        scale = self.coefficient_scale().view(1, -1, 1, 1)
        normalized_bicubic = bicubic_coefficients / scale
        base_hsi = self.stage1.decode(
            bicubic_coefficients,
            basis=basis,
        )
        base_msi = self.project_hsi_to_msi(base_hsi)
        msi_residual = hr_msi - base_msi

        anchor = self.analytical_coefficient_anchor(msi_residual)
        anchor_coefficients = (
            bicubic_coefficients + anchor["analytic_coefficient_residual"]
        )
        normalized_anchor = anchor_coefficients / scale
        anchor_hsi = self.stage1.decode(
            anchor_coefficients,
            basis=basis,
        )
        anchor_msi = self.project_hsi_to_msi(anchor_hsi)

        # Keep the original bicubic physical MSI for the frequency comparison.
        reliability = self.reliability(base_msi, hr_msi)
        low_discrepancy = (
            reliability["low_feature"] - reliability["physical_feature"]
        )
        correction = self._predict_normalized_residual(
            normalized_upsampled_coefficients=normalized_anchor,
            physical_feature=reliability["physical_feature"],
            low_discrepancy_feature=low_discrepancy,
            mid_feature=reliability["mid_feature"],
            reliable_high_feature=reliability["reliable_high_feature"],
            reliability_map=reliability["reliability_map"],
        )

        corrected_coefficients = (
            anchor_coefficients + correction["coefficient_residual"]
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
            "bicubic_coefficients": bicubic_coefficients,
            # Existing training utilities use this key as the residual target base.
            "upsampled_coefficients": anchor_coefficients,
            "normalized_bicubic_coefficients": normalized_bicubic,
            "normalized_upsampled_coefficients": normalized_anchor,
            "anchor_coefficients": anchor_coefficients,
            "base_hsi": base_hsi,
            "anchor_hsi": anchor_hsi,
            "base_msi": base_msi,
            "anchor_msi": anchor_msi,
            "msi_residual": msi_residual,
            "low_discrepancy_feature": low_discrepancy,
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": projected_msi,
            "actual_anchor_ridge": self.actual_anchor_ridge,
            **anchor,
            **reliability,
            **correction,
        }

        if compute_zero_msi:
            zero_feature = torch.zeros_like(reliability["mid_feature"])
            zero_map = torch.zeros_like(reliability["reliability_map"])
            zero_correction = self._predict_normalized_residual(
                normalized_upsampled_coefficients=normalized_bicubic,
                physical_feature=reliability["physical_feature"],
                low_discrepancy_feature=zero_feature,
                mid_feature=zero_feature,
                reliable_high_feature=zero_feature,
                reliability_map=zero_map,
            )
            zero_coefficients = (
                bicubic_coefficients
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
