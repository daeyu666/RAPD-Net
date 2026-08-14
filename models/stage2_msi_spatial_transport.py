"""MSI-guided spatial transport of LR-HSI null-space coefficients for Stage 2.

The HR-MSI is deliberately forbidden from synthesizing null-space coefficient
values. It predicts only a spatial redistribution kernel. The transported
spectral vectors themselves always come from the bicubic LR-HSI null-space
coefficient field:

    C_null^0 = P_null up(C_lr)
    w_p = softmax(G(Z_hr)_p)
    C_null^T(p) = sum_{q in N(p)} w_p(q) C_null^0(q)

The same scalar spatial weights are shared by all coefficient channels, so the
MSI controls where an LR-derived spectral state is moved, not what spectral
state is generated. Because spatial mixing commutes with the fixed coefficient
projector, the result remains in the exact null space (up to numerical error).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_spectral_basis import Stage1SpectralBasisNet


class TransportResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class MSILocalTransport(nn.Module):
    """Predict one shared local spatial mixing kernel per HR pixel from MSI."""

    def __init__(
        self,
        msi_channels: int,
        hidden_channels: int = 48,
        blocks: int = 3,
        kernel_size: int = 5,
        dilation: int = 2,
        identity_logit: float = 6.0,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be >= 8")
        if blocks < 1:
            raise ValueError("blocks must be >= 1")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.kernel_elements = self.kernel_size * self.kernel_size
        self.radius = self.dilation * (self.kernel_size - 1) // 2
        self.center_index = self.kernel_elements // 2
        self.identity_logit = float(identity_logit)

        self.stem = nn.Sequential(
            nn.Conv2d(msi_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[TransportResidualBlock(hidden_channels) for _ in range(blocks)]
        )
        self.logit_head = nn.Conv2d(
            hidden_channels,
            self.kernel_elements,
            3,
            padding=1,
        )

        nn.init.normal_(self.logit_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.logit_head.bias)
        with torch.no_grad():
            self.logit_head.bias[self.center_index] = self.identity_logit

        offsets = []
        half = self.kernel_size // 2
        for iy in range(-half, half + 1):
            for ix in range(-half, half + 1):
                offsets.append((iy * self.dilation, ix * self.dilation))
        offset_tensor = torch.tensor(offsets, dtype=torch.float32)
        self.register_buffer("offsets", offset_tensor, persistent=False)
        self.register_buffer(
            "offset_radius",
            torch.sqrt(offset_tensor.square().sum(dim=1)),
            persistent=False,
        )

    def forward(
        self,
        hr_msi: torch.Tensor,
        source_coefficients: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if hr_msi.ndim != 4 or source_coefficients.ndim != 4:
            raise ValueError("hr_msi and source_coefficients must be 4D tensors")
        if hr_msi.shape[-2:] != source_coefficients.shape[-2:]:
            raise ValueError(
                "MSI and coefficient fields must share spatial size, got "
                f"{tuple(hr_msi.shape[-2:])} and {tuple(source_coefficients.shape[-2:])}"
            )

        features = self.blocks(self.stem(hr_msi))
        logits = self.logit_head(features)
        weights = torch.softmax(logits, dim=1)

        radius = self.radius
        if source_coefficients.size(-2) <= radius or source_coefficients.size(-1) <= radius:
            raise ValueError(
                f"Spatial size {tuple(source_coefficients.shape[-2:])} is too small "
                f"for reflect padding radius={radius}"
            )
        padded = F.pad(
            source_coefficients,
            (radius, radius, radius, radius),
            mode="reflect",
        )
        patches = F.unfold(
            padded,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=1,
        )
        n, channels, height, width = source_coefficients.shape
        patches = patches.view(
            n,
            channels,
            self.kernel_elements,
            height,
            width,
        )
        transported = (patches * weights.unsqueeze(1)).sum(dim=2)

        eps = 1e-8
        entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=1)
        entropy = entropy / math.log(float(self.kernel_elements))
        expected_radius = (
            weights * self.offset_radius.view(1, -1, 1, 1).to(weights)
        ).sum(dim=1)
        max_weight = weights.max(dim=1).values
        center_weight = weights[:, self.center_index]

        return {
            "transported_coefficients": transported,
            "transport_weights": weights,
            "transport_logits": logits,
            "transport_center_weight": center_weight.mean(),
            "transport_max_weight": max_weight.mean(),
            "transport_entropy": entropy.mean(),
            "transport_expected_radius": expected_radius.mean(),
        }


class Stage2MSISpatialTransportNet(nn.Module):
    """SRF analytical observable anchor plus MSI-only null spatial transport."""

    def __init__(
        self,
        stage1_model: Stage1SpectralBasisNet,
        spectral_response: torch.Tensor,
        anchor_ridge_ratio: float = 1e-3,
        projector_tolerance: float = 1e-6,
        transport_hidden_channels: int = 48,
        transport_blocks: int = 3,
        transport_kernel_size: int = 5,
        transport_dilation: int = 2,
        transport_identity_logit: float = 6.0,
    ):
        super().__init__()
        if anchor_ridge_ratio <= 0:
            raise ValueError("anchor_ridge_ratio must be positive")
        if projector_tolerance <= 0:
            raise ValueError("projector_tolerance must be positive")
        if spectral_response.ndim != 2:
            raise ValueError("spectral_response must be [M, B]")

        self.stage1 = stage1_model
        self.n_bands = int(stage1_model.n_bands)
        self.basis_rank = int(stage1_model.basis_rank)
        self.msi_channels = int(spectral_response.size(0))
        self.anchor_ridge_ratio = float(anchor_ridge_ratio)
        self.projector_tolerance = float(projector_tolerance)

        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)
        self.stage1.eval()

        response = spectral_response.detach().float().contiguous()
        if response.size(1) != self.n_bands:
            raise ValueError(
                f"spectral_response bands={response.size(1)} != Stage1 bands={self.n_bands}"
            )
        self.register_buffer("spectral_response", response)

        with torch.no_grad():
            basis = self.stage1.get_basis().detach().float()
            reduced = response @ basis
            _, singular_values, vh = torch.linalg.svd(reduced, full_matrices=True)
            threshold = self.projector_tolerance * singular_values.max().clamp_min(1e-12)
            rank = int((singular_values > threshold).sum().item())
            row_basis = vh[:rank].transpose(0, 1).contiguous()
            observable = row_basis @ row_basis.transpose(0, 1)
            identity = torch.eye(
                self.basis_rank,
                dtype=observable.dtype,
                device=observable.device,
            )
            null = identity - observable

            gram = reduced @ reduced.transpose(0, 1)
            gram_scale = torch.trace(gram) / max(self.msi_channels, 1)
            actual_ridge = self.anchor_ridge_ratio * gram_scale
            regularized = gram + actual_ridge * torch.eye(
                self.msi_channels,
                dtype=gram.dtype,
                device=gram.device,
            )
            backprojector = reduced.transpose(0, 1) @ torch.linalg.solve(
                regularized,
                torch.eye(
                    self.msi_channels,
                    dtype=gram.dtype,
                    device=gram.device,
                ),
            )

        self.register_buffer("reduced_response", reduced.contiguous())
        self.register_buffer("exact_observable_projector", observable.contiguous())
        self.register_buffer("exact_null_projector", null.contiguous())
        self.register_buffer("coefficient_backprojector", backprojector.contiguous())
        self.register_buffer("observable_singular_values", singular_values.contiguous())
        self.register_buffer("observable_rank", torch.tensor(rank, dtype=torch.int64))
        self.register_buffer("actual_anchor_ridge", actual_ridge.reshape(()))

        self.transport = MSILocalTransport(
            msi_channels=self.msi_channels,
            hidden_channels=transport_hidden_channels,
            blocks=transport_blocks,
            kernel_size=transport_kernel_size,
            dilation=transport_dilation,
            identity_logit=transport_identity_logit,
        )

    @staticmethod
    def _project(
        projector: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("rk,nkhw->nrhw", projector, coefficients)

    def project_hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "mb,nbhw->nmhw",
            self.spectral_response.to(hsi),
            hsi,
        )

    def coefficient_scale(self) -> torch.Tensor:
        return self.stage1.coefficient_scale.detach().clamp_min(1e-8)

    def analytical_coefficient_residual(
        self,
        msi_residual: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "rm,nmhw->nrhw",
            self.coefficient_backprojector.to(msi_residual),
            msi_residual,
        )

    def projector_statistics(self) -> Dict[str, torch.Tensor]:
        observable = self.exact_observable_projector
        null = self.exact_null_projector
        identity = torch.eye(
            self.basis_rank,
            dtype=observable.dtype,
            device=observable.device,
        )
        return {
            "observable_projector_idempotence_error": (
                observable @ observable - observable
            ).abs().max(),
            "null_projector_idempotence_error": (
                null @ null - null
            ).abs().max(),
            "projector_complement_error": (observable + null - identity).abs().max(),
            "projector_orthogonality_error": (observable @ null).abs().max(),
            "reduced_response_null_leakage": (
                self.reduced_response @ null
            ).abs().max(),
        }

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if lr_hsi.ndim != 4 or lr_hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected LR-HSI [N, {self.n_bands}, h, w], got {tuple(lr_hsi.shape)}"
            )
        if hr_msi.ndim != 4 or hr_msi.size(1) != self.msi_channels:
            raise ValueError(
                f"Expected HR-MSI [N, {self.msi_channels}, H, W], got {tuple(hr_msi.shape)}"
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

        observable_anchor = self._project(
            self.exact_observable_projector.to(anchor_coefficients),
            anchor_coefficients,
        )
        null_seed = self._project(
            self.exact_null_projector.to(bicubic_coefficients),
            bicubic_coefficients,
        )

        transport = self.transport(hr_msi, null_seed)
        transported_null = self._project(
            self.exact_null_projector.to(transport["transported_coefficients"]),
            transport["transported_coefficients"],
        )
        corrected_coefficients = observable_anchor + transported_null
        reconstructed_hsi = self.stage1.decode(corrected_coefficients, basis=basis)
        projected_msi = self.project_hsi_to_msi(reconstructed_hsi)

        null_transport_residual = transported_null - null_seed
        scale = self.coefficient_scale().view(1, -1, 1, 1)

        return {
            "basis": basis,
            "mean_spectrum": mean_spectrum,
            "coefficient_scale": self.coefficient_scale(),
            "lr_coefficients": lr_coefficients,
            "bicubic_coefficients": bicubic_coefficients,
            "base_hsi": base_hsi,
            "base_msi": base_msi,
            "msi_residual": msi_residual,
            "analytic_coefficient_residual": analytic_residual,
            "anchor_coefficients": anchor_coefficients,
            "anchor_hsi": anchor_hsi,
            "observable_anchor_coefficients": observable_anchor,
            "null_seed_coefficients": null_seed,
            "transported_null_coefficients": transported_null,
            "null_transport_residual": null_transport_residual,
            "normalized_null_transport_residual": null_transport_residual / scale,
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": projected_msi,
            "observable_rank": self.observable_rank.to(hr_msi.device),
            "actual_anchor_ridge": self.actual_anchor_ridge,
            **{k: v for k, v in transport.items() if k != "transported_coefficients"},
            **self.projector_statistics(),
        }
