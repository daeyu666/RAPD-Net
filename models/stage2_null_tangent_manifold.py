"""LR-HSI local tangent-manifold constrained null-space refinement for Stage 2.

HR-MSI is not allowed to synthesize an arbitrary null-space coefficient vector.
The frozen LR-HSI null field defines, at each HR pixel, a signed local tangent
basis T_p by SVD of neighbor-minus-center null coefficient differences.  A small
predictor receives MSI geometry together with the LR-derived null state and the
fixed tangent descriptors, and predicts only d tangent coordinates:

    C_null^0 = P_null up(C_lr)
    T_p, sigma_p = Tangent(C_null^0)
    a_p = kappa * sigma_p * tanh(G(Z_hr, C_null^0, T_p, sigma_p))
    Delta C_null(p) = T_p a_p

The tangent basis is sign-canonicalized per pixel so coordinate channels are
deterministic.  The final residual is projected back to the exact null space for
numerical safety.  At zero predictor output the model reproduces the analytical
SRF anchor exactly.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_spectral_basis import Stage1SpectralBasisNet


class TangentPredictorBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.act(self.norm1(residual))
        residual = self.norm2(self.conv2(residual))
        return self.act(x + residual)


@torch.no_grad()
def build_local_tangent_field(
    null_seed: torch.Tensor,
    dimension: int,
    kernel_size: int,
    dilation: int,
    chunk_pixels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sign-canonicalized LR-null tangent basis and local variation scale.

    Returns:
        tangent_basis: [N, C, d, H, W], orthonormal spectral directions.
        tangent_scale: [N, d, H, W], RMS local variation along each direction.
        singular_values: [N, d, H, W], raw local singular values.
    """
    if null_seed.ndim != 4:
        raise ValueError(f"Expected null_seed [N,C,H,W], got {tuple(null_seed.shape)}")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    if dilation < 1:
        raise ValueError("dilation must be >= 1")
    if chunk_pixels < 1:
        raise ValueError("chunk_pixels must be >= 1")

    n, channels, height, width = null_seed.shape
    elements = kernel_size * kernel_size
    max_dimension = min(channels, elements)
    if dimension > max_dimension:
        raise ValueError(
            f"dimension={dimension} exceeds local rank bound {max_dimension}"
        )

    radius = dilation * (kernel_size - 1) // 2
    if height <= radius or width <= radius:
        raise ValueError(
            f"Spatial size {(height, width)} is too small for tangent radius={radius}"
        )

    padded = F.pad(
        null_seed,
        (radius, radius, radius, radius),
        mode="reflect",
    )
    patches = F.unfold(
        padded,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=0,
        stride=1,
    )
    patches = patches.view(n, channels, elements, height, width)
    differences = patches - null_seed.unsqueeze(2)
    matrices = (
        differences.permute(0, 3, 4, 1, 2)
        .reshape(n * height * width, channels, elements)
        .contiguous()
    )

    tangent_flat = null_seed.new_zeros(
        n * height * width,
        channels,
        dimension,
    )
    singular_flat = null_seed.new_zeros(n * height * width, dimension)

    for start in range(0, matrices.size(0), chunk_pixels):
        stop = min(start + chunk_pixels, matrices.size(0))
        matrix = matrices[start:stop].float()
        u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        tangent = u[:, :, :dimension]
        singular = singular_values[:, :dimension]

        # SVD vector signs are arbitrary. Canonicalize every local direction by
        # forcing its largest-magnitude coefficient entry to be positive.
        max_indices = tangent.abs().argmax(dim=1, keepdim=True)
        pivots = torch.gather(tangent, dim=1, index=max_indices).squeeze(1)
        signs = torch.sign(pivots)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        tangent = tangent * signs.unsqueeze(1)

        tangent_flat[start:stop] = tangent.to(tangent_flat.dtype)
        singular_flat[start:stop] = singular.to(singular_flat.dtype)

    tangent_basis = (
        tangent_flat.reshape(n, height, width, channels, dimension)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    singular_field = (
        singular_flat.reshape(n, height, width, dimension)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    tangent_scale = singular_field / math.sqrt(max(elements - 1, 1))
    return tangent_basis.detach(), tangent_scale.detach(), singular_field.detach()


class TangentCoordinatePredictor(nn.Module):
    """Predict only low-dimensional tangent coordinates, never spectral values."""

    def __init__(
        self,
        input_channels: int,
        dimension: int,
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
        self.head = nn.Conv2d(hidden_channels, dimension, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class Stage2NullTangentManifoldNet(nn.Module):
    """Analytical observable anchor plus LR-null tangent-constrained generation."""

    def __init__(
        self,
        stage1_model: Stage1SpectralBasisNet,
        spectral_response: torch.Tensor,
        anchor_ridge_ratio: float = 1e-3,
        projector_tolerance: float = 1e-6,
        tangent_dimension: int = 4,
        tangent_kernel_size: int = 5,
        tangent_dilation: int = 2,
        tangent_chunk_pixels: int = 2048,
        tangent_amplitude_multiplier: float = 8.0,
        predictor_hidden_channels: int = 96,
        predictor_blocks: int = 4,
    ):
        super().__init__()
        if anchor_ridge_ratio <= 0:
            raise ValueError("anchor_ridge_ratio must be positive")
        if projector_tolerance <= 0:
            raise ValueError("projector_tolerance must be positive")
        if tangent_amplitude_multiplier <= 0:
            raise ValueError("tangent_amplitude_multiplier must be positive")
        if spectral_response.ndim != 2:
            raise ValueError("spectral_response must be [M,B]")

        self.stage1 = stage1_model
        self.n_bands = int(stage1_model.n_bands)
        self.basis_rank = int(stage1_model.basis_rank)
        self.msi_channels = int(spectral_response.size(0))
        self.anchor_ridge_ratio = float(anchor_ridge_ratio)
        self.projector_tolerance = float(projector_tolerance)
        self.tangent_dimension = int(tangent_dimension)
        self.tangent_kernel_size = int(tangent_kernel_size)
        self.tangent_dilation = int(tangent_dilation)
        self.tangent_chunk_pixels = int(tangent_chunk_pixels)
        self.tangent_amplitude_multiplier = float(tangent_amplitude_multiplier)

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

        predictor_input_channels = (
            3 * self.msi_channels
            + self.basis_rank
            + self.basis_rank * self.tangent_dimension
            + self.tangent_dimension
        )
        self.coordinate_predictor = TangentCoordinatePredictor(
            input_channels=predictor_input_channels,
            dimension=self.tangent_dimension,
            hidden_channels=predictor_hidden_channels,
            blocks=predictor_blocks,
        )

    @staticmethod
    def _project(projector: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return torch.einsum("rk,nkhw->nrhw", projector, coefficients)

    def project_hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "mb,nbhw->nmhw",
            self.spectral_response.to(hsi),
            hsi,
        )

    def coefficient_scale(self) -> torch.Tensor:
        return self.stage1.coefficient_scale.detach().clamp_min(1e-8)

    def analytical_coefficient_residual(self, msi_residual: torch.Tensor) -> torch.Tensor:
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
            "reduced_response_null_leakage": (self.reduced_response @ null).abs().max(),
        }

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
        tangent_basis_channels = tangent_basis.reshape(
            tangent_basis.size(0),
            self.basis_rank * self.tangent_dimension,
            tangent_basis.size(-2),
            tangent_basis.size(-1),
        )
        global_scale = coefficient_scale.mean().clamp_min(1e-8)
        normalized_tangent_scale = tangent_scale / global_scale

        predictor_input = torch.cat(
            [
                hr_msi,
                base_msi,
                msi_residual,
                normalized_null_seed,
                tangent_basis_channels,
                normalized_tangent_scale,
            ],
            dim=1,
        )
        raw_coordinates = self.coordinate_predictor(predictor_input)
        normalized_coordinates = torch.tanh(raw_coordinates)
        coordinate_limit = (
            self.tangent_amplitude_multiplier * tangent_scale
        ).clamp_min(global_scale * 1e-6)
        tangent_coordinates = normalized_coordinates * coordinate_limit

        tangent_residual = torch.einsum(
            "nrdhw,ndhw->nrhw",
            tangent_basis,
            tangent_coordinates,
        )
        tangent_residual = self._project(
            self.exact_null_projector.to(tangent_residual),
            tangent_residual,
        )

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
            "tangent_coordinate_limit": coordinate_limit,
            "raw_tangent_coordinates": raw_coordinates,
            "normalized_tangent_coordinates": normalized_coordinates,
            "tangent_coordinates": tangent_coordinates,
            "tangent_residual": tangent_residual,
            "corrected_coefficients": corrected_coefficients,
            "reconstructed_hsi": reconstructed_hsi,
            "projected_msi": projected_msi,
            "observable_rank": self.observable_rank.to(hr_msi.device),
            "actual_anchor_ridge": self.actual_anchor_ridge,
            **self.projector_statistics(),
        }
