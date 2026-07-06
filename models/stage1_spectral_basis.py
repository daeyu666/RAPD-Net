"""Stage-1 scene-adaptive spectral basis extraction for RAPD-Net.

This stage uses LR-HSI only. It replaces the former endmember/abundance model
with an affine orthogonal spectral subspace:

    C_lr = U_r^T (Y_lr - mu)
    Y_hat = mu + U_r C_lr

where ``U_r`` has orthonormal columns. The basis is initialized from PCA over
training LR-HSI spectra and can then be refined by reconstruction, SAM, and
spectral-shape objectives. No HR-MSI or HR-HSI supervision is used.

The model deliberately separates three concepts:

* ``mu``: a fixed scene mean spectrum estimated from training LR-HSI;
* ``U_r``: a learnable orthonormal spectral basis defining the scene subspace;
* ``C_lr``: signed low-resolution spectral coefficients obtained by exact
  projection, not by a separately learned encoder.

The basis vectors are mathematical coordinates, not physical endmember curves.
Their signs are canonicalized only to keep exported coefficient channels stable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class Stage1SpectralBasisNet(nn.Module):
    """Learn an affine orthogonal spectral basis from LR-HSI only."""

    def __init__(
        self,
        n_bands: int,
        basis_rank: int = 32,
        eps: float = 1e-8,
    ):
        super().__init__()
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2")
        if basis_rank < 1 or basis_rank > n_bands:
            raise ValueError(
                f"basis_rank must be in [1, {n_bands}], got {basis_rank}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.n_bands = int(n_bands)
        self.basis_rank = int(basis_rank)
        self.eps = float(eps)

        initial = torch.randn(n_bands, basis_rank)
        initial, _ = torch.linalg.qr(initial, mode="reduced")
        self.raw_basis = nn.Parameter(initial)

        self.register_buffer(
            "mean_spectrum",
            torch.zeros(n_bands, dtype=torch.float32),
        )
        self.register_buffer(
            "reference_basis",
            initial.detach().clone(),
        )
        self.register_buffer(
            "coefficient_scale",
            torch.ones(basis_rank, dtype=torch.float32),
        )
        self.register_buffer(
            "pca_eigenvalues",
            torch.zeros(basis_rank, dtype=torch.float32),
        )
        self.register_buffer(
            "total_variance",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "initialized_from_pca",
            torch.tensor(False, dtype=torch.bool),
        )

    @staticmethod
    def _canonicalize_signs(basis: torch.Tensor) -> torch.Tensor:
        """Make the largest-magnitude entry of every basis vector positive.

        A basis and its sign-flipped version span the same subspace. Detached
        signs keep exported coefficient channels stable without changing the
        subspace gradient.
        """
        rank = basis.size(1)
        indices = basis.abs().argmax(dim=0)
        columns = torch.arange(rank, device=basis.device)
        signs = torch.sign(basis[indices, columns]).detach()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return basis * signs.view(1, -1)

    def get_basis(self) -> torch.Tensor:
        """Return the current orthonormal basis in layout ``[B, r]``."""
        basis, _ = torch.linalg.qr(self.raw_basis, mode="reduced")
        return self._canonicalize_signs(basis)

    def get_projector(self) -> torch.Tensor:
        """Return the orthogonal spectral projector ``P = U U^T``."""
        basis = self.get_basis()
        return basis @ basis.transpose(0, 1)

    def get_reference_projector(self) -> torch.Tensor:
        reference = self.reference_basis
        return reference @ reference.transpose(0, 1)

    @torch.no_grad()
    def initialize_from_pca(
        self,
        mean_spectrum: torch.Tensor,
        basis: torch.Tensor,
        coefficient_scale: torch.Tensor,
        eigenvalues: torch.Tensor,
        total_variance: torch.Tensor | float,
    ) -> None:
        """Initialize the affine basis from LR-HSI PCA statistics."""
        mean_spectrum = mean_spectrum.reshape(-1)
        coefficient_scale = coefficient_scale.reshape(-1)
        eigenvalues = eigenvalues.reshape(-1)

        if tuple(mean_spectrum.shape) != (self.n_bands,):
            raise ValueError(
                f"mean_spectrum must be [{self.n_bands}], got "
                f"{tuple(mean_spectrum.shape)}"
            )
        if tuple(basis.shape) != (self.n_bands, self.basis_rank):
            raise ValueError(
                f"basis must be [{self.n_bands}, {self.basis_rank}], got "
                f"{tuple(basis.shape)}"
            )
        if tuple(coefficient_scale.shape) != (self.basis_rank,):
            raise ValueError(
                f"coefficient_scale must be [{self.basis_rank}], got "
                f"{tuple(coefficient_scale.shape)}"
            )
        if tuple(eigenvalues.shape) != (self.basis_rank,):
            raise ValueError(
                f"eigenvalues must be [{self.basis_rank}], got "
                f"{tuple(eigenvalues.shape)}"
            )

        basis = basis.to(self.raw_basis)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        basis = self._canonicalize_signs(basis)

        self.raw_basis.copy_(basis)
        self.mean_spectrum.copy_(mean_spectrum.to(self.mean_spectrum))
        self.reference_basis.copy_(basis.detach())
        self.coefficient_scale.copy_(
            coefficient_scale.to(self.coefficient_scale).clamp_min(self.eps)
        )
        self.pca_eigenvalues.copy_(
            eigenvalues.to(self.pca_eigenvalues).clamp_min(0.0)
        )
        self.total_variance.copy_(
            torch.as_tensor(
                total_variance,
                dtype=self.total_variance.dtype,
                device=self.total_variance.device,
            ).clamp_min(0.0)
        )
        self.initialized_from_pca.fill_(True)

    @torch.no_grad()
    def set_coefficient_scale(self, scale: torch.Tensor) -> None:
        scale = scale.reshape(-1)
        if tuple(scale.shape) != (self.basis_rank,):
            raise ValueError(
                f"scale must be [{self.basis_rank}], got {tuple(scale.shape)}"
            )
        self.coefficient_scale.copy_(
            scale.to(self.coefficient_scale).clamp_min(self.eps)
        )

    def encode(
        self,
        lr_hsi: torch.Tensor,
        basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project LR-HSI onto the signed orthogonal spectral coordinates."""
        self._validate_hsi(lr_hsi)
        if basis is None:
            basis = self.get_basis()
        centered = lr_hsi - self.mean_spectrum.view(1, -1, 1, 1)
        return torch.einsum("br,nbhw->nrhw", basis, centered)

    def decode(
        self,
        coefficients: torch.Tensor,
        basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode signed coefficients back to the HSI spectral space."""
        if coefficients.ndim != 4 or coefficients.size(1) != self.basis_rank:
            raise ValueError(
                f"Expected coefficients [N, {self.basis_rank}, H, W], got "
                f"{tuple(coefficients.shape)}"
            )
        if basis is None:
            basis = self.get_basis()
        reconstruction = torch.einsum(
            "br,nrhw->nbhw", basis, coefficients
        )
        return reconstruction + self.mean_spectrum.view(1, -1, 1, 1)

    def project_in_basis(self, hsi_residual: torch.Tensor) -> torch.Tensor:
        """Project a spectral residual onto ``span(U_r)``."""
        self._validate_hsi(hsi_residual)
        basis = self.get_basis()
        coefficients = torch.einsum(
            "br,nbhw->nrhw", basis, hsi_residual
        )
        return torch.einsum("br,nrhw->nbhw", basis, coefficients)

    def project_out_of_basis(self, hsi_residual: torch.Tensor) -> torch.Tensor:
        """Project a spectral residual onto the orthogonal complement."""
        return hsi_residual - self.project_in_basis(hsi_residual)

    def subspace_statistics(self) -> Dict[str, torch.Tensor]:
        basis = self.get_basis()
        identity = torch.eye(
            self.basis_rank,
            device=basis.device,
            dtype=basis.dtype,
        )
        gram = basis.transpose(0, 1) @ basis
        projector = basis @ basis.transpose(0, 1)
        reference_projector = self.get_reference_projector().to(projector)
        retained_variance = self.pca_eigenvalues.sum()
        explained_ratio = retained_variance / self.total_variance.clamp_min(self.eps)
        return {
            "orthogonality_error": (gram - identity).pow(2).mean().sqrt(),
            "projector_idempotence_error": (
                projector @ projector - projector
            ).pow(2).mean().sqrt(),
            "projector_drift": (
                projector - reference_projector
            ).pow(2).mean().sqrt(),
            "pca_explained_variance_ratio": explained_ratio,
        }

    def forward(self, lr_hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._validate_hsi(lr_hsi)
        basis = self.get_basis()
        coefficients = self.encode(lr_hsi, basis=basis)
        reconstruction = self.decode(coefficients, basis=basis)
        normalized_coefficients = coefficients / self.coefficient_scale.view(
            1, -1, 1, 1
        )
        residual = lr_hsi - reconstruction
        return {
            "reconstruction": reconstruction,
            "residual": residual,
            "coefficients": coefficients,
            "normalized_coefficients": normalized_coefficients,
            "basis": basis,
            "mean_spectrum": self.mean_spectrum,
            "coefficient_scale": self.coefficient_scale,
            "projector": basis @ basis.transpose(0, 1),
            **self.subspace_statistics(),
        }

    def _validate_hsi(self, hsi: torch.Tensor) -> None:
        if hsi.ndim != 4 or hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected HSI [N, {self.n_bands}, H, W], got "
                f"{tuple(hsi.shape)}"
            )
