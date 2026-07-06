"""Stage-1 physical unmixing network for RAPD-Net.

A scene-level endmember bank is initialized from real LR-HSI pixels by the
training script. The network then jointly optimizes that bank and a pixel-wise
abundance estimator using LR-HSI reconstruction only.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualAbundanceBlock(nn.Module):
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


class Stage1UnmixingNet(nn.Module):
    """Learn a global endmember dictionary and LR abundance maps."""

    def __init__(
        self,
        n_bands: int,
        num_endmembers: int = 32,
        hidden_channels: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        if n_bands < 2 or num_endmembers < 2:
            raise ValueError("n_bands and num_endmembers must both be >= 2")

        self.n_bands = int(n_bands)
        self.num_endmembers = int(num_endmembers)
        initial = torch.empty(num_endmembers, n_bands).uniform_(0.15, 0.85)
        self.endmember_logits = nn.Parameter(torch.logit(initial))

        groups = _group_count(hidden_channels)
        self.spectral_stem = nn.Sequential(
            nn.Conv2d(n_bands, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.spatial_blocks = nn.Sequential(
            *[ResidualAbundanceBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.abundance_head = nn.Sequential(
            nn.Conv2d(
                hidden_channels + num_endmembers,
                hidden_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_endmembers, 1),
        )
        self.similarity_scale_raw = nn.Parameter(torch.tensor(0.5413))

    @torch.no_grad()
    def initialize_endmembers(self, endmembers: torch.Tensor) -> None:
        """Initialize from real spectra shaped ``[B, K]`` or ``[K, B]``."""
        if endmembers.ndim != 2:
            raise ValueError(
                f"endmembers must be 2D, got shape {tuple(endmembers.shape)}"
            )
        if endmembers.shape == (self.n_bands, self.num_endmembers):
            endmembers = endmembers.transpose(0, 1)

        expected = (self.num_endmembers, self.n_bands)
        if tuple(endmembers.shape) != expected:
            raise ValueError(
                f"Expected {expected} or its transpose, got {tuple(endmembers.shape)}"
            )

        endmembers = endmembers.to(self.endmember_logits).clamp(1e-4, 1 - 1e-4)
        self.endmember_logits.copy_(torch.logit(endmembers))

    def get_endmembers(self) -> torch.Tensor:
        """Return the dictionary in physical layout ``[B, K]``."""
        return torch.sigmoid(self.endmember_logits).transpose(0, 1)

    @staticmethod
    def spectral_similarity(
        lr_hsi: torch.Tensor,
        endmembers: torch.Tensor,
    ) -> torch.Tensor:
        pixels = F.normalize(lr_hsi, dim=1, eps=1e-8)
        spectra = F.normalize(endmembers.transpose(0, 1), dim=1, eps=1e-8)
        return torch.einsum("nbhw,kb->nkhw", pixels, spectra)

    def forward(self, lr_hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        if lr_hsi.ndim != 4 or lr_hsi.size(1) != self.n_bands:
            raise ValueError(
                f"Expected [N, {self.n_bands}, h, w], got {tuple(lr_hsi.shape)}"
            )

        endmembers = self.get_endmembers()
        similarity = self.spectral_similarity(lr_hsi, endmembers)
        features = self.spatial_blocks(self.spectral_stem(lr_hsi))
        learned_logits = self.abundance_head(
            torch.cat([features, similarity], dim=1)
        )
        abundance_logits = (
            learned_logits + F.softplus(self.similarity_scale_raw) * similarity
        )
        abundance = torch.softmax(abundance_logits, dim=1)
        reconstruction = torch.einsum("bk,nkhw->nbhw", endmembers, abundance)

        return {
            "reconstruction": reconstruction,
            "abundance": abundance,
            "abundance_logits": abundance_logits,
            "endmembers": endmembers,
            "similarity": similarity,
        }
