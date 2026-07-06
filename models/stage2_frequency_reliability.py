"""Stage-2 frequency reliability screening for RAPD-Net.

The module follows the SFSR frequency-refinement structure instead of replacing
it with a generic frequency attention block:

1. a weight-shared encoder maps the physical MSI baseline and HR-MSI reference
   into one feature space;
2. a Spectral Splitter (SSP) performs channel-wise 2-D Fourier decomposition
   with two learnable spectral boundaries per feature channel;
3. a Noise Splitter (NSP) aggregates LF and MF features, extracts Sobel edges,
   and isolates unsupported HF responses with the original binary edge mask;
4. the original LF alignment and noise-minimization losses are exposed for the
   Stage-2 objective.

No abundance correction is implemented in this file. The outputs preserve the
raw LF/MF/HF branches so the subsequent abundance-residual module can consume
them without hiding the original SSP/NSP structure behind a simplified gate.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _inverse_sigmoid(value: float, eps: float = 1e-6) -> float:
    value = min(max(value, eps), 1.0 - eps)
    return math.log(value / (1.0 - value))


class SharedEncoderResidualBlock(nn.Module):
    """Resolution-preserving residual block used by the shared encoder."""

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


class SharedMSIFeatureEncoder(nn.Module):
    """One encoder instance shared by physical MSI and observed HR-MSI."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        groups = _group_count(feature_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, feature_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                SharedEncoderResidualBlock(feature_channels)
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x)))


class ChannelWiseSpectralSplitter(nn.Module):
    """SFSR-style Spectral Splitter with learnable channel-wise boundaries.

    A centered 2-D FFT is quantized into ``num_frequency_bands`` radial bands.
    Each feature channel owns a low/mid boundary ``tau_low`` and a mid/high
    boundary ``tau_high``. The forward partition is hard, matching the paper's
    indicator-style band assignment, while a straight-through soft partition
    supplies gradients to the learnable boundaries.
    """

    def __init__(
        self,
        channels: int,
        num_frequency_bands: int = 20,
        init_low_boundary: float = 5.0,
        init_high_boundary: float = 18.0,
        boundary_temperature: float = 0.5,
        hard_partition: bool = True,
    ):
        super().__init__()
        if num_frequency_bands < 4:
            raise ValueError("num_frequency_bands must be >= 4")
        if not 0.0 <= init_low_boundary < init_high_boundary:
            raise ValueError("Require 0 <= init_low_boundary < init_high_boundary")
        if init_high_boundary > num_frequency_bands - 1:
            raise ValueError("init_high_boundary exceeds the last frequency band")
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive")

        self.channels = int(channels)
        self.num_frequency_bands = int(num_frequency_bands)
        self.boundary_temperature = float(boundary_temperature)
        self.hard_partition = bool(hard_partition)

        max_low = float(num_frequency_bands - 2)
        low_ratio = init_low_boundary / max_low
        remaining = max_low - init_low_boundary
        gap_ratio = (init_high_boundary - init_low_boundary - 1.0) / max(
            remaining, 1e-6
        )
        self.low_boundary_raw = nn.Parameter(
            torch.full((channels,), _inverse_sigmoid(low_ratio))
        )
        self.high_gap_raw = nn.Parameter(
            torch.full((channels,), _inverse_sigmoid(gap_ratio))
        )

        self._band_cache: Dict[
            Tuple[int, int, str, int | None, torch.dtype], torch.Tensor
        ] = {}

    def boundaries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_low = float(self.num_frequency_bands - 2)
        tau_low = max_low * torch.sigmoid(self.low_boundary_raw)
        tau_high = tau_low + 1.0 + (max_low - tau_low) * torch.sigmoid(
            self.high_gap_raw
        )
        return tau_low, tau_high

    def _radial_band_index(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        key = (height, width, x.device.type, x.device.index, x.dtype)
        cached = self._band_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=x.device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=x.device))
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-8)
        band_index = torch.round(
            radius * float(self.num_frequency_bands - 1)
        ).to(dtype=x.dtype)
        band_index = band_index.view(1, 1, height, width)
        self._band_cache[key] = band_index
        return band_index

    def _partition_masks(
        self,
        band_index: torch.Tensor,
        tau_low: torch.Tensor,
        tau_high: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau_low = tau_low.view(1, -1, 1, 1)
        tau_high = tau_high.view(1, -1, 1, 1)
        temperature = self.boundary_temperature

        soft_low = torch.sigmoid((tau_low - band_index) / temperature)
        soft_below_high = torch.sigmoid((tau_high - band_index) / temperature)
        soft_mid = (soft_below_high - soft_low).clamp(0.0, 1.0)
        soft_high = 1.0 - soft_below_high

        if not self.hard_partition:
            return soft_low, soft_mid, soft_high

        hard_low = (band_index < tau_low).to(dtype=band_index.dtype)
        hard_high = (band_index > tau_high).to(dtype=band_index.dtype)
        hard_mid = 1.0 - hard_low - hard_high

        low = hard_low + soft_low - soft_low.detach()
        mid = hard_mid + soft_mid - soft_mid.detach()
        high = hard_high + soft_high - soft_high.detach()
        return low, mid, high

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feature.ndim != 4 or feature.size(1) != self.channels:
            raise ValueError(
                f"Expected [N, {self.channels}, H, W], got {tuple(feature.shape)}"
            )

        spectrum = torch.fft.fftshift(
            torch.fft.fft2(feature, norm="ortho"), dim=(-2, -1)
        )
        band_index = self._radial_band_index(feature)
        tau_low, tau_high = self.boundaries()
        low_mask, mid_mask, high_mask = self._partition_masks(
            band_index, tau_low, tau_high
        )

        def inverse(mask: torch.Tensor) -> torch.Tensor:
            component = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)),
                norm="ortho",
            )
            return component.real

        low = inverse(low_mask)
        mid = inverse(mid_mask)
        high = inverse(high_mask)
        reconstructed = low + mid + high

        return {
            "low": low,
            "mid": mid,
            "high": high,
            "low_mask": low_mask,
            "mid_mask": mid_mask,
            "high_mask": high_mask,
            "band_index": band_index,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "partition_reconstruction": reconstructed,
        }


class NoiseSplitter(nn.Module):
    """SFSR-style Noise Splitter using LM-supported Sobel evidence."""

    def __init__(
        self,
        channels: int,
        edge_mask_threshold: float = 1e-5,
    ):
        super().__init__()
        if edge_mask_threshold < 0:
            raise ValueError("edge_mask_threshold must be non-negative")
        self.channels = int(channels)
        self.edge_mask_threshold = float(edge_mask_threshold)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if low.shape != mid.shape or low.shape != high.shape:
            raise ValueError("low, mid, and high features must have the same shape")
        if low.ndim != 4 or low.size(1) != self.channels:
            raise ValueError(
                f"Expected [N, {self.channels}, H, W], got {tuple(low.shape)}"
            )

        low_mid = low + mid
        grad_x = F.conv2d(
            low_mid,
            self.sobel_x.to(dtype=low_mid.dtype),
            padding=1,
            groups=self.channels,
        )
        grad_y = F.conv2d(
            low_mid,
            self.sobel_y.to(dtype=low_mid.dtype),
            padding=1,
            groups=self.channels,
        )
        edge_magnitude_channel = torch.sqrt(
            grad_x.square() + grad_y.square() + 1e-12
        )
        # The paper uses one fixed tau_mask and visualizes one spatial mask.
        # Channel evidence is therefore aggregated before the binary decision.
        edge_magnitude = edge_magnitude_channel.mean(dim=1, keepdim=True)

        # Original NSP definition: M = Ind(E < tau_mask).
        noise_mask = (edge_magnitude < self.edge_mask_threshold).to(low_mid.dtype)
        reliability_map = 1.0 - noise_mask
        noise_feature = high * noise_mask
        reliable_high = high * reliability_map

        return {
            "low_mid": low_mid,
            "edge_magnitude_channel": edge_magnitude_channel,
            "edge_magnitude": edge_magnitude,
            "noise_mask": noise_mask,
            "reliability_mask_channel": reliability_map.expand_as(high),
            "reliability_map": reliability_map,
            "noise_feature": noise_feature,
            "reliable_high": reliable_high,
        }


class FrequencyReliabilityScreen(nn.Module):
    """Shared encoder + SSP + NSP, adapted from SFSR for HR-MSI screening.

    ``physical_msi`` is the SRF projection of the Stage-1 physical baseline and
    plays the role of the stable LR/WFOV feature. ``hr_msi`` is the observed
    high-resolution MSI reference whose frequency responses are screened.
    """

    def __init__(
        self,
        msi_channels: int,
        feature_channels: int = 64,
        encoder_blocks: int = 3,
        num_frequency_bands: int = 20,
        init_low_boundary: float = 5.0,
        init_high_boundary: float = 18.0,
        boundary_temperature: float = 0.5,
        edge_mask_threshold: float = 1e-5,
        hard_partition: bool = True,
    ):
        super().__init__()
        self.msi_channels = int(msi_channels)
        self.feature_channels = int(feature_channels)

        self.shared_encoder = SharedMSIFeatureEncoder(
            in_channels=msi_channels,
            feature_channels=feature_channels,
            num_blocks=encoder_blocks,
        )
        self.spectral_splitter = ChannelWiseSpectralSplitter(
            channels=feature_channels,
            num_frequency_bands=num_frequency_bands,
            init_low_boundary=init_low_boundary,
            init_high_boundary=init_high_boundary,
            boundary_temperature=boundary_temperature,
            hard_partition=hard_partition,
        )
        self.noise_splitter = NoiseSplitter(
            channels=feature_channels,
            edge_mask_threshold=edge_mask_threshold,
        )

    def spectral_boundary_parameters(self) -> Iterable[nn.Parameter]:
        yield self.spectral_splitter.low_boundary_raw
        yield self.spectral_splitter.high_gap_raw

    def regular_parameters(self) -> Iterable[nn.Parameter]:
        boundary_ids = {
            id(parameter) for parameter in self.spectral_boundary_parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in boundary_ids:
                yield parameter

    @staticmethod
    def frequency_activation_ratios(
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        activation = torch.stack(
            [low.abs().mean(), mid.abs().mean(), high.abs().mean()]
        )
        return activation / activation.sum().clamp_min(1e-8)

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

        physical_feature = self.shared_encoder(physical_msi)
        reference_feature = self.shared_encoder(hr_msi)
        split = self.spectral_splitter(reference_feature)
        noise = self.noise_splitter(
            split["low"], split["mid"], split["high"]
        )

        # Original SFSR auxiliary objectives.
        low_frequency_alignment_loss = F.mse_loss(
            split["low"], physical_feature
        )
        noise_minimization_loss = noise["noise_feature"].square().mean()
        partition_reconstruction_loss = F.l1_loss(
            split["partition_reconstruction"], reference_feature
        )

        reliable_detail = torch.cat(
            [split["mid"], noise["reliable_high"]], dim=1
        )
        refined_reference = torch.cat(
            [split["low"], split["mid"], noise["reliable_high"]], dim=1
        )

        return {
            "physical_feature": physical_feature,
            "reference_feature": reference_feature,
            "low_feature": split["low"],
            "mid_feature": split["mid"],
            "high_feature": split["high"],
            "low_mid_feature": noise["low_mid"],
            "edge_magnitude": noise["edge_magnitude"],
            "noise_mask": noise["noise_mask"],
            "reliability_mask_channel": noise["reliability_mask_channel"],
            "reliability_map": noise["reliability_map"],
            "noise_feature": noise["noise_feature"],
            "reliable_high_feature": noise["reliable_high"],
            "reliable_detail_feature": reliable_detail,
            "refined_reference_feature": refined_reference,
            "tau_low": split["tau_low"],
            "tau_high": split["tau_high"],
            "low_mask": split["low_mask"],
            "mid_mask": split["mid_mask"],
            "high_mask": split["high_mask"],
            "frequency_activation_ratio": self.frequency_activation_ratios(
                split["low"], split["mid"], noise["reliable_high"]
            ),
            "low_frequency_alignment_loss": low_frequency_alignment_loss,
            "noise_minimization_loss": noise_minimization_loss,
            "partition_reconstruction_loss": partition_reconstruction_loss,
        }
