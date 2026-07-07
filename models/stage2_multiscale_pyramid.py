"""Residual-of-residual coarse-to-fine coefficient pyramid for Stage 2.

The trained symmetric-frequency model is preserved as a source predictor:

    Delta C_source = G_source(C_anchor, frequency differences).

Three new zero-initialized correction levels predict only the remaining target
error after the source residual:

    Delta C_final = Delta C_source
                  + Up(Delta C_quarter)
                  + Up(Delta C_half)
                  + Delta C_full_correction.

Each correction level retains the exact observable/null-space decomposition.
The source residual is detached when it is used as correction context so the
new branches cannot destabilize the mature source path through their inputs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import (
    CoefficientResidualBlock,
    _group_count,
)
from .stage2_symmetric_frequency import Stage2SymmetricFrequencyNet


def resize_antialiased(
    x: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize with anti-aliasing when the installed PyTorch supports it."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        try:
            return F.interpolate(x, antialias=True, **kwargs)
        except TypeError:
            return F.interpolate(x, **kwargs)
    return F.interpolate(x, **kwargs)


class PyramidDualScaleBranch(nn.Module):
    """One zero-initialized observable/null-space correction branch."""

    def __init__(
        self,
        basis_rank: int,
        feature_channels: int,
        fusion_channels: int,
        fusion_blocks: int,
        max_normalized_residual: float,
    ):
        super().__init__()
        self.basis_rank = int(basis_rank)
        self.max_normalized_residual = float(max_normalized_residual)
        feature_groups = _group_count(feature_channels)
        fusion_groups = _group_count(fusion_channels)

        self.coefficient_context = nn.Sequential(
            nn.Conv2d(
                basis_rank,
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
        self.fusion_trunk = nn.Sequential(
            nn.Conv2d(
                feature_channels * 5 + 1,
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
        self.observable_head = nn.Conv2d(
            fusion_channels,
            basis_rank,
            3,
            padding=1,
        )
        self.null_head = nn.Conv2d(
            fusion_channels,
            basis_rank,
            3,
            padding=1,
        )
        self.zero_heads()

    def zero_heads(self) -> None:
        nn.init.zeros_(self.observable_head.weight)
        nn.init.zeros_(self.observable_head.bias)
        nn.init.zeros_(self.null_head.weight)
        nn.init.zeros_(self.null_head.bias)

    @torch.no_grad()
    def initialize_trunk_from_full(
        self,
        parent: Stage2SymmetricFrequencyNet,
    ) -> None:
        self.coefficient_context.load_state_dict(
            parent.coefficient_context.state_dict()
        )
        self.physical_context_adapter.load_state_dict(
            parent.physical_context_adapter.state_dict()
        )
        self.low_discrepancy_adapter.load_state_dict(
            parent.low_discrepancy_adapter.state_dict()
        )
        self.mid_detail_adapter.load_state_dict(
            parent.mid_detail_adapter.state_dict()
        )
        self.high_detail_adapter.load_state_dict(
            parent.high_detail_adapter.state_dict()
        )
        self.fusion_trunk.load_state_dict(parent.fusion_trunk.state_dict())
        self.zero_heads()

    @staticmethod
    def _project(
        projector: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("rk,nkhw->nrhw", projector, coefficients)

    def forward(
        self,
        normalized_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
        coefficient_scale: torch.Tensor,
        observable_projector: torch.Tensor,
        null_projector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coefficient_feature = self.coefficient_context(normalized_coefficients)
        physical_feature = self.physical_context_adapter(physical_feature)
        low_feature = self.low_discrepancy_adapter(low_feature)
        mid_feature = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(high_feature)
        hidden = self.fusion_trunk(
            torch.cat(
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
        )

        raw_observable = self.observable_head(hidden)
        raw_null = self.null_head(hidden)
        bounded_observable = self.max_normalized_residual * torch.tanh(
            raw_observable
        )
        bounded_null = self.max_normalized_residual * torch.tanh(raw_null)

        scale = coefficient_scale.view(1, -1, 1, 1)
        unprojected_observable = bounded_observable * scale
        unprojected_null = bounded_null * scale
        observable = self._project(
            observable_projector.to(unprojected_observable),
            unprojected_observable,
        )
        null = self._project(
            null_projector.to(unprojected_null),
            unprojected_null,
        )
        total = observable + null
        return {
            "raw_observable": raw_observable,
            "raw_null": raw_null,
            "normalized_observable": observable / scale,
            "normalized_null": null / scale,
            "normalized_total": total / scale,
            "observable": observable,
            "null": null,
            "total": total,
        }


class Stage2MultiScalePyramidNet(Stage2SymmetricFrequencyNet):
    """Source predictor plus quarter/half/full residual correction pyramid."""

    def __init__(
        self,
        *args,
        pyramid_quarter_scale: float = 0.25,
        pyramid_half_scale: float = 0.5,
        **kwargs,
    ):
        if not 0.0 < pyramid_quarter_scale < pyramid_half_scale < 1.0:
            raise ValueError(
                "Require 0 < pyramid_quarter_scale < pyramid_half_scale < 1"
            )
        feature_channels = int(kwargs.get("feature_channels", 64))
        fusion_channels = int(kwargs.get("fusion_channels", 96))
        fusion_blocks = int(kwargs.get("fusion_blocks", 4))
        super().__init__(*args, **kwargs)
        self.pyramid_quarter_scale = float(pyramid_quarter_scale)
        self.pyramid_half_scale = float(pyramid_half_scale)

        branch_kwargs = {
            "basis_rank": self.basis_rank,
            "feature_channels": feature_channels,
            "fusion_channels": fusion_channels,
            "fusion_blocks": fusion_blocks,
            "max_normalized_residual": self.max_normalized_residual,
        }
        self.quarter_branch = PyramidDualScaleBranch(**branch_kwargs)
        self.half_branch = PyramidDualScaleBranch(**branch_kwargs)
        self.full_correction_branch = PyramidDualScaleBranch(**branch_kwargs)

    @torch.no_grad()
    def initialize_pyramid_from_full(self) -> None:
        self.quarter_branch.initialize_trunk_from_full(self)
        self.half_branch.initialize_trunk_from_full(self)
        self.full_correction_branch.initialize_trunk_from_full(self)

    @staticmethod
    def _scaled_size(
        full_size: Tuple[int, int],
        scale: float,
    ) -> Tuple[int, int]:
        return (
            max(4, int(round(full_size[0] * scale))),
            max(4, int(round(full_size[1] * scale))),
        )

    def _resize_feature_dict(
        self,
        size: Tuple[int, int],
        normalized_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "coefficients": resize_antialiased(
                normalized_coefficients,
                size,
                mode="bicubic",
            ),
            "physical": resize_antialiased(physical_feature, size),
            "low": resize_antialiased(low_feature, size),
            "mid": resize_antialiased(mid_feature, size),
            "high": resize_antialiased(high_feature, size),
            "reliability": resize_antialiased(reliability_map, size),
        }

    def _source_residual(
        self,
        normalized_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return super()._predict_normalized_residual(
            normalized_upsampled_coefficients=normalized_coefficients,
            physical_feature=physical_feature,
            low_discrepancy_feature=low_feature,
            mid_feature=mid_feature,
            reliable_high_feature=high_feature,
            reliability_map=reliability_map,
        )

    def _predict_normalized_residual(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_size = tuple(normalized_upsampled_coefficients.shape[-2:])
        quarter_size = self._scaled_size(
            full_size,
            self.pyramid_quarter_scale,
        )
        half_size = self._scaled_size(
            full_size,
            self.pyramid_half_scale,
        )
        scale = self.coefficient_scale()
        scale_view = scale.view(1, -1, 1, 1)

        # Preserve the mature single-scale path exactly. The detached copy is
        # only correction context; the non-detached source remains in the final
        # sum so low-rate joint fine-tuning can still update the source branch.
        source = self._source_residual(
            normalized_coefficients=normalized_upsampled_coefficients,
            physical_feature=physical_feature,
            low_feature=low_discrepancy_feature,
            mid_feature=mid_feature,
            high_feature=reliable_high_feature,
            reliability_map=reliability_map,
        )
        source_context = source["normalized_coefficient_residual"].detach()
        corrected_source_coefficients = (
            normalized_upsampled_coefficients + source_context
        )

        quarter_inputs = self._resize_feature_dict(
            quarter_size,
            corrected_source_coefficients,
            physical_feature,
            low_discrepancy_feature,
            mid_feature,
            reliable_high_feature,
            reliability_map,
        )
        quarter = self.quarter_branch(
            normalized_coefficients=quarter_inputs["coefficients"],
            physical_feature=quarter_inputs["physical"],
            low_feature=quarter_inputs["low"],
            mid_feature=quarter_inputs["mid"],
            high_feature=quarter_inputs["high"],
            reliability_map=quarter_inputs["reliability"],
            coefficient_scale=scale,
            observable_projector=self.exact_observable_projector,
            null_projector=self.exact_null_projector,
        )

        half_inputs = self._resize_feature_dict(
            half_size,
            corrected_source_coefficients,
            physical_feature,
            low_discrepancy_feature,
            mid_feature,
            reliable_high_feature,
            reliability_map,
        )
        quarter_to_half_normalized = resize_antialiased(
            quarter["normalized_total"],
            half_size,
            mode="bicubic",
        )
        half = self.half_branch(
            normalized_coefficients=(
                half_inputs["coefficients"] + quarter_to_half_normalized
            ),
            physical_feature=half_inputs["physical"],
            low_feature=half_inputs["low"],
            mid_feature=half_inputs["mid"],
            high_feature=half_inputs["high"],
            reliability_map=half_inputs["reliability"],
            coefficient_scale=scale,
            observable_projector=self.exact_observable_projector,
            null_projector=self.exact_null_projector,
        )
        half_cumulative_normalized = (
            quarter_to_half_normalized + half["normalized_total"]
        )
        half_to_full_normalized = resize_antialiased(
            half_cumulative_normalized,
            full_size,
            mode="bicubic",
        )

        full_correction = self.full_correction_branch(
            normalized_coefficients=(
                corrected_source_coefficients + half_to_full_normalized
            ),
            physical_feature=physical_feature,
            low_feature=low_discrepancy_feature,
            mid_feature=mid_feature,
            high_feature=reliable_high_feature,
            reliability_map=reliability_map,
            coefficient_scale=scale,
            observable_projector=self.exact_observable_projector,
            null_projector=self.exact_null_projector,
        )

        quarter_observable_full = resize_antialiased(
            quarter["observable"],
            full_size,
            mode="bicubic",
        )
        quarter_null_full = resize_antialiased(
            quarter["null"],
            full_size,
            mode="bicubic",
        )
        half_observable_full = resize_antialiased(
            half["observable"],
            full_size,
            mode="bicubic",
        )
        half_null_full = resize_antialiased(
            half["null"],
            full_size,
            mode="bicubic",
        )

        correction_observable = (
            quarter_observable_full
            + half_observable_full
            + full_correction["observable"]
        )
        correction_null = (
            quarter_null_full
            + half_null_full
            + full_correction["null"]
        )
        correction_total = correction_observable + correction_null

        observable_total = (
            source["observable_coefficient_residual"]
            + correction_observable
        )
        null_total = source["null_coefficient_residual"] + correction_null
        coefficient_total = observable_total + null_total

        return {
            "raw_normalized_coefficient_residual": source[
                "raw_normalized_coefficient_residual"
            ],
            "normalized_observable_coefficient_residual": (
                observable_total / scale_view
            ),
            "normalized_null_coefficient_residual": null_total / scale_view,
            "observable_coefficient_residual": observable_total,
            "null_coefficient_residual": null_total,
            "normalized_coefficient_residual": coefficient_total / scale_view,
            "coefficient_residual": coefficient_total,
            "observable_rank": self.observable_rank.to(coefficient_total.device),
            "pyramid_source_normalized_residual": source[
                "normalized_coefficient_residual"
            ],
            "pyramid_source_observable_residual": source[
                "observable_coefficient_residual"
            ],
            "pyramid_source_null_residual": source[
                "null_coefficient_residual"
            ],
            "pyramid_source_coefficient_residual": source[
                "coefficient_residual"
            ],
            "pyramid_quarter_normalized_residual": quarter["normalized_total"],
            "pyramid_quarter_observable_residual": quarter["observable"],
            "pyramid_quarter_null_residual": quarter["null"],
            "pyramid_half_increment_normalized_residual": half[
                "normalized_total"
            ],
            "pyramid_half_cumulative_normalized_residual": (
                half_cumulative_normalized
            ),
            "pyramid_half_increment_observable_residual": half["observable"],
            "pyramid_half_increment_null_residual": half["null"],
            "pyramid_full_correction_normalized_residual": full_correction[
                "normalized_total"
            ],
            "pyramid_full_correction_observable_residual": full_correction[
                "observable"
            ],
            "pyramid_full_correction_null_residual": full_correction["null"],
            "pyramid_correction_normalized_residual": (
                correction_total / scale_view
            ),
            "pyramid_correction_observable_residual": correction_observable,
            "pyramid_correction_null_residual": correction_null,
            "pyramid_correction_coefficient_residual": correction_total,
            "pyramid_quarter_size": coefficient_total.new_tensor(
                quarter_size,
                dtype=torch.int64,
            ),
            "pyramid_half_size": coefficient_total.new_tensor(
                half_size,
                dtype=torch.int64,
            ),
            **self.projector_statistics(),
        }
