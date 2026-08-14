"""Observable-guided masked null-space residual completion for Stage 2.

This controlled variant preserves the trained symmetric-frequency representation,
SRF analytical anchor, exact observable/null-space projectors, dual coefficient
heads, and the original losses. Only the null-space route is changed during
training:

1. The observable branch receives the complete symmetric-frequency evidence.
2. A blockwise spatial token mask is applied only to low/mid/high difference
   evidence before the shared fusion trunk is evaluated for the null branch.
3. The projected observable coefficient residual is encoded as a physically
   grounded context and injected into the null route through pooled
   cross-attention.

At evaluation/inference the mask is disabled, so a warm-started model initially
matches the source Stage2SymmetricFrequencyNet exactly because the cross-attention
gate is initialized to zero.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count
from .stage2_symmetric_frequency import Stage2SymmetricFrequencyNet


class ObservableGuidedNullAttention(nn.Module):
    """Efficient pooled cross-attention from observable coefficients to null tokens."""

    def __init__(
        self,
        basis_rank: int,
        hidden_channels: int,
        num_heads: int = 4,
        pool_size: int = 8,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels={hidden_channels} must be divisible by "
                f"num_heads={num_heads}"
            )
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")

        groups = _group_count(hidden_channels)
        self.pool_size = int(pool_size)
        self.observable_adapter = nn.Sequential(
            nn.Conv2d(basis_rank, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.query_norm = nn.LayerNorm(hidden_channels)
        self.context_norm = nn.LayerNorm(hidden_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        null_hidden: torch.Tensor,
        normalized_observable_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n, c, h, w = null_hidden.shape
        observable = self.observable_adapter(normalized_observable_residual)
        pool_h = min(self.pool_size, h)
        pool_w = min(self.pool_size, w)
        observable = F.adaptive_avg_pool2d(observable, (pool_h, pool_w))

        query = null_hidden.flatten(2).transpose(1, 2)
        context = observable.flatten(2).transpose(1, 2)
        query_norm = self.query_norm(query)
        context_norm = self.context_norm(context)
        attended, _ = self.attention(
            query_norm,
            context_norm,
            context_norm,
            need_weights=False,
        )
        attended = attended.transpose(1, 2).reshape(n, c, h, w)

        gate = torch.tanh(self.gate_logit)
        fused = null_hidden + gate * attended
        diagnostics = {
            "null_cross_gate": gate.detach(),
            "null_cross_attention_abs": attended.detach().abs().mean(),
            "observable_context_abs": observable.detach().abs().mean(),
            "null_hidden_abs": null_hidden.detach().abs().mean(),
        }
        return fused, diagnostics


class Stage2ObservableGuidedMaskedNullNet(Stage2SymmetricFrequencyNet):
    """Symmetric-frequency Stage 2 with masked, observable-guided null routing."""

    def __init__(
        self,
        *args,
        null_mask_ratio: float = 0.5,
        null_mask_block_size: int = 4,
        null_attention_heads: int = 4,
        null_attention_pool_size: int = 8,
        **kwargs,
    ) -> None:
        if not 0.0 <= null_mask_ratio < 1.0:
            raise ValueError("null_mask_ratio must be in [0, 1)")
        if null_mask_block_size <= 0:
            raise ValueError("null_mask_block_size must be positive")

        fusion_channels = int(kwargs.get("fusion_channels", 96))
        super().__init__(*args, **kwargs)
        self.null_mask_ratio = float(null_mask_ratio)
        self.null_mask_block_size = int(null_mask_block_size)
        self.observable_to_null = ObservableGuidedNullAttention(
            basis_rank=self.basis_rank,
            hidden_channels=fusion_channels,
            num_heads=int(null_attention_heads),
            pool_size=int(null_attention_pool_size),
        )

    def _sample_visibility_mask(self, reference: torch.Tensor) -> torch.Tensor:
        n, _, h, w = reference.shape
        if (not self.training) or self.null_mask_ratio <= 0:
            return reference.new_ones((n, 1, h, w))

        block = self.null_mask_block_size
        coarse_h = max((h + block - 1) // block, 1)
        coarse_w = max((w + block - 1) // block, 1)
        keep_probability = 1.0 - self.null_mask_ratio
        coarse = (
            torch.rand(
                (n, 1, coarse_h, coarse_w),
                device=reference.device,
                dtype=reference.dtype,
            )
            < keep_probability
        ).to(reference.dtype)
        return F.interpolate(coarse, size=(h, w), mode="nearest")

    def _build_fused_features(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coefficient_feature = self.coefficient_context(
            normalized_upsampled_coefficients
        )
        physical_context = self.physical_context_adapter(physical_feature)

        low_feature = self.low_discrepancy_adapter(low_discrepancy_feature)
        mid_adapted = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(reliable_high_feature)
        full_fused = torch.cat(
            [
                coefficient_feature,
                physical_context,
                low_feature,
                mid_adapted,
                high_feature,
                reliability_map,
            ],
            dim=1,
        )

        visibility = self._sample_visibility_mask(low_discrepancy_feature)
        masked_low = self.low_discrepancy_adapter(
            low_discrepancy_feature * visibility
        )
        masked_mid = self.mid_detail_adapter(mid_feature * visibility)
        masked_high = self.high_detail_adapter(
            reliable_high_feature * visibility
        )
        masked_fused = torch.cat(
            [
                coefficient_feature,
                physical_context,
                masked_low,
                masked_mid,
                masked_high,
                reliability_map * visibility,
            ],
            dim=1,
        )
        return {
            "full_fused": full_fused,
            "masked_fused": masked_fused,
            "null_visibility_mask": visibility,
        }

    def _predict_normalized_residual(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        routed = self._build_fused_features(
            normalized_upsampled_coefficients,
            physical_feature,
            low_discrepancy_feature,
            mid_feature,
            reliable_high_feature,
            reliability_map,
        )
        observable_hidden = self.fusion_trunk(routed["full_fused"])
        null_hidden = self.fusion_trunk(routed["masked_fused"])

        raw_observable = self.observable_normalized_residual_head(
            observable_hidden
        )
        bounded_observable = self.max_normalized_residual * torch.tanh(
            raw_observable
        )
        scale = self.coefficient_scale().view(1, -1, 1, 1)
        unprojected_observable = bounded_observable * scale
        observable_residual = self._project_coefficients(
            self.exact_observable_projector.to(unprojected_observable),
            unprojected_observable,
        )
        normalized_observable = observable_residual / scale

        null_hidden, attention_stats = self.observable_to_null(
            null_hidden,
            normalized_observable,
        )
        raw_null = self.null_normalized_residual_head(null_hidden)
        bounded_null = self.max_normalized_residual * torch.tanh(raw_null)
        unprojected_null = bounded_null * scale
        null_residual = self._project_coefficients(
            self.exact_null_projector.to(unprojected_null),
            unprojected_null,
        )

        coefficient_residual = observable_residual + null_residual
        normalized_null = null_residual / scale
        normalized_total = coefficient_residual / scale
        visibility = routed["null_visibility_mask"]
        actual_mask_ratio = 1.0 - visibility.mean()
        statistics = self.projector_statistics()

        return {
            "raw_normalized_observable_residual": raw_observable,
            "raw_normalized_null_residual": raw_null,
            "unprojected_normalized_observable_residual": bounded_observable,
            "unprojected_normalized_null_residual": bounded_null,
            "normalized_observable_coefficient_residual": normalized_observable,
            "normalized_null_coefficient_residual": normalized_null,
            "observable_coefficient_residual": observable_residual,
            "null_coefficient_residual": null_residual,
            "raw_normalized_coefficient_residual": 0.5
            * (raw_observable + raw_null),
            "normalized_coefficient_residual": normalized_total,
            "coefficient_residual": coefficient_residual,
            "observable_rank": self.observable_rank.to(observable_hidden.device),
            "null_visibility_mask": visibility,
            "null_actual_mask_ratio": actual_mask_ratio.detach(),
            "null_configured_mask_ratio": observable_hidden.new_tensor(
                self.null_mask_ratio
            ),
            **attention_stats,
            **statistics,
        }
