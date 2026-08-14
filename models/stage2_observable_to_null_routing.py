"""Observable-to-null cross-space routing for RAPD-Net Stage 2.

Controlled variant of Stage2SymmetricFrequencyNet:

* symmetric-frequency representation is unchanged;
* SRF analytical anchor and exact observable/null projectors are unchanged;
* observable and null coefficient heads are warm-started from the same source;
* no masking or degradation simulation is used;
* the null branch gets its own fusion trunk;
* the observable coefficient residual is detached and used as context for a
  lightweight pooled cross-attention module that guides the null branch;
* the cross-space residual gate starts from a small non-zero value so attention
  parameters receive gradients from the first optimization step.

The null branch consumes the same complete symmetric-frequency evidence as the
observable branch. Its input feature tensor is detached before the null-specific
trunk so null losses cannot perturb the shared observable feature adapters.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage2_coefficient_residual import _group_count
from .stage2_symmetric_frequency import Stage2SymmetricFrequencyNet


class ObservableToNullCrossAttention(nn.Module):
    """Pooled cross-attention using observable coefficients as null context."""

    def __init__(
        self,
        basis_rank: int,
        hidden_channels: int,
        num_heads: int = 4,
        pool_size: int = 8,
        init_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels={hidden_channels} must be divisible by "
                f"num_heads={num_heads}"
            )
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if not 0.0 < init_gate < 1.0:
            raise ValueError("init_gate must be in (0, 1)")

        groups = _group_count(hidden_channels)
        self.pool_size = int(pool_size)
        self.init_gate = float(init_gate)

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

        # tanh(gate_logit) == init_gate at initialization.
        initial_logit = 0.5 * math.log(
            (1.0 + self.init_gate) / (1.0 - self.init_gate)
        )
        self.gate_logit = nn.Parameter(torch.tensor(initial_logit))

    def gate_value(self) -> torch.Tensor:
        return torch.tanh(self.gate_logit)

    def forward(
        self,
        null_hidden: torch.Tensor,
        normalized_observable_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n, c, h, w = null_hidden.shape
        pool_h = min(self.pool_size, h)
        pool_w = min(self.pool_size, w)

        # The observable residual is detached by the caller. This adapter and
        # attention block can learn how to read it, but null losses cannot
        # back-propagate into the observable coefficient branch.
        observable = self.observable_adapter(normalized_observable_residual)

        query_map = F.adaptive_avg_pool2d(
            null_hidden,
            (pool_h, pool_w),
        )
        context_map = F.adaptive_avg_pool2d(
            observable,
            (pool_h, pool_w),
        )
        query = query_map.flatten(2).transpose(1, 2)
        context = context_map.flatten(2).transpose(1, 2)

        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        attended_map = attended.transpose(1, 2).reshape(
            n,
            c,
            pool_h,
            pool_w,
        )
        attended_map = F.interpolate(
            attended_map,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        gate = self.gate_value()
        fused = null_hidden + gate * attended_map
        diagnostics = {
            "null_cross_gate": gate.detach(),
            "null_cross_attention_abs": attended_map.detach().abs().mean(),
            "observable_context_abs": observable.detach().abs().mean(),
            "null_hidden_abs": null_hidden.detach().abs().mean(),
        }
        return fused, diagnostics


class Stage2ObservableToNullRoutingNet(Stage2SymmetricFrequencyNet):
    """Symmetric-frequency Stage 2 with one-way observable-to-null routing."""

    def __init__(
        self,
        *args,
        null_attention_heads: int = 4,
        null_attention_pool_size: int = 8,
        null_cross_init_gate: float = 0.1,
        **kwargs,
    ) -> None:
        fusion_channels = int(kwargs.get("fusion_channels", 96))
        super().__init__(*args, **kwargs)

        # Separate only the fusion trunk. Source checkpoint loading later copies
        # the trained observable trunk into this null trunk exactly.
        self.null_fusion_trunk = copy.deepcopy(self.fusion_trunk)
        self.observable_to_null = ObservableToNullCrossAttention(
            basis_rank=self.basis_rank,
            hidden_channels=fusion_channels,
            num_heads=int(null_attention_heads),
            pool_size=int(null_attention_pool_size),
            init_gate=float(null_cross_init_gate),
        )

    @torch.no_grad()
    def synchronize_null_trunk_from_observable(self) -> None:
        """Copy the trained source fusion trunk into the null-specific trunk."""
        self.null_fusion_trunk.load_state_dict(self.fusion_trunk.state_dict())

    def _build_fused_features(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> torch.Tensor:
        coefficient_feature = self.coefficient_context(
            normalized_upsampled_coefficients
        )
        physical_context = self.physical_context_adapter(physical_feature)
        low_feature = self.low_discrepancy_adapter(low_discrepancy_feature)
        mid_adapted = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(reliable_high_feature)

        return torch.cat(
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

    def _predict_normalized_residual(
        self,
        normalized_upsampled_coefficients: torch.Tensor,
        physical_feature: torch.Tensor,
        low_discrepancy_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        reliable_high_feature: torch.Tensor,
        reliability_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = self._build_fused_features(
            normalized_upsampled_coefficients,
            physical_feature,
            low_discrepancy_feature,
            mid_feature,
            reliable_high_feature,
            reliability_map,
        )

        # Observable branch remains the original source path.
        observable_hidden = self.fusion_trunk(fused)
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

        # One-way route: null loss cannot modify shared feature adapters or the
        # observable coefficient prediction. Only the null trunk, attention,
        # gate, and null head receive gradients from this route.
        null_hidden = self.null_fusion_trunk(fused.detach())
        null_hidden, attention_stats = self.observable_to_null(
            null_hidden,
            normalized_observable.detach(),
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
            **attention_stats,
            **statistics,
        }

    @torch.no_grad()
    def trunk_divergence(self) -> Dict[str, torch.Tensor]:
        """Mean/max absolute parameter difference between the two fusion trunks."""
        means = []
        maxima = []
        for observable, null in zip(
            self.fusion_trunk.parameters(),
            self.null_fusion_trunk.parameters(),
        ):
            difference = (observable - null).abs()
            means.append(difference.mean())
            maxima.append(difference.max())
        if not means:
            zero = self.exact_observable_projector.new_zeros(())
            return {
                "cross_trunk_parameter_l1": zero,
                "cross_trunk_parameter_max": zero,
            }
        return {
            "cross_trunk_parameter_l1": torch.stack(means).mean(),
            "cross_trunk_parameter_max": torch.stack(maxima).max(),
        }
