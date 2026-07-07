"""Stage-2 observable/null-space dual coefficient residual variant.

This is the next controlled improvement after the SRF analytical anchor. The
shared encoder, SSP/NSP reliability module, and single-scale fusion trunk remain
unchanged. Only the final coefficient residual head is split into two branches:

    delta C_obs  = P_obs  delta C_obs_raw
    delta C_null = P_null delta C_null_raw

where ``P_obs`` is the exact row-space projector of ``S = R U_r`` and
``P_null = I - P_obs``. The analytical SRF anchor remains unchanged.

Both branches see the same hidden feature tensor in this version. This isolates
the effect of coefficient-space decomposition before introducing branch-specific
feature routing or multi-scale prediction.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .stage2_srf_anchor import Stage2SRFAnchorNet


class Stage2DualSpaceNet(Stage2SRFAnchorNet):
    """SRF anchor plus exact observable/null-space residual decomposition."""

    def __init__(self, *args, projector_tolerance: float = 1e-6, **kwargs):
        if projector_tolerance <= 0:
            raise ValueError("projector_tolerance must be positive")
        self.projector_tolerance = float(projector_tolerance)
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            reduced = self.reduced_response.detach().float()
            _, singular_values, vh = torch.linalg.svd(
                reduced,
                full_matrices=True,
            )
            threshold = self.projector_tolerance * singular_values.max().clamp_min(
                1e-12
            )
            rank = int((singular_values > threshold).sum().item())
            row_basis = vh[:rank].transpose(0, 1).contiguous()
            observable = row_basis @ row_basis.transpose(0, 1)
            identity = torch.eye(
                self.basis_rank,
                device=observable.device,
                dtype=observable.dtype,
            )
            null = identity - observable

        self.register_buffer(
            "exact_observable_projector",
            observable.detach().contiguous(),
        )
        self.register_buffer(
            "exact_null_projector",
            null.detach().contiguous(),
        )
        self.register_buffer(
            "observable_singular_values",
            singular_values.detach().contiguous(),
        )
        self.register_buffer(
            "observable_rank",
            torch.tensor(rank, dtype=torch.int64),
        )

        # Replace the original single residual head with two structurally
        # identical heads. They are zero-initialized for training from scratch.
        old_head = self.normalized_residual_head
        self.observable_normalized_residual_head = nn.Conv2d(
            old_head.in_channels,
            old_head.out_channels,
            old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=old_head.bias is not None,
            padding_mode=old_head.padding_mode,
        )
        self.null_normalized_residual_head = nn.Conv2d(
            old_head.in_channels,
            old_head.out_channels,
            old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=old_head.bias is not None,
            padding_mode=old_head.padding_mode,
        )
        nn.init.zeros_(self.observable_normalized_residual_head.weight)
        nn.init.zeros_(self.null_normalized_residual_head.weight)
        if self.observable_normalized_residual_head.bias is not None:
            nn.init.zeros_(self.observable_normalized_residual_head.bias)
            nn.init.zeros_(self.null_normalized_residual_head.bias)
        del self.normalized_residual_head

    @torch.no_grad()
    def initialize_dual_heads_from_single(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        """Clone one trained head into both projected branches.

        Since ``P_obs + P_null = I``, cloning the same single-head prediction
        into both branches preserves the original total coefficient residual at
        initialization, up to floating-point projector error.
        """
        expected = self.observable_normalized_residual_head.weight.shape
        if tuple(weight.shape) != tuple(expected):
            raise ValueError(
                f"Single-head weight shape {tuple(weight.shape)} != {tuple(expected)}"
            )
        self.observable_normalized_residual_head.weight.copy_(weight)
        self.null_normalized_residual_head.weight.copy_(weight)
        if self.observable_normalized_residual_head.bias is not None:
            if bias is None:
                raise ValueError("Single-head checkpoint is missing bias")
            self.observable_normalized_residual_head.bias.copy_(bias)
            self.null_normalized_residual_head.bias.copy_(bias)

    @staticmethod
    def _project_coefficients(
        projector: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("rk,nkhw->nrhw", projector, coefficients)

    def projector_statistics(self) -> Dict[str, torch.Tensor]:
        observable = self.exact_observable_projector
        null = self.exact_null_projector
        identity = torch.eye(
            self.basis_rank,
            device=observable.device,
            dtype=observable.dtype,
        )
        return {
            "observable_projector_idempotence_error": (
                observable @ observable - observable
            ).abs().max(),
            "null_projector_idempotence_error": (
                null @ null - null
            ).abs().max(),
            "projector_complement_error": (
                observable + null - identity
            ).abs().max(),
            "projector_orthogonality_error": (
                observable @ null
            ).abs().max(),
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
        coefficient_feature = self.coefficient_context(
            normalized_upsampled_coefficients
        )
        physical_feature = self.physical_context_adapter(physical_feature)
        low_feature = self.low_discrepancy_adapter(low_discrepancy_feature)
        mid_feature = self.mid_detail_adapter(mid_feature)
        high_feature = self.high_detail_adapter(reliable_high_feature)

        fused = torch.cat(
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
        hidden = self.fusion_trunk(fused)

        raw_observable = self.observable_normalized_residual_head(hidden)
        raw_null = self.null_normalized_residual_head(hidden)
        bounded_observable = self.max_normalized_residual * torch.tanh(
            raw_observable
        )
        bounded_null = self.max_normalized_residual * torch.tanh(raw_null)

        scale = self.coefficient_scale().view(1, -1, 1, 1)
        unprojected_observable = bounded_observable * scale
        unprojected_null = bounded_null * scale
        observable_residual = self._project_coefficients(
            self.exact_observable_projector.to(unprojected_observable),
            unprojected_observable,
        )
        null_residual = self._project_coefficients(
            self.exact_null_projector.to(unprojected_null),
            unprojected_null,
        )
        coefficient_residual = observable_residual + null_residual

        normalized_observable = observable_residual / scale
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
            "observable_rank": self.observable_rank.to(hidden.device),
            **statistics,
        }
