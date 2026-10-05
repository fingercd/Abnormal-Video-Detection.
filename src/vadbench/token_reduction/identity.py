"""The no-op reducer used to prove bridge and layout equivalence first."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import (
    ReductionContext,
    ReductionResult,
    TokenLayout,
    TokenReductionContractError,
)


class IdentityReducer:
    """Return the existing token tensor and metadata without allocating a copy."""

    name = "identity"

    def reduce(
        self, tokens: torch.Tensor, layout: TokenLayout, context: ReductionContext
    ) -> ReductionResult:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise TokenReductionContractError("tokens 必须是 [B, N, D] torch.Tensor")
        if tuple(tokens.shape[:2]) != (layout.batch_size, layout.token_capacity):
            raise TokenReductionContractError("tokens 与 TokenLayout 的 [B, N] 不一致")
        if not isinstance(context, ReductionContext):
            raise TokenReductionContractError("context 必须为 ReductionContext")
        if context.budget is not None and context.budget != layout.token_capacity:
            raise TokenReductionContractError("identity budget must equal the dense token capacity")
        if layout.valid_mask.device != tokens.device:
            raise TokenReductionContractError("tokens and layout must reside on the same device")

        valid = layout.valid_mask
        offsets = torch.cat(
            (
                torch.zeros((layout.batch_size, 1), device=tokens.device, dtype=torch.int64),
                valid.to(dtype=torch.int64).cumsum(dim=1),
            ),
            dim=1,
        )
        members = torch.full_like(
            layout.original_token_ids, -1, dtype=torch.int64, device=tokens.device
        )
        for row in range(layout.batch_size):
            ids = layout.original_token_ids[row, valid[row]]
            members[row, : ids.numel()] = ids
        counts = valid.sum(dim=1, dtype=torch.int64).to(tokens.device)
        telemetry: dict[str, Any] = {
            "reducer": self.name,
            "layer_depth": context.layer_depth,
            "input_capacity": layout.token_capacity,
            "output_capacity": layout.token_capacity,
            "ratio": 1.0,
            "position_contract": layout.position_contract,
        }
        return ReductionResult(
            tokens=tokens,
            layout=layout,
            member_offsets=offsets,
            member_token_ids=members,
            input_token_counts=counts,
            output_token_counts=counts,
            telemetry=telemetry,
        )


__all__ = ["IdentityReducer"]
