from __future__ import annotations

import pytest
import torch

from vadbench.token_reduction import (
    IdentityReducer,
    ReductionContext,
    TokenLayout,
    TokenReductionContractError,
)


def _layout() -> TokenLayout:
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return TokenLayout(
        valid_mask=valid,
        original_token_ids=torch.tensor([[10, 11, 12, -1], [20, 21, -1, -1]]),
        special_token_mask=torch.tensor(
            [[True, False, False, False], [False, False, False, False]]
        ),
        mass=torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]]),
        source_coordinates=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        position_contract="actual-tubelet-t-h-w-v1",
        provenance={"coordinate_source": "patch_embed"},
    )


def test_identity_preserves_tokens_layout_coordinates_mass_and_gradients() -> None:
    tokens = torch.randn(2, 4, 3, requires_grad=True)
    layout = _layout()
    result = IdentityReducer().reduce(tokens, layout, ReductionContext(layer_depth=2, budget=4))

    assert result.tokens is tokens
    assert result.layout is layout
    assert result.layout.source_coordinates is layout.source_coordinates
    assert result.layout.mass is layout.mass
    assert result.input_token_counts.tolist() == [3, 2]
    assert result.output_token_counts.tolist() == [3, 2]
    assert result.member_offsets.tolist() == [[0, 1, 2, 3, 3], [0, 1, 2, 2, 2]]
    assert result.member_token_ids.tolist() == [[10, 11, 12, -1], [20, 21, -1, -1]]
    assert result.telemetry["ratio"] == 1.0

    result.tokens.sum().backward()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


@pytest.mark.parametrize(
    "metadata",
    [
        {"video_id": "forbidden"},
        {"labels": torch.tensor([1])},
        {"future_layer": torch.zeros(1)},
        {"nested": {"anomaly_category": "forbidden"}},
    ],
)
def test_reduction_context_rejects_deployment_forbidden_information(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(TokenReductionContractError, match="不得包含"):
        ReductionContext(layer_depth=0, metadata=metadata)


def test_layout_rejects_invalid_padding() -> None:
    with pytest.raises(TokenReductionContractError, match="original_token_ids"):
        TokenLayout(
            valid_mask=torch.tensor([[True, False]]),
            original_token_ids=torch.tensor([[0, 5]]),
            special_token_mask=torch.tensor([[False, False]]),
            mass=torch.tensor([[1.0, 0.0]]),
            source_coordinates=None,
            position_contract="verified",
        )


def test_identity_csr_handles_holes_and_rejects_compressed_budget():
    valid = torch.tensor([[True, False, True]])
    layout = TokenLayout(
        valid_mask=valid,
        original_token_ids=torch.tensor([[0, -1, 2]]),
        special_token_mask=torch.zeros_like(valid),
        mass=valid.float(),
        source_coordinates=None,
        position_contract="fixture",
    )
    tokens = torch.ones(1, 3, 2)
    result = IdentityReducer().reduce(tokens, layout, ReductionContext(0))
    assert result.member_offsets.tolist() == [[0, 1, 1, 2]]
    assert result.member_token_ids.tolist() == [[0, 2, -1]]
    with pytest.raises(TokenReductionContractError, match="budget"):
        IdentityReducer().reduce(tokens, layout, ReductionContext(0, budget=2))


def test_layout_rejects_non_real_coordinate_shape():
    with pytest.raises(TokenReductionContractError, match=r"\[B, N, 3\]"):
        TokenLayout(
            valid_mask=torch.tensor([[True]]),
            original_token_ids=torch.tensor([[0]]),
            special_token_mask=torch.tensor([[False]]),
            mass=torch.tensor([[1.0]]),
            source_coordinates=torch.zeros(1, 1, 2),
            position_contract="verified",
        )
