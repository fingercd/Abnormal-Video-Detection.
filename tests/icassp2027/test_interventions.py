from __future__ import annotations

import pytest
import torch

from vadbench.research.interventions import (
    InterventionScoreError,
    fixed_budget_indices,
    relative_branch_update,
    same_position_temporal_change,
)
from vadbench.token_reduction.contracts import TokenLayout


def _layout(*, timesformer: bool = False, padding: bool = False, duplicate: bool = False) -> TokenLayout:
    # CLS + four spatial tracks, each T=2 and represented in spatial-major order.
    coordinates = torch.tensor(
        [[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]]
    )
    if duplicate:
        coordinates[0, 4] = coordinates[0, 3]
    valid = torch.ones((1, 9), dtype=torch.bool)
    if padding:
        valid[0, -1] = False
    ids = torch.arange(9).unsqueeze(0)
    ids[~valid] = -1
    special = torch.zeros_like(valid)
    special[0, 0] = True
    return TokenLayout(
        valid_mask=valid,
        original_token_ids=ids,
        special_token_mask=special,
        mass=valid.float(),
        source_coordinates=coordinates,
        position_contract=(
            "hf-timesformer-absolute-spatial-and-temporal-position-before-divided-attention"
            if timesformer
            else "verified-patch-layout"
        ),
        provenance={"grid": [2, 2, 2], "coordinate_source": "fixture"},
    )


def test_fixed_budget_controls_identity_specials_order_and_seed():
    layout = _layout()
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    identity = fixed_budget_indices(scores, layout, 9, "score_low")
    assert identity.indices.tolist() == [list(range(9))]
    high = fixed_budget_indices(scores, layout, 5, "score_high")
    low = fixed_budget_indices(scores, layout, 5, "score_low")
    uniform = fixed_budget_indices(scores, layout, 5, "uniform")
    random_a = fixed_budget_indices(scores, layout, 5, "seeded_random", seed=7)
    random_b = fixed_budget_indices(scores, layout, 5, "seeded_random", seed=7)
    assert high.indices.tolist() == [[0, 5, 6, 7, 8]]
    assert low.indices.tolist() == [[0, 1, 2, 3, 4]]
    assert uniform.indices[0, 0].item() == 0
    assert torch.equal(random_a.indices, random_b.indices)
    for selection in (high, low, uniform, random_a):
        assert selection.indices.shape == (1, 5)
        assert torch.equal(selection.indices, torch.sort(selection.indices, dim=1).values)
    with pytest.raises(InterventionScoreError, match="未知 strategy"):
        fixed_budget_indices(scores, layout, 9, "bad")  # type: ignore[arg-type]
    with pytest.raises(InterventionScoreError, match="需要整数 seed"):
        fixed_budget_indices(scores, layout, 9, "seeded_random")


def test_timesformer_controls_keep_cls_complete_trajectories_and_width_aligned_budget():
    layout = _layout(timesformer=True)
    scores = torch.tensor([[0.0, 10.0, 10.0, 9.0, 9.0, 1.0, 1.0, 2.0, 2.0]])
    selection = fixed_budget_indices(scores, layout, 6, "score_high")
    # Requested 6 permits two complete trajectories after width alignment: CLS + 4 patches.
    assert selection.requested_budget == 6
    assert selection.effective_budget == 5
    assert selection.indices.tolist() == [[0, 1, 2, 3, 4]]
    for start in (1, 3):
        assert selection.indices[0, start : start + 2].tolist()[1] == selection.indices[0, start : start + 2].tolist()[0] + 1
    with pytest.raises(InterventionScoreError, match="完整且 patch-width"):
        fixed_budget_indices(scores, layout, 2, "uniform")


def test_relative_branch_update_is_differentiable_and_rejects_zero_or_nonfinite_input_norms():
    branch = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]], requires_grad=True)
    source = torch.tensor([[[6.0, 8.0], [1.0, 0.0]]], requires_grad=True)
    scores = relative_branch_update(branch, source)
    torch.testing.assert_close(scores, torch.tensor([[0.5, 2.0]]))
    scores.sum().backward()
    assert branch.grad is not None and source.grad is not None
    with pytest.raises(InterventionScoreError, match="为零"):
        relative_branch_update(branch.detach(), torch.zeros_like(source))
    bad = source.detach().clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(InterventionScoreError, match="非有限"):
        relative_branch_update(branch.detach(), bad)


def test_same_position_temporal_change_uses_verified_tracks_and_rejects_ambiguous_layouts():
    layout = _layout()
    tokens = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]],
        requires_grad=True,
    )
    result = same_position_temporal_change(tokens, layout)
    assert not result.available_mask[0, 0]
    assert result.available_mask[0, 1:].all()
    torch.testing.assert_close(result.scores[0, 1:3], torch.tensor([2.0, 2.0]))
    result.scores.sum().backward()
    assert tokens.grad is not None
    with pytest.raises(InterventionScoreError, match="padding"):
        same_position_temporal_change(tokens.detach(), _layout(padding=True))
    with pytest.raises(InterventionScoreError, match="重复来源"):
        same_position_temporal_change(tokens.detach(), _layout(duplicate=True))
    no_coordinates = TokenLayout(
        valid_mask=torch.ones((1, 2), dtype=torch.bool),
        original_token_ids=torch.tensor([[0, 1]]),
        special_token_mask=torch.zeros((1, 2), dtype=torch.bool),
        mass=torch.ones((1, 2)),
        source_coordinates=None,
        position_contract="fixture",
    )
    with pytest.raises(InterventionScoreError, match="真实坐标"):
        same_position_temporal_change(torch.ones((1, 2, 2)), no_coordinates)
    with pytest.raises(InterventionScoreError, match=r"浮点"):
        same_position_temporal_change(torch.ones((1, 9, 2), dtype=torch.int64), layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; no GPU smoke is run here")
def test_timesformer_cpu_layout_cuda_scores_keeps_native_indices_on_score_device():
    layout = _layout(timesformer=True)  # TokenLayout deliberately remains CPU.
    scores = torch.tensor(
        [[0.0, 10.0, 10.0, 9.0, 9.0, 1.0, 1.0, 2.0, 2.0]], device="cuda"
    )
    selection = fixed_budget_indices(scores, layout, 6, "score_high")
    assert selection.indices.device == scores.device
    assert selection.indices.tolist() == [[0, 1, 2, 3, 4]]
