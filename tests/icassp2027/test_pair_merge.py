from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from vadbench.token_reduction import (
    PairLinearGate,
    PairMergeError,
    PairMergePlanCache,
    PairWeightedMerge,
    horizontal_pair_merge_spec,
)
from vadbench.token_reduction.bridges import create_observation_bridge, indexed_gather
from vadbench.token_reduction.contracts import TokenLayout


def _patch_layout() -> TokenLayout:
    return TokenLayout(
        valid_mask=torch.ones((1, 4), dtype=torch.bool),
        original_token_ids=torch.arange(4, dtype=torch.int64).unsqueeze(0),
        special_token_mask=torch.zeros((1, 4), dtype=torch.bool),
        mass=torch.ones((1, 4)),
        source_coordinates=torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]]),
        position_contract="fixture-native",
        provenance={"grid": [1, 1, 4], "coordinate_source": "verified-fixture"},
    )


def _timesformer_layout() -> TokenLayout:
    return TokenLayout(
        valid_mask=torch.ones((1, 9), dtype=torch.bool),
        original_token_ids=torch.arange(9, dtype=torch.int64).unsqueeze(0),
        special_token_mask=torch.tensor([[True, False, False, False, False, False, False, False, False]]),
        mass=torch.ones((1, 9)),
        source_coordinates=torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]]
        ),
        position_contract="hf-timesformer-absolute",
        provenance={"grid": [2, 2, 2], "coordinate_source": "verified-fixture"},
    )


def test_pair_mean_uses_first_member_anchors_and_linear_gate_has_gradient():
    spec = horizontal_pair_merge_spec("videomae", _patch_layout())
    assert spec.output_indices.tolist() == [[0, 2]]
    assert spec.pair_indices.tolist() == [[[0, 1], [2, 3]]]
    assert spec.output_layout.mass.tolist() == [[2.0, 2.0]]
    assert spec.receipt()["input_valid_mass"] == spec.receipt()["output_valid_mass"] == [4.0]
    assert spec.receipt()["position_anchor"] == "first_member_native_index"
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [2.0, 1.0], [6.0, 5.0]]])
    mean = PairWeightedMerge(2)(hidden, spec)
    torch.testing.assert_close(mean, torch.tensor([[[2.0, 1.0], [4.0, 3.0]]]))

    gate = PairLinearGate(2)
    weighted = PairWeightedMerge(2, gate)(hidden, spec)
    torch.testing.assert_close(weighted, mean)  # w=0 is exactly the documented mean.
    weighted.square().sum().backward()
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0


def test_pair_plan_cache_reuses_verified_topology_without_video_identity():
    cache = PairMergePlanCache()
    first = cache.resolve("videomae", _patch_layout())
    second = cache.resolve("videomae", _patch_layout())
    assert first.pair_indices.data_ptr() == second.pair_indices.data_ptr()
    assert first.output_layout.mass.tolist() == second.output_layout.mass.tolist() == [[2.0, 2.0]]


def test_cache_binds_real_coordinate_order_not_just_grid_shape():
    cache = PairMergePlanCache()
    original = _patch_layout()
    first = cache.resolve("videomae", original)
    order = torch.tensor([0, 2, 1, 3])
    reordered = replace(
        original,
        source_coordinates=original.source_coordinates[:, order],
        original_token_ids=original.original_token_ids[:, order],
    )
    cached = cache.resolve("videomae", reordered)
    direct = horizontal_pair_merge_spec("videomae", reordered)
    assert cached.pair_indices.tolist() == direct.pair_indices.tolist() == [[[0, 2], [1, 3]]]
    assert cached.output_indices.tolist() == [[0, 1]]
    hidden = torch.tensor([[[0.], [20.], [10.], [30.]]])
    torch.testing.assert_close(PairWeightedMerge(1)(hidden, cached), torch.tensor([[[5.], [25.]]]))
    with pytest.raises(PairMergeError, match="topology"):
        first.with_layout(reordered)


def test_cache_does_not_reuse_a_plan_after_original_ids_change():
    cache = PairMergePlanCache()
    original = _patch_layout()
    first = cache.resolve("videomae", original)
    changed = replace(original, original_token_ids=original.original_token_ids + 8192)
    second = cache.resolve("videomae", changed)
    assert first.pair_indices.data_ptr() != second.pair_indices.data_ptr()
    assert second.output_layout.original_token_ids.tolist() == [[8192, 8194]]
    with pytest.raises(PairMergeError, match="topology"):
        first.with_layout(changed)


def test_cache_reuses_topology_but_recomputes_current_mass():
    cache = PairMergePlanCache()
    original = _patch_layout()
    first = cache.resolve("videomae", original)
    changed = replace(original, mass=torch.tensor([[1., 2., 3., 4.]]))
    second = cache.resolve("videomae", changed)
    assert first.pair_indices.data_ptr() == second.pair_indices.data_ptr()
    assert first.input_topology_sha256 == second.input_topology_sha256
    assert second.output_layout.mass.tolist() == [[3., 7.]]
    assert second.receipt()["input_valid_mass"] == second.receipt()["output_valid_mass"] == [10.]


def test_topology_signature_is_not_aliased_to_mutable_coordinate_tensor():
    layout = _patch_layout()
    cache = PairMergePlanCache()
    first = cache.resolve("videomae", layout)
    layout.source_coordinates[:, [1, 2]] = layout.source_coordinates[:, [2, 1]]
    second = cache.resolve("videomae", layout)
    assert first.input_topology_sha256 != second.input_topology_sha256
    assert second.pair_indices.tolist() == [[[0, 2], [1, 3]]]
    with pytest.raises(PairMergeError, match="topology"):
        first.with_layout(layout)


def test_cached_plan_cannot_turn_padding_or_extra_specials_into_real_mass():
    cache = PairMergePlanCache()
    original = _patch_layout()
    cache.resolve("videomae", original)
    padded = replace(original, valid_mask=torch.tensor([[True, True, True, False]]), original_token_ids=torch.tensor([[0, 1, 2, -1]]), mass=torch.tensor([[1., 1., 1., 0.]]))
    with pytest.raises(PairMergeError, match="padding"):
        cache.resolve("videomae", padded)
    special = replace(original, special_token_mask=torch.tensor([[True, False, False, False]]))
    with pytest.raises(PairMergeError, match="patch-only"):
        cache.resolve("videomae", special)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_gate_exactly_matches_mean_above_8192_native_indices(dtype):
    grid = (2, 64, 66)
    count = 2 * 64 * 66
    coordinates = torch.cartesian_prod(*(torch.arange(n) for n in grid)).unsqueeze(0)
    layout = TokenLayout(
        valid_mask=torch.ones((1, count), dtype=torch.bool),
        original_token_ids=torch.arange(count).unsqueeze(0),
        special_token_mask=torch.zeros((1, count), dtype=torch.bool),
        mass=torch.ones((1, count)),
        source_coordinates=coordinates,
        position_contract="fixture-native",
        provenance={"grid": list(grid), "coordinate_source": "verified-fixture"},
    )
    spec = horizontal_pair_merge_spec("vjepa2", layout)
    expected_ids = torch.arange(0, count, 2, dtype=torch.int64).unsqueeze(0)
    assert expected_ids.max() > 8192
    assert spec.output_indices.dtype == torch.int64
    assert torch.equal(spec.output_indices, expected_ids)
    assert torch.equal(spec.output_layout.original_token_ids, expected_ids)
    assert torch.equal(spec.output_layout.source_coordinates, coordinates[:, expected_ids[0]])
    hidden = torch.randn((1, count, 3), generator=torch.Generator().manual_seed(43)).to(dtype)
    if dtype == torch.float16:
        # Valid small fp16 activations expose premature product rounding.
        hidden[0, -2:, 0] = torch.tensor([2 ** -24, 2 ** -23], dtype=dtype)
    mean = PairWeightedMerge(3)(hidden, spec)
    gated = PairWeightedMerge(3, PairLinearGate(3).to(dtype))(hidden, spec)
    assert torch.equal(mean, gated)
    torch.testing.assert_close(mean[:, -1], hidden[:, -2:].mean(dim=1), rtol=0, atol=0)


def test_timesformer_pair_weights_are_tied_across_complete_trajectories():
    spec = horizontal_pair_merge_spec("timesformer", _timesformer_layout())
    assert spec.output_indices.tolist() == [[0, 1, 2, 5, 6]]
    assert spec.gate_group_ids.tolist() == [[0, 0, 1, 1]]
    assert spec.output_layout.mass.tolist() == [[1.0, 2.0, 2.0, 2.0, 2.0]]
    assert spec.receipt()["special_mass_preserved"]
    pairs = torch.tensor(
        [[[[1.0], [4.0]], [[3.0], [8.0]], [[2.0], [5.0]], [[9.0], [10.0]]]]
    )
    gate = PairLinearGate(1)
    with torch.no_grad():
        gate.weight.fill_(0.7)
    weights = gate(pairs, trajectory_length=spec.trajectory_length)
    torch.testing.assert_close(weights[:, 0], weights[:, 1])
    torch.testing.assert_close(weights[:, 2], weights[:, 3])
    expected = torch.softmax(torch.tensor([[2.0, 6.0], [5.5, 7.5]]) * .7, dim=-1)
    torch.testing.assert_close(weights[0, [0, 2]], expected)
    assert not torch.equal(weights[:, 0], weights[:, 2])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_timesformer_zero_gate_mean_preserve_cls_and_anchor_mass(dtype):
    layout = _timesformer_layout()
    spec = horizontal_pair_merge_spec("timesformer", layout)
    hidden = torch.arange(18, dtype=dtype).reshape(1, 9, 2) / 8
    if dtype == torch.float16:
        hidden[0, 1, 0] = 2 ** -24
        hidden[0, 3, 0] = 2 ** -23
    mean = PairWeightedMerge(2)(hidden, spec)
    gated = PairWeightedMerge(2, PairLinearGate(2).to(dtype))(hidden, spec)
    assert torch.equal(mean, gated)
    assert torch.equal(mean[:, 0], hidden[:, 0])
    assert spec.output_layout.original_token_ids.tolist() == [[0, 1, 2, 5, 6]]
    assert torch.equal(spec.output_layout.source_coordinates, layout.source_coordinates[:, [0, 1, 2, 5, 6]])
    assert spec.receipt()["input_valid_mass"] == spec.receipt()["output_valid_mass"] == [9.]


def test_learned_gate_and_hidden_gradients_match_finite_differences():
    spec = horizontal_pair_merge_spec("videomae", _patch_layout())
    merge = PairWeightedMerge(2, PairLinearGate(2)).double()
    hidden = torch.tensor([[[1., .2], [-.3, .7], [.4, -.9], [1.2, .5]]], dtype=torch.double, requires_grad=True)
    weight = torch.tensor([.2, -.4], dtype=torch.double, requires_grad=True)

    def apply(h, w):
        return torch.func.functional_call(merge, {"gate.weight": w}, (h, spec))

    assert torch.autograd.gradcheck(apply, (hidden, weight))
    apply(hidden, weight).square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and weight.grad.abs().sum() > 0


def test_low_precision_accumulation_keeps_gate_and_hidden_gradients():
    spec = horizontal_pair_merge_spec("videomae", _patch_layout())
    hidden = torch.tensor([[[1., 0.], [3., 2.], [2., 1.], [6., 5.]]], dtype=torch.float16, requires_grad=True)
    gate = PairLinearGate(2).half()
    PairWeightedMerge(2, gate)(hidden, spec).float().square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all()
    assert gate.weight.grad.abs().sum() > 0


class _Block(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Block(), _Block()])
        self.calls = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        for block in self.encoder.layer:
            hidden = block(hidden)
        return hidden


def test_indexed_transform_merges_current_block_output_in_one_real_forward():
    layout = _patch_layout()
    spec = horizontal_pair_merge_spec("videomae", layout)
    model = _Model()
    bridge = create_observation_bridge("videomae", model)
    merge = PairWeightedMerge(2)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    with indexed_gather(
        bridge,
        0,
        spec.output_indices,
        layout,
        transform=lambda current, _indices: merge(current, spec),
    ) as intervention:
        output = model(hidden)
        receipt = intervention.validate_execution()
    assert model.calls == 1
    torch.testing.assert_close(output, torch.tensor([[[4.0, 5.0], [8.0, 9.0]]]))
    assert receipt["gathered_shape"] == [1, 2, 2]


class _VJBlock(nn.Module):
    def forward(self, hidden: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return hidden + 1


class _VJModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_VJBlock(), _VJBlock()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.encoder.layer:
            hidden = block(hidden)
        return hidden


def test_vjepa_position_ids_are_passed_without_cpu_mask_dump_for_formal_runs():
    layout = _patch_layout()
    spec = horizontal_pair_merge_spec("vjepa2", layout)
    bridge = create_observation_bridge("vjepa2", _VJModel())
    with indexed_gather(
        bridge,
        0,
        spec.output_indices,
        layout,
        transform=lambda current, _indices: PairWeightedMerge(2)(current, spec),
        record_position_masks=False,
    ) as intervention:
        output = bridge.model(torch.ones((1, 4, 2)))
        receipt = intervention.validate_execution()
    assert output.shape == (1, 2, 2)
    assert receipt["vjepa2_position_injections"] == 1
    assert receipt["recorded_position_masks"] is False
    assert intervention.suffix_position_masks == {}
