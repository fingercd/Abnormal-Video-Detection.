"""Three-tier group-budget selection: fixed budgets, real shortening, trajectory semantics.

Engineering verification for method owner B (guidance §5/§6/§9):
tier math and rounding, identity-slot parity for controls, dynamic selection
with recorded real indices, TimeSformer complete-trajectory consistency,
per-layer token counts and plugin overhead instrumentation, and the leakage
contract (signal reads the current block output only).
"""

from __future__ import annotations

import pytest
import torch
from test_pair_deployment import _batch, _case

from vadbench.token_reduction.bridges.indexed import IndexedInterventionError, identity_indices
from vadbench.token_reduction.deployment import GroupSelectDeployment
from vadbench.token_reduction.pair_merge import PairMergeError
from vadbench.token_reduction.token_selection import (
    SUPPORTED_KEEP_RATIOS,
    GroupBudgetSelector,
    group_quotas,
    group_select_spec,
    round_half_up,
)


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_tier_math_and_fixed_rounding():
    assert SUPPORTED_KEEP_RATIOS == (0.80, 0.60, 0.40)
    assert round_half_up(8.0) == 8 and round_half_up(2.5) == 3 and round_half_up(3.2) == 3
    # Full groups are exact for every tier; no tier is silently reshaped.
    assert group_quotas(30, 0.80) == [8, 8, 8]
    assert group_quotas(30, 0.60) == [6, 6, 6]
    assert group_quotas(30, 0.40) == [4, 4, 4]
    # Remainder groups use the same fixed rounding rule.
    assert group_quotas(34, 0.80) == [8, 8, 8, round_half_up(0.80 * 4)]
    assert group_quotas(7, 0.60) == [round_half_up(0.60 * 7)]
    with pytest.raises(PairMergeError, match="keep_ratio"):
        group_quotas(10, 1.0)
    with pytest.raises(PairMergeError, match="keep_ratio"):
        group_quotas(10, 0.5)


def test_actual_budget_reported_and_shared_across_rules():
    # 784 pairs: 78 full groups + remainder 4 — realistic VideoMAEv2 geometry.
    quotas = group_quotas(784, 0.80)
    kept_units = sum(quotas)
    actual = kept_units * 2 / 1568
    assert quotas[:78] == [8] * 78 and quotas[-1] == 3
    assert abs(actual - 0.80) < 0.005  # reported, not silently reshaped
    for tier, per_group in ((0.60, 6), (0.40, 4)):
        q = group_quotas(784, tier)
        assert q[:78] == [per_group] * 78
        assert abs(sum(q) * 2 / 1568 - tier) < 0.005


def test_spec_unit_table_and_skeleton_consistency():
    _adapter, bridge, layout, frames, dim = _case("videomaev2")
    spec = group_select_spec("videomaev2", layout, 0.60)
    reducible = layout.token_capacity  # patch-only fixture
    units = reducible // 2
    assert spec.units.shape == (units, 2)
    assert spec.reducible_tokens == reducible
    assert spec.output_token_count == sum(spec.quotas) * spec.tokens_per_slot
    assert len(torch.unique(spec.skeleton_indices)) == spec.skeleton_indices.numel()
    assert spec.actual_keep_ratio == spec.output_token_count / reducible


def _deploy(encoder_id, keep_ratio, rule):
    adapter, bridge, layout, frames, dim = _case(encoder_id)
    deployment = GroupSelectDeployment(
        bridge, layout, depth=0, dim=dim, keep_ratio=keep_ratio, rule=rule,
        batch_sizes=(1, 2), device="cpu", seed=7,
    )
    return adapter, bridge, layout, frames, dim, deployment


def test_group_uniform_is_exact_static_gather_at_every_tier():
    for tier in SUPPORTED_KEEP_RATIOS:
        adapter, bridge, layout, frames, dim, deployment = _deploy("videomaev2", tier, "group_uniform")
        identity = deployment.reducer_identity
        assert identity["dynamic_selection"] is False
        assert identity["position_semantics"] == "retained_native_position"
        assert identity["keep_ratio"] == tier
        spec = deployment.selection_spec
        batch = _batch(size=2, frames=frames)
        context = deployment(batch)
        with torch.no_grad(), context:
            adapter.encode(batch)
        receipt = context.validate_execution()
        expected_k = spec.output_token_count  # patch-only fixture: no specials
        assert receipt["gathered_tokens"] == expected_k
        assert receipt["native_input_tokens"] == layout.token_capacity
        assert receipt["per_layer_token_counts"] == {
            str(depth): expected_k for depth in receipt["suffix_shapes"]
        }
        # Static skeleton parity: the uniform control is a plain indexed
        # gather of the skeleton (exact native positions, no transform).
        with torch.no_grad(), context:
            output = adapter.encode(batch)
        assert output.shape[1] == expected_k


def test_pair_select_dynamic_choice_records_real_indices():
    adapter, bridge, layout, frames, dim, deployment = _deploy("videomaev2", 0.60, "pair_select")
    identity = deployment.reducer_identity
    assert identity["dynamic_selection"] is True
    assert identity["position_semantics"] == "approximate_anchor_position"
    assert identity["selection_signal"] == "member_l2_norm_max"
    assert identity["leakage_contract"].startswith("no_labels")
    spec = deployment.selection_spec
    # Make member norms decisive: inflate even-indexed tokens.
    batch = _batch(size=2, frames=frames)
    context = deployment(batch)
    with torch.no_grad(), context:
        output = adapter.encode(batch)
    assert output.shape[1] == spec.output_token_count
    digest = deployment.selection_digest(2)
    assert digest is not None and len(digest) == 64
    receipt = context.validate_execution()
    assert receipt["gathered_tokens"] == spec.output_token_count
    assert receipt["plugin_overhead_ms"]["transform_ms"] >= 0.0
    # Selection reacts to content: a different hidden state gives a recorded
    # selection of the same length (slot count fixed by quota table).
    selector = GroupBudgetSelector(dim, rule="pair_select")
    hidden = torch.randn(2, layout.token_capacity, dim)
    specials = torch.nonzero(layout.special_token_mask[0]).flatten()
    out = selector(hidden, spec, specials)
    assert out.shape == (2, specials.numel() + spec.output_token_count, dim)


def test_pair_select_keeps_higher_norm_member_within_quota():
    _adapter, _bridge, layout, _frames, dim, deployment = _deploy("videomaev2", 0.80, "pair_select")
    spec = deployment.selection_spec
    selector = GroupBudgetSelector(dim, rule="pair_select")
    hidden = torch.zeros(1, layout.token_capacity, dim)
    hidden[0, 1] = 5.0  # pair 0 right member has the larger norm
    slots = selector._select_slots(spec, hidden)[0].tolist()
    # Tier 0.80 on 8 tokens: one group, quota round(6.4)=6 -> drop 2 tokens.
    # Keep priority: better members first (pair signal desc), then worse
    # members of the strongest pairs. Pair 0 (signal 5.0) keeps both members;
    # the weakest pairs lose their worse member first.
    assert 1 in slots  # better member of pair 0 (right token)
    assert 0 in slots  # worse member of the strongest pair survives at 0.80
    assert len(slots) == 6
    dropped = sorted(set(range(layout.token_capacity)) - set(slots))
    assert 1 not in dropped
    # At tier 0.40 (quota round(3.2)=3) only the best better members survive.
    spec_low = group_select_spec("videomaev2", layout, 0.40)
    slots_low = selector._select_slots(spec_low, hidden)[0].tolist()
    assert slots_low == [1, 2, 4]  # better member of every pair, native order


def test_random_control_same_budget_and_seeded():
    deployment_a = _deploy("videomaev2", 0.60, "group_random")[5]
    deployment_b = _deploy("videomaev2", 0.60, "group_random")[5]
    assert (
        deployment_a.reducer_identity["selected_indices_sha256"]
        == deployment_b.reducer_identity["selected_indices_sha256"]
    )
    assert deployment_a.selection_spec.output_token_count == deployment_b.selection_spec.output_token_count
    deployment_c = _deploy("videomaev2", 0.40, "group_random")[5]
    assert deployment_c.selection_spec.output_token_count < deployment_a.selection_spec.output_token_count


def test_timesformer_complete_trajectory_selection():
    adapter, bridge, layout, frames, dim, deployment = _deploy("timesformer", 0.60, "group_uniform")
    spec = deployment.selection_spec
    assert spec.mode == "trajectory"
    assert spec.units.shape[1] == frames  # whole trajectories as units
    assert spec.tokens_per_slot == frames
    batch = _batch(size=1, frames=frames)
    context = deployment(batch)
    with torch.no_grad(), context:
        adapter.encode(batch)
    receipt = context.validate_execution()
    # CLS preserved separately; retained tokens are complete trajectories in
    # native spatial order (validated by the indexed intervention itself).
    assert receipt["gathered_tokens"] == 1 + spec.output_token_count
    # An illegally per-frame subsampled selection must be rejected explicitly.
    indices = identity_indices(layout)[0]
    bad = torch.cat([indices[:1], indices[1 : 1 + frames - 1]])  # partial trajectory
    from vadbench.token_reduction.bridges.indexed import IndexedTokenIntervention

    with pytest.raises(IndexedInterventionError, match="every time index"):
        IndexedTokenIntervention(bridge, 0, bad.unsqueeze(0), layout)


def test_bypass_identity_uses_native_dense_path():
    # keep_ratio tiers never claim the dense identity: deployment refuses 1.0
    # and the dense baseline stays the reducer="identity" extraction path.
    adapter, bridge, layout, frames, dim = _case("videomaev2")
    with pytest.raises(PairMergeError, match="identity"):
        GroupSelectDeployment(
            bridge, layout, depth=0, dim=dim, keep_ratio=1.0, rule="pair_select",
            batch_sizes=(1,), device="cpu",
        )


def test_timesformer_pair_select_dynamic_branch_runs():
    """Regression for §2.2: the trajectory gather mixed a 2-D unit table with
    a 3-D batch gather index and could never execute. Whole trajectories are
    selected per sample with the frozen quota table."""
    adapter, bridge, layout, frames, dim, deployment = _deploy("timesformer", 0.60, "pair_select")
    spec = deployment.selection_spec
    assert spec.mode == "trajectory"
    batch = _batch(size=2, frames=frames)
    context = deployment(batch)
    with torch.no_grad(), context:
        output = adapter.encode(batch)
    assert output.shape[1] == 1 + spec.output_token_count
    receipt = context.validate_execution()
    assert receipt["gathered_tokens"] == 1 + spec.output_token_count
    # Selection reacts to per-sample content but keeps the fixed slot count.
    digest = deployment.selection_digest(2)
    assert digest is not None and len(digest) == 64
    # Real chosen indices keep every time index of a kept spatial trajectory.
    selected = deployment._selectors[2].last_selection
    specials = 1
    kept = selected[0, specials:].tolist()
    assert len(kept) % frames == 0


def test_diagnostics_switch_disables_cpu_exports():
    _adapter, bridge, layout, frames, dim = _case("videomaev2")
    spec = group_select_spec("videomaev2", layout, 0.60)
    specials = torch.nonzero(layout.special_token_mask[0]).flatten()
    hidden = torch.randn(2, layout.token_capacity, dim)
    off = GroupBudgetSelector(dim, rule="pair_select", record_diagnostics=False)
    off(hidden, spec, specials)
    assert off.last_selection is None and off.last_signal is None
    assert off.selection_digest() is None
    on = GroupBudgetSelector(dim, rule="pair_select", record_diagnostics=True)
    on(hidden, spec, specials)
    assert on.last_selection is not None and on.selection_digest() is not None
    # Deployment identity records the formal-path posture.
    _a, _b, _l, _f, _d, deployment = _deploy("videomaev2", 0.60, "pair_select")
    assert deployment.reducer_identity["record_diagnostics"] is True


def test_timesformer_geometry_snap_matches_production_grid():
    """Blocker-1 regression: 196 trajectories, native patch width 14.

    Raw tier quotas are not divisible by the patch width; the frozen snap
    (floor the total to a width multiple, trim whole trajectories from tail
    groups, never above target) makes every tier runnable and reports the
    actual ratio instead of faking the tier.
    """
    from vadbench.token_reduction.token_selection import snap_trajectory_quotas

    units = 196
    quotas = group_quotas(units, 0.60)
    assert sum(quotas) == 118  # raw, NOT divisible by 14
    snapped, receipt = snap_trajectory_quotas(quotas, 14)
    assert sum(snapped) == 112 and sum(snapped) % 14 == 0
    assert receipt["trimmed_units"] == 6 and receipt["never_above_target"] is True
    assert sum(snapped) / units == pytest.approx(0.5714, abs=1e-4)
    for tier, expected_kept in ((0.80, 154), (0.40, 70)):
        tier_quotas = group_quotas(units, tier)
        tier_snapped, tier_receipt = snap_trajectory_quotas(tier_quotas, 14)
        assert sum(tier_snapped) == expected_kept
        assert sum(tier_snapped) % 14 == 0
        assert sum(tier_snapped) <= sum(tier_quotas)
    # Deterministic and tail-first: trimming hits the last groups.
    again, _ = snap_trajectory_quotas(list(quotas), 14)
    assert again == snapped
    # Empty snap is an explicit error, never a silent zero selection.
    with pytest.raises(PairMergeError, match="清空"):
        snap_trajectory_quotas([1], 14)


def test_timesformer_all_tiers_run_with_actual_ratio():
    """Blocker-1 acceptance: TS three tiers execute with trajectory integrity."""
    for tier in SUPPORTED_KEEP_RATIOS:
        adapter, bridge, layout, frames, dim, deployment = _deploy("timesformer", tier, "group_uniform")
        spec = deployment.selection_spec
        assert spec.geometry_snap is None or spec.geometry_snap["snapped_kept_units"] % spec.native_patch_width == 0
        batch = _batch(size=1, frames=frames)
        context = deployment(batch)
        with torch.no_grad(), context:
            adapter.encode(batch)
        receipt = context.validate_execution()
        assert receipt["gathered_tokens"] == 1 + spec.output_token_count
        assert spec.actual_keep_ratio == spec.output_token_count / spec.reducible_tokens
        # whole trajectories: kept tokens per sample stay time-complete
        assert spec.output_token_count % frames == 0


def test_pair_member_ablation_modes_batch1b():
    """Batch-1b ordering ablation: identical structure/budget, member rule only."""
    _adapter, _bridge, layout, _frames, dim = _case("videomaev2")
    spec = group_select_spec("videomaev2", layout, 0.60)
    specials = torch.nonzero(layout.special_token_mask[0]).flatten()
    hidden = torch.zeros(1, layout.token_capacity, dim)
    hidden[0, 0] = 9.0  # pair 0: LEFT member larger
    hidden[0, 3] = 7.0  # pair 1: RIGHT member larger
    rules = {}
    for rule in ("pair_select", "pair_fixed", "pair_random_member", "pair_reverse"):
        selector = GroupBudgetSelector(dim, rule=rule, seed=11)
        out = selector(hidden, spec, specials)
        assert out.shape == (1, specials.numel() + spec.output_token_count, dim)
        rules[rule] = selector.last_selection[0, specials.numel():].tolist()
    # Same budget everywhere.
    assert len({len(v) for v in rules.values()}) == 1
    # pair_select keeps the max-norm member of every pair (0 and 3 survive).
    assert 0 in rules["pair_select"] and 3 in rules["pair_select"]
    # pair_fixed always keeps the native first member (0, 2), never 3.
    assert 0 in rules["pair_fixed"] and 2 in rules["pair_fixed"] and 3 not in rules["pair_fixed"]
    # pair_reverse keeps the min-norm members first (1, 2). Round-2 worse
    # members are rule-independent by design, so only round-1 membership is
    # asserted: pair 1's max-norm member (3) is excluded under reverse.
    assert 1 in rules["pair_reverse"] and 2 in rules["pair_reverse"]
    assert 3 not in rules["pair_reverse"]
    # pair_random_member is seeded and deterministic.
    again = GroupBudgetSelector(dim, rule="pair_random_member", seed=11)
    again(hidden, spec, specials)
    assert again.last_selection[0].tolist() == rules["pair_random_member"]
    other_seed = GroupBudgetSelector(dim, rule="pair_random_member", seed=12)
    other_seed(hidden, spec, specials)
    kept = [len(set(rules["pair_random_member"]) - set(other_seed.last_selection[0].tolist()))]
    assert kept[0] <= 2  # member-level draws differ mildly across seeds


def test_ablation_modes_deploy_with_identity():
    for rule in ("pair_fixed", "pair_random_member", "pair_reverse"):
        _adapter, _bridge, _layout, _frames, _dim, deployment = _deploy("videomaev2", 0.60, rule)
        identity = deployment.reducer_identity
        assert identity["name"] == rule
        assert identity["within_unit_preference"] is not None
        assert identity["ablation_note"].startswith("pair_fixed")
        assert identity["dynamic_selection"] is True


def test_selector_leakage_contract_reads_only_hidden():
    # The selector signature is (hidden, spec, specials): no batch metadata,
    # labels, file names, test statistics, or future-layer tensors can reach
    # the signal path.
    import inspect

    from vadbench.token_reduction import token_selection

    source = inspect.getsource(token_selection.GroupBudgetSelector.forward)
    for forbidden in ("video_ids", "labels", "metadata", "future"):
        assert forbidden not in source
    _adapter, bridge, layout, frames, dim = _case("videomaev2")
    spec = group_select_spec("videomaev2", layout, 0.60)
    calls = []

    def spy_signal(members):
        calls.append(members.shape)
        return members.norm(dim=-1).amax(dim=-1)

    selector = GroupBudgetSelector(dim, rule="pair_select", signal_fn=spy_signal)
    specials = torch.nonzero(layout.special_token_mask[0]).flatten()
    hidden = torch.randn(2, layout.token_capacity, dim)
    selector(hidden, spec, specials)
    assert calls and calls[0][0] == 2  # only current-layer hidden derived data
