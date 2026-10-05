"""Unified three-tier group-budget token selection at one native encoder depth.

Fixed budget rule (definition frozen 2026-09-20, method owner B; the pairselect
candidate is validated against this rule, it is not a settled method):

* ``keep_ratio`` ∈ {0.80, 0.60, 0.40} names the target share of *reducible*
  tokens. Special tokens are preserved separately and never counted.
* Budget accounting unit: the native patch token (VideoMAEv2/VideoMAE) or the
  complete spatial trajectory (TimeSformer — a trajectory is kept or dropped
  as a whole, never per frame).
* Groups are consecutive in native order with ``GROUP_TOKENS = 10`` budget
  slots per group (VideoMAEv2/VideoMAE: 10 tokens = 5 native adjacent pairs;
  TimeSformer: 10 trajectories). Full groups keep exactly
  ``round_half_up(r*10)`` slots (8 / 6 / 4 — exact, no tier is silently
  reshaped). The trailing remainder group of ``R`` slots keeps
  ``round_half_up(r*R)``.
* Rounding: ``round_half_up(x) = floor(x + 0.5)`` clamped to
  ``[0, slots]``; the actually kept count K is reported as
  ``actual_keep_ratio`` and is identical for every rule at the same tier.
* VideoMAEv2/VideoMAE pair-member priority (one selection signal, multi-round):
  per pair the signal is the member L2 norms; the better member always
  outranks the worse member of the same pair. Within a group, keep priority
  is: better members of all 5 pairs ordered by pair signal descending, then
  worse members in the same pair order (round 2 for quotas below one per
  pair). ``keep_ratio=0.50`` (not a tier) would be exactly "one member per
  pair"; tiers 0.80/0.60 keep additional worse members of the strongest
  pairs, tier 0.40 keeps only the best 4 better members.
* TimeSformer trajectory priority: keep the top-q trajectories of the group
  by signal (max member L2 norm), ties by native trajectory order.
* Slot layout: kept slots fill in native order for every rule; controls use
  the identical quota table — ``group_uniform`` keeps the first slots in
  native order, ``group_random`` a fixed seeded per-group draw.

Position semantics (disclosed): uniform/random controls are exact static
gathers — retained tokens keep their native positions
(``retained_native_position``). ``pair_select`` is dynamic (per forward):
kept slots fill the uniform skeleton's quota slots, so a token chosen by
signal sits at a skeleton slot coordinate —
``approximate_anchor_position``, the same disclosure level as the pair-merge
anchor. The real per-forward chosen indices are recorded by the selector
(``selection_digest``) and never silently claimed as native positions.

Leakage contract: the selector reads only the current block output tensor
(and the static spec). No labels, file names, test statistics, future-layer
activations, or dense-teacher signals participate. Method owner A's one-page
signal definition can replace ``signal_fn`` without changing this rule.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import TokenLayout
from .pair_merge import PairMergeError, horizontal_pair_merge_spec

SUPPORTED_KEEP_RATIOS = (0.80, 0.60, 0.40)
GROUP_TOKENS = 10
SIGNAL_NAME = "member_l2_norm_max"


def round_half_up(value: float) -> int:
    """The single fixed rounding rule: floor(x + 0.5), clamped nonnegative."""

    if not math.isfinite(value):
        raise PairMergeError("quota rounding requires a finite value")
    return max(0, int(math.floor(value + 0.5)))


def group_quotas(slot_count: int, keep_ratio: float) -> list[int]:
    """Per-group kept-slot quotas for the frozen three-tier budget rule."""

    if type(slot_count) is not int or slot_count <= 0:
        raise PairMergeError("slot_count 必须为正整数")
    if keep_ratio not in SUPPORTED_KEEP_RATIOS:
        raise PairMergeError(
            f"keep_ratio 必须是 {SUPPORTED_KEEP_RATIOS} 之一；dense 基线请用 identity 路径"
        )
    quotas: list[int] = []
    full, remainder = divmod(slot_count, GROUP_TOKENS)
    quotas.extend(round_half_up(keep_ratio * GROUP_TOKENS) for _ in range(full))
    if remainder:
        quotas.append(round_half_up(keep_ratio * remainder))
    return quotas


def default_signal(members: torch.Tensor) -> torch.Tensor:
    """Selection signal: max member L2 norm within each unit.

    ``members`` is [B,U,W,D] (pairs or trajectories); the score is [B,U].
    This is the current pair-select candidate signal; method owner A's
    one-page definition may replace it through ``signal_fn``.
    """

    if members.ndim != 4:
        raise PairMergeError("signal 需要 [B,U,W,D] members")
    return members.norm(dim=-1).amax(dim=-1)


def snap_trajectory_quotas(
    quotas: list[int], native_patch_width: int
) -> tuple[list[int], dict[str, Any]]:
    """Align the kept-trajectory total with TimeSformer divided-spatial geometry.

    TimeSformer's divided spatial attention reshapes retained locations by the
    native patch width, so the retained spatial-token count must be divisible
    by ``native_patch_width``. The frozen rule cannot hit every ratio exactly
    (e.g. 196 trajectories, width 14, r=0.60 -> raw 118); the single fixed
    adjustment is conservative: snap the TOTAL down to the nearest multiple of
    the patch width (never above the target budget) and take whole trajectories
    from the tail groups in reverse order. ``actual_keep_ratio`` is reported
    from the snapped table — never silently reshaped to fake a tier.
    """

    if type(native_patch_width) is not int or native_patch_width <= 0:
        raise PairMergeError("native_patch_width 必须为正整数")
    raw = sum(quotas)
    snapped_total = native_patch_width * (raw // native_patch_width)
    receipt = {
        "rule": "floor_total_to_patch_width_multiple_trim_tail_groups",
        "native_patch_width": native_patch_width,
        "raw_kept_units": raw,
        "snapped_kept_units": snapped_total,
        "trimmed_units": raw - snapped_total,
        "never_above_target": True,
    }
    if snapped_total == raw:
        return list(quotas), receipt
    if snapped_total < 1:
        raise PairMergeError("TimeSformer 几何对齐把保留集合清空；该档位在此几何下不可行")
    adjusted = list(quotas)
    remaining = raw - snapped_total
    for index in range(len(adjusted) - 1, -1, -1):
        take = min(adjusted[index], remaining)
        adjusted[index] -= take
        remaining -= take
        if remaining == 0:
            break
    _require = remaining == 0
    if not _require:
        raise PairMergeError("trajectory quota snap 无法收敛")
    return adjusted, receipt


@dataclass(frozen=True)
class GroupSelectSpec:
    """Static unit table, quota slots and skeleton for one native geometry."""

    encoder_id: str
    input_token_count: int
    keep_ratio: float
    mode: str  # "pair_member" (videomaev2/videomae) or "trajectory" (timesformer)
    units: torch.Tensor  # [U,W] member token indices per unit
    group_size_units: int  # units per group (5 pairs or 10 trajectories)
    quotas: tuple[int, ...]  # kept slots per group (tokens or trajectories)
    slot_count: int  # total reducible slots (tokens for pair_member, trajectories for trajectory)
    skeleton_indices: torch.Tensor  # [K] uniform-choice slot sources (B=1)
    tokens_per_slot: int
    reducible_tokens: int
    kept_units: int
    input_topology_sha256: str
    native_patch_width: int | None = None
    geometry_snap: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.keep_ratio not in SUPPORTED_KEEP_RATIOS:
            raise PairMergeError("keep_ratio 不在冻结三档内")
        if self.mode not in {"pair_member", "trajectory"}:
            raise PairMergeError("mode 必须是 pair_member 或 trajectory")
        if self.units.ndim != 2 or self.units.dtype != torch.int64:
            raise PairMergeError("units 必须为 int64 [U,W]")
        if self.mode == "pair_member" and self.units.shape[1] != 2:
            raise PairMergeError("pair_member 模式需要 [P,2] pair 表")
        units = self.units.shape[0]
        slot_per_group = 2 if self.mode == "pair_member" else 1
        if self.slot_count != units * slot_per_group:
            raise PairMergeError("slot_count 与 unit 表不一致")
        expected_groups = self.slot_count // GROUP_TOKENS + (1 if self.slot_count % GROUP_TOKENS else 0)
        if len(self.quotas) != expected_groups:
            raise PairMergeError("quotas 与分组不一致")
        if (self.units < 0).any() or (self.units >= self.input_token_count).any():
            raise PairMergeError("units 含超出 native token 范围的索引")
        kept = sum(self.quotas)
        expected_tokens = kept * self.tokens_per_slot
        if self.kept_units != kept or self.skeleton_indices.numel() != expected_tokens:
            raise PairMergeError("skeleton 与配额总和不一致")
        if (self.skeleton_indices < 0).any() or (self.skeleton_indices >= self.input_token_count).any():
            raise PairMergeError("skeleton 含超出 native token 范围的索引")
        if len(torch.unique(self.skeleton_indices)) != self.skeleton_indices.numel():
            raise PairMergeError("skeleton indices 不可重复")
        if self.reducible_tokens <= 0:
            raise PairMergeError("reducible_tokens 必须为正")

    @property
    def output_token_count(self) -> int:
        return self.skeleton_indices.numel()

    @property
    def actual_keep_ratio(self) -> float:
        return self.output_token_count / self.reducible_tokens

    def budget_receipt(self) -> dict[str, Any]:
        return {
            "budget_rule": "group_budget_v1",
            "group_tokens": GROUP_TOKENS,
            "group_size_units": self.group_size_units,
            "keep_ratio": self.keep_ratio,
            "rounding": "round_half_up_floor_x_plus_0.5",
            "selection_unit": "native_patch_token" if self.mode == "pair_member" else "complete_spatial_trajectory",
            "tokens_per_slot": self.tokens_per_slot,
            "units_total": self.units.shape[0],
            "quotas": list(self.quotas),
            "kept_units": self.kept_units,
            "reducible_tokens": self.reducible_tokens,
            "kept_tokens": self.output_token_count,
            "actual_keep_ratio": self.actual_keep_ratio,
            "same_budget_for_all_controls": True,
            **({"geometry_snap": dict(self.geometry_snap)} if self.geometry_snap else {}),
        }


def _units_from_layout(encoder_id: str, layout: TokenLayout) -> tuple[torch.Tensor, int, str, int]:
    """Unit table and accounting mode from the verified topology.

    VideoMAEv2/VideoMAE: one unit per native adjacent pair (from the verified
    pair table); budget slots are single tokens (5 pairs = 10 slots/group).
    TimeSformer: one unit per complete spatial trajectory — every time index
    of one spatial position, from verified real coordinates; budget slots are
    whole trajectories (10 per group).
    """

    if encoder_id == "timesformer":
        if layout.source_coordinates is None:
            raise PairMergeError("TimeSformer trajectory 划分需要已验证的真实坐标")
        grid = layout.provenance.get("grid")
        if not isinstance(grid, list) or len(grid) != 3:
            raise PairMergeError("TimeSformer 需要已验证的 [T,H,W] grid")
        frames = grid[0]
        trajectories: dict[tuple[int, int], list[int]] = {}
        for index in torch.nonzero(~layout.special_token_mask[0], as_tuple=False).flatten().tolist():
            time, row, column = (int(value) for value in layout.source_coordinates[0, index].tolist())
            trajectories.setdefault((row, column), []).append((time, index))
        units: list[torch.Tensor] = []
        for position in sorted(trajectories):
            entries = sorted(trajectories[position])
            if [time for time, _index in entries] != list(range(frames)):
                raise PairMergeError("TimeSformer 空间轨迹缺少完整时间覆盖")
            units.append(torch.tensor([index for _time, index in entries], dtype=torch.int64))
        width = units[0].numel()
        if any(unit.numel() != width for unit in units):
            raise PairMergeError("TimeSformer 轨迹 unit 宽度不一致")
        return torch.stack(units), width, "trajectory", 10
    pair_spec = horizontal_pair_merge_spec(encoder_id, layout)
    return pair_spec.pair_indices[0], 1, "pair_member", 5


def group_select_spec(
    encoder_id: str, layout: TokenLayout, keep_ratio: float
) -> GroupSelectSpec:
    """Build the static group-budget selection spec for a verified layout."""

    if layout.source_coordinates is None:
        raise PairMergeError("group select 缺少已验证的真实 token 坐标")
    if bool((~layout.valid_mask).any()):
        raise PairMergeError("group select 不支持 padding 或 ragged layout")
    if encoder_id not in {"videomaev2", "timesformer", "videomae"}:
        raise PairMergeError(f"不支持的 encoder={encoder_id!r}")
    if encoder_id == "timesformer":
        special = layout.special_token_mask
        if not bool(special[:, 0].all()) or bool(special.sum(dim=1).ne(1).any()):
            raise PairMergeError("TimeSformer group select 需要唯一 index-0 CLS")
    elif bool(layout.special_token_mask.any()):
        raise PairMergeError(f"{encoder_id} 当前 group select 仅支持 patch-only layout")
    units, tokens_per_slot, mode, group_size_units = _units_from_layout(encoder_id, layout)
    slot_count = units.shape[0] * (2 if mode == "pair_member" else 1)
    quotas = group_quotas(slot_count, keep_ratio)
    native_patch_width = None
    geometry_snap = None
    if mode == "trajectory":
        grid = layout.provenance.get("grid")
        if not isinstance(grid, list) or len(grid) != 3:
            raise PairMergeError("TimeSformer 几何对齐需要已验证的 [T,H,W] grid")
        native_patch_width = grid[2]
        quotas, geometry_snap = snap_trajectory_quotas(quotas, native_patch_width)
    skeleton = []
    start = 0
    for quota in quotas:
        size = min(GROUP_TOKENS, slot_count - start)
        if mode == "pair_member":
            skeleton.append(torch.arange(start, start + quota, dtype=torch.int64))
        else:
            for unit in units[start : start + quota]:
                skeleton.append(unit)
        start += size
    skeleton_indices = (
        torch.sort(torch.cat(skeleton)).values.contiguous() if skeleton else torch.empty(0, dtype=torch.int64)
    )
    topology = horizontal_pair_merge_spec(encoder_id, layout)
    reducible = int((~layout.special_token_mask[0]).sum().item())
    return GroupSelectSpec(
        encoder_id=encoder_id,
        input_token_count=layout.token_capacity,
        keep_ratio=keep_ratio,
        mode=mode,
        units=units,
        group_size_units=group_size_units,
        quotas=tuple(quotas),
        slot_count=slot_count,
        skeleton_indices=skeleton_indices,
        tokens_per_slot=tokens_per_slot,
        reducible_tokens=reducible,
        kept_units=sum(quotas),
        input_topology_sha256=topology.input_topology_sha256,
        native_patch_width=native_patch_width,
        geometry_snap=geometry_snap,
    )


PAIR_MEMBER_RULES = ("pair_select", "pair_fixed", "pair_random_member", "pair_reverse")


class GroupBudgetSelector(nn.Module):
    """Dynamic per-forward slot selection under the frozen group quotas.

    ``rule`` decides which units fill the quota slots inside every group:
    ``pair_select`` ranks pairs by the signal and prefers the max-norm member
    (the batch-1 candidate); ``pair_fixed`` always keeps the native first
    member of every pair; ``pair_random_member`` keeps a fixed seeded member
    per pair; ``pair_reverse`` prefers the min-norm member. These four share
    the identical pair table, quota table, pair ordering, tie-breaking and
    slot layout — only the within-pair member preference differs (batch-1b
    ordering ablation). ``group_uniform`` keeps the first slots in native
    order; ``group_random`` a fixed seeded per-group draw. All rules produce
    identical slot counts at the same keep_ratio.
    """

    def __init__(
        self,
        dim: int,
        *,
        rule: str = "pair_select",
        signal_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 0,
        record_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise PairMergeError("dim 必须为正整数")
        if rule not in set(PAIR_MEMBER_RULES) | {"group_uniform", "group_random"}:
            raise PairMergeError(
                f"rule 必须是 {PAIR_MEMBER_RULES} 之一或 group_uniform/group_random"
            )
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairMergeError("seed 必须为 [0, 2**63) 内的整数")
        if not isinstance(record_diagnostics, bool):
            raise PairMergeError("record_diagnostics 必须为布尔")
        self.dim = dim
        self.rule = rule
        self.signal_fn = signal_fn or default_signal
        self.seed = seed
        self.record_diagnostics = record_diagnostics
        self._random_choice: torch.Tensor | None = None
        self._member_choice: torch.Tensor | None = None
        self.last_selection: torch.Tensor | None = None
        self.last_signal: torch.Tensor | None = None

    def _random_slots(self, spec: GroupSelectSpec, device: torch.device) -> torch.Tensor:
        if self._random_choice is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            keep = torch.zeros(spec.input_token_count, dtype=torch.bool)
            start = 0  # slot units: tokens (pair_member) or trajectories (trajectory)
            for quota in spec.quotas:
                group_slots = min(GROUP_TOKENS, spec.slot_count - start)
                choice = torch.randperm(group_slots, generator=generator)[:quota]
                if spec.mode == "pair_member":
                    keep[start + choice] = True
                else:
                    for unit_index in choice.tolist():
                        keep[spec.units[start + unit_index]] = True
                start += group_slots
            self._random_choice = torch.nonzero(keep).flatten()
        return self._random_choice.to(device)

    def _select_slots(self, spec: GroupSelectSpec, hidden: torch.Tensor) -> torch.Tensor:
        """[B,K] kept token indices per sample (specials excluded)."""

        batch = hidden.shape[0]
        device = hidden.device
        if self.rule == "group_random":
            chosen = self._random_slots(spec, device)
            return chosen[None].expand(batch, -1)
        if self.rule == "group_uniform":
            chosen = spec.skeleton_indices.to(device)
            return chosen[None].expand(batch, -1)
        members = hidden[:, spec.units.flatten()].reshape(
            batch, spec.units.shape[0], spec.units.shape[1], self.dim
        )
        scores = self.signal_fn(members)
        if scores.ndim != 2 or scores.shape != (batch, spec.units.shape[0]) or not torch.is_floating_point(scores):
            raise PairMergeError("signal_fn 必须返回浮点 [B,U]")
        if not bool(torch.isfinite(scores).all()):
            raise PairMergeError("选择信号必须有限")
        rows = []
        units_device = spec.units.to(device)
        start = 0
        for quota in spec.quotas:
            size_units = min(spec.group_size_units, spec.units.shape[0] - start)
            if spec.mode == "pair_member":
                pair_scores = scores[:, start : start + size_units]
                order = torch.argsort(pair_scores, dim=1, descending=True, stable=True)
                pair_table = units_device[start : start + size_units]  # [G,2]
                left_norm = hidden[:, pair_table[:, 0]].norm(dim=-1)  # [B,G]
                right_norm = hidden[:, pair_table[:, 1]].norm(dim=-1)
                if self.rule == "pair_fixed":
                    better = pair_table[:, 0][None].expand(batch, -1)
                    worse = pair_table[:, 1][None].expand(batch, -1)
                elif self.rule == "pair_random_member":
                    if self._member_choice is None:
                        generator = torch.Generator(device="cpu").manual_seed(self.seed)
                        self._member_choice = torch.randint(
                            2, (spec.units.shape[0],), generator=generator
                        ).to(device)
                    choice = self._member_choice[start : start + size_units][None].expand(batch, -1)
                    first, second = pair_table[:, 0][None].expand(batch, -1), pair_table[:, 1][None].expand(batch, -1)
                    better = torch.where(choice == 1, second, first)
                    worse = torch.where(choice == 1, first, second)
                else:
                    # pair_select: max-norm member is better; pair_reverse: min-norm.
                    take_right = right_norm > left_norm
                    if self.rule == "pair_reverse":
                        take_right = ~take_right
                    better = torch.where(
                        take_right.unsqueeze(-1), pair_table[:, 1:], pair_table[:, :1]
                    ).squeeze(-1)
                    worse = torch.where(
                        take_right.unsqueeze(-1), pair_table[:, :1], pair_table[:, 1:]
                    ).squeeze(-1)
                # Keep priority: better members in pair-signal order, then worse.
                kept = torch.cat(
                    [
                        better.gather(1, order)[:, :quota],
                        worse.gather(1, order)[:, : max(0, quota - size_units)],
                    ],
                    dim=1,
                )
                rows.append(kept)
            else:
                group_scores = scores[:, start : start + size_units]
                order = torch.argsort(group_scores, dim=1, descending=True, stable=True)
                group_units = units_device[start : start + size_units]  # [G,W]
                # [B,q,W]: expand the 2-D unit table across the batch before
                # gathering — a 2-D tensor cannot take a 3-D gather index.
                kept_units = group_units[None].expand(batch, -1, -1).gather(
                    1,
                    order[:, :quota].unsqueeze(-1).expand(-1, -1, group_units.shape[1]),
                )
                rows.append(kept_units.reshape(batch, -1))
            start += size_units
        return torch.cat(rows, dim=1)

    def forward(self, hidden: torch.Tensor, spec: GroupSelectSpec, specials: torch.Tensor) -> torch.Tensor:
        """Select slots per sample; specials are preserved unmodified.

        Output rows are ``cat(special tokens, kept slot tokens)`` in ascending
        token order — the same slot layout for every rule at the same tier.
        """

        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[1] != spec.input_token_count
            or hidden.shape[2] != self.dim
        ):
            raise PairMergeError("hidden 必须与 GroupSelectSpec 对齐为 [B,N,D]")
        if not torch.is_floating_point(hidden):
            raise PairMergeError("hidden 必须为浮点 tensor")
        batch = hidden.shape[0]
        selected_slots = self._select_slots(spec, hidden)  # [B,K]
        if selected_slots.shape[1] != spec.output_token_count:
            raise PairMergeError("选择结果与冻结配额不一致")
        specials = specials.to(hidden.device)
        selected_index = torch.sort(
            torch.cat([specials[None].expand(batch, -1), selected_slots], dim=1), dim=1
        ).values
        output = torch.gather(
            hidden, 1, selected_index.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        if self.record_diagnostics:
            # Dev-only audit exports; formal timing keeps the algorithm cost
            # alone and leaves these CPU transfers disabled.
            self.last_selection = selected_index.detach().cpu()
            members = hidden[:, spec.units.flatten()].reshape(
                batch, spec.units.shape[0], spec.units.shape[1], self.dim
            )
            self.last_signal = self.signal_fn(members).detach().cpu()
        return output

    def selection_digest(self) -> str | None:
        """SHA-256 of the real per-forward chosen token indices (audit)."""

        if self.last_selection is None:
            return None
        return hashlib.sha256(self.last_selection.numpy().tobytes()).hexdigest()


__all__ = [
    "GROUP_TOKENS",
    "PAIR_MEMBER_RULES",
    "SIGNAL_NAME",
    "SUPPORTED_KEEP_RATIOS",
    "GroupBudgetSelector",
    "GroupSelectSpec",
    "default_signal",
    "group_quotas",
    "group_select_spec",
    "round_half_up",
    "snap_trajectory_quotas",
]
