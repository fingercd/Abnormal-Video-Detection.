"""Label-free, current-layer index controls for neutral token interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional

from vadbench.token_reduction.contracts import TokenLayout


class InterventionScoreError(ValueError):
    """A score or verified layout cannot support a neutral index control."""


Strategy = Literal["uniform", "seeded_random", "score_high", "score_low"]


@dataclass(frozen=True)
class IndexSelection:
    """Sorted native indices and the honest effective budget used for them."""

    indices: torch.Tensor
    requested_budget: int
    effective_budget: int
    strategy: Strategy


@dataclass(frozen=True)
class TemporalChange:
    """Per-token adjacent-time cosine dissimilarity and availability mask."""

    scores: torch.Tensor
    available_mask: torch.Tensor


def _require_scores(scores: torch.Tensor, layout: TokenLayout) -> None:
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise InterventionScoreError("scores 必须为 [B,N] tensor")
    if tuple(scores.shape) != tuple(layout.valid_mask.shape):
        raise InterventionScoreError("scores 必须与已验证 layout 的 [B,N] 一致")
    if not torch.is_floating_point(scores) or not torch.isfinite(scores).all():
        raise InterventionScoreError("scores 必须是有限浮点 tensor")


def _same_valid_count(layout: TokenLayout) -> int:
    counts = layout.valid_token_counts
    if not bool((counts == counts[0]).all()):
        raise InterventionScoreError("固定 budget control 不接受 batch 内不同有效 token 数")
    return int(counts[0])


def _is_timesformer(layout: TokenLayout) -> bool:
    return layout.position_contract.startswith("hf-timesformer-")


def _validate_strategy(strategy: Strategy, seed: int | None) -> None:
    if strategy not in {"uniform", "seeded_random", "score_high", "score_low"}:
        raise InterventionScoreError(f"未知 strategy={strategy!r}")
    if strategy == "seeded_random" and type(seed) is not int:
        raise InterventionScoreError("seeded_random 需要整数 seed")


def _choose(
    candidates: torch.Tensor, values: torch.Tensor, count: int, strategy: Strategy, seed: int | None
) -> torch.Tensor:
    if count == len(candidates):
        return candidates
    if strategy == "uniform":
        slots = torch.linspace(0, len(candidates) - 1, count, device=candidates.device).round().long()
        return candidates[slots]
    if strategy == "seeded_random":
        assert type(seed) is int
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(candidates), generator=generator, device="cpu")[:count].to(candidates.device)
        return candidates[order]
    descending = strategy == "score_high"
    selected = torch.argsort(values, descending=descending, stable=True)[:count]
    return candidates[selected]


def _timesformer_selection(scores: torch.Tensor, layout: TokenLayout, budget: int, strategy: Strategy, seed: int | None) -> IndexSelection:
    grid = layout.provenance.get("grid")
    if not isinstance(grid, list) or len(grid) != 3 or not all(type(value) is int and value > 0 for value in grid):
        raise InterventionScoreError("TimeSformer selection 需要已验证 [T,H,W] grid")
    frames, height, width = grid
    special = layout.special_token_mask
    if not bool(special[:, 0].all()) or bool(special.sum(dim=1).ne(1).any()):
        raise InterventionScoreError("TimeSformer selection 需要唯一 index-0 CLS")
    patch_budget = budget - 1
    if patch_budget < 0:
        raise InterventionScoreError("budget 小于必须保留的 CLS")
    trajectories = patch_budget // frames
    trajectories -= trajectories % width
    effective = 1 + trajectories * frames
    if trajectories <= 0:
        raise InterventionScoreError("budget 不能保留完整且 patch-width 对齐的 TimeSformer 轨迹")
    expected_patches = frames * height * width
    if layout.token_capacity != 1 + expected_patches or _same_valid_count(layout) != layout.token_capacity:
        raise InterventionScoreError("TimeSformer control 只接受无 padding 的完整 CLS+T×H×W layout")
    coordinates = layout.source_coordinates
    if coordinates is None:
        raise InterventionScoreError("TimeSformer trajectory selection 缺少已验证坐标")
    rows: list[torch.Tensor] = []
    for batch in range(layout.batch_size):
        spatial: list[tuple[tuple[int, int], torch.Tensor]] = []
        for row in range(height):
            for column in range(width):
                ids = torch.nonzero(
                    (coordinates[batch, :, 1] == row)
                    & (coordinates[batch, :, 2] == column)
                    & ~special[batch],
                    as_tuple=False,
                ).flatten()
                if len(ids) != frames:
                    raise InterventionScoreError("TimeSformer 每个空间轨迹必须含完整时间 token")
                ordered = ids[torch.argsort(coordinates[batch, ids, 0])]
                if not torch.equal(coordinates[batch, ordered, 0], torch.arange(frames, device=ordered.device)):
                    raise InterventionScoreError("TimeSformer 轨迹时间坐标必须为连续原生顺序")
                spatial.append(((row, column), ordered))
        trajectory_ids = torch.stack([ids for _, ids in spatial]).to(scores.device)
        trajectory_scores = scores[batch, trajectory_ids].mean(dim=1)
        choices = torch.arange(len(spatial), device=scores.device)
        chosen = _choose(choices, trajectory_scores, trajectories, strategy, seed)
        indices = torch.cat(
            (
                torch.zeros(1, device=scores.device, dtype=torch.long),
                trajectory_ids[chosen].reshape(-1),
            )
        )
        rows.append(torch.sort(indices).values)
    return IndexSelection(torch.stack(rows), budget, effective, strategy)


def fixed_budget_indices(
    scores: torch.Tensor,
    layout: TokenLayout,
    budget: int,
    strategy: Strategy,
    *,
    seed: int | None = None,
) -> IndexSelection:
    """Choose sorted current-layer native indices without labels or future tensors."""

    _require_scores(scores, layout)
    _validate_strategy(strategy, seed)
    if type(budget) is not int or budget <= 0:
        raise InterventionScoreError("budget 必须为正整数")
    valid_count = _same_valid_count(layout)
    if budget > valid_count:
        raise InterventionScoreError("budget 不可超过有效 token 数")
    if _is_timesformer(layout):
        return _timesformer_selection(scores, layout, budget, strategy, seed)
    special_count = int(layout.special_token_mask[0].sum())
    if budget < special_count:
        raise InterventionScoreError("budget 小于必须保留的 special token 数")
    chosen_count = budget - special_count
    rows: list[torch.Tensor] = []
    for batch in range(layout.batch_size):
        candidates = torch.nonzero(
            layout.valid_mask[batch] & ~layout.special_token_mask[batch], as_tuple=False
        ).flatten().to(scores.device)
        specials = torch.nonzero(layout.special_token_mask[batch], as_tuple=False).flatten().to(scores.device)
        if chosen_count > len(candidates):
            raise InterventionScoreError("budget 超出可选择 patch token 数")
        selected = _choose(candidates, scores[batch, candidates], chosen_count, strategy, seed)
        rows.append(torch.sort(torch.cat((specials, selected))).values)
    return IndexSelection(torch.stack(rows), budget, budget, strategy)


def relative_branch_update(branch_output: torch.Tensor, branch_input: torch.Tensor) -> torch.Tensor:
    """Return differentiable per-token ``||U|| / max(||X||, eps)`` scores."""

    if (
        not isinstance(branch_output, torch.Tensor)
        or not isinstance(branch_input, torch.Tensor)
        or branch_output.ndim != 3
        or branch_output.shape != branch_input.shape
        or branch_output.device != branch_input.device
        or not torch.is_floating_point(branch_output)
        or not torch.is_floating_point(branch_input)
    ):
        raise InterventionScoreError("branch tensors 必须为同 device、同 shape 的 [B,N,D] 浮点 tensor")
    denominator = torch.linalg.vector_norm(branch_input, dim=-1)
    numerator = torch.linalg.vector_norm(branch_output, dim=-1)
    if not torch.isfinite(denominator).all() or not torch.isfinite(numerator).all():
        raise InterventionScoreError("branch norm 含非有限值")
    if bool((denominator <= 0).any()):
        raise InterventionScoreError("branch input norm 为零；拒绝静默 epsilon 替代")
    epsilon = torch.finfo(branch_input.dtype).eps
    return numerator / denominator.clamp_min(epsilon)


def same_position_temporal_change(tokens: torch.Tensor, layout: TokenLayout) -> TemporalChange:
    """Adjacent-time cosine dissimilarity on verified dense ``(t,h,w)`` provenance."""

    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or not torch.is_floating_point(tokens)
        or tuple(tokens.shape[:2]) != tuple(layout.valid_mask.shape)
    ):
        raise InterventionScoreError("tokens 必须为与 layout 对齐的浮点 [B,N,D]")
    if layout.source_coordinates is None:
        raise InterventionScoreError("same-position temporal change 缺少真实坐标")
    if bool((~layout.valid_mask).any()):
        raise InterventionScoreError("same-position temporal change 不接受 padding layout")
    if not torch.isfinite(tokens).all():
        raise InterventionScoreError("tokens 含非有限值")
    scores = torch.zeros(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)
    available = torch.zeros_like(layout.valid_mask, device=tokens.device)
    coordinates = layout.source_coordinates.to(tokens.device)
    special = layout.special_token_mask.to(tokens.device)
    for batch in range(layout.batch_size):
        tracks: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for index in torch.nonzero(~special[batch], as_tuple=False).flatten().tolist():
            time, row, column = (int(value) for value in coordinates[batch, index].tolist())
            tracks.setdefault((row, column), []).append((time, index))
        seen: set[tuple[int, int, int]] = set()
        for (row, column), track in tracks.items():
            track.sort()
            for time, _index in track:
                key = (time, row, column)
                if key in seen:
                    raise InterventionScoreError("真实坐标存在重复来源，拒绝 temporal change")
                seen.add(key)
            if len(track) < 2:
                continue
            times = [time for time, _index in track]
            if times != list(range(times[0], times[0] + len(times))):
                raise InterventionScoreError("same-position temporal change 需要相邻时间坐标")
            ids = torch.tensor([index for _time, index in track], device=tokens.device)
            values = tokens[batch, ids]
            norms = torch.linalg.vector_norm(values, dim=-1)
            if bool((norms <= 0).any()):
                raise InterventionScoreError("temporal cosine 的 token norm 为零")
            normalized = functional.normalize(values, dim=-1)
            pair_change = 1 - (normalized[:-1] * normalized[1:]).sum(dim=-1)
            track_scores = torch.cat((pair_change[:1], (pair_change[:-1] + pair_change[1:]) / 2, pair_change[-1:]))
            scores[batch, ids] = track_scores
            available[batch, ids] = True
    return TemporalChange(scores=scores, available_mask=available)


__all__ = [
    "IndexSelection",
    "InterventionScoreError",
    "TemporalChange",
    "fixed_budget_indices",
    "relative_branch_update",
    "same_position_temporal_change",
]
