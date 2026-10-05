"""Deterministic cache merge and baseline compression policies."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .contracts import (
    CacheUpdate,
    CacheUpdateMode,
    CacheView,
    ContractError,
    TokenTimeline,
)


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raw = np.asarray(value).shape
    return tuple(int(dim) for dim in raw)


def _to_numpy(value: Any) -> np.ndarray:
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concatenate_tensors(left: Any, right: Any, axis: int, name: str) -> Any:
    left_shape = _shape(left)
    right_shape = _shape(right)
    if len(left_shape) != len(right_shape):
        raise ContractError(f"cache tensor {name!r} append 前后维数不同")
    if any(
        left_dim != right_dim
        for index, (left_dim, right_dim) in enumerate(zip(left_shape, right_shape, strict=True))
        if index != axis
    ):
        raise ContractError(
            f"cache tensor {name!r} 除 sequence 轴外必须同形，实际 {left_shape}/{right_shape}"
        )

    left_root = type(left).__module__.split(".", 1)[0]
    right_root = type(right).__module__.split(".", 1)[0]
    try:
        if left_root == right_root == "torch":
            torch = importlib.import_module("torch")
            return torch.cat((left, right), dim=axis)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.concatenate((left, right), axis=axis)
    except Exception as exc:
        raise ContractError(f"cache tensor {name!r} 无法沿 sequence 轴拼接: {exc}") from exc
    raise ContractError(f"cache tensor {name!r} append 要求两侧同为 numpy.ndarray 或 torch.Tensor")


def _concatenate_timelines(left: TokenTimeline, right: TokenTimeline) -> TokenTimeline:
    if left.batch_size != right.batch_size:
        raise ContractError("cache timeline append 前后 batch 不一致")

    def concatenate_values(left_value: Any, right_value: Any) -> np.ndarray:
        return np.concatenate((_to_numpy(left_value), _to_numpy(right_value)), axis=1)

    valid_mask: np.ndarray | None = None
    if left.valid_mask is not None or right.valid_mask is not None:
        left_valid = (
            np.ones((left.batch_size, left.num_tokens), dtype=bool)
            if left.valid_mask is None
            else _to_numpy(left.valid_mask).astype(bool, copy=False)
        )
        right_valid = (
            np.ones((right.batch_size, right.num_tokens), dtype=bool)
            if right.valid_mask is None
            else _to_numpy(right.valid_mask).astype(bool, copy=False)
        )
        valid_mask = np.concatenate((left_valid, right_valid), axis=1)

    left_has_frames = left.source_frame_start is not None
    right_has_frames = right.source_frame_start is not None
    if left_has_frames != right_has_frames:
        raise ContractError("cache append 前后必须一致地提供 source frame provenance")

    return TokenTimeline(
        start_s=concatenate_values(left.start_s, right.start_s),
        end_s=concatenate_values(left.end_s, right.end_s),
        valid_mask=valid_mask,
        source_frame_start=(
            concatenate_values(left.source_frame_start, right.source_frame_start)
            if left_has_frames
            else None
        ),
        source_frame_end=(
            concatenate_values(left.source_frame_end, right.source_frame_end)
            if left_has_frames
            else None
        ),
    )


def append_cache_views(current: CacheView, appended: CacheView) -> CacheView:
    """Append normalized cache tensors after validating structural compatibility."""

    if current.kind is not appended.kind:
        raise ContractError(
            f"cache kind 不能混合 append：{current.kind.value}/{appended.kind.value}"
        )
    if current.sequence_axis != appended.sequence_axis:
        raise ContractError("cache append 前后 sequence_axis 必须一致")
    if current.tensors.keys() != appended.tensors.keys():
        raise ContractError(
            "cache append 前后 tensor 键必须完全一致："
            f"{sorted(current.tensors)}/{sorted(appended.tensors)}"
        )

    tensors = {
        name: _concatenate_tensors(
            current.tensors[name],
            appended.tensors[name],
            current.sequence_axis,
            name,
        )
        for name in current.tensors
    }
    metadata = dict(current.metadata)
    metadata.update(appended.metadata)
    return CacheView(
        kind=current.kind,
        tensors=tensors,
        sequence_axis=current.sequence_axis,
        timeline=_concatenate_timelines(current.timeline, appended.timeline),
        metadata=metadata,
    )


def merge_cache_update(current: CacheView | None, update: CacheUpdate) -> CacheView:
    """Apply the update semantics without compression."""

    if not isinstance(update, CacheUpdate):
        raise ContractError("update 必须是 CacheUpdate")
    if update.mode is CacheUpdateMode.REPLACE or current is None:
        return update.view
    return append_cache_views(current, update.view)


@dataclass(frozen=True, slots=True)
class IdentityCachePolicy:
    """Lossless reference policy: merge updates and retain every cache token."""

    name: ClassVar[str] = "identity"

    def compress(self, view: CacheView) -> CacheView:
        if not isinstance(view, CacheView):
            raise ContractError("view 必须是 CacheView")
        return view

    def apply(self, current: CacheView | None, update: CacheUpdate) -> CacheView:
        return self.compress(merge_cache_update(current, update))


@dataclass(frozen=True, slots=True)
class KeepRecentCachePolicy:
    """Keep only the most recent ``max_tokens`` on the declared sequence axis."""

    max_tokens: int
    name: ClassVar[str] = "keep_recent"

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ContractError("max_tokens 必须是正整数")

    def compress(self, view: CacheView) -> CacheView:
        if not isinstance(view, CacheView):
            raise ContractError("view 必须是 CacheView")
        if view.sequence_length <= self.max_tokens:
            return view
        start = view.sequence_length - self.max_tokens
        return view.slice_sequence(start, view.sequence_length)

    def apply(self, current: CacheView | None, update: CacheUpdate) -> CacheView:
        return self.compress(merge_cache_update(current, update))


def build_cache_policy(
    name: str, *, max_tokens: int | None = None
) -> IdentityCachePolicy | KeepRecentCachePolicy:
    """Construct a built-in policy from experiment configuration."""

    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"identity", "none"}:
        if max_tokens is not None:
            raise ContractError("identity policy 不接受 max_tokens")
        return IdentityCachePolicy()
    if normalized == "keep_recent":
        if max_tokens is None:
            raise ContractError("keep_recent policy 必须提供 max_tokens")
        return KeepRecentCachePolicy(max_tokens=max_tokens)
    raise ContractError(f"未知 cache policy：{name!r}")


__all__ = [
    "IdentityCachePolicy",
    "KeepRecentCachePolicy",
    "append_cache_views",
    "build_cache_policy",
    "merge_cache_update",
]
