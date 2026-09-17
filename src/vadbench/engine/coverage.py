"""Shared frame-coverage validation for prediction and evaluation stages."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_frame_coverage(
    *,
    video_id: str,
    clip_indices: Any,
    frame_starts: Any,
    frame_ends: Any,
    num_frames: int | None,
    fps: float | None = None,
    require_fps: bool = False,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate half-open frame intervals and return an auditable summary.

    Complete coverage means that the union of intervals is exactly
    ``[0, num_frames)`` and no frame is assigned more than once.
    """

    if num_frames is None:
        raise ValueError(f"{video_id}: strict coverage requires manifest num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError(f"{video_id}: manifest num_frames must be a positive integer")
    if require_fps and fps is None:
        raise ValueError(f"{video_id}: strict coverage requires manifest fps")
    if fps is not None and (
        isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0
    ):
        raise ValueError(f"{video_id}: manifest fps must be finite and positive")

    indices = np.asarray(clip_indices)
    starts = np.asarray(frame_starts)
    ends = np.asarray(frame_ends)
    for name, value in (("clip_indices", indices), ("frame_starts", starts), ("frame_ends", ends)):
        if value.ndim != 1:
            raise ValueError(f"{video_id}: {name} must be one-dimensional")
        if value.dtype.kind not in "iu":
            raise ValueError(f"{video_id}: {name} must contain integers")
    if not indices.size or starts.shape != indices.shape or ends.shape != indices.shape:
        raise ValueError(f"{video_id}: coverage arrays must have the same non-empty shape")
    if np.any(indices < 0):
        raise ValueError(f"{video_id}: clip_index values must be non-negative")
    if len(set(int(item) for item in indices)) != indices.size:
        raise ValueError(f"{video_id}: strict coverage requires unique clip_index values")
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError(f"{video_id}: prediction frame ranges must satisfy 0 <= start < end")
    if np.any(ends > num_frames):
        raise ValueError(f"{video_id}: prediction frame range exceeds num_frames={num_frames}")

    events: dict[int, int] = {0: 0, num_frames: 0}
    for start, end in zip(starts, ends, strict=True):
        start_value, end_value = int(start), int(end)
        events[start_value] = events.get(start_value, 0) + 1
        events[end_value] = events.get(end_value, 0) - 1
    active = 0
    previous = 0
    covered_frames = 0
    gap_frames = 0
    overlap_frames = 0
    for position in sorted(events):
        width = position - previous
        if active == 0:
            gap_frames += width
        elif active == 1:
            covered_frames += width
        else:
            covered_frames += width
            overlap_frames += width
        active += events[position]
        previous = position

    complete = gap_frames == 0 and overlap_frames == 0
    if require_complete and gap_frames:
        raise ValueError(f"{video_id}: frame coverage has gap frames={gap_frames}")
    if require_complete and overlap_frames:
        raise ValueError(f"{video_id}: frame coverage overlaps frames={overlap_frames}")
    return {
        "video_id": video_id,
        "num_frames": num_frames,
        "intervals": int(indices.size),
        "covered_frames": covered_frames,
        "gap_frames": gap_frames,
        "overlap_frames": overlap_frames,
        "coverage_ratio": covered_frames / num_frames,
        "complete": complete,
    }


def aggregate_coverage(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-video coverage without discarding incomplete identities."""

    if not summaries:
        raise ValueError("coverage summaries must not be empty")
    total_frames = sum(int(item["num_frames"]) for item in summaries)
    covered_frames = sum(int(item["covered_frames"]) for item in summaries)
    gap_frames = sum(int(item["gap_frames"]) for item in summaries)
    overlap_frames = sum(int(item["overlap_frames"]) for item in summaries)
    incomplete = [str(item["video_id"]) for item in summaries if not item["complete"]]
    return {
        "num_videos": len(summaries),
        "total_frames": total_frames,
        "covered_frames": covered_frames,
        "gap_frames": gap_frames,
        "overlap_frames": overlap_frames,
        "coverage_ratio": covered_frames / total_frames,
        "complete": not incomplete,
        "incomplete_video_ids": incomplete,
    }


__all__ = ["aggregate_coverage", "validate_frame_coverage"]
