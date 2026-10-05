"""One-video frozen-detector timing primitives built from existing VAD paths.

The module deliberately keeps no FeatureStore and never evaluates labels.  It
encodes every dense window of one decoded video, invokes the real frozen
TopKMIL task, and uses the repository's dense interval aggregation before
returning a frame-coverage receipt.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vadbench.data.dense_sampling import DenseClipSample, aggregate_interval_scores
from vadbench.data.video import build_clip_batch
from vadbench.engine.coverage import validate_frame_coverage
from vadbench.data.batches import clean_encoder_batch

try:  # Keep CLI help/catalog imports usable without the optional train runtime.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


VIDEO_EFFICIENCY_SCHEMA_VERSION = "vadbench.icassp2027.frozen-video-efficiency.v1"


@dataclass(frozen=True, slots=True)
class VideoTimingSettings:
    warmup: int = 1
    repeat: int = 10
    batch_size: int = 8

    def __post_init__(self) -> None:
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        if type(self.repeat) is not int or self.repeat <= 0:
            raise ValueError("repeat must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size != 8:
            raise ValueError("frozen full-video benchmark requires fixed batch_size=8")


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("frozen video detector timing requires PyTorch")
    return torch


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("summary observations must be finite")
    return {
        "median": float(np.percentile(array, 50)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _chunks(items: Sequence[DenseClipSample], size: int) -> tuple[tuple[DenseClipSample, ...], ...]:
    return tuple(tuple(items[index : index + size]) for index in range(0, len(items), size))


def _scores_from_existing_task(task: Any, features: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the established task and predictor score adapter, never hand-sigmoid."""

    torch_module = _require_torch()
    from vadbench.engine.predict import _scores  # Existing checkpoint predictor conversion.

    valid = torch_module.ones(features.shape[:2], dtype=torch_module.bool, device=features.device)
    step = task.prediction_step({"features": features, "valid_mask": valid})
    values = _scores(step, "wsvad")
    if values.shape != tuple(features.shape[:2]) or not np.isfinite(values).all():
        raise RuntimeError("existing weak-MIL predictor did not emit finite [B,S] snippet scores")
    predictions = step.predictions
    logits = getattr(predictions, "snippet_logits", None)
    if not torch_module.is_tensor(logits) or tuple(logits.shape) != tuple(features.shape[:2]):
        raise RuntimeError("existing weak-MIL task did not expose aligned snippet logits")
    return values, {
        "task": "weak_mil",
        "head_output": type(predictions).__qualname__,
        "snippet_logits_shape": list(logits.shape),
        "snippet_logits_dtype": str(logits.dtype),
        "video_logits_shape": list(predictions.video_logits.shape),
    }


def predictor_path_qa(task: Any, *, feature_dim: int, device: str) -> dict[str, Any]:
    """Small pre-timing QA against the existing task/predictor score path.

    The synthetic feature bag has no video or label.  It proves that the
    loaded head's task output, the existing predictor score adapter, and the
    existing interval aggregator agree on shape/finite behavior before an
    expensive full decoded video is measured.
    """

    torch_module = _require_torch()
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    with torch_module.inference_mode():
        features = torch_module.zeros((1, 4, feature_dim), dtype=torch_module.float32, device=device)
        scores, task_receipt = _scores_from_existing_task(task, features)
    dense = aggregate_interval_scores(
        np.asarray([0, 2, 4, 6], dtype=np.int64),
        np.asarray([3, 5, 7, 8], dtype=np.int64),
        scores[0],
        num_frames=8,
        reduction="mean",
    )
    coverage = validate_frame_coverage(
        video_id="synthetic-predictor-qa",
        clip_indices=np.arange(8, dtype=np.int64),
        frame_starts=np.arange(8, dtype=np.int64),
        frame_ends=np.arange(1, 9, dtype=np.int64),
        num_frames=8,
        require_complete=True,
    )
    receipt = {
        "status": "passed",
        "synthetic_labels_used": False,
        "predictor_task": task_receipt,
        "existing_interval_aggregation": {
            "reduction": dense.reduction,
            "scores_finite": bool(np.isfinite(dense.scores).all()),
            "contributors_min": int(dense.contributors.min()),
            "contributors_max": int(dense.contributors.max()),
        },
        "coverage": coverage,
    }
    del features
    return receipt


def _frame_intervals(scores: np.ndarray, contributors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compact an already frame-aligned aggregate; no second aggregation occurs."""

    starts: list[int] = [0]
    for index in range(1, scores.size):
        if scores[index] != scores[index - 1] or contributors[index] != contributors[index - 1]:
            starts.append(index)
    ends = np.asarray([*starts[1:], scores.size], dtype=np.int64)
    starts_array = np.asarray(starts, dtype=np.int64)
    return np.column_stack((starts_array, ends)), scores[starts_array]


def run_full_video_detector(
    *,
    adapter: Any,
    detector_task: Any,
    video_path: str,
    video_id: str,
    samples: Sequence[DenseClipSample],
    num_frames: int,
    fps: float,
    context_factory: Any | None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Decode all dense windows, score their pooled features, and cover every frame.

    ``context_factory`` is a prepared :class:`PairMergeDeployment` for one
    method or ``None`` for dense.  It is entered per real encoder batch, so
    reducer hook/gather/merge overhead is part of the timed path.
    """

    torch_module = _require_torch()
    if type(batch_size) is not int or batch_size != 8:
        raise ValueError("full-video detector benchmark fixes encoder batch_size=8")
    if not samples:
        raise ValueError("full-video detector needs nonempty dense samples")
    chunks = _chunks(samples, batch_size)
    pooled_batches: list[Any] = []
    token_lengths: list[int] = []
    reduction_receipts: list[dict[str, Any]] = []
    for group in chunks:
        raw = build_clip_batch(video_path, video_id, [sample.clip for sample in group])
        batch = clean_encoder_batch(raw)
        with torch_module.inference_mode():
            if context_factory is None:
                output = adapter.encode(batch, train=False)
                execution = None
            else:
                with context_factory(batch) as context:
                    output = adapter.encode(batch, train=False)
                    execution = dict(context.validate_execution())
            pooled = output.pooled
            if not torch_module.is_tensor(pooled) or tuple(pooled.shape[:1]) != (len(group),):
                raise RuntimeError("adapter pooled output does not align with the real dense batch")
            if pooled.dtype != torch_module.float32 or output.features.dtype != torch_module.float32:
                raise RuntimeError("formal full-video detector requires actual adapter features and pooled output in FP32")
            if not bool(torch_module.isfinite(pooled).all()):
                raise RuntimeError("adapter pooled output is non-finite")
            pooled_batches.append(pooled)
            token_lengths.extend([int(output.features.shape[1])] * len(group))
            if execution is not None:
                reduction_receipts.append(execution)
        del output, raw, batch
    # The adapter forwards above run in inference mode. Keep concatenation and
    # the frozen TopKMIL task in that same mode: prediction_step itself does
    # not open a no-grad context.
    with torch_module.inference_mode():
        features = torch_module.cat(pooled_batches, dim=0).unsqueeze(0)
        scores, head_receipt = _scores_from_existing_task(detector_task, features)
    if scores.shape != (1, len(samples)):
        raise RuntimeError("frozen detector scores do not align with all real dense windows")
    aggregated = aggregate_interval_scores(
        np.asarray([sample.score_frame_start for sample in samples], dtype=np.int64),
        np.asarray([sample.score_frame_end for sample in samples], dtype=np.int64),
        scores[0],
        num_frames=num_frames,
        reduction="mean",
    )
    intervals, interval_scores = _frame_intervals(aggregated.scores, aggregated.contributors)
    coverage = validate_frame_coverage(
        video_id=video_id,
        clip_indices=np.arange(len(intervals), dtype=np.int64),
        frame_starts=intervals[:, 0],
        frame_ends=intervals[:, 1],
        num_frames=num_frames,
        fps=fps,
        require_fps=True,
        require_complete=True,
    )
    receipt = {
        "video_id": video_id,
        "windows": len(samples),
        "encoder_batches": [len(group) for group in chunks],
        "final_batch_size": len(chunks[-1]),
        "pooled_feature_shape": list(features.shape),
        "pooled_feature_dtype": str(features.dtype),
        "token_lengths": {"unique": sorted(set(token_lengths)), "per_window": token_lengths},
        "head": head_receipt,
        "frame_scores_finite": bool(np.isfinite(aggregated.scores).all()),
        "final_intervals": int(len(intervals)),
        "coverage": coverage,
        "aggregation": {
            "reduction": aggregated.reduction,
            "contributors_min": int(aggregated.contributors.min()),
            "contributors_max": int(aggregated.contributors.max()),
            "overlap_before_aggregation": bool(aggregated.contributors.max() > 1),
            "final_intervals_nonoverlapping": True,
        },
        "reduction_executions": reduction_receipts,
    }
    del features, pooled_batches
    return receipt


def measure_full_video_detector(
    operation: Callable[[], Mapping[str, Any]],
    *,
    torch_module: Any,
    device: str,
    settings: VideoTimingSettings,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Synchronized wall-clock timing of one entire decoded-video detector path."""

    if not str(device).lower().startswith("cuda") or not torch_module.cuda.is_available():
        raise RuntimeError("formal full-video detector benchmark requires CUDA")
    cuda = torch_module.cuda

    def call(name: str) -> Any:
        function = getattr(cuda, name)
        try:
            return function(device)
        except TypeError:
            return function()

    def event() -> Any:
        return cuda.Event(enable_timing=True)

    call("synchronize")
    cuda.empty_cache()
    call("synchronize")
    resident = {"allocated": int(call("memory_allocated")), "reserved": int(call("memory_reserved"))}

    def once(index: int, *, include: bool) -> dict[str, Any] | None:
        call("synchronize")
        call("reset_peak_memory_stats")
        start_event, end_event = event(), event()
        start_event.record()
        started = clock()
        payload = dict(operation())
        end_event.record()
        call("synchronize")
        wall = max(0.0, float(clock() - started))
        event_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
        if not math.isfinite(event_seconds) or event_seconds < 0:
            raise RuntimeError("CUDA event timing is invalid")
        if not include:
            return None
        return {
            "repeat_index": index,
            "wall_seconds": wall,
            "cuda_event_seconds": event_seconds,
            "peak_allocated_bytes": int(call("max_memory_allocated")),
            "peak_reserved_bytes": int(call("max_memory_reserved")),
            "video_receipt": payload,
        }

    for index in range(settings.warmup):
        once(-(settings.warmup - index), include=False)
    samples = [once(index, include=True) for index in range(settings.repeat)]
    completed = [item for item in samples if item is not None]
    if len(completed) != settings.repeat:
        raise RuntimeError("formal timing did not retain every repeat")
    return {
        "status": "completed",
        "settings": {"warmup": settings.warmup, "repeat": settings.repeat, "batch_size": settings.batch_size},
        "measurement_boundary": {
            "included": "video decode, adapter/encoder/reducer, frozen refit head, dense interval aggregation, frame coverage validation",
            "excluded": "model/head loading, source verification, reducer setup, checkpoint/hash QA, FeatureStore writes",
            "primary_duration": "synchronized wall clock after CUDA end event",
            "cuda_event_duration": "diagnostic only",
            "io_context": "same local path is reopened each iteration; result is warm-OS-cache video I/O, not a cold-disk claim",
        },
        "model_resident_baseline_gpu_memory_bytes": resident,
        "samples": completed,
        "summary": {
            "wall_seconds": _summary([item["wall_seconds"] for item in completed]),
            "cuda_event_seconds": _summary([item["cuda_event_seconds"] for item in completed]),
            "peak_allocated_bytes": _summary([item["peak_allocated_bytes"] for item in completed]),
            "peak_reserved_bytes": _summary([item["peak_reserved_bytes"] for item in completed]),
        },
    }


__all__ = [
    "VIDEO_EFFICIENCY_SCHEMA_VERSION",
    "VideoTimingSettings",
    "measure_full_video_detector",
    "predictor_path_qa",
    "run_full_video_detector",
]
