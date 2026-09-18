"""Formal CUDA timing primitives for the frozen ICASSP reducer comparison.

This module intentionally does not create adapters, decode videos, fit gates, or
write feature stores.  The command entrypoint supplies those concrete operations.
Keeping timing here small makes the measured boundary auditable: an operation is
called once per warmup/repeat, CUDA is synchronized around it, and CUDA-event
durations are retained only as a diagnostic alongside wall-clock results.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EFFICIENCY_SCHEMA_VERSION = "vadbench.icassp2027.frozen-efficiency.v1"


@dataclass(frozen=True, slots=True)
class FrozenTimingSettings:
    """Fixed formal timing controls; changing either count is explicit evidence."""

    warmup: int = 5
    repeat: int = 30

    def __post_init__(self) -> None:
        for name in ("warmup", "repeat"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "warmup" else 1):
                raise ValueError(f"{name} must be {'nonnegative' if name == 'warmup' else 'positive'} int")


class CudaTimingRuntime:
    """Narrow CUDA facade, deliberately injectable for deterministic CPU tests."""

    def __init__(self, torch_module: Any, *, device: str) -> None:
        self.torch = torch_module
        self.device = device
        self.cuda = getattr(torch_module, "cuda", None)
        available = callable(getattr(self.cuda, "is_available", None)) and self.cuda.is_available()
        if not available or not str(device).lower().startswith("cuda"):
            raise RuntimeError("formal frozen-reducer timing requires an available CUDA device")

    def _call(self, name: str, default: Any = None) -> Any:
        function = getattr(self.cuda, name, None)
        if not callable(function):
            return default
        try:
            return function(self.device)
        except TypeError:
            return function()

    def synchronize(self) -> None:
        self._call("synchronize")

    def empty_cache(self) -> None:
        self._call("empty_cache")

    def reset_peak(self) -> None:
        self._call("reset_peak_memory_stats")

    def memory(self, name: str) -> int | None:
        value = self._call(name)
        return None if value is None else max(0, int(value))

    def event(self) -> Any | None:
        factory = getattr(self.cuda, "Event", None)
        return factory(enable_timing=True) if callable(factory) else None


def summary(values: Sequence[float | int]) -> dict[str, float]:
    """Return the mandated robust distribution summary, never a fabricated zero."""

    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("timing samples must be finite")
    return {
        "median": float(np.percentile(array, 50)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _event_seconds(start: Any | None, end: Any | None) -> float | None:
    if start is None or end is None:
        return None
    elapsed = start.elapsed_time(end)
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed < 0:
        raise RuntimeError("CUDA event elapsed time is invalid")
    return float(elapsed) / 1000.0


def _memory(runtime: CudaTimingRuntime) -> dict[str, int | None]:
    return {
        "allocated": runtime.memory("memory_allocated"),
        "reserved": runtime.memory("memory_reserved"),
        "peak_allocated": runtime.memory("max_memory_allocated"),
        "peak_reserved": runtime.memory("max_memory_reserved"),
    }


def measure_scope(
    operation: Callable[[], Mapping[str, Any] | None],
    *,
    runtime: CudaTimingRuntime,
    settings: FrozenTimingSettings,
    scope: str,
    batch_size: int,
    sampled_frames: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure one already-prepared operation.

    ``operation`` owns the exact scope boundary.  For a reduced operation it
    must enter/exit :class:`PairMergeDeployment` itself, which intentionally
    includes temporary hook registration, gather/merge, suffix-shape checking,
    and V-JEPA positional injection in the timing sample.
    """

    if scope not in {"pure_model", "adapter", "end_to_end"}:
        raise ValueError("scope must be pure_model, adapter, or end_to_end")
    if type(batch_size) is not int or batch_size <= 0 or type(sampled_frames) is not int or sampled_frames <= 0:
        raise ValueError("batch_size and sampled_frames must be positive ints")

    # Each method/scope starts from the same loaded-model state with allocator
    # cache released.  The resident model allocation remains visible below.
    runtime.synchronize()
    runtime.empty_cache()
    runtime.synchronize()
    resident = {
        "allocated": runtime.memory("memory_allocated"),
        "reserved": runtime.memory("memory_reserved"),
    }

    def invoke(*, include: bool, index: int) -> dict[str, Any] | None:
        runtime.synchronize()
        runtime.reset_peak()
        before = _memory(runtime)
        event_start, event_end = runtime.event(), runtime.event()
        if event_start is not None:
            event_start.record()
        started = clock()
        result = operation()
        if event_end is not None:
            event_end.record()
        runtime.synchronize()
        # CUDA launches are asynchronous.  The stopping clock belongs after
        # the end event and its device synchronization, otherwise wall time
        # only measures host-side enqueue work.
        wall = max(0.0, float(clock() - started))
        event = _event_seconds(event_start, event_end)
        if not include:
            return None
        after = _memory(runtime)
        return {
            "repeat_index": index,
            "wall_seconds": wall,
            "cuda_event_seconds": event,
            "throughput": {
                "clips_per_second": None if wall == 0 else float(batch_size / wall),
                "sampled_frames_per_second": None if wall == 0 else float(sampled_frames / wall),
            },
            "gpu_memory_bytes": {"before": before, "after": after},
            "operation_receipt": None if result is None else dict(result),
        }

    for index in range(settings.warmup):
        invoke(include=False, index=-(settings.warmup - index))
    samples = [invoke(include=True, index=index) for index in range(settings.repeat)]
    assert all(sample is not None for sample in samples)
    completed = [sample for sample in samples if sample is not None]

    wall = [sample["wall_seconds"] for sample in completed]
    events = [sample["cuda_event_seconds"] for sample in completed]
    event_values = [value for value in events if value is not None]
    clip_rates = [sample["throughput"]["clips_per_second"] for sample in completed]
    frame_rates = [sample["throughput"]["sampled_frames_per_second"] for sample in completed]
    peak_allocated = [sample["gpu_memory_bytes"]["after"]["peak_allocated"] for sample in completed]
    peak_reserved = [sample["gpu_memory_bytes"]["after"]["peak_reserved"] for sample in completed]
    allocated_values = [value for value in peak_allocated if value is not None]
    reserved_values = [value for value in peak_reserved if value is not None]
    return {
        "status": "completed",
        "scope": scope,
        "settings": {"warmup": settings.warmup, "repeat": settings.repeat},
        "measurement_boundary": {
            "included": "operation callable; reduced operation includes deployment context registration and validation",
            "excluded": "model load, setup geometry, teacher forward, calibration, CPU dumps, feature-store writes",
            "synchronization": "CUDA synchronize before/reset-after every measured operation",
            "cuda_events": "stream elapsed-time diagnostic, including possible host enqueue gaps; not summed kernel time; wall_seconds is primary",
            "reserved_memory": "PyTorch caching-allocator reserved bytes; empty_cache runs before each scope, never within warmups/repeats",
            "end_to_end_io": (
                "same local video path is reopened per iteration; this is a warm-OS-cache deployment path, not a cold-disk claim"
                if scope == "end_to_end" else None
            ),
        },
        "model_resident_baseline_gpu_memory_bytes": resident,
        "samples": completed,
        "summary": {
            "wall_seconds": summary(wall),
            "cuda_event_seconds": None if not event_values else summary(event_values),
            "clips_per_second": summary(clip_rates),
            "sampled_frames_per_second": summary(frame_rates),
            "peak_allocated_bytes": None if not allocated_values else summary(allocated_values),
            "peak_reserved_bytes": None if not reserved_values else summary(reserved_values),
        },
    }


def failed_scope(*, scope: str, error: BaseException) -> dict[str, Any]:
    """Persist a real failure without turning it into a zero-valued result."""

    return {
        "status": "failed",
        "scope": scope,
        "samples": None,
        "summary": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


__all__ = [
    "CudaTimingRuntime",
    "EFFICIENCY_SCHEMA_VERSION",
    "FrozenTimingSettings",
    "failed_scope",
    "measure_scope",
    "summary",
]
