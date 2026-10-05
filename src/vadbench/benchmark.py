"""可复现的视频编码器性能基准执行器。

本模块只依赖公共 encoder contract，并通过 :class:`BenchmarkCase` 注入
adapter、解码与预处理函数。这样真实视频/GPU 基准和无 PyTorch 的 CPU
fake 测试共用同一条计时、缓存遥测与公平性检查路径。

``encoder_seconds`` 是 adapter 调用的同步 wall time，包含 adapter 内部的
原生压缩；``native_compression_seconds`` 是从 ``StreamStep.telemetry`` 读取的
嵌套子项，不能再次加到端到端 wall time。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    EncoderOutput,
    StreamState,
    validate_encoder_adapter,
    validate_encoder_output,
    validate_stream_step,
)

PERFORMANCE_SCHEMA_VERSION = "vadbench.performance-result.v1"
_AUTO_TORCH = object()


def _json_safe(value: Any) -> Any:
    """Return deterministic, JSON-compatible provenance without model objects."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, (str, int, float, bool)):
        return _json_safe(raw_value)
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"{name} 必须是{qualifier}整数")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Execution controls shared by all cases in one comparison."""

    warmup: int = 1
    repeat: int = 5
    synchronize_cuda: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.warmup, "warmup", allow_zero=True)
        _positive_int(self.repeat, "repeat")
        if type(self.synchronize_cuda) is not bool:
            raise ValueError("synchronize_cuda 必须是 bool")
        if self.device is not None and (not isinstance(self.device, str) or not self.device):
            raise ValueError("device 必须是非空字符串或 None")


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    """One repeatable decode -> preprocess workload.

    ``decode`` is called again for every warmup/repeat. ``preprocess`` must return
    one :class:`ClipBatch` or a finite iterable of batches. For streaming mode,
    each batch is one B=1 chronological chunk; video changes trigger a state
    finalize/reset and never reuse cache across videos.
    """

    name: str
    mode: str
    decode: Callable[[], Any]
    preprocess: Callable[[Any], ClipBatch | Iterable[ClipBatch]]
    sampling: Mapping[str, Any] = field(default_factory=dict)
    video_seconds: float | None = None
    task: str = "encoder_performance_only"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("workload.name 必须是非空字符串")
        if self.mode not in {"fixed", "streaming"}:
            raise ValueError("workload.mode 必须是 fixed 或 streaming")
        if not callable(self.decode) or not callable(self.preprocess):
            raise ValueError("decode/preprocess 必须可调用")
        if not isinstance(self.sampling, Mapping):
            raise ValueError("sampling 必须是 mapping")
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task 必须是非空字符串")
        if self.video_seconds is not None and (
            not isinstance(self.video_seconds, (int, float))
            or not math.isfinite(float(self.video_seconds))
            or float(self.video_seconds) <= 0
        ):
            raise ValueError("video_seconds 必须是有限正数或 None")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Adapter plus workload and case-specific reproducibility metadata."""

    name: str
    adapter: Any
    workload: BenchmarkWorkload
    compression: Any | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("case.name 必须是非空字符串")
        if not isinstance(self.workload, BenchmarkWorkload):
            raise ValueError("workload 必须是 BenchmarkWorkload")
        if not isinstance(self.config, Mapping):
            raise ValueError("config 必须是 mapping")


class _TorchRuntime:
    """Small optional PyTorch/CUDA facade; ``torch_module=None`` disables it."""

    def __init__(
        self,
        torch_module: Any | None | object,
        *,
        device: str | None,
        synchronize_cuda: bool,
    ) -> None:
        if torch_module is _AUTO_TORCH:
            try:
                import torch  # type: ignore
            except ImportError:  # pragma: no cover - depends on optional environment
                torch = None
            torch_module = torch
        self.torch = torch_module
        self.device = device
        self.synchronize_cuda = synchronize_cuda
        self.cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
        self.cuda_available = bool(
            self.cuda is not None
            and callable(getattr(self.cuda, "is_available", None))
            and self.cuda.is_available()
        )
        self.cuda_active = self.cuda_available and (
            device is None or str(device).lower().startswith("cuda")
        )

    def _cuda_call(self, name: str, default: Any = None) -> Any:
        if not self.cuda_active:
            return default
        function = getattr(self.cuda, name, None)
        if not callable(function):
            return default
        if self.device is not None:
            try:
                return function(self.device)
            except TypeError:
                pass
        return function()

    def synchronize(self) -> None:
        if self.synchronize_cuda:
            self._cuda_call("synchronize")

    def reset_peak_memory(self) -> None:
        self._cuda_call("reset_peak_memory_stats")

    def peak_memory_bytes(self) -> int | None:
        value = self._cuda_call("max_memory_allocated")
        return None if value is None else max(0, int(value))

    def peak_reserved_memory_bytes(self) -> int | None:
        value = self._cuda_call("max_memory_reserved")
        return None if value is None else max(0, int(value))

    def allocated_memory_bytes(self) -> int | None:
        value = self._cuda_call("memory_allocated")
        return None if value is None else max(0, int(value))

    def reserved_memory_bytes(self) -> int | None:
        value = self._cuda_call("memory_reserved")
        return None if value is None else max(0, int(value))

    def inference_context(self) -> Any:
        inference_mode = None if self.torch is None else getattr(self.torch, "inference_mode", None)
        return inference_mode() if callable(inference_mode) else nullcontext()

    def provenance(self) -> dict[str, Any]:
        torch_module = self.torch
        if torch_module is None:
            return {
                "torch_available": False,
                "torch_version": None,
                "cuda_available": False,
                "cuda_version": None,
                "cudnn_version": None,
                "device_requested": self.device,
                "device_name": None,
                "device_capability": None,
                "device_total_memory_bytes": None,
            }

        version_namespace = getattr(torch_module, "version", None)
        backends = getattr(torch_module, "backends", None)
        cudnn = getattr(backends, "cudnn", None)
        cudnn_version = None
        if cudnn is not None and callable(getattr(cudnn, "version", None)):
            try:
                cudnn_version = cudnn.version()
            except Exception:  # pragma: no cover - driver-specific introspection
                cudnn_version = None

        device_name = self._cuda_call("get_device_name")
        capability = self._cuda_call("get_device_capability")
        properties = self._cuda_call("get_device_properties")
        total_memory = getattr(properties, "total_memory", None)
        return {
            "torch_available": True,
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "cuda_available": self.cuda_available,
            "cuda_version": (
                None if version_namespace is None else getattr(version_namespace, "cuda", None)
            ),
            "cudnn_version": cudnn_version,
            "device_requested": self.device,
            "device_name": None if device_name is None else str(device_name),
            "device_capability": (
                None if capability is None else [int(item) for item in capability]
            ),
            "device_total_memory_bytes": (None if total_memory is None else int(total_memory)),
        }


@contextmanager
def _adapter_inference(adapter: Any, runtime: _TorchRuntime) -> Iterable[None]:
    """Put injected model owners in eval mode and restore their prior state."""

    candidates: list[Any] = []
    for candidate in (adapter, getattr(adapter, "model", None), getattr(adapter, "encoder", None)):
        if (
            candidate is not None
            and all(candidate is not existing for existing in candidates)
            and callable(getattr(candidate, "eval", None))
        ):
            candidates.append(candidate)
    previous_training = [getattr(candidate, "training", None) for candidate in candidates]
    for candidate in candidates:
        candidate.eval()
    try:
        with runtime.inference_context():
            yield
    finally:
        for candidate, training in zip(candidates, previous_training, strict=True):
            train = getattr(candidate, "train", None)
            if isinstance(training, bool) and callable(train):
                train(training)


def _normalise_batches(value: ClipBatch | Iterable[ClipBatch]) -> tuple[ClipBatch, ...]:
    batches = (value,) if isinstance(value, ClipBatch) else tuple(value)
    if not batches:
        raise ValueError("preprocess 必须至少返回一个 ClipBatch")
    if any(not isinstance(batch, ClipBatch) for batch in batches):
        raise TypeError("preprocess 只能返回 ClipBatch 或 ClipBatch iterable")
    return batches


def _measure(
    operation: Callable[[], Any], runtime: _TorchRuntime, clock: Callable[[], float]
) -> tuple[Any, float]:
    runtime.synchronize()
    started = clock()
    value = operation()
    runtime.synchronize()
    elapsed = max(0.0, float(clock() - started))
    return value, elapsed


def _batch_video_seconds(batch: ClipBatch, declared_sample_fps: float | None) -> float:
    timestamps = np.asarray(batch.timestamps_s)
    valid_lengths = batch.valid_lengths
    seconds = 0.0
    for row, length_value in enumerate(valid_lengths):
        length = int(length_value)
        times = timestamps[row, :length].astype(np.float64, copy=False)
        if length > 1:
            positive_steps = np.diff(times)
            positive_steps = positive_steps[positive_steps > 0]
            if positive_steps.size:
                frame_period = float(np.median(positive_steps))
            elif declared_sample_fps:
                frame_period = 1.0 / declared_sample_fps
            else:
                frame_period = 0.0
            seconds += float(times[-1] - times[0]) + frame_period
        elif length == 1 and declared_sample_fps:
            seconds += 1.0 / declared_sample_fps
    return seconds


def _sampling_observation(
    batches: Sequence[ClipBatch],
    *,
    sampling: Mapping[str, Any],
    explicit_video_seconds: float | None,
) -> dict[str, Any]:
    coordinate_rows: list[dict[str, Any]] = []
    spatial_sizes: set[tuple[int, int]] = set()
    batch_valid_lengths: list[list[int]] = []
    sampled_frames = 0
    all_frame_indices_present = True
    declared_sample_fps = sampling.get("sample_fps")
    try:
        sample_fps = float(declared_sample_fps) if declared_sample_fps is not None else None
    except (TypeError, ValueError):
        sample_fps = None
    if sample_fps is not None and (not math.isfinite(sample_fps) or sample_fps <= 0):
        sample_fps = None

    derived_seconds = 0.0
    for batch in batches:
        spatial_sizes.add(tuple(int(item) for item in batch.spatial_size))
        lengths = [int(item) for item in batch.valid_lengths]
        batch_valid_lengths.append(lengths)
        sampled_frames += sum(lengths)
        derived_seconds += _batch_video_seconds(batch, sample_fps)
        timestamps = np.asarray(batch.timestamps_s)
        frame_indices = None if batch.frame_indices is None else np.asarray(batch.frame_indices)
        all_frame_indices_present = all_frame_indices_present and frame_indices is not None
        for row, (video_id, length) in enumerate(zip(batch.video_ids, lengths, strict=True)):
            for column in range(length):
                coordinate_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": (
                            None if frame_indices is None else int(frame_indices[row, column])
                        ),
                        # float.hex is stable and does not hide sampling differences by rounding.
                        "timestamp_hex": float(timestamps[row, column]).hex(),
                    }
                )

    return {
        "sampled_frames": sampled_frames,
        "video_seconds": (
            float(explicit_video_seconds)
            if explicit_video_seconds is not None
            else float(derived_seconds)
        ),
        "spatial_sizes": [list(size) for size in sorted(spatial_sizes)],
        "batch_valid_lengths": batch_valid_lengths,
        "video_ids": list(dict.fromkeys(row["video_id"] for row in coordinate_rows)),
        "coordinate_sha256": _sha256_json(coordinate_rows),
        "coordinate_basis": "video_id+frame_index+timestamp",
        "all_frame_indices_present": all_frame_indices_present,
        "layout": ClipBatch.layout,
        "dtype": ClipBatch.dtype,
    }


def _numeric_telemetry(value: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        raw = value.get(name)
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            return max(0.0, float(raw))
    return default


def _cache_step_record(
    telemetry: Mapping[str, Any], state: StreamState, *, step_index: int
) -> dict[str, Any]:
    cache_bytes_fallback = sum(int(view.nbytes) for view in state.caches.values())
    before = int(
        _numeric_telemetry(
            telemetry, "decoder_kv_tokens_before_max", "kv_tokens_before", default=0.0
        )
    )
    after = int(
        _numeric_telemetry(telemetry, "decoder_kv_tokens_after_max", "kv_tokens_after", default=0.0)
    )
    return {
        "step_index": step_index,
        "cache_owner": (
            str(telemetry["cache_owner"]) if telemetry.get("cache_owner") is not None else None
        ),
        "is_vision_encoder_kv": bool(telemetry.get("is_vision_encoder_kv", False)),
        "decoder_kv_layers": int(_numeric_telemetry(telemetry, "decoder_kv_layers", default=0.0)),
        "decoder_kv_input_tokens": int(_numeric_telemetry(telemetry, "input_tokens", default=0.0)),
        "output_tokens": int(_numeric_telemetry(telemetry, "output_tokens", default=0.0)),
        "frames_encoded": int(_numeric_telemetry(telemetry, "frames_encoded", default=0.0)),
        "projected_visual_tokens": int(
            _numeric_telemetry(telemetry, "projected_visual_tokens", default=0.0)
        ),
        "feature_stage": (
            str(telemetry["feature_stage"]) if telemetry.get("feature_stage") is not None else None
        ),
        "feature_cache_conditioned": bool(telemetry.get("feature_cache_conditioned", False)),
        "cache_hit": bool(telemetry.get("cache_hit", False)),
        "kv_tokens_before_min": int(
            _numeric_telemetry(telemetry, "decoder_kv_tokens_before_min", default=float(before))
        ),
        "kv_tokens_before": before,
        "kv_tokens_after_min": int(
            _numeric_telemetry(telemetry, "decoder_kv_tokens_after_min", default=float(after))
        ),
        "kv_tokens_after": after,
        "kv_tokens_after_total": int(
            _numeric_telemetry(telemetry, "decoder_kv_tokens_after_total", default=float(after))
        ),
        "reused_tokens": int(_numeric_telemetry(telemetry, "reused_tokens", default=0.0)),
        "cache_bytes": int(
            _numeric_telemetry(telemetry, "cache_bytes", default=float(cache_bytes_fallback))
        ),
        "decoder_kv_bytes_after": int(
            _numeric_telemetry(
                telemetry, "decoder_kv_bytes_after", default=float(cache_bytes_fallback)
            )
        ),
        "decoder_kv_replaced": bool(telemetry.get("decoder_kv_replaced", False)),
        "decoder_kv_replaced_by_policy": bool(
            telemetry.get("decoder_kv_replaced_by_policy", False)
        ),
        "external_cache_policy": (
            str(telemetry["external_cache_policy"])
            if telemetry.get("external_cache_policy") is not None
            else None
        ),
        "external_cache_policy_applied": bool(
            telemetry.get("external_cache_policy_applied", False)
        ),
        "native_compression_seconds": _numeric_telemetry(
            telemetry,
            "native_hermes_compression_ms",
            "native_compression_ms",
            default=0.0,
        )
        / 1000.0,
        "native_compression_called": bool(
            telemetry.get(
                "native_hermes_compression_called",
                telemetry.get("native_compression_called", False),
            )
        ),
        "native_compression_applied": bool(
            telemetry.get(
                "native_hermes_compression_applied",
                telemetry.get("native_compression_applied", False),
            )
        ),
        "native_compression_enabled": bool(
            telemetry.get(
                "native_hermes_compression_enabled",
                telemetry.get("native_compression_enabled", False),
            )
        ),
        "native_compression_mode": (
            str(telemetry["native_hermes_compression_mode"])
            if telemetry.get("native_hermes_compression_mode") is not None
            else None
        ),
        "native_visual_budget_tokens": int(
            _numeric_telemetry(telemetry, "native_hermes_visual_budget_tokens", default=0.0)
        ),
        "native_protected_prefix_tokens": int(
            _numeric_telemetry(telemetry, "native_hermes_protected_prefix_tokens", default=0.0)
        ),
        "native_effective_total_budget_tokens": int(
            _numeric_telemetry(
                telemetry, "native_hermes_effective_total_budget_tokens", default=0.0
            )
        ),
        "native_tokens_before_max": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_before_max", default=float(before))
        ),
        "native_tokens_before_min": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_before_min", default=float(before))
        ),
        "native_tokens_before_total": int(
            _numeric_telemetry(
                telemetry, "native_hermes_tokens_before_total", default=float(before)
            )
        ),
        "native_tokens_after_min": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_after_min", default=float(after))
        ),
        "native_tokens_after_max": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_after_max", default=float(after))
        ),
        "native_tokens_after_total": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_after_total", default=float(after))
        ),
        "native_tokens_evicted_total": int(
            _numeric_telemetry(telemetry, "native_hermes_tokens_evicted_total", default=0.0)
        ),
    }


def _valid_output_tokens(output: EncoderOutput | None) -> int:
    if output is None:
        return 0
    return int(output.timeline.valid_lengths.sum())


def _feature_stage(output: EncoderOutput | None, *, fallback: str) -> str:
    if output is None:
        return fallback
    value = output.aux.get("feature_stage")
    return fallback if value is None else str(value)


def _encode_fixed(
    adapter: Any, batches: Sequence[ClipBatch]
) -> tuple[int, list[dict[str, Any]], list[str]]:
    if not callable(getattr(adapter, "encode", None)):
        raise TypeError("fixed adapter 必须实现 encode")
    output_tokens = 0
    feature_stages: list[str] = []
    for batch in batches:
        output = adapter.encode(batch, train=False)
        validate_encoder_output(output, batch)
        output_tokens += _valid_output_tokens(output)
        feature_stages.append(_feature_stage(output, fallback="fixed_encoder_output"))
    return output_tokens, [], feature_stages


def _encode_streaming(
    adapter: Any,
    batches: Sequence[ClipBatch],
    compression: Any | None,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    for name in ("init_state", "encode_step", "finalize"):
        if not callable(getattr(adapter, name, None)):
            raise TypeError(f"streaming adapter 必须实现 {name}")

    current_video_id: str | None = None
    state: StreamState | None = None
    output_tokens = 0
    cache_steps: list[dict[str, Any]] = []
    feature_stages: list[str] = []
    for batch in batches:
        if batch.batch_size != 1:
            raise ValueError("streaming benchmark 的每个 chunk 必须满足 B=1")
        video_id = batch.video_ids[0]
        if video_id != current_video_id:
            if state is not None:
                final_output = adapter.finalize(state)
                output_tokens += _valid_output_tokens(final_output)
                if final_output is not None:
                    feature_stages.append(
                        _feature_stage(final_output, fallback="streaming_encoder_output")
                    )
            state = adapter.init_state(video_id)
            if not isinstance(state, StreamState) or state.video_id != video_id:
                raise TypeError("init_state 必须返回匹配 video_id 的 StreamState")
            current_video_id = video_id
        assert state is not None
        previous_state = state
        step = adapter.encode_step(batch, state, train=False, compression=compression)
        validate_stream_step(
            step,
            previous_state=previous_state,
            chunk=batch,
            capabilities=adapter.capabilities,
        )
        state = step.state
        output_tokens += _valid_output_tokens(step.output)
        if step.output is not None:
            feature_stages.append(_feature_stage(step.output, fallback="streaming_encoder_output"))
        cache_steps.append(_cache_step_record(step.telemetry, state, step_index=state.step_index))
    if state is not None:
        final_output = adapter.finalize(state)
        output_tokens += _valid_output_tokens(final_output)
        if final_output is not None:
            feature_stages.append(_feature_stage(final_output, fallback="streaming_encoder_output"))
    return output_tokens, cache_steps, feature_stages


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _one_repeat(
    case: BenchmarkCase,
    runtime: _TorchRuntime,
    *,
    clock: Callable[[], float],
    repeat_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime.synchronize()
    baseline_allocated = runtime.allocated_memory_bytes()
    baseline_reserved = runtime.reserved_memory_bytes()
    runtime.reset_peak_memory()
    wall_started = clock()

    decoded, decode_seconds = _measure(case.workload.decode, runtime, clock)
    batches, preprocess_seconds = _measure(
        lambda: _normalise_batches(case.workload.preprocess(decoded)), runtime, clock
    )
    observation = _sampling_observation(
        batches,
        sampling=case.workload.sampling,
        explicit_video_seconds=case.workload.video_seconds,
    )

    if case.workload.mode == "fixed":
        if case.compression is not None:
            raise ValueError("fixed benchmark 不接受 cache compression")
        (output_tokens, cache_steps, feature_stages), encoder_seconds = _measure(
            lambda: _encode_fixed(case.adapter, batches), runtime, clock
        )
    else:
        (output_tokens, cache_steps, feature_stages), encoder_seconds = _measure(
            lambda: _encode_streaming(case.adapter, batches, case.compression), runtime, clock
        )

    runtime.synchronize()
    wall_seconds = max(0.0, float(clock() - wall_started))
    peak_memory = runtime.peak_memory_bytes()
    peak_reserved_memory = runtime.peak_reserved_memory_bytes()
    steady_allocated = runtime.allocated_memory_bytes()
    steady_reserved = runtime.reserved_memory_bytes()
    native_seconds = sum(step["native_compression_seconds"] for step in cache_steps)
    decoder_kv_input_tokens = sum(step["decoder_kv_input_tokens"] for step in cache_steps)
    frames = int(observation["sampled_frames"])
    source_video_seconds = float(observation["video_seconds"])
    repeat = {
        "repeat_index": repeat_index,
        "timings_seconds": {
            "decode": decode_seconds,
            "preprocess": preprocess_seconds,
            "encoder": encoder_seconds,
            "encoder_excluding_native_compression": max(0.0, encoder_seconds - native_seconds),
            "native_compression": native_seconds,
            "wall": wall_seconds,
        },
        "counts": {
            "frames": frames,
            # This is source-time coverage, rather than wall duration.  It is
            # explicit so a clipped/temporally sparse workload cannot be read as
            # having processed the entire source video.
            "source_video_seconds": source_video_seconds,
            "output_tokens": output_tokens,
            "decoder_kv_input_tokens": decoder_kv_input_tokens,
        },
        "throughput": {
            "end_to_end_frames_per_second": _safe_rate(frames, wall_seconds),
            "end_to_end_source_video_seconds_per_second": _safe_rate(
                source_video_seconds, wall_seconds
            ),
            "end_to_end_tokens_per_second": _safe_rate(output_tokens, wall_seconds),
            "encoder_frames_per_second": _safe_rate(frames, encoder_seconds),
            "encoder_source_video_seconds_per_second": _safe_rate(
                source_video_seconds, encoder_seconds
            ),
            "encoder_tokens_per_second": _safe_rate(output_tokens, encoder_seconds),
            "end_to_end_decoder_kv_tokens_per_second": _safe_rate(
                decoder_kv_input_tokens, wall_seconds
            ),
            "encoder_decoder_kv_tokens_per_second": _safe_rate(
                decoder_kv_input_tokens, encoder_seconds
            ),
        },
        "peak_gpu_memory_bytes": peak_memory,
        "gpu_memory_bytes": {
            "allocated": {
                "baseline": baseline_allocated,
                "peak": peak_memory,
                "steady": steady_allocated,
            },
            "reserved": {
                "baseline": baseline_reserved,
                "peak": peak_reserved_memory,
                "steady": steady_reserved,
            },
        },
        "token_semantics": {
            "frame_input": "decoded BTHWC uint8 frames",
            "output_tokens": "adapter feature tokens; per-case diagnostic",
            "decoder_kv": (
                "language model decoder KV tokens" if case.workload.mode == "streaming" else None
            ),
        },
        "feature_stages": list(dict.fromkeys(feature_stages)),
        "cache": {
            "kv_tokens_before_max": max(
                (step["kv_tokens_before"] for step in cache_steps), default=0
            ),
            "kv_tokens_after_max": max(
                (step["kv_tokens_after"] for step in cache_steps), default=0
            ),
            "reused_tokens_total": sum(step["reused_tokens"] for step in cache_steps),
            "cache_bytes_peak": max((step["cache_bytes"] for step in cache_steps), default=0),
            "native_compression_calls": sum(
                int(step["native_compression_called"]) for step in cache_steps
            ),
            "native_compression_applied_steps": sum(
                int(step["native_compression_applied"]) for step in cache_steps
            ),
            "steps": cache_steps,
        },
    }
    return repeat, observation


def _stats(values: Sequence[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("不能聚合空序列")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_repeats(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sections: dict[str, dict[str, dict[str, float]]] = {}
    for section_name in ("timings_seconds", "counts", "throughput"):
        keys = list(repeats[0][section_name])
        section: dict[str, dict[str, float]] = {}
        for key in keys:
            values = [repeat[section_name][key] for repeat in repeats]
            numeric_values = [value for value in values if isinstance(value, (int, float))]
            if numeric_values:
                section[key] = _stats(numeric_values)
        sections[section_name] = section

    memory_metrics: dict[str, dict[str, float]] = {}
    for family in ("allocated", "reserved"):
        for point in ("baseline", "peak", "steady"):
            values = [
                repeat["gpu_memory_bytes"][family][point]
                for repeat in repeats
                if repeat["gpu_memory_bytes"][family][point] is not None
            ]
            if values:
                memory_metrics[f"{family}_{point}_bytes"] = _stats(values)
    sections["memory"] = memory_metrics
    sections["cache"] = {
        key: _stats([repeat["cache"][key] for repeat in repeats])
        for key in (
            "kv_tokens_before_max",
            "kv_tokens_after_max",
            "reused_tokens_total",
            "cache_bytes_peak",
            "native_compression_calls",
            "native_compression_applied_steps",
        )
    }
    return sections


def _adapter_provenance(adapter: Any) -> dict[str, Any]:
    capabilities = getattr(adapter, "capabilities", None)
    return {
        "module": type(adapter).__module__,
        "class": type(adapter).__qualname__,
        "capabilities": _json_safe(capabilities),
    }


def _machine_provenance() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": uname.node,
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }


def run_encoder_benchmark(
    case: BenchmarkCase,
    settings: BenchmarkSettings | None = None,
    *,
    torch_module: Any | None | object = _AUTO_TORCH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one injected adapter case and return a schema-ready case record."""

    settings = settings or BenchmarkSettings()
    capabilities = validate_encoder_adapter(case.adapter)
    if case.workload.mode == "fixed":
        capabilities.require("supports_fixed_clip")
    else:
        capabilities.require("supports_streaming")
        if case.compression is not None:
            capabilities.require("supports_external_cache_policy")
    runtime = _TorchRuntime(
        torch_module,
        device=settings.device,
        synchronize_cuda=settings.synchronize_cuda,
    )

    # Warmups execute the full decode/preprocess/encode path and reset streaming
    # state, but do not contaminate repeat statistics. PyTorch modules remain in
    # eval + inference_mode for the complete case and are restored afterwards.
    repeats: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    with _adapter_inference(case.adapter, runtime):
        for warmup_index in range(settings.warmup):
            _one_repeat(
                case,
                runtime,
                clock=clock,
                repeat_index=-(settings.warmup - warmup_index),
            )

        for repeat_index in range(settings.repeat):
            repeat, observation = _one_repeat(case, runtime, clock=clock, repeat_index=repeat_index)
            repeats.append(repeat)
            observations.append(observation)

    reference = observations[0]
    consistency_reasons: list[str] = []
    for repeat_index, observation in enumerate(observations[1:], start=1):
        for key in ("coordinate_sha256", "sampled_frames", "spatial_sizes", "video_seconds"):
            if observation[key] != reference[key]:
                consistency_reasons.append(f"repeat {repeat_index} 的 {key} 与 repeat 0 不一致")

    config_snapshot = _json_safe(case.config)
    feature_stages = list(
        dict.fromkeys(stage for repeat in repeats for stage in repeat["feature_stages"])
    )
    if "projected_visual" in feature_stages:
        # A projected visual token is emitted before decoder context is applied.
        # Its cache policy may affect runtime, but not this feature value, so it
        # cannot support a cache-conditioned accuracy claim in either mode.
        cache_conditioned = False
        accuracy_eligible = False
        accuracy_reasons = [
            "feature_stage=projected_visual is performance-only and cannot support "
            "a cache-conditioned accuracy comparison"
        ]
    elif case.workload.mode == "streaming":
        cache_conditioned = feature_stages == ["decoder_contextual"]
        accuracy_eligible = cache_conditioned
        accuracy_reasons = (
            []
            if accuracy_eligible
            else [
                "streaming accuracy comparison requires feature_stage=decoder_contextual; "
                f"observed {feature_stages!r}"
            ]
        )
    else:
        cache_conditioned = False
        accuracy_eligible = True
        accuracy_reasons = []
    return {
        "name": case.name,
        "mode": case.workload.mode,
        "workload": {
            "name": case.workload.name,
            "task": case.workload.task,
            "declared_sampling": _json_safe(case.workload.sampling),
            "observed_sampling": reference,
            "sampling_consistent_across_repeats": not consistency_reasons,
            "sampling_consistency_reasons": consistency_reasons,
        },
        "repeats": repeats,
        "aggregate": _aggregate_repeats(repeats),
        "accuracy_eligibility": {
            "eligible": accuracy_eligible,
            "reasons": accuracy_reasons,
            "observed_feature_stages": feature_stages,
            # projected_visual is intentionally a performance-only diagnostic:
            # it does not expose the decoder-cache-conditioned representation.
            "performance_only": not accuracy_eligible,
            "cache_conditioned": cache_conditioned,
        },
        "timing_semantics": {
            "encoder_includes_native_compression": True,
            "native_compression_source": "adapter_telemetry_host_wall",
            "encoder_excluding_native_compression_is_approximate": True,
            "throughput_wall_time_source": "cuda_synchronized_perf_counter",
        },
        "provenance": {
            "adapter": _adapter_provenance(case.adapter),
            "torch_device": runtime.provenance(),
            "config": config_snapshot,
            "config_sha256": _sha256_json(config_snapshot),
        },
    }


def assess_sampling_comparability(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that compared cases consumed the same ordered sampled frames.

    Fixed clips and streaming chunks may group frames differently. They are only
    marked comparable when their flattened video/frame/timestamp coordinates,
    spatial sizes, frame count, and processed video duration match. Missing or
    nondeterministic evidence fails closed with explicit reasons.
    """

    if not case_results:
        raise ValueError("至少需要一个 case result")
    reasons: list[str] = []
    pairwise: list[dict[str, Any]] = []
    reference = case_results[0]
    reference_name = str(reference["name"])
    reference_workload = reference["workload"]
    if not reference_workload["sampling_consistent_across_repeats"]:
        reasons.extend(
            f"{reference_name}: {reason}"
            for reason in reference_workload["sampling_consistency_reasons"]
        )

    comparison_keys = (
        "coordinate_sha256",
        "sampled_frames",
        "spatial_sizes",
        "video_seconds",
        "layout",
        "dtype",
    )
    reference_observation = reference_workload["observed_sampling"]
    if len(case_results) > 1 and not reference_observation.get("all_frame_indices_present"):
        reasons.append(
            f"{reference_name} 缺少 frame_indices，不能仅凭 timestamp 证明跨编码器采样相同"
        )
    for candidate in case_results[1:]:
        candidate_name = str(candidate["name"])
        candidate_workload = candidate["workload"]
        pair_reasons: list[str] = []
        if not reference_observation.get("all_frame_indices_present"):
            pair_reasons.append(
                f"{reference_name} 缺少 frame_indices，不能仅凭 timestamp 证明跨编码器采样相同"
            )
        if candidate_workload.get("task") != reference_workload.get("task"):
            pair_reasons.append(f"{reference_name} 与 {candidate_name} 的 benchmark task 不一致")
        if _canonical_json(candidate_workload.get("declared_sampling")) != _canonical_json(
            reference_workload.get("declared_sampling")
        ):
            pair_reasons.append(f"{reference_name} 与 {candidate_name} 的声明采样协议不一致")
        if not candidate_workload["sampling_consistent_across_repeats"]:
            pair_reasons.extend(
                f"{candidate_name}: {reason}"
                for reason in candidate_workload["sampling_consistency_reasons"]
            )
        candidate_observation = candidate_workload["observed_sampling"]
        reference_stages = reference.get("accuracy_eligibility", {}).get(
            "observed_feature_stages", []
        )
        candidate_stages = candidate.get("accuracy_eligibility", {}).get(
            "observed_feature_stages", []
        )
        if candidate_stages != reference_stages:
            pair_reasons.append(
                f"{reference_name} 与 {candidate_name} 的 feature_stage 不一致："
                f"{reference_stages!r} != {candidate_stages!r}"
            )
        if not candidate_observation.get("all_frame_indices_present"):
            pair_reasons.append(
                f"{candidate_name} 缺少 frame_indices，不能仅凭 timestamp 证明跨编码器采样相同"
            )
        for key in comparison_keys:
            if candidate_observation.get(key) != reference_observation.get(key):
                pair_reasons.append(f"{reference_name} 与 {candidate_name} 的实际采样 {key} 不一致")
        pairwise.append(
            {
                "left": reference_name,
                "right": candidate_name,
                "comparable": not pair_reasons,
                "reasons": pair_reasons,
            }
        )
        reasons.extend(pair_reasons)

    accuracy_reasons: list[str] = []
    for case_result in case_results:
        eligibility = case_result.get("accuracy_eligibility", {})
        if not eligibility.get("eligible", False):
            accuracy_reasons.extend(
                f"{case_result['name']}: {reason}" for reason in eligibility.get("reasons", [])
            )
    if reasons:
        accuracy_reasons.append("performance sampling/task comparison is not comparable")

    return {
        "comparable": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "pairwise": pairwise,
        "criterion": (
            "same ordered video/frame/timestamp coordinates with non-null frame indices, "
            "spatial size, frame count, "
            "video duration, BTHWC layout and uint8 dtype"
        ),
        "token_rate_scope": (
            "per-case diagnostic only; token semantics are not asserted comparable across encoders"
        ),
        "accuracy_comparable": not accuracy_reasons,
        "accuracy_reasons": list(dict.fromkeys(accuracy_reasons)),
    }


def run_benchmark_suite(
    cases: Sequence[BenchmarkCase],
    settings: BenchmarkSettings | None = None,
    *,
    torch_module: Any | None | object = _AUTO_TORCH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one or more cases and emit a versioned performance result."""

    if not cases:
        raise ValueError("cases 不能为空")
    settings = settings or BenchmarkSettings()
    results = [
        run_encoder_benchmark(
            case,
            settings,
            torch_module=torch_module,
            clock=clock,
        )
        for case in cases
    ]
    return {
        "schema_version": PERFORMANCE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": _json_safe(settings),
        "comparison": assess_sampling_comparability(results),
        "provenance": {"machine": _machine_provenance()},
        "cases": results,
    }


def write_performance_result(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write UTF-8 JSON atomically enough for a single benchmark process."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


__all__ = [
    "PERFORMANCE_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkSettings",
    "BenchmarkWorkload",
    "assess_sampling_comparability",
    "run_benchmark_suite",
    "run_encoder_benchmark",
    "write_performance_result",
]
