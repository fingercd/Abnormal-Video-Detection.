"""Run one benchmark case, locally or inside its declared encoder Python.

The controller sends only JSON configuration to an isolated interpreter. Video
decode, adapter construction, warmups, repeats, CUDA synchronization, and
memory accounting all remain in that interpreter so a case never reports the
parent process's CUDA allocator state.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.benchmark import (
    BenchmarkCase,
    BenchmarkSettings,
    BenchmarkWorkload,
    run_encoder_benchmark,
)
from vadbench.config import load_experiment
from vadbench.contracts import ClipBatch
from vadbench.data.video import VideoInfo, decode_rgb_frames, probe_video
from vadbench.environment_registry import (
    EncoderRuntime,
    build_encoder_runtime_environment,
    resolve_encoder_runtime,
)
from vadbench.integrations.worker_protocol import (
    SidecarStore,
    WorkerProtocolError,
    ensure_json_value,
)
from vadbench.orchestration import compression_from_experiment, create_encoder_from_experiment

_RUNTIME_SCHEMA_VERSION = 1


class BenchmarkRuntimeError(RuntimeError):
    """An isolated benchmark case could not produce a trustworthy result."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sample_indices(info: VideoInfo, sample_fps: float, sampled_frames: int) -> np.ndarray:
    if sample_fps <= 0 or sampled_frames <= 0:
        raise ValueError("sample_fps 与 sampled_frames 必须大于 0")
    stride = max(1, int(round(info.fps / sample_fps)))
    indices = np.arange(sampled_frames, dtype=np.int64) * stride
    if int(indices[-1]) >= info.num_frames:
        raise ValueError(
            f"视频只有 {info.num_frames} 帧，无法以 stride={stride} 采 {sampled_frames} 帧"
        )
    return indices


def batch_groups(
    decoded: np.ndarray,
    *,
    indices: np.ndarray,
    fps: float,
    video_id: str,
    mode: str,
    units: int,
    frames_per_unit: int,
) -> ClipBatch | tuple[ClipBatch, ...]:
    expected = units * frames_per_unit
    if decoded.shape[0] != expected or indices.shape[0] != expected:
        raise ValueError(f"预处理期望 {expected} 帧，实际 {decoded.shape[0]}/{indices.shape[0]}")
    frames = decoded.reshape(units, frames_per_unit, *decoded.shape[1:])
    frame_indices = indices.reshape(units, frames_per_unit)
    timestamps = frame_indices.astype(np.float64) / float(fps)

    def one(start: int, stop: int) -> ClipBatch:
        clip_indices = list(range(start, stop))
        return ClipBatch(
            frames=frames[start:stop],
            timestamps_s=timestamps[start:stop],
            video_ids=(video_id,) * (stop - start),
            frame_indices=frame_indices[start:stop],
            metadata={
                "clip_ids": [f"{video_id}:benchmark-{index:04d}" for index in clip_indices],
                "clip_indices": clip_indices,
                "sampling_kind": "shared_ordered_frames",
            },
        )

    if mode == "fixed":
        return one(0, units)
    if mode == "streaming":
        return tuple(one(index, index + 1) for index in range(units))
    raise ValueError(f"未知 benchmark mode：{mode}")


def case_experiment(
    case_spec: Mapping[str, Any], root: Path, *, device: str | None
) -> dict[str, Any]:
    experiment = load_experiment(_resolve(root, str(case_spec["experiment"])))
    encoder = dict(case_spec.get("encoder", {}))
    if device is not None:
        encoder["device"] = device
    streaming = {
        **dict(experiment.get("streaming", {})),
        "enabled": str(case_spec["mode"]) == "streaming",
    }
    if case_spec.get("compression") is not None:
        streaming["compression"] = dict(case_spec["compression"])
    return _deep_merge(experiment, {"encoder": encoder, "streaming": streaming})


def release_runtime() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover - optional dependency
        pass


def run_benchmark_case(
    case_spec: Mapping[str, Any],
    *,
    root: Path,
    selected_video: Path,
    input_spec: Mapping[str, Any],
    settings: BenchmarkSettings,
    encoder_factory: Callable[..., tuple[Any, Mapping[str, Any]]] = create_encoder_from_experiment,
    probe_fn: Callable[..., VideoInfo] = probe_video,
    decode_fn: Callable[..., np.ndarray] = decode_rgb_frames,
) -> dict[str, Any]:
    """Construct and measure one case; all repeat work stays in this process."""

    experiment = case_experiment(case_spec, root, device=settings.device)
    adapter, definition = encoder_factory(experiment, project_root=root)
    try:
        mode = str(case_spec["mode"])
        grouping = case_spec["grouping"]
        units = int(grouping["units"])
        frames_per_unit = int(grouping["frames_per_unit"])
        sampled_frames = int(input_spec["sampled_frames"])
        if units * frames_per_unit != sampled_frames:
            raise ValueError(f"{case_spec['name']}: grouping 与 sampled_frames 不一致")

        info = probe_fn(selected_video)
        indices = sample_indices(info, float(input_spec["sample_fps"]), sampled_frames)
        actual_stride = int(indices[1] - indices[0]) if len(indices) > 1 else 1
        video_seconds = float((int(indices[-1]) - int(indices[0]) + actual_stride) / info.fps)
        preprocess = partial(
            batch_groups,
            indices=indices,
            fps=info.fps,
            video_id=selected_video.stem,
            mode=mode,
            units=units,
            frames_per_unit=frames_per_unit,
        )
        workload = BenchmarkWorkload(
            name=str(input_spec.get("sampling_protocol", "benchmark")),
            mode=mode,
            decode=partial(decode_fn, selected_video, indices),
            preprocess=preprocess,
            sampling={
                **input_spec,
                "source_fps": info.fps,
                "actual_frame_stride": actual_stride,
                "source_video_path": str(selected_video),
            },
            video_seconds=video_seconds,
            task=str(input_spec.get("task", "encoder_performance_only")),
        )
        compression = compression_from_experiment(experiment) if mode == "streaming" else None
        case = BenchmarkCase(
            name=str(case_spec["name"]),
            adapter=adapter,
            workload=workload,
            compression=compression,
            config={
                "case": case_spec,
                "experiment": experiment,
                "encoder_definition": definition,
            },
        )
        return run_encoder_benchmark(case, settings)
    finally:
        del adapter
        release_runtime()


def record_runtime_provenance(
    case_result: Mapping[str, Any], runtime: EncoderRuntime
) -> dict[str, Any]:
    """Attach the interpreter selection to a schema-versioned case record."""

    result = dict(case_result)
    provenance = dict(result["provenance"])
    provenance["runtime"] = {
        "group": runtime.group.id,
        "python_executable": str(runtime.python),
        "overlay": None if runtime.overlay is None else str(runtime.overlay),
    }
    result["provenance"] = provenance
    return result


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkRuntimeError(f"{label} 必须是对象")
    actual = set(value)
    if actual != expected:
        raise BenchmarkRuntimeError(
            f"{label} 字段非法：missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkRuntimeError(f"{label} 必须是对象")
    return value


def _runtime_request(
    value: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any], BenchmarkSettings]:
    data = _exact_mapping(
        value,
        {"schema_version", "project_root", "video", "input", "case", "settings"},
        "benchmark request",
    )
    if data["schema_version"] != _RUNTIME_SCHEMA_VERSION:
        raise BenchmarkRuntimeError("unsupported benchmark runtime request schema_version")
    if not isinstance(data["project_root"], str) or not isinstance(data["video"], str):
        raise BenchmarkRuntimeError("project_root 与 video 必须是字符串")
    root = Path(data["project_root"]).resolve()
    video = Path(data["video"]).resolve()
    if root not in video.parents:
        raise BenchmarkRuntimeError("video 必须位于 project_root 内")
    input_spec = _mapping(data["input"], "benchmark input")
    case_spec = _mapping(data["case"], "benchmark case")
    settings_data = _exact_mapping(
        data["settings"], {"warmup", "repeat", "synchronize_cuda", "device"}, "benchmark settings"
    )
    return root, video, input_spec, case_spec, BenchmarkSettings(**settings_data)


def run_runtime_once(exchange_root: str | Path, request_path: str, response_path: str) -> int:
    """Execute one JSON-sidecar request in the selected model interpreter."""

    store = SidecarStore(exchange_root, create=True)
    try:
        root, video, input_spec, case_spec, settings = _runtime_request(
            store.read_json(request_path)
        )
        result = run_benchmark_case(
            case_spec,
            root=root,
            selected_video=video,
            input_spec=input_spec,
            settings=settings,
        )
        store.write_json(
            response_path,
            {"schema_version": _RUNTIME_SCHEMA_VERSION, "status": "ok", "result": result},
        )
        return 0
    except Exception as exc:
        response = {
            "schema_version": _RUNTIME_SCHEMA_VERSION,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": (str(exc) or type(exc).__name__)[:2048],
            },
        }
        store.write_json(response_path, response)
        return 1


def _read_runtime_response(store: SidecarStore, response_path: str) -> Mapping[str, Any]:
    response = store.read_json(response_path)
    if (
        not isinstance(response, Mapping)
        or response.get("schema_version") != _RUNTIME_SCHEMA_VERSION
    ):
        raise BenchmarkRuntimeError("isolated benchmark response is invalid")
    status = response.get("status")
    if status == "ok":
        _exact_mapping(response, {"schema_version", "status", "result"}, "benchmark response")
        if not isinstance(response.get("result"), Mapping):
            raise BenchmarkRuntimeError("isolated benchmark response result is invalid")
        return response["result"]
    if status == "error":
        _exact_mapping(response, {"schema_version", "status", "error"}, "benchmark response")
        if not isinstance(response.get("error"), Mapping):
            raise BenchmarkRuntimeError("isolated benchmark response error is invalid")
        message = response["error"].get("message", "isolated benchmark failed")
        raise BenchmarkRuntimeError(f"isolated benchmark failed: {message}")
    raise BenchmarkRuntimeError("isolated benchmark response has invalid status")


def _diagnostic_bundle(
    diagnostics_dir: Path | None, case_name: str
) -> tuple[Path, Path | None, bool]:
    if diagnostics_dir is None:
        return Path(tempfile.mkdtemp(prefix="vadbench-benchmark-")), None, False
    base = diagnostics_dir.resolve() / "benchmark-runtime"
    created_base = not base.exists()
    base.mkdir(parents=True, exist_ok=True)
    safe_name = (
        "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in case_name
        ).strip("-")
        or "case"
    )
    return Path(tempfile.mkdtemp(prefix=f"{safe_name}-", dir=base)), base, created_base


def _write_process_logs(bundle: Path, *, stdout: str, stderr: str) -> None:
    (bundle / "stdout.log").write_text(stdout, encoding="utf-8")
    (bundle / "stderr.log").write_text(stderr, encoding="utf-8")


def _diagnostic_error(
    message: str,
    *,
    case_name: str,
    runtime: EncoderRuntime,
    bundle: Path,
    root: Path,
    stderr: str,
) -> BenchmarkRuntimeError:
    try:
        artifact_path = bundle.relative_to(root).as_posix()
    except ValueError:
        artifact_path = str(bundle)
    detail = (
        f"case={case_name!r}; group={runtime.group.id!r}; "
        f"python={runtime.python}; diagnostics={artifact_path}"
    )
    stderr = stderr.strip()
    if stderr:
        detail += f"; stderr={stderr[-4096:]}"
    return BenchmarkRuntimeError(f"{message}; {detail}")


def run_isolated_benchmark_case(
    case_spec: Mapping[str, Any],
    *,
    root: Path,
    selected_video: Path,
    input_spec: Mapping[str, Any],
    settings: BenchmarkSettings,
    runtime: EncoderRuntime | None = None,
    executor: Callable[[EncoderRuntime, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    diagnostics_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure one real encoder in the Python selected by the v2 registry."""

    encoder = case_spec.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(encoder.get("adapter"), str):
        raise ValueError("benchmark case 缺少 encoder.adapter")
    selected_runtime = runtime or resolve_encoder_runtime(encoder["adapter"], project_root=root)
    request = {
        "schema_version": _RUNTIME_SCHEMA_VERSION,
        "project_root": str(root),
        "video": str(selected_video),
        "input": dict(input_spec),
        "case": dict(case_spec),
        "settings": asdict(settings),
    }
    try:
        request = ensure_json_value(request, name="benchmark request")
    except WorkerProtocolError as exc:
        raise BenchmarkRuntimeError(f"benchmark request is not portable JSON: {exc}") from exc
    if executor is not None:
        return record_runtime_provenance(executor(selected_runtime, request), selected_runtime)
    if not selected_runtime.python.is_file():
        raise BenchmarkRuntimeError(
            f"encoder environment Python is unavailable: {selected_runtime.python}"
        )

    case_name = str(case_spec.get("name", encoder["adapter"]))
    bundle, diagnostics_base, created_diagnostics_base = _diagnostic_bundle(
        None if diagnostics_dir is None else Path(diagnostics_dir), case_name
    )
    success = False
    completed = None
    expected_parent = bundle.parent.resolve()
    try:
        store = SidecarStore(bundle, create=True)
        request_id = uuid.uuid4().hex
        request_path = f"input/{request_id}/request.json"
        response_path = f"output/{request_id}/response.json"
        store.write_json(request_path, request)
        completed = subprocess.run(
            [
                str(selected_runtime.python),
                "-m",
                "vadbench.benchmark_runtime",
                "--bundle-root",
                str(bundle),
                "--request",
                request_path,
                "--response",
                response_path,
            ],
            cwd=root,
            env=build_encoder_runtime_environment(selected_runtime, project_root=root),
            text=True,
            capture_output=True,
            check=False,
        )
        result = _read_runtime_response(store, response_path)
        if completed.returncode != 0:
            raise BenchmarkRuntimeError(
                f"worker exited {completed.returncode} despite a success response"
            )
        result = record_runtime_provenance(result, selected_runtime)
        success = True
        return result
    except Exception as exc:
        stdout = "" if completed is None else completed.stdout
        stderr = str(exc) if completed is None else completed.stderr
        _write_process_logs(bundle, stdout=stdout, stderr=stderr)
        raise _diagnostic_error(
            str(exc),
            case_name=case_name,
            runtime=selected_runtime,
            bundle=bundle,
            root=root,
            stderr=stderr,
        ) from exc
    finally:
        if success:
            if bundle.is_symlink() or bundle.resolve().parent != expected_parent:
                raise RuntimeError(f"refusing to remove unexpected benchmark bundle: {bundle}")
            shutil.rmtree(bundle)
            if created_diagnostics_base and diagnostics_base is not None:
                with suppress(OSError):
                    diagnostics_base.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated VADBench benchmark case")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_runtime_once(args.bundle_root, args.request, args.response)


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())


__all__ = [
    "BenchmarkRuntimeError",
    "batch_groups",
    "case_experiment",
    "main",
    "record_runtime_provenance",
    "release_runtime",
    "run_benchmark_case",
    "run_isolated_benchmark_case",
    "run_runtime_once",
    "sample_indices",
]
