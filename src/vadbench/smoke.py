"""真实权重 encoder 的单视频冒烟执行器。"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jsonschema import Draft202012Validator

from vadbench.config import load_yaml
from vadbench.contracts import (
    EncoderOutput,
    StreamState,
    StreamStep,
    validate_encoder_output,
    validate_stream_step,
)
from vadbench.data.manifest import VideoManifestRecord
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import (
    build_clip_batch,
    iter_streaming_chunk_batches,
    probe_video,
)
from vadbench.hashing import sha256_file as _sha256_file
from vadbench.integrations.catalog import (
    IntegrationCatalog,
    IntegrationRecord,
    load_default_integration_catalog,
)
from vadbench.integrations.common import (
    OutputHealthError,
    inspect_output_health,
)
from vadbench.orchestration import (
    compression_from_experiment,
    create_encoder_from_experiment,
    resolve_encoder_config,
)
from vadbench.resources import package_resource_path


def _gpu_peak_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except ImportError:  # pragma: no cover - optional dependency
        pass
    return None


SMOKE_V2_SCHEMA_VERSION = "vadbench.encoder-smoke.v2"
CATALOG_V1_VERSION = "vadbench.encoder-integrations.v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|cookie|credential|authorization)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _json_safe(value: Any, *, redact: bool = False) -> Any:
    """Make metadata strict-JSON without importing model packages."""

    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>"
            if redact and _SECRET_KEY.search(str(key))
            else _json_safe(item, redact=redact)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, redact=redact) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not __import__("math").isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item(), redact=redact)
        except Exception:
            pass
    return f"<{type(value).__module__}.{type(value).__name__}>"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value, redact=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_identity(root: Path) -> tuple[str, bool]:
    """Return schema-safe git identity; seven zeroes means git was unavailable."""

    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    if commit is None or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        commit = "0000000"
    return commit, dirty


def _small_values(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
            value = value.detach().cpu()
        import numpy as np

        array = np.asarray(value)
        return [_json_safe(item) for item in array.reshape(-1).tolist()]
    except Exception:
        return []


def _batch_identity(batch: Any, batch_index: int) -> dict[str, Any]:
    shape = [int(item) for item in getattr(batch.frames, "shape", ())]
    dtype = str(getattr(batch.frames, "dtype", "uint8"))
    frame_indices = batch.frame_indices
    if frame_indices is None:
        frame_indices = [list(range(int(shape[1]))) for _ in range(int(shape[0]))]
    return {
        "batch_index": int(batch_index),
        "shape": shape,
        "dtype": dtype,
        "layout": "BTHWC",
        "video_ids": [str(item) for item in batch.video_ids],
        "frame_indices": [int(item) for item in _small_values(frame_indices)],
        "timestamps_seconds": [float(item) for item in _small_values(batch.timestamps_s)],
    }


def _video_identity(path: Path, root: Path, info: Any | None) -> dict[str, Any]:
    if info is None:
        return {
            "path": _relative_path(path, root),
            "sha256": "0" * 64,
            "num_frames": 1,
            "fps": 1.0,
            "duration_seconds": 1.0,
            "width": 1,
            "height": 1,
        }
    return {
        "path": _relative_path(path, root),
        "sha256": _sha256_file(path),
        "num_frames": int(info.num_frames),
        "fps": float(info.fps),
        "duration_seconds": float(info.duration_seconds),
        "width": int(info.width),
        "height": int(info.height),
    }


def _command_list(command: Sequence[str] | str | None) -> list[str]:
    if command is None:
        return [str(item) for item in (sys.argv or ["vadbench"])]
    if isinstance(command, str):
        return [command]
    return [str(item) for item in command] or ["vadbench"]


def _environment_identity(profile: str, device: str) -> dict[str, Any]:
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu: dict[str, Any] | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = (
            None if getattr(torch.version, "cuda", None) is None else str(torch.version.cuda)
        )
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            gpu = {
                "name": str(torch.cuda.get_device_name(index)),
                "index": index,
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
    except Exception:
        pass
    packages: dict[str, str | None] = {}
    for name in ("numpy", "torch", "transformers", "torchvision", "pytorchvideo", "opencv-python"):
        try:
            packages[name] = str(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "profile": profile or "unknown",
        "hostname": platform.node() or "unknown-host",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "device": device or "cpu",
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu": gpu,
        "packages": packages,
    }


def _load_record(
    config: Mapping[str, Any],
    project_root: Path,
    catalog: IntegrationCatalog | None,
    integration_id: str | None,
) -> IntegrationRecord | None:
    encoder = config.get("encoder", {})
    encoder = encoder if isinstance(encoder, Mapping) else {}
    adapter_id = integration_id or str(encoder.get("adapter", ""))
    if not adapter_id:
        return None
    if catalog is None:
        try:
            catalog = load_default_integration_catalog(project_root)
        except Exception:
            return None
    try:
        return catalog.get(adapter_id)
    except Exception:
        return None


def _load_definition(
    config: Mapping[str, Any],
    record: IntegrationRecord | None,
    project_root: Path,
) -> dict[str, Any]:
    if record is None:
        return {}
    return resolve_encoder_config(config, project_root=project_root)


def _asset_identity(
    definition: Mapping[str, Any],
    record: IntegrationRecord | None,
    project_root: Path,
) -> dict[str, Any]:
    constructor = definition.get("constructor", {})
    constructor = constructor if isinstance(constructor, Mapping) else {}
    checkpoint = definition.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    lock_data: Mapping[str, Any] = {}
    if record is not None:
        lock_path = project_root / record.upstream_lock
        if not lock_path.is_file():
            with suppress(FileNotFoundError):
                lock_path = package_resource_path(record.upstream_lock)
        if lock_path.is_file():
            try:
                loaded = load_yaml(lock_path)
                if isinstance(loaded, Mapping):
                    lock_data = loaded
            except Exception:
                pass
    source = lock_data.get("source", {})
    source = source if isinstance(source, Mapping) else {}
    weights = lock_data.get("weights", {})
    weights = weights if isinstance(weights, Mapping) else {}
    source_license = lock_data.get("license", {})
    source_license = source_license if isinstance(source_license, Mapping) else {}
    upstream_repo = source.get("repository") or source.get("repo")
    upstream_revision = source.get("commit") or source.get("revision")
    upstream_license = source_license.get("spdx") or source_license.get("license")
    checkout = constructor.get("checkout_path")
    checkout_path = None
    if checkout is not None:
        checkout_value = Path(str(checkout))
        if not checkout_value.is_absolute():
            checkout_value = project_root / checkout_value
        checkout_path = _relative_path(checkout_value, project_root)
    upstream = {
        "repo": "unknown" if upstream_repo is None else str(upstream_repo),
        "revision": "unknown" if upstream_revision is None else str(upstream_revision),
        "license": "unknown" if upstream_license is None else str(upstream_license),
        "checkout_path": checkout_path,
    }
    cp_repo = checkpoint.get("model_id") or checkpoint.get("repo_id") or weights.get("model_id")
    cp_revision = checkpoint.get("revision") or weights.get("revision")
    cp_license = checkpoint.get("license") or weights.get("license")
    cp_path = (
        checkpoint.get("local_path") or weights.get("local_dir") or constructor.get("model_path")
    )
    cp_path_text = None
    if cp_path is not None:
        cp_path_value = Path(str(cp_path))
        if not cp_path_value.is_absolute():
            cp_path_value = project_root / cp_path_value
        cp_path_text = _relative_path(cp_path_value, project_root)
    cp_hash: str | None = None
    locked_hashes = weights.get("sha256")
    if isinstance(locked_hashes, Mapping):
        for value in locked_hashes.values():
            if isinstance(value, str) and _HEX64.fullmatch(value):
                cp_hash = value
                break
    if cp_hash is None and cp_path is not None:
        resolved = Path(str(cp_path))
        if not resolved.is_absolute():
            resolved = project_root / resolved
        if resolved.is_file():
            with suppress(OSError):
                cp_hash = _sha256_file(resolved)
    size_bytes = None
    if cp_path is not None:
        resolved = Path(str(cp_path))
        if not resolved.is_absolute():
            resolved = project_root / resolved
        try:
            if resolved.is_file():
                size_bytes = int(resolved.stat().st_size)
        except OSError:
            pass
    return {
        "upstream": upstream,
        "checkpoint": {
            "repo": None if cp_repo is None else str(cp_repo),
            "revision": None if cp_revision is None else str(cp_revision),
            "license": None if cp_license is None else str(cp_license),
            "path": cp_path_text,
            "sha256": cp_hash,
            "size_bytes": size_bytes,
        },
    }


def _error_payload(
    exc: BaseException,
    *,
    stage: str,
    evidence: Mapping[str, Any] | None = None,
    blocked: bool = False,
) -> dict[str, Any]:
    if blocked:
        category = "missing_asset" if isinstance(exc, FileNotFoundError) else "blocked"
    elif isinstance(exc, OutputHealthError):
        category = "output_health"
    else:
        category = "runtime_error"
    text = str(exc).strip() or type(exc).__name__
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "category": category,
        "stage": stage,
        "message": text[:4096],
        "recoverable": not blocked,
        "traceback": stack[-16000:] or None,
        "evidence": _json_safe(dict(evidence or {})),
    }


def _is_blocking_exception(exc: BaseException) -> bool:
    return isinstance(exc, (FileNotFoundError, ModuleNotFoundError, ImportError, PermissionError))


def _output_record(
    output: EncoderOutput,
    *,
    step_index: int,
    info: Any,
    feature_stage: str | None,
) -> tuple[dict[str, Any], Any]:
    health = inspect_output_health(
        output,
        video_duration_seconds=float(info.duration_seconds),
        video_num_frames=int(info.num_frames),
        feature_stage=feature_stage,
        require_pooled=True,
        require_video_bounds=True,
    )
    payload = health.to_dict()
    payload.update({"step_index": int(step_index), "aux": _json_safe(dict(output.aux))})
    return payload, health


def run_encoder_smoke_v2(
    config: Mapping[str, Any],
    video_path: str | Path,
    *,
    project_root: str | Path = ".",
    max_chunks: int | None = None,
    run_id: str | None = None,
    command: Sequence[str] | str | None = None,
    log_path: str | Path | None = None,
    integration_id: str | None = None,
    catalog: IntegrationCatalog | None = None,
    adapter_instance: Any | None = None,
) -> dict[str, Any]:
    """Run one encoder and return a schema-v2, failure-preserving document."""

    root = Path(project_root).expanduser().resolve()
    raw_path = Path(video_path).expanduser()
    path = (root / raw_path if not raw_path.is_absolute() else raw_path).resolve()
    if max_chunks is not None and (
        isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0
    ):
        raise ValueError("max_chunks 必须是正整数或 None")
    started_at = _utc_now()
    started = time.perf_counter()
    run_id = run_id or (
        f"smoke-{_datetime.datetime.now(_datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{uuid.uuid4().hex[:8]}"
    )
    runtime_config = config
    record = _load_record(runtime_config, root, catalog, integration_id)
    encoder_cfg = runtime_config.get("encoder", {})
    encoder_cfg = encoder_cfg if isinstance(encoder_cfg, Mapping) else {}
    adapter_id = integration_id or str(encoder_cfg.get("adapter", "unknown"))
    definition: dict[str, Any] = {}
    assets = _asset_identity(definition, record, root)
    commit, dirty = _git_identity(root)
    env_cfg = runtime_config.get("environment", {})
    env_cfg = env_cfg if isinstance(env_cfg, Mapping) else {}
    profile = (
        str(record.environment.profile)
        if record is not None
        else str(env_cfg.get("profile", "unknown"))
    )
    params_cfg = encoder_cfg.get("params", {})
    params_cfg = params_cfg if isinstance(params_cfg, Mapping) else {}
    device = str(encoder_cfg.get("device") or params_cfg.get("device") or "cpu")
    info = None
    batches: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    stream_summary: dict[str, Any] | None = None
    streaming_hint = bool(
        (
            runtime_config.get("streaming", {}).get("enabled", False)
            if isinstance(runtime_config.get("streaming"), Mapping)
            else False
        )
        or (record is not None and record.run_mode == "streaming")
    )
    partial_state_steps: list[int] = []
    partial_cache_kinds: set[str] = set()
    requested_chunks = max(2, int(max_chunks or 2))
    error: dict[str, Any] | None = None
    status = "smoke_pass"
    exit_code: int | None = 0
    stage = "probe"
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        info = probe_video(path)
        stage = "adapter_load"
        if adapter_instance is None:
            adapter, loaded_definition = create_encoder_from_experiment(
                runtime_config, project_root=root
            )
            if loaded_definition:
                definition = loaded_definition
                assets = _asset_identity(definition, record, root)
        else:
            adapter = adapter_instance
            definition = _load_definition(runtime_config, record, root)
            assets = _asset_identity(definition, record, root)
        capabilities = getattr(adapter, "capabilities", None)
        if capabilities is None:
            raise TypeError("adapter.capabilities 缺失")
        feature_stage = str(
            encoder_cfg.get("feature_stage")
            or params_cfg.get("feature_stage")
            or (record.feature_stage if record is not None else "pooled")
        )
        streaming_cfg = runtime_config.get("streaming", {})
        streaming_cfg = streaming_cfg if isinstance(streaming_cfg, Mapping) else {}
        is_streaming = bool(streaming_cfg.get("enabled", False)) or (
            record is not None and record.run_mode == "streaming"
        )
        if is_streaming:
            requested = int(
                max_chunks
                or streaming_cfg.get("chunks")
                or (record.smoke_profile.chunks if record is not None else 2)
                or 2
            )
            requested = max(2, requested)
            requested_chunks = requested
            chunk_frames = int(
                streaming_cfg.get("chunk_frames")
                or (record.smoke_profile.clip_frames if record is not None else 16)
            )
            sample_fps = streaming_cfg.get("sample_fps")
            frame_stride = streaming_cfg.get("frame_stride")
            if sample_fps is None and frame_stride is None and record is not None:
                frame_stride = record.smoke_profile.frame_stride
            manifest_record = VideoManifestRecord(
                video_id=path.stem,
                path=path.name,
                split="test",
                category="Smoke",
                is_anomaly=False,
                num_frames=info.num_frames,
                fps=info.fps,
                duration_seconds=info.duration_seconds,
                metadata={"smoke_input": True},
            )
            chunks = iter_streaming_chunk_batches(
                (manifest_record,),
                path.parent,
                chunk_frames=chunk_frames,
                frame_stride=None if frame_stride is None else int(frame_stride),
                sample_fps=None if sample_fps is None else float(sample_fps),
            )
            state = adapter.init_state(manifest_record.video_id)
            if not isinstance(state, StreamState):
                raise TypeError("init_state 必须返回 StreamState")
            policy = compression_from_experiment(runtime_config)
            state_steps: list[int] = []
            cache_kinds: set[str] = set()
            stage = "forward"
            for index, chunk in enumerate(islice(chunks, requested)):
                batches.append(_batch_identity(chunk, index))
                step = adapter.encode_step(chunk, state, train=False, compression=policy)
                if not isinstance(step, StreamStep):
                    raise TypeError("encode_step 必须返回 StreamStep")
                validate_stream_step(
                    step,
                    previous_state=state,
                    chunk=chunk,
                    capabilities=capabilities,
                )
                state = step.state
                state_steps.append(int(state.step_index))
                partial_state_steps[:] = state_steps
                for view in state.caches.values():
                    value = getattr(view.kind, "value", str(view.kind))
                    if value in {"vision_tokens", "visual_memory", "decoder_kv"}:
                        cache_kinds.add(value)
                partial_cache_kinds.update(cache_kinds)
                for update in step.cache_updates.values():
                    value = getattr(update.view.kind, "value", str(update.view.kind))
                    if value in {"vision_tokens", "visual_memory", "decoder_kv"}:
                        cache_kinds.add(value)
                if step.output is not None:
                    record_output, health = _output_record(
                        step.output,
                        step_index=state.step_index,
                        info=info,
                        feature_stage=feature_stage,
                    )
                    record_output["aux"]["stream_telemetry"] = _json_safe(dict(step.telemetry))
                    outputs.append(record_output)
                    if not health.passed:
                        raise OutputHealthError("stream 输出健康检查失败", health=health.to_dict())
            final_output = adapter.finalize(state)
            if final_output is not None:
                record_output, health = _output_record(
                    final_output,
                    step_index=state.step_index,
                    info=info,
                    feature_stage=feature_stage,
                )
                outputs.append(record_output)
                if not health.passed:
                    raise OutputHealthError(
                        "final stream 输出健康检查失败", health=health.to_dict()
                    )
            completed = len(state_steps)
            stream_summary = {
                "chunks_requested": requested,
                "chunks_completed": completed,
                "state_steps": state_steps,
                "state_present": True,
                "cache_kinds": sorted(cache_kinds),
            }
            if completed < 2:
                raise ValueError(f"streaming smoke 至少需要 2 个 chunks，实际完成 {completed}")
        else:
            frames = getattr(capabilities, "fixed_num_frames", None)
            sampler = runtime_config.get("sampler", {})
            sampler = sampler if isinstance(sampler, Mapping) else {}
            frames = int(
                frames
                or sampler.get("clip_frames")
                or (record.smoke_profile.clip_frames if record is not None else 16)
            )
            stride = int(
                sampler.get("frame_stride")
                or (record.smoke_profile.frame_stride if record is not None else 1)
            )
            sample = sample_fixed_clip(
                info.num_frames,
                clip_frames=frames,
                frame_stride=stride,
                position="center",
            )
            batch = build_clip_batch(
                path,
                path.stem,
                (sample,),
                metadata={"clip_ids": [f"{path.stem}:smoke"], "clip_indices": [0]},
            )
            batches.append(_batch_identity(batch, 0))
            stage = "forward"
            output = adapter.encode(batch, train=False)
            validate_encoder_output(output, batch)
            record_output, health = _output_record(
                output,
                step_index=0,
                info=info,
                feature_stage=feature_stage,
            )
            outputs.append(record_output)
            if not health.passed:
                raise OutputHealthError("fixed 输出健康检查失败", health=health.to_dict())
    except Exception as exc:
        status = "blocked" if _is_blocking_exception(exc) else "failed"
        exit_code = None if status == "blocked" else 1
        error = _error_payload(
            exc,
            stage=stage,
            evidence={
                "output_health": getattr(exc, "health", None),
                "batches_completed": len(batches),
                "outputs_completed": len(outputs),
            },
            blocked=status == "blocked",
        )
        if streaming_hint and stream_summary is None:
            stream_summary = {
                "chunks_requested": requested_chunks,
                "chunks_completed": len(partial_state_steps),
                "state_steps": list(partial_state_steps),
                "state_present": bool(partial_state_steps),
                "cache_kinds": sorted(partial_cache_kinds),
            }
    finished_at = _utc_now()
    elapsed = max(0.0, time.perf_counter() - started)
    input_identity = {"video": _video_identity(path, root, info), "batches": batches}
    if record is not None:
        display_name = record.display_name
        backend = record.backend
        run_mode = "streaming" if record.run_mode == "streaming" else "fixed"
        output_stage = record.feature_stage
        adapter_target = record.adapter_target
    else:
        display_name = str(encoder_cfg.get("display_name") or adapter_id)
        backend = str(encoder_cfg.get("backend") or adapter_id)
        run_mode = "streaming" if stream_summary is not None else "fixed"
        output_stage = str(encoder_cfg.get("feature_stage") or "pooled")
        adapter_target = str(encoder_cfg.get("adapter", adapter_id))
    execution_log = (
        _relative_path(log_path, root)
        if log_path is not None
        else f"outputs/encoder-integration/{run_id}/run.log"
    )
    result: dict[str, Any] = {
        "schema_version": SMOKE_V2_SCHEMA_VERSION,
        "generated_at_utc": finished_at,
        "run_id": run_id,
        "status": status,
        "encoder": {
            "id": adapter_id,
            "display_name": display_name,
            "adapter": adapter_target,
            "backend": backend,
            "run_mode": run_mode,
            "feature_stage": output_stage,
        },
        "input": input_identity,
        "outputs": outputs,
        "streaming": stream_summary,
        "environment": _environment_identity(profile, device),
        "assets": assets,
        "execution": {
            "command": _command_list(command),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed,
            "exit_code": exit_code,
            "log_path": execution_log,
            "peak_gpu_memory_bytes": _gpu_peak_bytes(),
        },
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "config_sha256": _canonical_sha256(config),
            "catalog_version": CATALOG_V1_VERSION,
        },
        "error": error,
    }
    if log_path is not None or error is not None:
        actual_log = Path(log_path) if log_path is not None else root / execution_log
        if not actual_log.is_absolute():
            actual_log = root / actual_log
        try:
            actual_log = actual_log.resolve()
            if actual_log != root and root not in actual_log.parents:
                raise ValueError("log_path 必须位于 project_root 内")
            actual_log.parent.mkdir(parents=True, exist_ok=True)
            line = f"{finished_at} status={status} run_id={run_id}"
            if error is not None:
                line += f" error={error['category']}:{error['message']}"
            actual_log.write_text(line + "\n", encoding="utf-8")
        except (OSError, ValueError):
            pass
    return result


def _assert_contained(path: Path, root: Path) -> None:
    resolved_root = root.expanduser().resolve()
    resolved_path = path.expanduser().resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"输出路径越出 output_root：{resolved_path}")


_SMOKE_SCHEMA_RELATIVE = "schemas/encoder-smoke-v2.schema.json"


@lru_cache(maxsize=1)
def _smoke_result_validator() -> Draft202012Validator:
    # Load after model construction: eager jsonschema/rpds + OpenCV can crash
    # the pinned foundation runtime while importing the model stack.
    from jsonschema import Draft202012Validator, FormatChecker

    source_schema = Path(__file__).resolve().parents[2] / _SMOKE_SCHEMA_RELATIVE
    schema_path = (
        source_schema if source_schema.is_file() else package_resource_path(_SMOKE_SCHEMA_RELATIVE)
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_smoke_result_v2(
    result: Mapping[str, Any],
    *,
    expected_record: IntegrationRecord | None = None,
) -> dict[str, Any]:
    """Validate schema and, when supplied, the catalog identity for one result."""

    if not isinstance(result, Mapping):
        raise ValueError("smoke result 顶层必须是对象")
    payload = dict(result)
    errors = sorted(
        _smoke_result_validator().iter_errors(payload),
        key=lambda error: repr(list(error.absolute_path)),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ValueError(f"smoke result 不符合 v2 schema（{location}）：{error.message}")

    if expected_record is not None:
        encoder = payload["encoder"]
        expected = {
            "id": expected_record.id,
            "adapter": expected_record.adapter_target,
            "backend": expected_record.backend,
            "run_mode": "streaming" if expected_record.run_mode == "streaming" else "fixed",
            "feature_stage": expected_record.feature_stage,
        }
        mismatches = {
            field: {"expected": value, "actual": encoder[field]}
            for field, value in expected.items()
            if encoder[field] != value
        }
        if mismatches:
            raise ValueError(f"smoke result 与 catalog identity 不一致：{mismatches}")
        if payload["status"] == "smoke_pass":
            upstream = payload["assets"]["upstream"]
            checkpoint = payload["assets"]["checkpoint"]
            if any(
                upstream[field] in {None, "", "unknown"}
                for field in ("repo", "revision", "license")
            ):
                raise ValueError("登记路线 smoke_pass 缺少明确 upstream 身份")
            if any(
                checkpoint[field] in {None, "", "unknown"}
                for field in ("repo", "revision", "license", "path", "sha256")
            ):
                raise ValueError("登记路线 smoke_pass 缺少明确 checkpoint 身份")
    return payload


def read_smoke_result_v2(
    path: str | Path,
    *,
    expected_record: IntegrationRecord | None = None,
) -> dict[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"smoke result 不是合法 JSON：{selected}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"smoke result 顶层必须是对象：{selected}")
    return validate_smoke_result_v2(value, expected_record=expected_record)


def write_smoke_result_v2(
    result: Mapping[str, Any],
    path: str | Path,
    *,
    output_root: str | Path | None = None,
    overwrite_success: bool = False,
) -> Path:
    """Validate and atomically write one result; overwrite requires an explicit flag."""

    output = Path(path).expanduser()
    root = Path(output_root).expanduser() if output_root is not None else output.parent
    _assert_contained(output, root)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite_success:
        raise FileExistsError(f"smoke result 已存在；显式允许覆盖后才能写入：{output}")
    payload_value = validate_smoke_result_v2(result)
    payload = json.dumps(payload_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp_name: str | None = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    finally:
        if temp_name is not None:
            with suppress(FileNotFoundError):
                Path(temp_name).unlink()
    return output


__all__ = [
    "CATALOG_V1_VERSION",
    "SMOKE_V2_SCHEMA_VERSION",
    "read_smoke_result_v2",
    "run_encoder_smoke_v2",
    "validate_smoke_result_v2",
    "write_smoke_result_v2",
]
