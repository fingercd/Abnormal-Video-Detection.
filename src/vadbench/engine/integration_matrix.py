"""统一的 25 项 encoder integration 预检、调度和产物聚合。

矩阵层只负责编排，不在同一进程预加载 25 个模型。具体 forward 可以通过
run_one、in_process_runner 或 external_python_runner 注入；默认路径调用
vadbench.smoke.run_encoder_smoke_v2。
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from vadbench.config import load_yaml
from vadbench.environment_registry import resolve_encoder_runtime
from vadbench.integrations.catalog import (
    IntegrationCatalog,
    IntegrationRecord,
    load_default_integration_catalog,
    load_integration_catalog,
)
from vadbench.smoke import (
    CATALOG_V1_VERSION,
    SMOKE_V2_SCHEMA_VERSION,
    read_smoke_result_v2,
    run_encoder_smoke_v2,
    validate_smoke_result_v2,
    write_smoke_result_v2,
)

MATRIX_SCHEMA_VERSION = "vadbench.encoder-integration-matrix.v1"
_STATUSES = frozenset(
    {"planned", "preflight_pass", "acquiring", "integrated", "smoke_pass", "failed", "blocked"}
)
_RUN_MODES = frozenset({"fixed", "long", "streaming"})
_RUNTIMES = frozenset({"in_process", "external_python"})


def _utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _safe(value.item())
        except Exception:
            pass
    return f"<{type(value).__module__}.{type(value).__name__}>"


def _config_hash(config: Mapping[str, Any] | None) -> str:
    payload = json.dumps(
        _safe(config or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> tuple[str, bool]:
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
    if commit is None or re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
        commit = "0000000"
    return commit, dirty


def _coerce_catalog(
    catalog: IntegrationCatalog | str | Path | None,
    project_root: Path,
) -> IntegrationCatalog:
    if isinstance(catalog, IntegrationCatalog):
        return catalog
    if catalog is not None:
        return load_integration_catalog(catalog, project_root=project_root)
    return load_default_integration_catalog(project_root)


def filter_integrations(
    catalog: IntegrationCatalog | str | Path | None = None,
    *,
    project_root: str | Path = ".",
    integration_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    run_modes: Sequence[str] | None = None,
    runtimes: Sequence[str] | None = None,
    limit: int | None = None,
) -> tuple[IntegrationRecord, ...]:
    """按 ID、状态、模型族、运行模式和 runtime 过滤 catalog。"""

    root = Path(project_root).expanduser().resolve()
    selected_catalog = _coerce_catalog(catalog, root)
    selected_ids = set(integration_ids or ())
    selected_statuses = set(statuses or ())
    selected_families = set(families or ())
    selected_modes = set(run_modes or ())
    selected_runtimes = set(runtimes or ())
    if selected_statuses - _STATUSES:
        raise ValueError(f"未知 integration status：{sorted(selected_statuses - _STATUSES)}")
    if selected_modes - _RUN_MODES:
        raise ValueError(f"未知 integration run_mode：{sorted(selected_modes - _RUN_MODES)}")
    if selected_runtimes - _RUNTIMES:
        raise ValueError(f"未知 integration runtime：{sorted(selected_runtimes - _RUNTIMES)}")
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit 必须是正整数或 None")
    records = tuple(
        record
        for record in selected_catalog.integrations
        if (not selected_ids or record.id in selected_ids)
        and (not selected_statuses or record.status in selected_statuses)
        and (not selected_families or record.family in selected_families)
        and (not selected_modes or record.run_mode in selected_modes)
        and (not selected_runtimes or record.environment.runtime in selected_runtimes)
    )
    unknown = selected_ids - set(selected_catalog.ids)
    if unknown:
        raise ValueError(f"catalog 中不存在 integration：{sorted(unknown)}")
    return records if limit is None else records[:limit]


def _invoke_hook(
    hook: Callable[..., Any], record: IntegrationRecord, context: Mapping[str, Any]
) -> Any:
    """Invoke hooks with a deterministic compatibility surface."""

    candidates = (
        ((record,), {}),
        ((record, context), {}),
        ((), {"record": record, **dict(context)}),
        ((record,), dict(context)),
    )
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return hook(record, context)
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return hook(*args, **kwargs)
    raise TypeError(f"hook {hook!r} 不接受 record/context 参数")


def _default_preflight(record: IntegrationRecord, project_root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "definition": False,
        "upstream_lock": False,
        "checkpoint": record.checkpoint.status,
        "adapter_target": bool(record.adapter_target),
    }
    reasons: list[str] = []
    definition = (project_root / record.definition).resolve()
    lock = (project_root / record.upstream_lock).resolve()
    if definition.is_file():
        checks["definition"] = True
    else:
        reasons.append(f"definition 不存在：{record.definition}")
    if lock.is_file():
        checks["upstream_lock"] = True
    else:
        reasons.append(f"upstream_lock 不存在：{record.upstream_lock}")
    if record.checkpoint.status == "planned":
        reasons.append(f"checkpoint 尚未登记：{record.checkpoint.registry_id}")
    else:
        registry = project_root / "registry" / "checkpoints.yaml"
        try:
            data = load_yaml(registry)
            checkpoints = data.get("checkpoints", {}) if isinstance(data, Mapping) else {}
            if (
                not isinstance(checkpoints, Mapping)
                or record.checkpoint.registry_id not in checkpoints
            ):
                reasons.append(f"checkpoint registry 缺少：{record.checkpoint.registry_id}")
            else:
                entry = checkpoints[record.checkpoint.registry_id]
                checks["checkpoint"] = (
                    str(entry.get("status", record.checkpoint.status))
                    if isinstance(entry, Mapping)
                    else record.checkpoint.status
                )
                if isinstance(entry, Mapping):
                    local_path = entry.get("local_path") or entry.get("local_dir")
                    if local_path:
                        candidate = Path(str(local_path))
                        if not candidate.is_absolute():
                            candidate = project_root / candidate
                        if not candidate.exists():
                            reasons.append(f"checkpoint 文件不存在：{local_path}")
        except Exception as exc:
            reasons.append(f"checkpoint registry 无法读取：{exc}")
    return {
        "integration_id": record.id,
        "status": "preflight_pass" if not reasons else "blocked",
        "runtime": record.environment.runtime,
        "run_mode": record.run_mode,
        "checks": checks,
        "reasons": reasons,
    }


def preflight_integration(
    record: IntegrationRecord,
    *,
    project_root: str | Path = ".",
    hook: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """预检单项引用和资产；不导入模型、不加载权重。"""

    if not isinstance(record, IntegrationRecord):
        raise TypeError("record 必须是 IntegrationRecord")
    root = Path(project_root).expanduser().resolve()
    if hook is None:
        return _default_preflight(record, root)
    result = _invoke_hook(hook, record, {"project_root": root})
    if isinstance(result, Mapping):
        payload = dict(result)
        payload.setdefault("integration_id", record.id)
        payload.setdefault("runtime", record.environment.runtime)
        payload.setdefault("run_mode", record.run_mode)
        payload.setdefault("checks", {})
        payload.setdefault("reasons", [])
        payload.setdefault("status", "preflight_pass")
        return _safe(payload)
    passed = bool(result)
    return {
        "integration_id": record.id,
        "status": "preflight_pass" if passed else "blocked",
        "runtime": record.environment.runtime,
        "run_mode": record.run_mode,
        "checks": {"hook": passed},
        "reasons": [] if passed else ["preflight hook 返回 false"],
    }


def preflight_integrations(
    catalog: IntegrationCatalog | str | Path | None = None,
    *,
    project_root: str | Path = ".",
    integration_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    run_modes: Sequence[str] | None = None,
    runtimes: Sequence[str] | None = None,
    limit: int | None = None,
    hook: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    records = filter_integrations(
        catalog,
        project_root=project_root,
        integration_ids=integration_ids,
        statuses=statuses,
        families=families,
        run_modes=run_modes,
        runtimes=runtimes,
        limit=limit,
    )
    return tuple(
        preflight_integration(record, project_root=project_root, hook=hook) for record in records
    )


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    value = load_yaml(config)
    if not isinstance(value, Mapping):
        raise ValueError("matrix config 必须是 YAML 对象")
    return dict(value)


def _item_config(base: Mapping[str, Any], record: IntegrationRecord) -> dict[str, Any]:
    config = dict(base)
    integrations = config.get("integrations")
    if isinstance(integrations, Mapping) and isinstance(integrations.get(record.id), Mapping):
        config = {**config, **dict(integrations[record.id])}
    cases = config.get("cases")
    if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            case_id = case.get("integration_id", case.get("id", case.get("name")))
            if case_id == record.id:
                config = {**config, **dict(case)}
                break
    encoder = dict(config.get("encoder", {})) if isinstance(config.get("encoder"), Mapping) else {}
    encoder.setdefault("adapter", record.id)
    encoder.setdefault("definition", record.definition)
    config["encoder"] = encoder
    streaming = (
        dict(config.get("streaming", {})) if isinstance(config.get("streaming"), Mapping) else {}
    )
    streaming.setdefault("enabled", record.run_mode == "streaming")
    if record.run_mode == "streaming":
        streaming.setdefault("chunk_frames", record.smoke_profile.clip_frames)
        streaming.setdefault("frame_stride", record.smoke_profile.frame_stride)
    config["streaming"] = streaming
    return config


def build_experiment_config(
    record: IntegrationRecord,
    *,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """由 integration 记录构造一个可交给统一 resolver 的最小 experiment。"""

    if not isinstance(record, IntegrationRecord):
        raise TypeError("record 必须是 IntegrationRecord")
    config = _item_config(dict(base_config or {}), record)
    sampler = dict(config.get("sampler", {}))
    sampler.setdefault("clip_frames", record.smoke_profile.clip_frames)
    sampler.setdefault("frame_stride", record.smoke_profile.frame_stride)
    config["sampler"] = sampler
    config.setdefault("schema_version", 1)
    config.setdefault("dataset", {})
    config.setdefault("task", {"kind": "encoder_smoke", "supervision": "video"})
    config.setdefault("output", {"root": "outputs", "run_name": f"smoke-{record.id}"})
    return config


def _path_inside(path: Path, root: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"输出路径越出 output_root：{resolved}")
    return resolved


def _failure_document(
    record: IntegrationRecord,
    *,
    status: str,
    message: str,
    stage: str,
    root: Path,
    video_path: str | Path | None,
    config: Mapping[str, Any],
    run_id: str,
    log_path: Path,
    exit_code: int | None,
    exception: BaseException | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    video = Path(video_path).expanduser().resolve() if video_path is not None else None
    video_rel = (
        "missing-video"
        if video is None
        else (video.relative_to(root).as_posix() if root in video.parents else video.as_posix())
    )
    video_identity = {
        "path": video_rel,
        "sha256": "0" * 64,
        "num_frames": 1,
        "fps": 1.0,
        "duration_seconds": 1.0,
        "width": 1,
        "height": 1,
    }
    if video is not None and video.is_file():
        with suppress(OSError):
            video_identity["sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    commit, dirty = _git_commit(root)
    stack = None
    if exception is not None:
        stack = "".join(
            traceback.format_exception(type(exception), exception, exception.__traceback__)
        )
    summary = None
    if record.run_mode == "streaming":
        summary = {
            "chunks_requested": max(2, record.smoke_profile.chunks),
            "chunks_completed": 0,
            "state_steps": [],
            "state_present": False,
            "cache_kinds": sorted(
                getattr(item, "value", str(item)) for item in record.capabilities.cache_kinds
            ),
        }
    return {
        "schema_version": SMOKE_V2_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "status": status,
        "encoder": {
            "id": record.id,
            "display_name": record.display_name,
            "adapter": record.adapter_target,
            "backend": record.backend,
            "run_mode": "streaming" if record.run_mode == "streaming" else "fixed",
            "feature_stage": record.feature_stage,
        },
        "input": {"video": video_identity, "batches": []},
        "outputs": [],
        "streaming": summary,
        "environment": {
            "profile": record.environment.profile,
            "hostname": platform.node() or "unknown-host",
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "device": "unknown",
            "torch_version": None,
            "cuda_version": None,
            "gpu": None,
            "packages": {},
        },
        "assets": {
            "upstream": {
                "repo": "unknown",
                "revision": "unknown",
                "license": "unknown",
                "checkout_path": None,
            },
            "checkpoint": {
                "repo": None,
                "revision": None,
                "license": None,
                "path": None,
                "sha256": None,
                "size_bytes": None,
            },
        },
        "execution": {
            "command": ["vadbench", "integrations", "matrix", record.id],
            "started_at_utc": _utc_now(),
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": 0.0,
            "exit_code": exit_code,
            "log_path": str(log_path.relative_to(root).as_posix())
            if root in log_path.parents
            else str(log_path),
            "peak_gpu_memory_bytes": None,
        },
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "config_sha256": _config_hash(config),
            "catalog_version": CATALOG_V1_VERSION,
        },
        "error": {
            "category": "preflight" if stage == "preflight" else "runtime_error",
            "stage": stage,
            "message": message[:4096],
            "recoverable": status == "failed",
            "traceback": stack[-16000:] if stack else None,
            "evidence": _safe(dict(evidence or {})),
        },
    }


def _coerce_runner_result(
    value: Any,
    record: IntegrationRecord,
    output_path: Path,
) -> tuple[dict[str, Any], bool]:
    if isinstance(value, Mapping):
        return validate_smoke_result_v2(value, expected_record=record), False
    if isinstance(value, (str, Path)):
        selected = Path(value)
        if not selected.is_file():
            raise FileNotFoundError(selected)
        result = read_smoke_result_v2(selected, expected_record=record)
        return result, selected.resolve() == output_path.resolve()
    if value is None:
        raise ValueError("runner 未返回 smoke result")
    raise TypeError(f"runner 返回不支持的类型：{type(value).__name__}")


def run_integration_matrix(
    catalog: IntegrationCatalog | str | Path | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    video_path: str | Path | None = None,
    *,
    project_root: str | Path = ".",
    integration_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    run_modes: Sequence[str] | None = None,
    runtimes: Sequence[str] | None = None,
    limit: int | None = None,
    output_root: str | Path = "outputs/encoder-integration",
    matrix_path: str | Path | None = None,
    run_id: str | None = None,
    run_one: Callable[..., Any] | None = None,
    in_process_runner: Callable[..., Any] | None = None,
    external_python_runner: Callable[..., Any] | None = None,
    preflight_hook: Callable[..., Any] | None = None,
    skip_preflight: bool = False,
    command: Sequence[str] | None = None,
    write_results: bool = True,
    execute: bool = True,
) -> dict[str, Any]:
    """串行运行选中的 integration；单项失败不会中断后续项。"""

    root = Path(project_root).expanduser().resolve()
    base_config = _load_config(config)
    selected = filter_integrations(
        catalog,
        project_root=root,
        integration_ids=integration_ids,
        statuses=statuses,
        families=families,
        run_modes=run_modes,
        runtimes=runtimes,
        limit=limit,
    )
    output_dir = Path(output_root).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    if output_dir == root or root not in output_dir.parents:
        raise ValueError(f"output_root 必须位于 project_root 内：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if matrix_path is None:
        resolved_matrix_path = output_dir / "matrix.json"
    else:
        resolved_matrix_path = Path(matrix_path)
        if not resolved_matrix_path.is_absolute():
            resolved_matrix_path = root / resolved_matrix_path
    resolved_matrix_path = _path_inside(resolved_matrix_path, output_dir)
    if write_results:
        existing_paths = [
            output_dir / record.id / "result.json"
            for record in selected
            if (output_dir / record.id / "result.json").exists()
        ]
        if resolved_matrix_path.exists():
            existing_paths.append(resolved_matrix_path)
        if existing_paths:
            listed = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(f"matrix 输出已存在；请使用新的 output_root：{listed}")
    run_id = run_id or (
        f"matrix-{_datetime.datetime.now(_datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{uuid.uuid4().hex[:8]}"
    )
    items: list[dict[str, Any]] = []
    for record in selected:
        item_dir = _path_inside(output_dir / record.id, output_dir)
        item_dir.mkdir(parents=True, exist_ok=True)
        result_path = _path_inside(item_dir / "result.json", output_dir)
        log_path = _path_inside(item_dir / "run.log", output_dir)
        item = {
            "integration_id": record.id,
            "display_name": record.display_name,
            "runtime": record.environment.runtime,
            "run_mode": record.run_mode,
            "result_path": result_path.relative_to(root).as_posix()
            if root in result_path.parents
            else result_path.as_posix(),
            "log_path": log_path.relative_to(root).as_posix()
            if root in log_path.parents
            else log_path.as_posix(),
            "preflight": None,
            "error": None,
        }
        preflight_result = None
        if not skip_preflight:
            try:
                preflight_result = preflight_integration(
                    record, project_root=root, hook=preflight_hook
                )
            except Exception as exc:
                preflight_result = {
                    "integration_id": record.id,
                    "status": "blocked",
                    "runtime": record.environment.runtime,
                    "run_mode": record.run_mode,
                    "checks": {},
                    "reasons": [str(exc)],
                }
        item["preflight"] = preflight_result
        if preflight_result is not None and preflight_result.get("status") != "preflight_pass":
            result = _failure_document(
                record,
                status="blocked",
                message="；".join(str(item) for item in preflight_result.get("reasons", []))
                or "preflight 未通过",
                stage="preflight",
                root=root,
                video_path=video_path,
                config=base_config,
                run_id=f"{run_id}-{record.id}",
                log_path=log_path,
                exit_code=None,
                evidence={"preflight": preflight_result},
            )
            if write_results:
                write_smoke_result_v2(result, result_path, output_root=output_dir)
            item.update(status="blocked", error=result["error"])
            items.append(item)
            continue
        if not execute:
            item["status"] = (
                preflight_result.get("status", "preflight_pass")
                if preflight_result is not None
                else "preflight_pass"
            )
            items.append(item)
            continue
        item_config = build_experiment_config(record, base_config=base_config)
        context: dict[str, Any] = {
            "record": record,
            "integration": record,
            "config": item_config,
            "video_path": video_path,
            "project_root": root,
            "output_path": result_path,
            "log_path": log_path,
            "run_id": f"{run_id}-{record.id}",
            "command": command,
            "preflight": preflight_result,
        }
        runner = run_one
        if runner is None:
            runner = (
                in_process_runner
                if record.environment.runtime == "in_process"
                else external_python_runner
            )
        if runner is None and record.environment.runtime == "external_python":
            current_prefix = Path(sys.prefix).resolve()
            try:
                expected_prefix = resolve_encoder_runtime(
                    record.id, project_root=root
                ).group.prefix.resolve()
                runtime_error = None
            except Exception as exc:
                expected_prefix = None
                runtime_error = str(exc) or type(exc).__name__
            if runtime_error is not None or current_prefix != expected_prefix:
                result = _failure_document(
                    record,
                    status="blocked",
                    message=(
                        f"external_python 目标必须在声明环境中执行：current={current_prefix}，"
                        f"expected={expected_prefix or 'unresolved'}；"
                        "请使用 server native runner"
                    ),
                    stage="runtime_environment",
                    root=root,
                    video_path=video_path,
                    config=item_config,
                    run_id=context["run_id"],
                    log_path=log_path,
                    exit_code=None,
                    evidence={
                        "current_prefix": str(current_prefix),
                        "expected_prefix": None
                        if expected_prefix is None
                        else str(expected_prefix),
                        "runtime_error": runtime_error,
                    },
                )
                if write_results:
                    write_smoke_result_v2(result, result_path, output_root=output_dir)
                item.update(status="blocked", error=result["error"])
                items.append(item)
                continue
        try:
            if runner is None:
                value = run_encoder_smoke_v2(
                    item_config,
                    video_path,
                    project_root=root,
                    run_id=context["run_id"],
                    command=command,
                    log_path=log_path,
                    integration_id=record.id,
                )
            else:
                value = _invoke_hook(runner, record, context)
            result, written_by_runner = _coerce_runner_result(value, record, result_path)
            if write_results and not written_by_runner:
                write_smoke_result_v2(result, result_path, output_root=output_dir)
            item_status = str(result.get("status", "failed"))
            item_error = result.get("error")
        except Exception as exc:
            result = _failure_document(
                record,
                status="failed",
                message=str(exc) or type(exc).__name__,
                stage="runner",
                root=root,
                video_path=video_path,
                config=item_config,
                run_id=context["run_id"],
                log_path=log_path,
                exit_code=1,
                exception=exc,
                evidence={"runtime": record.environment.runtime},
            )
            if write_results:
                failure_path = result_path
                if result_path.exists():
                    failure_path = _path_inside(item_dir / "failure.json", output_dir)
                    item["result_path"] = (
                        failure_path.relative_to(root).as_posix()
                        if root in failure_path.parents
                        else failure_path.as_posix()
                    )
                write_smoke_result_v2(result, failure_path, output_root=output_dir)
            item_status = "failed"
            item_error = result["error"]
        item.update(status=item_status, error=item_error)
        items.append(item)
    counts: dict[str, int] = {}
    for item in items:
        key = str(item["status"])
        counts[key] = counts.get(key, 0) + 1
    git_commit, git_dirty = _git_commit(root)
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "status": "completed",
        "output_root": output_dir.relative_to(root).as_posix()
        if root in output_dir.parents
        else output_dir.as_posix(),
        "selected_count": len(selected),
        "counts": counts,
        "items": items,
        "provenance": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "config_sha256": _config_hash(base_config),
            "catalog_version": CATALOG_V1_VERSION,
        },
    }
    if write_results:
        write_matrix_result(matrix, resolved_matrix_path, output_root=output_dir)
    matrix["matrix_path"] = (
        resolved_matrix_path.relative_to(root).as_posix()
        if root in resolved_matrix_path.parents
        else resolved_matrix_path.as_posix()
    )
    return matrix


def write_matrix_result(
    result: Mapping[str, Any],
    path: str | Path,
    *,
    output_root: str | Path | None = None,
    overwrite_success: bool = False,
) -> Path:
    """以临时文件 + replace 原子写入矩阵汇总；覆盖必须显式授权。"""

    output = Path(path).expanduser()
    root = Path(output_root).expanduser() if output_root is not None else output.parent
    _path_inside(output, root)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite_success:
        raise FileExistsError(f"matrix result 已存在；显式允许覆盖后才能写入：{output}")
    payload = json.dumps(dict(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
    "MATRIX_SCHEMA_VERSION",
    "build_experiment_config",
    "filter_integrations",
    "preflight_integration",
    "preflight_integrations",
    "run_integration_matrix",
    "write_matrix_result",
]
