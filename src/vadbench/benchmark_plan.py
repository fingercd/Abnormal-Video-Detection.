"""Orchestrate versioned benchmark YAML as isolated real-encoder cases."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.artifacts import file_identity, record_stage
from vadbench.benchmark import (
    PERFORMANCE_SCHEMA_VERSION,
    BenchmarkSettings,
    assess_sampling_comparability,
    write_performance_result,
)
from vadbench.benchmark_runtime import run_benchmark_case, run_isolated_benchmark_case
from vadbench.config import load_yaml
from vadbench.data.video import VideoInfo, decode_rgb_frames, probe_video


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_case_result(result: Mapping[str, Any], requirements: Mapping[str, Any]) -> None:
    for field in ("native_compression_calls", "native_compression_applied_steps"):
        requirement = f"{field}_min"
        if requirement in requirements:
            observed = min(int(repeat["cache"][field]) for repeat in result["repeats"])
            expected = int(requirements[requirement])
            if observed < expected:
                raise RuntimeError(
                    f"{result.get('name')}: {field} 每 repeat 最小值 {observed} < {expected}"
                )


def run_benchmark_plan(
    plan_path: str | Path,
    *,
    project_root: str | Path = ".",
    video: str | Path | None = None,
    device: str | None = None,
    warmup: int | None = None,
    repeat_count: int | None = None,
    output: str | Path | None = None,
    encoder_factory: Callable[..., tuple[Any, Mapping[str, Any]]] | None = None,
    probe_fn: Callable[..., VideoInfo] = probe_video,
    decode_fn: Callable[..., np.ndarray] = decode_rgb_frames,
    isolated_executor: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run every real case in its registered Python; injected factories stay local.

    Tests and narrow local adapters can inject ``encoder_factory``. The default
    path uses the v2 environment registry, so CUDA telemetry comes from the
    interpreter that loaded the model.
    """

    root = Path(project_root).resolve()
    resolved_plan_path = _resolve(root, plan_path)
    plan = load_yaml(resolved_plan_path)
    benchmark = dict(plan.get("benchmark", {}))
    input_spec = dict(plan.get("input", {}))
    case_specs = list(plan.get("cases", ()))
    if not case_specs:
        raise ValueError("benchmark plan 至少需要一个 case")
    selected_video = _resolve(root, video or str(input_spec["video"]))
    settings = BenchmarkSettings(
        warmup=int(benchmark.get("warmup", 1) if warmup is None else warmup),
        repeat=int(benchmark.get("repeat", 5) if repeat_count is None else repeat_count),
        synchronize_cuda=bool(benchmark.get("synchronize_cuda", True)),
        device=device or benchmark.get("device"),
    )
    destination = _resolve(root, output or str(benchmark["output"]))
    with record_stage(
        destination.parent,
        "benchmark",
        config=plan,
        inputs={"plan": resolved_plan_path, "video": selected_video},
        project_root=root,
    ) as stage:
        case_results: list[dict[str, Any]] = []
        for case_spec in case_specs:
            if encoder_factory is None:
                result = run_isolated_benchmark_case(
                    case_spec,
                    root=root,
                    selected_video=selected_video,
                    input_spec=input_spec,
                    settings=settings,
                    executor=isolated_executor,
                    diagnostics_dir=destination.parent,
                )
            else:
                result = run_benchmark_case(
                    case_spec,
                    root=root,
                    selected_video=selected_video,
                    input_spec=input_spec,
                    settings=settings,
                    encoder_factory=encoder_factory,
                    probe_fn=probe_fn,
                    decode_fn=decode_fn,
                )
            _require_case_result(result, case_spec.get("result_requirements", {}))
            case_results.append(result)

        result = {
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "settings": asdict(settings),
            "comparison": assess_sampling_comparability(case_results),
            "provenance": {
                "machine": {
                    "hostname": platform.node(),
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "python_implementation": platform.python_implementation(),
                    "python_executable": sys.executable,
                },
                "plan": str(resolved_plan_path),
            },
            "cases": case_results,
        }
        write_performance_result(result, destination)
        stage["outputs"] = {"performance": file_identity(destination, project_root=root)}
        stage["summary"] = {
            "case_count": len(case_results),
            "comparison_comparable": result["comparison"]["comparable"],
        }
    return result


__all__ = ["run_benchmark_plan"]
