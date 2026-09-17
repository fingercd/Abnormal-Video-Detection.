from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator

import vadbench.benchmark_runtime as benchmark_runtime
from vadbench.benchmark import PERFORMANCE_SCHEMA_VERSION
from vadbench.benchmark_plan import run_benchmark_plan
from vadbench.benchmark_runtime import BenchmarkRuntimeError, run_runtime_once
from vadbench.contracts import EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.video import VideoInfo
from vadbench.environment_registry import EncoderEnvironmentGroup, EncoderRuntime
from vadbench.integrations.worker_protocol import SidecarStore

ROOT = Path(__file__).resolve().parents[1]


class FakeFixed:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        fixed_num_frames=2,
        min_frames=2,
        max_frames=2,
    )

    def encode(self, batch, train=False):
        timeline = TokenTimeline(
            start_s=np.min(batch.timestamps_s, axis=1, keepdims=True),
            end_s=np.max(batch.timestamps_s, axis=1, keepdims=True) + 1.0,
        )
        return EncoderOutput(
            features=np.ones((batch.batch_size, 1, 3), dtype=np.float32),
            pooled=np.ones((batch.batch_size, 3), dtype=np.float32),
            timeline=timeline,
            aux={"feature_stage": "fixed"},
        )


def test_plan_runs_serial_case_and_writes_result(tmp_path: Path) -> None:
    experiment = {
        "schema_version": 1,
        "dataset": {"root": "data", "train_manifest": "train", "test_manifest": "test"},
        "encoder": {"adapter": "fake"},
        "streaming": {"enabled": False},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": "outputs", "run_name": "fake"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    plan = {
        "schema_version": 1,
        "benchmark": {"warmup": 0, "repeat": 2, "device": "cpu", "output": "result.json"},
        "input": {
            "task": "encoder_performance_only",
            "sampling_protocol": "same",
            "video": "video.mp4",
            "sample_fps": 2.0,
            "sampled_frames": 4,
        },
        "cases": [
            {
                "name": "fixed",
                "mode": "fixed",
                "experiment": "experiment.yaml",
                "encoder": {"adapter": "fake"},
                "grouping": {"units": 2, "frames_per_unit": 2},
                "compression": None,
            }
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    created = []

    def factory(config, project_root):
        created.append(config)
        return FakeFixed(), {"name": "fake"}

    info = VideoInfo(path=tmp_path / "video.mp4", num_frames=20, fps=4.0, width=8, height=8)
    result = run_benchmark_plan(
        plan_path,
        project_root=tmp_path,
        encoder_factory=factory,
        probe_fn=lambda path: info,
        decode_fn=lambda path, indices: np.zeros((4, 8, 8, 3), dtype=np.uint8),
    )
    assert result["schema_version"] == PERFORMANCE_SCHEMA_VERSION
    assert len(created) == 1
    assert len(result["cases"][0]["repeats"]) == 2
    assert (tmp_path / "result.json").is_file()
    stages = list((tmp_path / "provenance" / "stages").glob("benchmark-*.json"))
    assert len(stages) == 1
    stage = json.loads(stages[0].read_text(encoding="utf-8"))
    assert stage["status"] == "completed"
    assert stage["inputs"]["plan"]["sha256"]
    assert stage["outputs"]["performance"]["sha256"]
    assert stage["summary"]["case_count"] == 1
    schema = json.loads(
        Path("schemas/performance-result-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)


def test_plan_rejects_grouping_mismatch(tmp_path: Path) -> None:
    experiment = {
        "schema_version": 1,
        "dataset": {"root": "data", "train_manifest": "train", "test_manifest": "test"},
        "encoder": {"adapter": "fake"},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": "outputs", "run_name": "fake"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    plan = {
        "benchmark": {"warmup": 0, "repeat": 1, "device": "cpu", "output": "out.json"},
        "input": {"video": "video.mp4", "sample_fps": 1.0, "sampled_frames": 4},
        "cases": [
            {
                "name": "bad",
                "mode": "fixed",
                "experiment": "experiment.yaml",
                "encoder": {"adapter": "fake"},
                "grouping": {"units": 1, "frames_per_unit": 2},
            }
        ],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    info = VideoInfo(path=tmp_path / "video.mp4", num_frames=20, fps=4.0, width=8, height=8)
    try:
        run_benchmark_plan(
            path,
            project_root=tmp_path,
            encoder_factory=lambda config, project_root: (FakeFixed(), {}),
            probe_fn=lambda path: info,
            decode_fn=lambda path, indices: np.zeros((4, 8, 8, 3), dtype=np.uint8),
        )
    except ValueError as exc:
        assert "grouping" in str(exc)
        stages = list((tmp_path / "provenance" / "stages").glob("benchmark-*.json"))
        assert len(stages) == 1
        assert json.loads(stages[0].read_text(encoding="utf-8"))["status"] == "failed"
    else:  # pragma: no cover
        raise AssertionError("expected grouping mismatch")


def test_default_plan_path_selects_registered_runtime_and_records_it(tmp_path: Path) -> None:
    local_plan = {
        "benchmark": {"warmup": 0, "repeat": 1, "device": "cpu", "output": "local.json"},
        "input": {"video": "video.mp4", "sample_fps": 1.0, "sampled_frames": 2},
        "cases": [
            {
                "name": "local-fixed",
                "mode": "fixed",
                "experiment": "experiment.yaml",
                "encoder": {"adapter": "fake"},
                "grouping": {"units": 1, "frames_per_unit": 2},
            }
        ],
    }
    experiment = {
        "schema_version": 1,
        "dataset": {"root": "data", "train_manifest": "train", "test_manifest": "test"},
        "encoder": {"adapter": "fake"},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": "outputs", "run_name": "fake"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    (tmp_path / "local.yaml").write_text(yaml.safe_dump(local_plan), encoding="utf-8")
    info = VideoInfo(path=tmp_path / "video.mp4", num_frames=20, fps=4.0, width=8, height=8)
    local = run_benchmark_plan(
        tmp_path / "local.yaml",
        project_root=tmp_path,
        encoder_factory=lambda config, project_root: (FakeFixed(), {}),
        probe_fn=lambda path: info,
        decode_fn=lambda path, indices: np.zeros((2, 8, 8, 3), dtype=np.uint8),
    )

    runtime_plan = {
        **local_plan,
        "benchmark": {**local_plan["benchmark"], "output": "runtime.json"},
        "cases": [
            {
                **local_plan["cases"][0],
                "name": "registered-fixed",
                "encoder": {"adapter": "videomaev2"},
            }
        ],
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(runtime_plan), encoding="utf-8")
    observed: list[dict[str, object]] = []

    def executor(runtime, request):
        observed.append({"group": runtime.group.id, "request": request})
        return local["cases"][0]

    result = run_benchmark_plan(
        path,
        project_root=ROOT,
        output=tmp_path / "runtime.json",
        isolated_executor=executor,
    )
    runtime = result["cases"][0]["provenance"]["runtime"]
    assert observed[0]["group"] == "foundation-video-v2"
    assert observed[0]["request"]["case"]["encoder"]["adapter"] == "videomaev2"
    assert runtime["group"] == "foundation-video-v2"
    assert (
        Path(runtime["python_executable"])
        == (ROOT / ".encoder-envs/v2/foundation-video-v2/bin/python").resolve()
    )
    schema = json.loads(
        Path("schemas/performance-result-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)


def test_runtime_worker_writes_structured_error_for_invalid_request(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    store.write_json("input/request.json", {"schema_version": 1})
    assert run_runtime_once(tmp_path, "input/request.json", "output/response.json") == 1
    response = store.read_json("output/response.json")
    assert response["status"] == "error"
    assert "benchmark request" in response["error"]["message"]


def test_isolated_failure_keeps_worker_exchange_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "environment" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture", encoding="utf-8")
    runtime = EncoderRuntime(
        encoder_id="fake",
        group=EncoderEnvironmentGroup(
            id="fixture-v2",
            prefix=python.parents[1],
            seed=tmp_path / "seed",
            python_version="3.11",
            packages={},
            encoders=("fake",),
        ),
        python=python,
        overlay=None,
    )

    def failed_subprocess(command, **_kwargs):
        bundle = Path(command[command.index("--bundle-root") + 1])
        response_path = command[command.index("--response") + 1]
        SidecarStore(bundle).write_json(
            response_path,
            {
                "schema_version": 1,
                "status": "error",
                "error": {"type": "RuntimeError", "message": "cuDNN failed"},
            },
        )
        return SimpleNamespace(returncode=1, stdout="worker stdout\n", stderr="worker stderr\n")

    monkeypatch.setattr(
        benchmark_runtime, "build_encoder_runtime_environment", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(benchmark_runtime.subprocess, "run", failed_subprocess)

    with pytest.raises(BenchmarkRuntimeError) as captured:
        benchmark_runtime.run_isolated_benchmark_case(
            {"name": "failing case", "encoder": {"adapter": "fake"}},
            root=tmp_path,
            selected_video=tmp_path / "video.mp4",
            input_spec={},
            settings=benchmark_runtime.BenchmarkSettings(warmup=0, repeat=1),
            runtime=runtime,
            diagnostics_dir=tmp_path / "run",
        )

    message = str(captured.value)
    assert "case='failing case'" in message
    assert "group='fixture-v2'" in message
    assert "worker stderr" in message
    bundles = list((tmp_path / "run" / "benchmark-runtime").glob("failing-case-*"))
    assert len(bundles) == 1
    assert (bundles[0] / "stdout.log").read_text(encoding="utf-8") == "worker stdout\n"
    assert (bundles[0] / "stderr.log").read_text(encoding="utf-8") == "worker stderr\n"
    assert list((bundles[0] / "input").rglob("request.json"))
    assert list((bundles[0] / "output").rglob("response.json"))

    def successful_subprocess(command, **_kwargs):
        bundle = Path(command[command.index("--bundle-root") + 1])
        response_path = command[command.index("--response") + 1]
        SidecarStore(bundle).write_json(
            response_path,
            {"schema_version": 1, "status": "ok", "result": {"provenance": {}}},
        )
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(benchmark_runtime.subprocess, "run", successful_subprocess)
    result = benchmark_runtime.run_isolated_benchmark_case(
        {"name": "successful case", "encoder": {"adapter": "fake"}},
        root=tmp_path,
        selected_video=tmp_path / "video.mp4",
        input_spec={},
        settings=benchmark_runtime.BenchmarkSettings(warmup=0, repeat=1),
        runtime=runtime,
        diagnostics_dir=tmp_path / "successful-run",
    )
    assert result["provenance"]["runtime"]["group"] == "fixture-v2"
    assert not (tmp_path / "successful-run" / "benchmark-runtime").exists()
