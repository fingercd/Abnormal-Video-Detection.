"""Dense-only dispatch contract for the frozen encoder timing entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.icassp2027 import benchmark_frozen_reducers as timing_cli

ROOT = Path(__file__).resolve().parents[2]


def _dry_run(*arguments: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/icassp2027/benchmark_frozen_reducers.py"),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_dense_only_dry_run_has_one_identity_method_and_no_calibration_dependency() -> None:
    plan = _dry_run("--encoder", "videomaev2", "--dense-only", "--runtime-origin", str(ROOT))

    assert plan["dense_only"] is True
    assert plan["methods"] == ["current_original_encoder_dense_baseline"]
    assert plan["calibration_root"] is None
    assert plan["benchmark_identity"] == {
        "kind": "current_original_encoder_dense_baseline",
        "reducer": "identity",
        "analysis_hooks_enabled": False,
        "clip_scopes": ["pure_model", "adapter", "end_to_end"],
        "not_a_urdmu_detector_benchmark": True,
        "not_a_new_reducer_or_method": True,
    }
    assert plan["batch_sizes"] == [1, 8]
    assert plan["timing"]["warmup"] == 5
    assert plan["timing"]["repeat"] == 30
    assert plan["frozen_precision"] == "float32"


def test_default_dispatch_preserves_all_legacy_methods_and_calibration_requirement() -> None:
    plan = _dry_run("--encoder", "timesformer")

    assert plan["dense_only"] is False
    assert plan["methods"] == [
        "dense",
        "global_uniform",
        "paired_random_seed0",
        "pair_linear_gatefit128",
    ]
    assert plan["calibration_root"] == str(timing_cli.DEFAULT_CALIBRATION_ROOT)
    assert plan["benchmark_identity"]["kind"] == "legacy_frozen_reducer_comparison"


def test_dense_dispatch_never_constructs_a_reducer_spec_with_calibration() -> None:
    specs = timing_cli._selected_method_specs(
        dense_only=True,
        calibration_root=Path("must-not-be-read"),
    )
    assert specs == (("current_original_encoder_dense_baseline", "identity", None),)


def test_runtime_receipt_requires_the_python_resolved_from_the_explicit_origin(monkeypatch) -> None:
    expected_python = Path(sys.executable).resolve()
    captured: dict[str, object] = {}

    def resolve(encoder: str, *, project_root: Path):
        captured.update({"encoder": encoder, "origin": project_root})
        return SimpleNamespace(
            python=expected_python,
            encoder_id=encoder,
            group=SimpleNamespace(id="foundation-video-v2", prefix=Path(sys.prefix).resolve()),
            overlay=None,
        )

    monkeypatch.setattr(timing_cli, "resolve_encoder_runtime", resolve)
    receipt = timing_cli._runtime_receipt(
        encoder="videomaev2",
        runtime_origin=ROOT,
        selection="dense_only_origin_verified",
    )

    assert captured == {"encoder": "videomaev2", "origin": ROOT}
    assert receipt == {
        "selection": "dense_only_origin_verified",
        "origin": str(ROOT),
        "encoder": "videomaev2",
        "group": "foundation-video-v2",
        "overlay": None,
        "actual_runtime": {
            "sys_executable": str(expected_python),
            "sys_prefix": str(Path(sys.prefix).resolve()),
        },
        "expected_runtime": {
            "python_executable": str(expected_python),
            "prefix": str(Path(sys.prefix).resolve()),
        },
    }


def test_runtime_dispatch_fails_before_cuda_work_when_the_wrong_python_is_used(monkeypatch) -> None:
    wrong_python = ROOT / "wrong-python"
    runtime = SimpleNamespace(
        python=wrong_python,
        encoder_id="videomae",
        group=SimpleNamespace(id="classic-video-v2", prefix=ROOT / "wrong-prefix"),
        overlay=None,
    )
    monkeypatch.setattr(timing_cli, "resolve_encoder_runtime", lambda *_args, **_kwargs: runtime)

    with pytest.raises(RuntimeError, match="--runtime-origin"):
        timing_cli._runtime_receipt(
            encoder="videomae",
            runtime_origin=ROOT,
            selection="dense_only_origin_verified",
        )


def test_runtime_receipt_rejects_a_shared_python_binary_with_the_wrong_environment_prefix(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(
        python=Path(sys.executable).resolve(),
        encoder_id="videomae",
        group=SimpleNamespace(id="classic-video-v2", prefix=ROOT / "different-env-prefix"),
        overlay=None,
    )
    monkeypatch.setattr(timing_cli, "resolve_encoder_runtime", lambda *_args, **_kwargs: runtime)

    with pytest.raises(RuntimeError, match="prefix"):
        timing_cli._runtime_receipt(
            encoder="videomae",
            runtime_origin=ROOT,
            selection="dense_only_origin_verified",
        )


def test_legacy_execution_without_origin_preserves_caller_managed_dispatch() -> None:
    receipt = timing_cli._execution_runtime_receipt(
        encoder="videomae", dense_only=False, runtime_origin=None
    )

    assert receipt == {
        "selection": "legacy_caller_managed",
        "origin": None,
        "encoder": "videomae",
        "group": None,
        "overlay": None,
        "actual_runtime": {
            "sys_executable": str(Path(sys.executable).resolve()),
            "sys_prefix": str(Path(sys.prefix).resolve()),
        },
        "expected_runtime": None,
    }


def test_dense_only_execute_requires_an_explicit_runtime_origin_before_cuda_initialization() -> None:
    with pytest.raises(ValueError, match="--runtime-origin"):
        timing_cli.main(
            [
                "--encoder",
                "videomae",
                "--dense-only",
                "--execute",
                "--dataset-root",
                "dataset-root-is-not-opened",
                "--fit-manifest",
                "fit-manifest-is-not-opened",
                "--role-lock",
                "role-lock-is-not-opened",
                "--fit-video-id",
                "fit-video-is-not-opened",
            ]
        )
