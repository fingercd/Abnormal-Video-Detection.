from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import vadbench.engine.integration_matrix as matrix_module
from vadbench.engine.integration_matrix import (
    filter_integrations,
    preflight_integrations,
    run_integration_matrix,
    write_matrix_result,
)
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG


def _smoke_result(record):
    streaming = record.run_mode == "streaming"
    return {
        "schema_version": "vadbench.encoder-smoke.v2",
        "generated_at_utc": "2026-09-11T00:00:00Z",
        "run_id": f"unit-{record.id}",
        "status": "smoke_pass",
        "encoder": {
            "id": record.id,
            "display_name": record.display_name,
            "adapter": record.adapter_target,
            "backend": record.backend,
            "run_mode": "streaming" if streaming else "fixed",
            "feature_stage": record.feature_stage,
        },
        "input": {
            "video": {
                "path": "video.mp4",
                "sha256": "0" * 64,
                "num_frames": 4,
                "fps": 2.0,
                "duration_seconds": 2.0,
                "width": 4,
                "height": 3,
            },
            "batches": [],
        },
        "outputs": [
            {
                "step_index": 1 if streaming else 0,
                "passed": True,
                "feature_stage": record.feature_stage,
                "sequence_source": "unit",
                "identity_inferred": False,
                "features": {
                    "shape": [1, 1, 2],
                    "dtype": "float32",
                    "finite": True,
                    "non_finite_count": 0,
                },
                "pooled": {
                    "shape": [1, 2],
                    "dtype": "float32",
                    "finite": True,
                    "non_finite_count": 0,
                },
                "timeline": {
                    "token_count": 1,
                    "valid_tokens_per_batch": [1],
                    "token_count_matches": True,
                    "monotonic": True,
                    "in_video_range": True,
                    "video_bounds_checked": True,
                    "has_source_frames": True,
                    "min_start_seconds": 0.0,
                    "max_end_seconds": 1.0,
                    "min_source_frame": 0,
                    "max_source_frame_end": 1,
                },
                "aux": {},
            }
        ],
        "streaming": (
            {
                "chunks_requested": 2,
                "chunks_completed": 2,
                "state_steps": [1, 2],
                "state_present": True,
                "cache_kinds": ["decoder_kv"],
            }
            if streaming
            else None
        ),
        "environment": {
            "profile": record.environment.profile,
            "hostname": "unit-host",
            "python_version": "3.11.0",
            "python_executable": "/python",
            "device": "cpu",
            "torch_version": None,
            "cuda_version": None,
            "gpu": None,
            "packages": {},
        },
        "assets": {
            "upstream": {
                "repo": "upstream",
                "revision": "abc1234",
                "license": "mit",
                "checkout_path": None,
            },
            "checkpoint": {
                "repo": "checkpoint",
                "revision": "abc1234",
                "license": "mit",
                "path": "weights/model.bin",
                "sha256": "1" * 64,
                "size_bytes": 1,
            },
        },
        "execution": {
            "command": ["vadbench", "smoke"],
            "started_at_utc": "2026-09-11T00:00:00Z",
            "finished_at_utc": "2026-09-11T00:00:01Z",
            "elapsed_seconds": 1.0,
            "exit_code": 0,
            "log_path": "run.log",
            "peak_gpu_memory_bytes": None,
        },
        "provenance": {
            "git_commit": "abcdef0",
            "git_dirty": False,
            "config_sha256": "2" * 64,
            "catalog_version": "vadbench.encoder-integrations.v1",
        },
        "error": None,
    }


def test_filter_and_preflight_support_id_runtime_and_mode() -> None:
    selected = filter_integrations(
        DEFAULT_INTEGRATION_CATALOG,
        integration_ids=["videomaev2", "hermes_llava_ov"],
    )
    assert [item.id for item in selected] == ["videomaev2", "hermes_llava_ov"]
    assert len(preflight_integrations(DEFAULT_INTEGRATION_CATALOG, run_modes=["streaming"])) == 5


def test_matrix_uses_runtime_hooks_and_continues_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(record):
        calls.append(record.id)
        if record.id == "videomaev2":
            raise RuntimeError("synthetic failure")
        return _smoke_result(record)

    result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {"schema_version": 1},
        tmp_path / "video.mp4",
        project_root=tmp_path,
        integration_ids=["videomaev2", "hermes_llava_ov"],
        skip_preflight=True,
        run_one=runner,
        output_root=tmp_path / "matrix",
    )
    assert calls == ["videomaev2", "hermes_llava_ov"]
    assert [item["status"] for item in result["items"]] == ["failed", "smoke_pass"]
    assert (tmp_path / "matrix" / "matrix.json").is_file()
    assert (tmp_path / "matrix" / "hermes_llava_ov" / "result.json").is_file()


def test_matrix_rejects_existing_result_before_running_and_rejects_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "matrix"
    output_root.mkdir()
    existing = output_root / "videomaev2" / "result.json"
    existing.parent.mkdir()
    existing.write_text(json.dumps({"status": "smoke_pass", "sentinel": 1}), encoding="utf-8")
    calls: list[str] = []

    def runner(record):
        calls.append(record.id)
        return _smoke_result(record)

    with pytest.raises(FileExistsError, match="新的 output_root"):
        run_integration_matrix(
            DEFAULT_INTEGRATION_CATALOG,
            {},
            None,
            project_root=tmp_path,
            integration_ids=["videomaev2"],
            skip_preflight=True,
            run_one=runner,
            output_root=output_root,
        )
    assert calls == []
    assert json.loads(existing.read_text())["sentinel"] == 1
    with pytest.raises(ValueError):
        write_matrix_result(
            {"status": "completed"}, tmp_path / "escape.json", output_root=output_root
        )


def test_matrix_rejects_incomplete_or_wrong_identity_runner_success(tmp_path: Path) -> None:
    def incomplete(record):
        return {"status": "smoke_pass", "encoder": {"id": record.id}}

    incomplete_result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=incomplete,
        output_root=tmp_path / "incomplete",
    )
    assert incomplete_result["items"][0]["status"] == "failed"
    assert "不符合 v2 schema" in incomplete_result["items"][0]["error"]["message"]

    def wrong_identity(record):
        result = deepcopy(_smoke_result(record))
        result["encoder"]["id"] = "another_encoder"
        return result

    wrong_result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=wrong_identity,
        output_root=tmp_path / "wrong-identity",
    )
    assert wrong_result["items"][0]["status"] == "failed"
    assert "catalog identity 不一致" in wrong_result["items"][0]["error"]["message"]

    def unknown_assets(record):
        result = deepcopy(_smoke_result(record))
        result["assets"]["upstream"]["repo"] = "unknown"
        return result

    unknown_result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=unknown_assets,
        output_root=tmp_path / "unknown-assets",
    )
    assert unknown_result["items"][0]["status"] == "failed"
    assert "缺少明确 upstream 身份" in unknown_result["items"][0]["error"]["message"]


def test_matrix_accepts_validated_result_written_to_requested_path(tmp_path: Path) -> None:
    def runner(record, context):
        path = context["output_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_smoke_result(record)), encoding="utf-8")
        return path

    result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=runner,
        output_root=tmp_path / "runner-writes",
    )

    assert result["items"][0]["status"] == "smoke_pass"
    saved = tmp_path / "runner-writes" / "videomaev2" / "result.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["run_id"] == "unit-videomaev2"


def test_matrix_preserves_invalid_runner_file_and_records_validation_failure(
    tmp_path: Path,
) -> None:
    def runner(record, context):
        path = context["output_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "smoke_pass"}), encoding="utf-8")
        return path

    result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=runner,
        output_root=tmp_path / "invalid-runner-file",
    )

    item = result["items"][0]
    assert item["status"] == "failed"
    assert item["result_path"].endswith("failure.json")
    invalid = tmp_path / "invalid-runner-file" / "videomaev2" / "result.json"
    assert json.loads(invalid.read_text(encoding="utf-8")) == {"status": "smoke_pass"}
    failure = tmp_path / item["result_path"]
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "failed"


def test_default_external_runtime_blocks_wrong_environment_but_injected_runner_bypasses_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_prefix = tmp_path / "expected-env"
    monkeypatch.setattr(matrix_module.sys, "prefix", str(tmp_path / "current-env"))
    monkeypatch.setattr(
        matrix_module,
        "resolve_encoder_runtime",
        lambda *args, **kwargs: SimpleNamespace(group=SimpleNamespace(prefix=expected_prefix)),
    )

    blocked = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["c3d"],
        skip_preflight=True,
        output_root=tmp_path / "wrong-environment",
    )
    assert blocked["items"][0]["status"] == "blocked"
    assert blocked["items"][0]["error"]["stage"] == "runtime_environment"

    injected = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["c3d"],
        skip_preflight=True,
        run_one=_smoke_result,
        output_root=tmp_path / "injected-runner",
    )
    assert injected["items"][0]["status"] == "smoke_pass"
