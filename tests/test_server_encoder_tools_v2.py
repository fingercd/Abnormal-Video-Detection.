from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vadbench.checkpoints import CheckpointSpec
from vadbench.environment_registry import (
    load_encoder_candidates,
    load_encoder_environment_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts/server" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_policy_skips_manual_candidate_and_license_gate() -> None:
    runner = load_script("run_native_encoder_matrix_v2.py")
    candidates = {item["id"]: item for item in load_encoder_candidates(ROOT)}
    assert runner.candidate_is_runnable(candidates["c3d"], False) == (
        False,
        "manual_asset_missing",
    )
    assert runner.candidate_is_runnable(candidates["videochat_online"], False) == (
        False,
        "license_blocked",
    )
    assert runner.candidate_is_runnable(candidates["videochat_online"], True) == (True, None)
    assert runner.candidate_is_runnable(candidates["videomae"], False) == (
        True,
        None,
    )


def test_runner_rejects_busy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("run_native_encoder_matrix_v2.py")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "1025\n")
    with pytest.raises(SystemExit, match="already uses 1025 MiB"):
        runner.require_available_gpu("cuda:3")


def _runner_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = load_script("run_native_encoder_matrix_v2.py")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture", encoding="utf-8")
    group = SimpleNamespace(id="unit-v2", prefix=python.parents[1])
    runtime = SimpleNamespace(group=group, python=python, overlay=None)
    candidate = {
        "id": "unit",
        "registration_state": "registered_local",
        "license_state": "verified",
    }
    record = SimpleNamespace(id="unit")
    catalog = SimpleNamespace(ids=("unit",), get=lambda encoder_id: record)
    monkeypatch.setattr(runner, "load_encoder_environment_registry", lambda root: SimpleNamespace())
    monkeypatch.setattr(runner, "load_default_integration_catalog", lambda root: catalog)
    monkeypatch.setattr(runner, "load_encoder_candidates", lambda root: (candidate,))
    monkeypatch.setattr(runner, "resolve_encoder_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(runner, "build_encoder_runtime_environment", lambda *args, **kwargs: {})
    return runner


def test_runner_requires_fresh_output_and_a_zero_exit_valid_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "matrix"

    def successful_subprocess(command, **kwargs):
        result = Path(command[-1]) / "unit" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", successful_subprocess)
    monkeypatch.setattr(
        runner, "read_smoke_result_v2", lambda path, **kwargs: {"status": "smoke_pass"}
    )
    assert runner.main(["--id", "unit", "--output-root", str(output_root)]) == 0
    matrix = json.loads((output_root / "matrix-v2.json").read_text(encoding="utf-8"))
    assert matrix["items"][0]["status"] == "smoke_pass"

    with pytest.raises(FileExistsError, match="already exists"):
        runner.main(["--id", "unit", "--output-root", str(output_root)])


def test_runner_nonzero_subprocess_cannot_promote_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_fixture(tmp_path, monkeypatch)
    output_root = tmp_path / "matrix"
    read_called = False

    def failed_subprocess(command, **kwargs):
        result = Path(command[-1]) / "unit" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text('{"status":"smoke_pass"}', encoding="utf-8")
        return SimpleNamespace(returncode=17, stdout="", stderr="failed")

    def forbidden_reader(path, **kwargs):
        nonlocal read_called
        read_called = True
        return {"status": "smoke_pass"}

    monkeypatch.setattr(runner.subprocess, "run", failed_subprocess)
    monkeypatch.setattr(runner, "read_smoke_result_v2", forbidden_reader)
    assert runner.main(["--id", "unit", "--output-root", str(output_root)]) == 1
    matrix = json.loads((output_root / "matrix-v2.json").read_text(encoding="utf-8"))
    item = matrix["items"][0]
    assert item["status"] == "failed"
    assert item["reason"] == "launcher_exit_nonzero"
    assert item["exit_code"] == 17
    assert read_called is False


def test_asset_verifier_accepts_exact_hash_and_rejects_mismatch(tmp_path: Path) -> None:
    assets = load_script("fetch_encoder_assets_v2.py")
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"native-checkpoint")
    digest = assets.sha256_file(payload)
    exact = {
        "local_path": str(payload),
        "sha256": {"model.bin": digest},
    }
    spec = CheckpointSpec(
        id="unit",
        adapter="unit",
        source="local",
        repo_id="unit/repo",
        revision="unit-revision",
        license="mit",
        allow_patterns=(),
        sha256={"model.bin": digest},
    )
    assert assets.verify_entry(exact, spec)["status"] == "verified"
    wrong_spec = replace(spec, sha256={"model.bin": "0" * 64})
    assert assets.verify_entry(exact, wrong_spec)["status"] == "missing_or_mismatch"


def test_asset_checkout_requires_git_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets = load_script("fetch_encoder_assets_v2.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    candidate = {"upstream_lock": "lock.yaml"}

    def fake_load_yaml(path: Path):
        if path.name == "definition.yaml":
            return {"constructor": {"checkout_path": str(checkout)}}
        return {"source": {"repository": "https://example.invalid/repo", "commit": "abc"}}

    monkeypatch.setattr(assets, "load_yaml", fake_load_yaml)
    assert assets.verify_checkout(candidate, "definition.yaml")["status"] == "git_metadata_missing"
    (checkout / ".git").mkdir()
    monkeypatch.setattr(
        assets.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )
    assert assets.verify_checkout(candidate, "definition.yaml")["status"] == "head_unavailable"


def test_asset_main_returns_nonzero_for_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = load_script("fetch_encoder_assets_v2.py")
    monkeypatch.setattr(assets, "PROJECT_ROOT", tmp_path)
    candidate = {
        "id": "unit",
        "registration_state": "registered_local",
        "license_state": "verified",
        "checkpoint": {
            "registry_id": "unit-checkpoint",
            "repo_url": None,
            "revision": "r",
            "license": "mit",
        },
    }
    entry = {"local_path": "weights/unit", "source": "huggingface"}
    spec = CheckpointSpec("unit-checkpoint", "unit", "huggingface", "unit/repo", "r", "mit", (), {})
    monkeypatch.setattr(
        assets,
        "load_encoder_environment_registry",
        lambda root: SimpleNamespace(new_external_root=tmp_path),
    )
    monkeypatch.setattr(assets, "load_encoder_candidates", lambda root: (candidate,))
    monkeypatch.setattr(
        assets, "load_yaml", lambda path: {"checkpoints": {"unit-checkpoint": entry}}
    )
    monkeypatch.setattr(assets, "load_checkpoint_registry", lambda path: {"unit-checkpoint": spec})
    monkeypatch.setattr(
        assets,
        "load_default_integration_catalog",
        lambda root: SimpleNamespace(
            integrations=(SimpleNamespace(id="unit", definition="unit.yaml"),)
        ),
    )
    monkeypatch.setattr(
        assets, "verify_checkout", lambda candidate, definition: {"status": "not_required"}
    )
    monkeypatch.setattr(
        assets,
        "verify_entry",
        lambda entry, spec: {"status": "missing_or_mismatch", "files": [], "error": "missing"},
    )
    assert assets.main(["--id", "unit", "--output-root", str(tmp_path / "audit")]) == 1


def test_huggingface_asset_is_verified_before_it_replaces_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = load_script("fetch_encoder_assets_v2.py")
    monkeypatch.setattr(assets, "PROJECT_ROOT", tmp_path)
    payload = b"native-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    spec = CheckpointSpec(
        "unit", "unit", "huggingface", "unit/repo", "r", "mit", (), {"model.bin": digest}
    )
    hub = ModuleType("huggingface_hub")

    def snapshot_download(*, local_dir: str, **kwargs) -> None:
        Path(local_dir, "model.bin").write_bytes(payload)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(assets, "disk_guard", lambda expected_bytes=0: None)
    result = assets.acquire_huggingface(
        {
            "id": "unit",
            "checkpoint": {
                "repo_id": "unit/repo",
                "revision": "r",
                "expected_size_bytes": 0,
                "allow_patterns": ["model.bin"],
            },
        },
        {"local_path": "weights/unit"},
        spec,
        tmp_path / "cache",
    )
    assert result["status"] == "downloaded_verified"
    assert (tmp_path / "weights" / "unit" / "model.bin").read_bytes() == payload


def test_environment_manager_fingerprints_missing_prefix(tmp_path: Path) -> None:
    manager = load_script("manage_encoder_envs_v2.py")
    result = manager.fingerprint_environment(tmp_path / "missing")
    assert result == {
        "prefix": str((tmp_path / "missing").resolve()),
        "exists": False,
    }


def test_environment_verify_returns_nonzero_when_group_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = load_script("manage_encoder_envs_v2.py")
    monkeypatch.setattr(
        manager,
        "verify",
        lambda output_root: {"groups": [{"status": "missing"}]},
    )
    assert manager.main(["verify", "--output-root", str(tmp_path)]) == 1


def test_all_server_tools_use_v2_roots() -> None:
    registry = load_encoder_environment_registry(ROOT)
    assert registry.root == (ROOT / ".encoder-envs/v2").resolve()
    for name in (
        "manage_encoder_envs_v2.py",
        "fetch_encoder_assets_v2.py",
        "run_native_encoder_matrix_v2.py",
        "prepare_encoder_overlays_v2.py",
        "consolidate_encoder_v2_results.py",
    ):
        text = (ROOT / "scripts/server" / name).read_text(encoding="utf-8")
        assert (
            ".encoder-envs/v2" in text
            or "load_encoder_environment_registry" in text
            or "load_encoder_candidates" in text
        )


def test_consolidator_uses_only_the_explicit_current_run() -> None:
    consolidate = load_script("consolidate_encoder_v2_results.py")
    current_run = {
        "run_id": "current",
        "matrix_path": "outputs/encoder-v2/current/matrix-v2.json",
        "items": [{"integration_id": "videomae", "status": "failed"}],
    }
    payload = consolidate.consolidate(current_run)
    assert payload["target_count"] == 25
    assert payload["current_run"] == {
        "run_id": "current",
        "matrix_path": "outputs/encoder-v2/current/matrix-v2.json",
        "success": False,
    }
    videomae = next(item for item in payload["items"] if item["integration_id"] == "videomae")
    assert videomae["status"] == "failed"
    assert payload["history_matrix_paths"] == []


def test_overlay_specs_include_observed_runtime_dependencies() -> None:
    overlays = load_script("prepare_encoder_overlays_v2.py")
    videomae_extra = overlays.COPY_SPECS["videomaev2"]["extra_sources"]
    assert any("easydict" in names for _, names in videomae_extra)
    assert "logzero" in overlays.COPY_SPECS["hermes_llava_ov"]["names"]


def test_overlay_hash_changes_when_same_size_content_changes(tmp_path: Path) -> None:
    overlays = load_script("prepare_encoder_overlays_v2.py")
    payload = tmp_path / "package" / "module.py"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"a")
    first = overlays.hash_overlay(tmp_path)
    payload.write_bytes(b"b")
    assert overlays.hash_overlay(tmp_path) != first
