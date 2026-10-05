from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vadbench.paper.profile import load_project, output_path
from vadbench.paper.resolve import resolve_probe, status
from vadbench.paper.stages import run_probe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    for directory in (
        "configs/encoders",
        "registry",
        "projects/icassp2027",
        "configs/papers/icassp2027",
    ):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    return load_project(tmp_path / "projects/icassp2027/profile.yaml")


def change_yaml(path, mutate):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_profile_reuses_catalog_without_loading_models(project):
    from vadbench.registry import ENCODER_REGISTRY

    before = tuple(ENCODER_REGISTRY)
    report = status(project)
    assert [row["id"] for row in report["encoders"]] == [
        "videomaev2",
        "videomae",
        "timesformer",
    ]
    assert all(
        row["probe_ready"] is None and not row["reduction_ready"] for row in report["encoders"]
    )
    assert tuple(ENCODER_REGISTRY) == before
    assert len(before) > 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(ignored=True),
        lambda p: p["active_encoders"].append(dict(p["active_encoders"][0])),
        lambda p: p["active_encoders"][0].update(checkpoint="timesformer-default"),
        lambda p: p["active_encoders"][0].update(definition="../outside.yaml"),
        lambda p: p.update(annotation_policy="diagnostic_development"),
    ],
)
def test_invalid_profile_fails_before_any_model_creation(project, mutate):
    change_yaml(project.path, mutate)
    with pytest.raises(ValueError):
        load_project(project.path)


def test_dry_run_reports_blockers_cost_and_label_access_without_outputs(project):
    plan = resolve_probe(project, "configs/papers/icassp2027/suites/probe-pilot.yaml")
    assert plan["plan_only"] and plan["cost_upper_bound"]["clip_windows"] == 64
    assert plan["cost_upper_bound"]["forward_windows"] == 192
    assert plan["label_access"]["official_test"] == "denied"
    assert plan["blockers"]
    assert not (project.root / "assets/experiments/icassp2027/runs").exists()
    assert plan == resolve_probe(project, "configs/papers/icassp2027/suites/probe-pilot.yaml")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.update(method="invented"), "unimplemented method"),
        (lambda s: s.update(encoder_ids=["hermes"]), "not active"),
        (lambda s: s.update(partition="test"), "test"),
        (lambda s: s.update(role="confirm"), "disagree"),
        (lambda s: s.update(reduction={"keep_ratio": 0.5}), "unknown"),
    ],
)
def test_suite_rejects_unimplemented_or_leaking_requests(project, mutate, message):
    suite = project.root / "configs/papers/icassp2027/suites/probe-pilot.yaml"
    change_yaml(suite, mutate)
    with pytest.raises(ValueError, match=message):
        resolve_probe(project, str(suite))


def test_failed_attempt_has_its_own_failure_receipt(project):
    plan = resolve_probe(project, "configs/papers/icassp2027/suites/probe-pilot.yaml")
    with pytest.raises(ValueError, match="missing"):
        run_probe(project, plan)
    runs = output_path(project.root, project.profile["output_root"])
    receipts = list(runs.rglob("provenance/stages/*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"
    assert not list(runs.rglob("summary.json"))


def test_custom_in_repository_output_root_still_works(project):
    change_yaml(project.path, lambda p: p.update(output_root="outputs/custom-runs"))
    reloaded = load_project(project.path)
    plan = resolve_probe(reloaded, "configs/papers/icassp2027/suites/probe-pilot.yaml")
    assert plan["output_template"] == str(project.root / "outputs/custom-runs/<unique-run-id>")
    assert not (project.root / "outputs").exists()


@pytest.mark.parametrize("value", ["../external-runs", "assets/experiments/../../external-runs"])
def test_output_root_rejects_parent_escape(project, value):
    change_yaml(project.path, lambda p: p.update(output_root=value))
    with pytest.raises(ValueError, match="output path"):
        load_project(project.path)


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")


def test_external_assets_mount_is_accepted_without_output_writes(project):
    external = project.root.parent / f"{project.root.name}-external-assets"
    external.mkdir(exist_ok=True)
    (project.root / "assets").mkdir()
    _symlink_or_skip(project.root / "assets/experiments", external)
    reloaded = load_project(project.path)
    assert status(reloaded)["encoders"]
    plan = resolve_probe(reloaded, "configs/papers/icassp2027/suites/probe-pilot.yaml")
    assert plan["output_template"] == str(external / "icassp2027/runs/<unique-run-id>")
    assert list(external.iterdir()) == []


def test_external_protocol_symlink_is_still_rejected(project):
    protocol = project.root / "projects/icassp2027/protocol.yaml"
    external = project.root.parent / f"{project.root.name}-external-protocol.yaml"
    external.write_bytes(protocol.read_bytes())
    protocol.unlink()
    _symlink_or_skip(protocol, external)
    with pytest.raises(ValueError, match="escapes project root"):
        load_project(project.path)


def test_external_assets_root_link_and_nested_escape(project):
    external = project.root.parent / f"{project.root.name}-mounted-assets"
    (external / "experiments").mkdir(parents=True)
    _symlink_or_skip(project.root / "assets", external)
    reloaded = load_project(project.path)
    assert output_path(reloaded.root, reloaded.profile["output_root"]) == (
        external / "experiments/icassp2027/runs"
    )
    assert not (external / "experiments/icassp2027").exists()
    escaped = project.root.parent / f"{project.root.name}-unrelated-data"
    escaped.mkdir()
    _symlink_or_skip(external / "experiments/icassp2027", escaped)
    with pytest.raises(ValueError, match="output path escapes"):
        load_project(project.path)


def test_status_and_dry_run_are_lightweight_in_fresh_process(project):
    code = """
import sys
from vadbench.paper.cli import main
assert main(['status', '--project', sys.argv[1]]) == 0
assert main(['probe', '--project', sys.argv[1], '--suite', 'configs/papers/icassp2027/suites/probe-pilot.yaml', '--dry-run']) == 0
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(project.path)],
        cwd=project.root,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
