from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vadbench.paper.profile import load_project
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
        "timesformer",
        "vjepa2",
        "videomae",
    ]
    assert all(
        row["probe_ready"] is None and not row["reduction_ready"] for row in report["encoders"]
    )
    assert tuple(ENCODER_REGISTRY) == before
    assert len(before) > 4


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
    assert not (project.root / "outputs").exists()
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
    receipts = list((project.root / "outputs").rglob("provenance/stages/*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"
    assert not list((project.root / "outputs").rglob("summary.json"))


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
