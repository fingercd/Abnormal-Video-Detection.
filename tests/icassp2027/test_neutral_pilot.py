from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.research.intervention_runner import (
    InterventionReceipt,
    InterventionResult,
    PooledComparison,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/icassp2027/run_neutral_pilot.py"
SPEC = importlib.util.spec_from_file_location("neutral_pilot_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class _Reader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.info = SimpleNamespace(num_frames=200, fps=10.0, height=2, width=3, path=path)

    def read_indices(self, indices):
        # Pixel values identify the actual decoded frame, not just sampler metadata.
        return np.broadcast_to(np.asarray(indices, dtype=np.uint8)[:, None, None, None], (len(indices), 2, 3, 3)).copy()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _assets(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path / "data"
    categories = [f"Category{index:02d}" for index in range(13)]
    records = []
    cases = []
    for index in range(26):
        anomaly = index < 13
        category = categories[index] if anomaly else "Normal"
        name = f"video-{index:02d}.mp4"
        relative = f"{category}/{name}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"frozen-{index}".encode()
        path.write_bytes(payload)
        records.append(
            VideoManifestRecord(
                video_id=f"video-{index:02d}",
                path=relative,
                split="train",
                category=category,
                is_anomaly=anomaly,
                num_frames=200,
                fps=10.0,
            )
        )
        cases.append(
            {
                "video_id": f"video-{index:02d}",
                "path": relative,
                "weak_label": int(anomaly),
                "category_for_analysis_only": category,
                "partition": "fit",
                "window_indices": [1, 6],
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = write_manifest_jsonl(records, tmp_path / "manifest.jsonl")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    role_lock = tmp_path / "roles.json"
    role_lock.write_text(json.dumps({"schema_version": 1, "basis": "complete_official_train_source_groups", "partitions": {**{f"unused-{i:04}": "confirm" for i in range(1584)}, **{c["video_id"]: "fit" for c in cases}}}), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "videos": 26,
                "clips_per_encoder": 52,
                "windows_per_video_in_source_sampler": 8,
                "selected_window_indices": [1, 6],
                "relative_depth": 0.5,
                "budget_ratio": 0.5,
                "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "dataset_root": str(root),
            }
        ),
        encoding="utf-8",
    )
    return plan, cases_path, manifest, root, role_lock


def _rebind_plan(plan: Path, cases: Path, manifest: Path) -> None:
    payload = json.loads(plan.read_text(encoding="utf8"))
    payload.update(cases_sha256=pilot.sha256_file(cases), source_manifest_sha256=pilot.sha256_file(manifest))
    plan.write_text(json.dumps(payload), encoding="utf8")


def _receipt(candidate: str, encoder: str) -> InterventionReceipt:
    results = {}
    for name in pilot.REQUIRED_CONTROLS:
        budget = 16 if name in ("dense", "identity") else 8
        shape = (1, budget, 4)
        results[name] = InterventionResult(
            name=name, indices_sha256=None if name == "dense" else "0" * 64,
            effective_budget=budget, gathered_shape=shape, suffix_shapes={6: shape, 7: shape},
            pooled=PooledComparison((1, 4), (1, budget, 4), 0.0 if name in ("dense", "identity") else 0.01, 1.0, 0.0),
        )
    return InterventionReceipt(encoder, candidate, 5, 8, True, results)


def _cli(tmp_path: Path, monkeypatch, *, frames: int = 16):
    plan, cases, manifest, root, role_lock = _assets(tmp_path)
    project = SimpleNamespace(root=tmp_path, encoder=lambda _name: {"definition": {"constructor": {"num_frames": frames}}})
    monkeypatch.setattr(pilot, "load_project", lambda _path: project)
    monkeypatch.setattr(pilot, "OpenCVVideoReader", _Reader)
    # Bind the trusted fixture lock bytes, exercising the same hash and role checks.
    # No production count, branch, or validation function is replaced.
    monkeypatch.setattr(pilot, "ROLE_LOCK_SHA256", pilot.sha256_file(role_lock))
    model_create = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(pilot.ENCODER_REGISTRY, "create", model_create)
    monkeypatch.setattr(pilot, "encoder_identity", lambda *a, **kw: {"synthetic_model": True})
    monkeypatch.setattr(pilot, "_adapter_details", lambda *a: {"synthetic_model": True})
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", "videomaev2", "--dataset-root", str(root), "--plan", str(plan), "--cases", str(cases), "--manifest", str(manifest), "--role-lock", str(role_lock), "--output", str(output), "--execute"])
    calls = []

    def diagnostic(**kwargs):
        ordinal = len(calls) // 2
        case_index, window = divmod(ordinal, 2)
        selected_window = (1, 6)[window]
        expected = pilot.sample_uniform_full_clips(200, num_segments=8, clip_frames=frames, frame_stride=2)[selected_window]
        batch = kwargs["batch"]
        assert batch.video_ids == (f"video-{case_index:02}",)
        assert batch.frames.shape == (1, frames, 2, 3, 3)
        np.testing.assert_array_equal(batch.frame_indices[0], expected.frame_indices)
        np.testing.assert_array_equal(batch.frames[0, :, 0, 0, 0], expected.frame_indices)
        np.testing.assert_allclose(batch.timestamps_s[0], np.asarray(expected.frame_indices) / 10.)
        assert batch.valid_mask.all()
        assert kwargs["candidate"] == pilot.CANDIDATES[len(calls) % 2]
        assert kwargs["seed"] == 20260918 + ordinal
        assert kwargs["relative_depth"] == kwargs["budget_ratio"] == 0.5
        assert kwargs["include_paired"] is True
        calls.append(kwargs)
        return _receipt(kwargs["candidate"], kwargs["encoder_id"])

    monkeypatch.setattr(pilot, "run_intervention_diagnostic", diagnostic)
    return SimpleNamespace(plan=plan, cases=cases, manifest=manifest, root=root, role_lock=role_lock, output=output, calls=calls, diagnostic=diagnostic, model_create=model_create)


def test_frozen_assets_bind_exactly_26_fit_cases_and_52_jobs(tmp_path: Path, monkeypatch) -> None:
    plan, cases, manifest, root, role_lock = _assets(tmp_path)
    monkeypatch.setattr(pilot, "OpenCVVideoReader", _Reader)

    frozen_plan, frozen_cases, records = pilot._validate_assets(
        plan_path=plan, cases_path=cases, manifest_path=manifest, dataset_root=root,
        role_lock_path=role_lock, role_lock_sha256=pilot.sha256_file(role_lock),
    )

    assert frozen_plan["videos"] == 26
    assert len(records) == 26
    jobs = pilot._clip_jobs(frozen_cases)
    assert len(jobs) == 52
    assert [jobs[0][2], jobs[1][2]] == [1, 6]
    with pytest.raises(ValueError, match="exactly 26"):
        pilot._clip_jobs(frozen_cases[:1])


def test_identity_mismatch_rejects_before_any_model_initialization(tmp_path: Path, monkeypatch) -> None:
    plan, cases, manifest, root, role_lock = _assets(tmp_path)
    monkeypatch.setattr(pilot, "OpenCVVideoReader", _Reader)
    payload = json.loads(cases.read_text(encoding="utf-8"))
    payload[0]["sha256"] = "0" * 64
    cases.write_text(json.dumps(payload), encoding="utf-8")
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["cases_sha256"] = hashlib.sha256(cases.read_bytes()).hexdigest()
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(ValueError, match="content identity"):
        pilot._validate_assets(plan_path=plan, cases_path=cases, manifest_path=manifest, dataset_root=root, role_lock_path=role_lock, role_lock_sha256=pilot.sha256_file(role_lock))


def test_partial_progress_is_atomic_and_never_claims_completion(tmp_path: Path) -> None:
    failures = [{"ordinal": 3, "type": "RuntimeError", "message": "fixture failure"}]
    pilot._write_progress(tmp_path, completed=3, failures=failures)

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress == {"expected_clips": 52, "completed_clips": 3, "failures": failures}
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("frames", [8, 16, 64])
def test_production_scale_cli_decodes_frozen_windows_and_runs_all_controls(tmp_path: Path, monkeypatch, frames: int) -> None:
    run = _cli(tmp_path, monkeypatch, frames=frames)
    assert pilot.main() == 0
    assert len(run.calls) == 52 * 2
    run.model_create.assert_called_once()
    shards = sorted((run.output / "shards").glob("*.json"))
    assert len(shards) == 52
    for ordinal, shard in enumerate(shards):
        data = json.loads(shard.read_text(encoding="utf8"))
        assert data["status"] == "completed"
        assert data["ordinal"] == ordinal
        assert data["window_index"] == (1, 6)[ordinal % 2]
        assert data["seed"] == 20260918 + ordinal
        assert set(data["results"]) == set(pilot.CANDIDATES)
        sample = pilot.sample_uniform_full_clips(200, num_segments=8, clip_frames=frames, frame_stride=2)[data["window_index"]]
        assert data["frame_indices"] == list(sample.frame_indices)
        for result in data["results"].values():
            assert set(result["results"]) == pilot.REQUIRED_CONTROLS
            assert result["requested_budget"] == 8
    stage = list((run.output / "provenance/stages").glob("*.json"))
    assert len(stage) == 1
    assert json.loads(stage[0].read_text(encoding="utf8"))["status"] == "completed"
    summary = json.loads((run.output / "summary.json").read_text(encoding="utf8"))
    assert summary["status"] == "completed" and summary["clips"] == 52
    resolved = json.loads((run.output / "resolved.json").read_text(encoding="utf8"))
    assert resolved["data_fingerprints"]["role_lock_sha256"] == pilot.sha256_file(run.role_lock)
    assert resolved["source_sha256"]["run_neutral_pilot"] == pilot.sha256_file(SCRIPT)


@pytest.mark.parametrize("field,value", [("weak_label", True), ("weak_label", "1"), ("weak_label", 2), ("weak_label", 0), ("category_for_analysis_only", "Wrong"), ("partition", "confirm"), ("video_id", "replacement")])
def test_case_identity_errors_fail_before_model_init(tmp_path: Path, monkeypatch, field: str, value) -> None:
    run = _cli(tmp_path, monkeypatch)
    cases = json.loads(run.cases.read_text(encoding="utf8"))
    cases[0][field] = value
    run.cases.write_text(json.dumps(cases), encoding="utf8")
    _rebind_plan(run.plan, run.cases, run.manifest)
    with pytest.raises(ValueError):
        pilot.main()
    run.model_create.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("mutation", ["cases_hash", "role_hash", "role_confirm", "role_missing", "manifest_test", "temporal", "caption"])
def test_frozen_hash_role_and_temporal_gates_precede_model_init(tmp_path: Path, monkeypatch, mutation: str) -> None:
    run = _cli(tmp_path, monkeypatch)
    if mutation == "cases_hash":
        run.cases.write_text(run.cases.read_text(encoding="utf8") + "\n", encoding="utf8")
    elif mutation.startswith("role_"):
        lock = json.loads(run.role_lock.read_text(encoding="utf8"))
        if mutation == "role_missing":
            del lock["partitions"]["video-00"]
        else:
            lock["partitions"]["video-00"] = "confirm"
        run.role_lock.write_text(json.dumps(lock), encoding="utf8")
        if mutation != "role_hash":
            monkeypatch.setattr(pilot, "ROLE_LOCK_SHA256", pilot.sha256_file(run.role_lock))
    else:
        rows = [json.loads(line) for line in run.manifest.read_text(encoding="utf8").splitlines()]
        if mutation == "manifest_test":
            rows[0]["split"] = "test"
        elif mutation == "temporal":
            rows[0]["annotations"] = [{"scope": "frame", "is_anomaly": True, "span": {"unit": "frame", "start": 1, "end": 3}}]
        else:
            rows[0]["annotations"] = [{"scope": "caption", "is_anomaly": None, "text": "event detail"}]
        run.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
        _rebind_plan(run.plan, run.cases, run.manifest)
    expected_message = "video-level labels only" if mutation in ("temporal", "caption") else None
    with pytest.raises(ValueError, match=expected_message):
        pilot.main()
    run.model_create.assert_not_called()
    assert not run.output.exists()


@pytest.mark.parametrize("mutation", ["exception", "missing_control", "budget", "suffix", "nan"])
def test_real_partial_run_records_failed_stage_without_completed_summary(tmp_path: Path, monkeypatch, mutation: str) -> None:
    run = _cli(tmp_path, monkeypatch)

    def partial(**kwargs):
        receipt = run.diagnostic(**kwargs)
        # Fail the second candidate of job 3, so following job ordinals remain exact.
        if kwargs["seed"] == 20260921 and kwargs["candidate"] == pilot.CANDIDATES[1]:
            if mutation == "exception":
                raise RuntimeError("synthetic diagnostic failure")
            controls = dict(receipt.results)
            if mutation == "missing_control":
                del controls["paired_low"]
            elif mutation == "budget":
                controls["paired_low"] = replace(controls["paired_low"], effective_budget=9)
            elif mutation == "suffix":
                controls["paired_low"] = replace(controls["paired_low"], suffix_shapes={6: (1, 16, 4), 7: (1, 16, 4)})
            else:
                controls["paired_low"] = replace(controls["paired_low"], pooled=replace(controls["paired_low"].pooled, relative_l2=float("nan")))
            return replace(receipt, results=controls)
        return receipt

    monkeypatch.setattr(pilot, "run_intervention_diagnostic", partial)
    with pytest.raises(RuntimeError, match="partial"):
        pilot.main()
    assert len(run.calls) == 104
    progress = json.loads((run.output / "progress.json").read_text(encoding="utf8"))
    assert progress["completed_clips"] == 51
    assert len(progress["failures"]) == 1
    assert progress["failures"][0]["ordinal"] == 3
    assert not (run.output / "summary.json").exists()
    stage = next((run.output / "provenance/stages").glob("*.json"))
    provenance = json.loads(stage.read_text(encoding="utf8"))
    assert provenance["status"] == "failed"
    assert "partial" in provenance["error"]["message"]
    failed = json.loads((run.output / "shards/003-video-01-window6.json").read_text(encoding="utf8"))
    assert failed["status"] == "failed"
    # JSON parser must never encounter a NaN/Infinity numeric extension.
    for path in run.output.rglob("*.json"):
        json.loads(path.read_text(encoding="utf8"), parse_constant=lambda value: pytest.fail(f"nonfinite JSON {value}"))


def test_summary_write_failure_marks_record_stage_failed(tmp_path: Path, monkeypatch) -> None:
    run = _cli(tmp_path, monkeypatch)
    original_write = pilot.atomic_write_json

    def write(path, value):
        if Path(path).name == "summary.json":
            raise OSError("synthetic summary storage failure")
        original_write(path, value)

    monkeypatch.setattr(pilot, "atomic_write_json", write)
    with pytest.raises(OSError, match="storage failure"):
        pilot.main()
    assert len(run.calls) == 104
    assert not (run.output / "summary.json").exists()
    stage = next((run.output / "provenance/stages").glob("*.json"))
    assert json.loads(stage.read_text(encoding="utf8"))["status"] == "failed"


def test_oom_aborts_actual_stage_and_keeps_partial_receipt(tmp_path: Path, monkeypatch) -> None:
    run = _cli(tmp_path, monkeypatch)

    def oom(**kwargs):
        receipt = run.diagnostic(**kwargs)
        if kwargs["seed"] == 20260921:
            raise pilot.torch.cuda.OutOfMemoryError("synthetic OOM; no GPU is used")
        return receipt

    monkeypatch.setattr(pilot, "run_intervention_diagnostic", oom)
    with pytest.raises(pilot.torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        pilot.main()
    assert len(run.calls) == 7
    progress = json.loads((run.output / "progress.json").read_text(encoding="utf8"))
    assert progress["completed_clips"] == 3 and len(progress["failures"]) == 1
    assert not (run.output / "summary.json").exists()
    stage = next((run.output / "provenance/stages").glob("*.json"))
    provenance = json.loads(stage.read_text(encoding="utf8"))
    assert provenance["status"] == "failed"
    assert provenance["error"]["type"] == "OutOfMemoryError"
