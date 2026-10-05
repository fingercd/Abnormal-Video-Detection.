"""official_frame method-freeze contract gate: fail-closed with a real key.

The simulated contracts here list the toy fixture runtime id on purpose: the
gate mechanics (SHA bindings, markdown and evidence receipts, frozen scope)
are identical to production, and scope-coverage rejection is tested by
excluding that id.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_urdmu_evaluation import _request, _source

from vadbench.checkpoints import sha256_file
from vadbench.paper.urdmu_backend import UPSTREAM_COMMIT
from vadbench.paper.urdmu_evaluation import (
    URDMUEvaluationRequest,
    run_urdmu_evaluation,
)


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def _freeze_contract(tmp_path: Path, *, encoders=("toy",), upstream=UPSTREAM_COMMIT, mutate=None) -> Path:
    markdown = tmp_path / "freeze.md"
    markdown.write_text("# fixture freeze\n", encoding="utf8")
    receipt = tmp_path / "evidence.json"
    receipt.write_text('{"finding": "fixture"}', encoding="utf8")
    document = {
        "schema": "icassp2027.method-freeze-contract/v1",
        "status": "active",
        "frozen_at": "2026-09-20",
        "frozen_by": "fixture",
        "markdown_contract": {"path": str(markdown), "sha256": sha256_file(markdown)},
        "method_scope": {
            "encoders": list(encoders),
            "backend": {
                "name": "UR-DMU",
                "upstream_commit": upstream,
                "head": "dense",
                "steps": 3000,
                "bags_per_class": 64,
                "seed_policy": "fixed",
                "checkpoint_selection": "fixed_final_step_no_test_selection",
            },
            "operator_families": [],
            "controls": [],
        },
        "property_evidence": [
            {"id": "fixture-evidence", "role": "fixture", "path": str(receipt), "sha256": sha256_file(receipt)}
        ],
        "evaluation_protocol": {"ucf_crime": "frame_roc_auc_primary"},
    }
    if mutate:
        mutate(document)
    return _write(tmp_path / "method-freeze-contract.json", document)


def _retag_head_formal(run: Path) -> None:
    checkpoint = run / "checkpoints" / "final.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["metadata"]["run_mode"] = "formal"
    payload["metadata"]["data_role"] = "official-fulltrain-final"
    payload["metadata"]["development_role"] = None
    torch.save(payload, checkpoint)
    digest = sha256_file(checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": digest, "size_bytes": checkpoint.stat().st_size})
    result = json.loads((run / "result.json").read_text())
    result["checkpoint_sha256"] = digest
    _write(run / "result.json", result)
    qa = json.loads((run / "training_qa.json").read_text())
    qa["checkpoint"]["sha256"] = digest
    _write(run / "training_qa.json", qa)
    stage_path = run / "provenance" / "stages" / "training.json"
    stage = json.loads(stage_path.read_text())
    stage["config"]["run_mode"] = "formal"
    _write(stage_path, stage)


def _official_request(tmp_path, run, features, manifest, contract_path, contract_sha, **overrides):
    values = {
        "dataset": "ucf_crime",
        "phase": "official_frame",
        "mode": "direct_insert",
        "trained_run": str(run),
        "feature_store": str(features),
        "evaluation_manifest": str(manifest),
        "upstream_dir": "must-not-load-in-unit-test",
        "output_root": str(tmp_path / "official-scores"),
        "device": "cpu",
        "freeze_path": str(_write(tmp_path / "freeze-receipt.json", {"fixture": True})),
        "audit_report": str(_write(tmp_path / "audit-report.json", {"fixture": True})),
        "method_freeze_contract_path": str(contract_path),
        "method_freeze_contract_sha256": contract_sha,
        "feature_contract_path": None,
        "feature_contract_sha256": None,
        "run_id": "official-scores",
    }
    values.update(overrides)
    return URDMUEvaluationRequest(**values)


def _fake_scores(source, target, request):
    return (
        {
            record.video_id: np.full(len(target.rows_by_video[record.video_id]), 0.4)
            for record in target.records
        },
        {},
    )


def test_official_frame_passes_with_valid_freeze_contract_and_records_audit_chain(tmp_path, monkeypatch):
    run, features, manifest, _contract, _authority_contract, _records = _source(tmp_path)
    _retag_head_formal(run)
    contract = _freeze_contract(tmp_path)
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", _fake_scores)
    result = run_urdmu_evaluation(
        _official_request(tmp_path, run, features, manifest, contract, sha256_file(contract))
    )
    assert result.completed and result.phase == "official_frame"
    resolved = json.loads((Path(result.run_dir) / "resolved.json").read_text())
    assert resolved["method_freeze_contract"]["sha256"] == sha256_file(contract)
    assert resolved["official_gate_sha256"][str(Path(resolved["method_freeze_contract"]["path"]).resolve())]
    document = json.loads((Path(result.run_dir) / "result.json").read_text())
    assert document["method_freeze_contract"]["sha256"] == sha256_file(contract)
    rows = [json.loads(line) for line in Path(result.predictions).read_text().splitlines()]
    assert rows and all(row["metadata"]["score_level"] == "frame" for row in rows)


def test_official_frame_rejects_wrong_contract_sha(tmp_path, monkeypatch):
    run, features, manifest, _c, _a, _r = _source(tmp_path)
    _retag_head_formal(run)
    contract = _freeze_contract(tmp_path)
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", _fake_scores)
    with pytest.raises(ValueError, match="method freeze contract SHA differs"):
        run_urdmu_evaluation(
            _official_request(tmp_path, run, features, manifest, contract, "3" * 64)
        )


def test_official_frame_rejects_encoder_outside_frozen_scope(tmp_path, monkeypatch):
    run, features, manifest, _c, _a, _r = _source(tmp_path)
    _retag_head_formal(run)
    contract = _freeze_contract(tmp_path, encoders=("videomaev2",))
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", _fake_scores)
    with pytest.raises(ValueError, match="outside the frozen method scope"):
        run_urdmu_evaluation(
            _official_request(tmp_path, run, features, manifest, contract, sha256_file(contract))
        )


def test_official_frame_rejects_backend_outside_frozen_scope(tmp_path, monkeypatch):
    run, features, manifest, _c, _a, _r = _source(tmp_path)
    _retag_head_formal(run)
    contract = _freeze_contract(tmp_path, upstream="0" * 40)
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", _fake_scores)
    with pytest.raises(ValueError, match="outside the frozen method scope"):
        run_urdmu_evaluation(
            _official_request(tmp_path, run, features, manifest, contract, sha256_file(contract))
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(schema="icassp2027.method-freeze-contract/v2"), "schema differs"),
        (lambda d: d.update(status="draft"), "not active"),
        (lambda d: d["markdown_contract"].update(sha256="4" * 64), "markdown SHA differs"),
        (lambda d: d["property_evidence"][0].update(sha256="5" * 64), "property evidence receipt SHA differs"),
        (lambda d: d.update(property_evidence=[]), "lacks property evidence"),
    ],
)
def test_official_frame_rejects_invalid_contract_documents(tmp_path, monkeypatch, mutate, match):
    run, features, manifest, _c, _a, _r = _source(tmp_path)
    _retag_head_formal(run)
    contract = _freeze_contract(tmp_path, mutate=mutate)
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", _fake_scores)
    with pytest.raises(ValueError, match=match):
        run_urdmu_evaluation(
            _official_request(tmp_path, run, features, manifest, contract, sha256_file(contract))
        )


def test_official_frame_contract_required_but_forbidden_elsewhere(tmp_path):
    run, features, manifest, contract, _a, _r = _source(tmp_path)
    base = _request(tmp_path, run, features, manifest, contract).__dict__
    with pytest.raises(ValueError, match="method freeze contract"):
        URDMUEvaluationRequest(**{**base, "method_freeze_contract_path": "x", "method_freeze_contract_sha256": "y" * 64})
    with pytest.raises(ValueError, match="provided together"):
        URDMUEvaluationRequest(
            **{
                **base,
                "phase": "official_frame",
                "freeze_path": "f",
                "audit_report": "a",
                "method_freeze_contract_path": "x",
            }
        )
