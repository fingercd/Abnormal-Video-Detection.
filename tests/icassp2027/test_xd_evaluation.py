"""CPU synthetic contract checks only; none of these scores are research results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_xd_raw_alignment import _opencv, _timing, pipeline  # noqa: F401

from vadbench.paper import xd_evaluation as xd
from vadbench.research import xd_feature_grid as grid


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf8")
    return path


@pytest.fixture
def sealed(pipeline, monkeypatch):  # noqa: F811
    """Generate the complete proof graph with the actual dataset-only producer."""
    from vadbench.research import xd_raw_alignment as producer

    preparation, _, _ = pipeline

    def probe(path, directory, runtime):
        directory.mkdir(parents=True)
        _write(directory / "ffprobe.json", _timing(35))
        _write(directory / "opencv.json", _opencv(35))
        (directory / "ffprobe.stderr.txt").write_bytes(b"")

    result = producer.run_raw_alignment_audit(
        preparation, probe=probe, runtime_factory=lambda _: {}
    )
    assert result["status"] == "sealed"
    root = Path(result["run_dir"])
    for name, value in (
        ("CANONICAL_METADATA_SHA256", producer.CANONICAL_SHA),
        ("GT_FILE_SHA256", producer.GT_SHA),
        ("OFFICIAL_VIDEO_COUNT", 2),
        ("OFFICIAL_GT_LENGTH", 64),
    ):
        monkeypatch.setattr(xd, name, value)
    request = xd.FrozenXDEvaluationRequest(
        encoder="videomaev2",
        device="cpu",
        dataset_root=preparation.raw_root,
        test_manifest=str(root / "private/test.jsonl"),
        audit_report=result["seal"]["path"],
        method_source_run=str(root / "unused-source"),
        output_root=str(root / "evaluation"),
        canonical_metadata=preparation.canonical_metadata,
        ground_truth=str(Path(preparation.prior_alignment_root) / "sealed-inputs/gt.npy"),
        raw_coordinate_receipt_sha256=result["seal"]["sha256"],
        head_data_contract_sha256="a" * 64,
        original_role_lock_path=str(root / "unused-lock"),
    )
    return request, json.loads(Path(request.audit_report).read_text(encoding="utf8"))


def _reseal(request, receipt):
    _write(Path(request.audit_report), receipt)
    return replace(request, raw_coordinate_receipt_sha256=_sha(request.audit_report))


def _metadata_only_sealed(case, monkeypatch):
    """Produce a real dataset-only seal carrying the exact accepted diagnostic."""
    from vadbench.research import xd_raw_alignment as producer

    preparation, _, _ = case
    diagnostic = (
        b"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7fd60e013700] "
        b"Referenced QT chapter track not found\n"
    )

    def probe(path, directory, runtime):
        directory.mkdir(parents=True)
        _write(directory / "ffprobe.json", _timing(35))
        _write(directory / "opencv.json", _opencv(35))
        (directory / "ffprobe.stderr.txt").write_bytes(diagnostic)

    result = producer.run_raw_alignment_audit(
        preparation, probe=probe, runtime_factory=lambda _: {}
    )
    assert result["status"] == "sealed"
    root = Path(result["run_dir"])
    for name, value in (
        ("CANONICAL_METADATA_SHA256", producer.CANONICAL_SHA),
        ("GT_FILE_SHA256", producer.GT_SHA),
        ("OFFICIAL_VIDEO_COUNT", 2),
        ("OFFICIAL_GT_LENGTH", 64),
    ):
        monkeypatch.setattr(xd, name, value)
    request = xd.FrozenXDEvaluationRequest(
        encoder="videomaev2",
        device="cpu",
        dataset_root=preparation.raw_root,
        test_manifest=str(root / "private/test.jsonl"),
        audit_report=result["seal"]["path"],
        method_source_run=str(root / "unused-source"),
        output_root=str(root / "evaluation"),
        canonical_metadata=preparation.canonical_metadata,
        ground_truth=str(Path(preparation.prior_alignment_root) / "sealed-inputs/gt.npy"),
        raw_coordinate_receipt_sha256=result["seal"]["sha256"],
        head_data_contract_sha256="a" * 64,
        original_role_lock_path=str(root / "unused-lock"),
    )
    return request, json.loads(Path(request.audit_report).read_text(encoding="utf8")), root


def test_sealed_metadata_only_ffprobe_diagnostic_round_trips_to_coordinate_consumer(
    pipeline, monkeypatch  # noqa: F811 - imported pytest fixture is injected by name
):
    request, _, root = _metadata_only_sealed(pipeline, monkeypatch)
    probe = json.loads((root / "private/probes/0000/receipt.json").read_text(encoding="utf8"))
    assert probe["ffprobe_stderr_classification"] == "metadata_only_qt_chapter_track_missing"
    assert len(xd.verify_raw_coordinates(request).manifest) == 2


def test_sealed_metadata_diagnostic_rejects_tampered_receipt_classification(pipeline, monkeypatch):  # noqa: F811
    request, receipt, root = _metadata_only_sealed(pipeline, monkeypatch)
    probe_path = root / "private/probes/0000/receipt.json"
    probe = json.loads(probe_path.read_text(encoding="utf8"))
    probe["ffprobe_stderr_classification"] = "none"
    _write(probe_path, probe)
    timing_path = root / "audits/raw_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf8"))
    timing["probe_receipt_sha256"]["private/probes/0000/receipt.json"] = _sha(probe_path)
    _write(timing_path, timing)
    receipt["audits"]["raw_timing"]["sha256"] = _sha(timing_path)
    with pytest.raises(ValueError, match="stderr classification"):
        xd.verify_raw_coordinates(_reseal(request, receipt))


def _predictions():
    return [
        SimpleNamespace(
            video_id=f"opaque{i}",
            clip_index=j,
            frame_start=start,
            frame_end=end,
            anomaly_score=score if i == 0 else 0.1,
        )
        for i in range(2)
        for j, (start, end, score) in enumerate(
            [(0, 3, 0.1), (3, 10, 0.9), (10, 32, 0.1), (32, 35, 0.99)]
        )
    ]


def test_prefix_scoring_uses_real_frames_and_drops_only_suffix(sealed):
    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    result = xd.score_xd_predictions(_predictions(), coordinates, request.ground_truth)
    assert result["frame_pr_auc"] == 1.0
    assert result["gt_length"] == 64
    assert result["prediction_representation"] == "raw_frame_dense_overlap_mean"


@pytest.mark.parametrize(
    "mutation", ["pending", "short", "fps", "pts", "suffix", "order", "name", "missing"]
)
def test_raw_coordinate_failures_prevent_score_access(sealed, monkeypatch, mutation):
    request, receipt = sealed
    if mutation == "pending":
        receipt["status"] = "pending"
    elif mutation == "missing":
        receipt["records"].pop()
    elif mutation == "order":
        receipt["records"].reverse()
    elif mutation == "name":
        receipt["records"][0]["feature_name"] = "other__0.npy"
    else:
        field, value = {
            "short": ("num_frames", 16),
            "fps": ("fps", 30.0),
            "pts": ("pts_verified", False),
            "suffix": ("suffix_audit_passed", False),
        }[mutation]
        receipt["records"][0][field] = value
    request = _reseal(request, receipt)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("GT values opened before raw gate"))
    with pytest.raises(ValueError):
        xd.verify_raw_coordinates(request)


def test_seal_and_raw_bytes_cannot_be_changed(sealed):
    request, receipt = sealed
    receipt["status"] = "pending"
    _write(Path(request.audit_report), receipt)
    with pytest.raises(ValueError, match="external seal"):
        xd.verify_raw_coordinates(request)
    request, receipt = request, {**receipt, "status": "sealed"}
    request = _reseal(request, receipt)
    (Path(request.dataset_root) / receipt["records"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="XD raw video SHA"):
        xd.verify_raw_coordinates(request)


@pytest.mark.parametrize("mutation", ["gap", "overlap", "foreign", "nan"])
def test_prediction_coverage_fails_before_gt_values(sealed, monkeypatch, mutation):
    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    records = _predictions()
    if mutation == "gap":
        records.pop(0)
    elif mutation == "overlap":
        records[1].frame_start = 2
    elif mutation == "foreign":
        records[0].video_id = "foreign"
    else:
        records[0].anomaly_score = float("nan")
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("GT values opened before coverage gate")
    )
    with pytest.raises(ValueError):
        xd.score_xd_predictions(records, coordinates, request.ground_truth)


def test_audit_tamper_and_gt_checksum_rejected(sealed):
    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    (Path(request.audit_report).parent / "audits/raw_timing.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="external seal"):
        xd.score_xd_predictions(_predictions(), coordinates, request.ground_truth)


def test_inactive_xd_encoders_rejected(sealed):
    request, _ = sealed
    with pytest.raises(ValueError, match="only videomaev2 and timesformer"):
        replace(request, encoder="vjepa2")


def test_pending_raw_gate_stops_before_checkpoint_or_adapter(sealed, monkeypatch):
    request, receipt = sealed
    receipt["status"] = "pending"
    request = _reseal(request, receipt)
    monkeypatch.setattr(
        xd, "load_frozen_xd_detector_source", lambda *a, **k: pytest.fail("checkpoint opened")
    )
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("GT values opened"))
    with pytest.raises(ValueError, match="pending"):
        xd.run_frozen_xd_evaluation(request, adapter_factory=lambda *a: pytest.fail("model loaded"))
    assert not Path(request.output_root).exists()


def test_completed_audit_attestation_cannot_be_missing(sealed):
    request, receipt = sealed
    audit_path = Path(request.audit_report).parent / "audits/raw_timing.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    audit["model_scores_read"] = True
    _write(audit_path, audit)
    receipt["audits"]["raw_timing"]["sha256"] = _sha(audit_path)
    with pytest.raises(ValueError, match="raw_timing audit"):
        xd.verify_raw_coordinates(_reseal(request, receipt))


def test_frame_predictions_are_not_averaged_into_sixteen_frame_bins(sealed):
    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    assert (
        xd.score_xd_predictions(_predictions(), coordinates, request.ground_truth)["frame_pr_auc"]
        == 1.0
    )
    # The positive interval is strictly inside one 16-frame bin. Averaging and
    # repeating that bin would score unrelated negative frames equally.
    hypothetical = np.r_[np.repeat((3 * 0.1 + 7 * 0.9 + 6 * 0.1) / 16, 16), np.repeat(0.1, 48)]
    assert grid.precision_recall_trapezoid_auc(np.load(request.ground_truth), hypothetical) < 1.0


def test_actual_metadata_freeze_matches_code_anchor():
    path = Path(
        "outputs/icassp2027/xd_feature_grid_freeze/20260917T212610Z-497e8abc/frozen-grid-metadata.json"
    )
    if not path.exists():
        pytest.skip("local read-only frozen metadata mirror is unavailable")
    metadata = json.loads(path.read_text(encoding="utf8"))
    assert grid.feature_grid_metadata_sha256(metadata) == xd.CANONICAL_METADATA_SHA256
    assert len(grid._verified_rows(metadata, xd.CANONICAL_METADATA_SHA256)[0]) == 800


@pytest.mark.parametrize("mutation", ["remove_probe", "change_pts"])
def test_sealed_timing_proof_graph_cannot_be_deleted_or_modified(sealed, mutation):
    request, _ = sealed
    directory = Path(request.audit_report).parent / "private/probes/0000"
    if mutation == "remove_probe":
        (directory / "receipt.json").unlink()
    else:
        path = directory / "ffprobe.json"
        document = json.loads(path.read_text(encoding="utf8"))
        document["frames"][1] = dict(document["frames"][0])
        _write(path, document)
    with pytest.raises((ValueError, FileNotFoundError)):
        xd.verify_raw_coordinates(request)


def test_raw_change_after_gate_is_detected_before_gt_values(sealed, monkeypatch):
    request, receipt = sealed
    coordinates = xd.verify_raw_coordinates(request)
    (Path(request.dataset_root) / receipt["records"][0]["path"]).write_bytes(b"changed after gate")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("GT opened after raw changed"))
    with pytest.raises(ValueError, match="external seal"):
        xd.score_xd_predictions(_predictions(), coordinates, request.ground_truth)


def test_even_resealed_invalid_pts_cannot_pass_timing_recomputation(sealed):
    request, receipt = sealed
    root = Path(request.audit_report).parent
    directory = root / "private/probes/0000"
    path = directory / "ffprobe.json"
    document = json.loads(path.read_text(encoding="utf8"))
    document["frames"][1] = dict(document["frames"][0])
    _write(path, document)
    probe_path = directory / "receipt.json"
    probe = json.loads(probe_path.read_text(encoding="utf8"))
    probe["artifact_sha256"]["ffprobe.json"] = _sha(path)
    _write(probe_path, probe)
    timing_path = root / "audits/raw_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf8"))
    timing["probe_receipt_sha256"]["private/probes/0000/receipt.json"] = _sha(probe_path)
    _write(timing_path, timing)
    receipt["audits"]["raw_timing"]["sha256"] = _sha(timing_path)
    with pytest.raises(ValueError, match="PTS_NOT_STRICTLY_INCREASING"):
        xd.verify_raw_coordinates(_reseal(request, receipt))


def test_extracted_content_evidence_must_match_sealed_raw_bytes(sealed):
    from vadbench.paper.extraction import make_data_content_evidence

    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    evidence = make_data_content_evidence(coordinates.manifest, dataset_root=request.dataset_root)
    xd.validate_xd_data_content_evidence(evidence, coordinates)
    evidence["videos"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA/bytes differ"):
        xd.validate_xd_data_content_evidence(evidence, coordinates)


def test_xd_runner_binds_extraction_callback_to_coordinates(sealed, monkeypatch):
    from vadbench.paper import evaluation
    from vadbench.paper.extraction import make_data_content_evidence

    request, _ = sealed
    coordinates = xd.verify_raw_coordinates(request)
    evidence = make_data_content_evidence(coordinates.manifest, dataset_root=request.dataset_root)

    def run_shared(request, **kwargs):
        kwargs["validate_extraction"]({"data_content_evidence": evidence})
        evidence["videos"][0]["size_bytes"] += 1
        with pytest.raises(ValueError, match="SHA/bytes differ"):
            kwargs["validate_extraction"]({"data_content_evidence": evidence})
        return "extraction callback checked"

    monkeypatch.setattr(evaluation, "_run_frozen_evaluation", run_shared)
    assert xd.run_frozen_xd_evaluation(request) == "extraction callback checked"


@pytest.fixture
def xd_source(tmp_path, monkeypatch):
    from test_frozen_evaluation import _freeze_document, _source_run

    from vadbench.data import xd_materialization
    from vadbench.data.manifest import load_manifest_jsonl
    from vadbench.paper import evaluation

    source, _ = _source_run(tmp_path, "xd-source", {"name": "identity"})
    freeze = _freeze_document()
    freeze_path = _write(tmp_path / "method-freeze.json", freeze)
    plan = _write(tmp_path / "derived-plan.json", {"purpose": "synthetic XD roles"})
    original = _write(tmp_path / "original-lock.json", {"purpose": "synthetic original lock"})
    contract = _write(tmp_path / "xd-contract.json", {"status": "synthetic fixture"})
    scope = _write(
        tmp_path / "scope.json",
        {
            "secondary_encoders": ["toy"],
            "derived_role_plan_file_sha256": _sha(plan),
            "original_role_lock_file_sha256": _sha(original),
            "methods": ["identity"],
            "head_seeds": [0, 1, 2],
            "global_uniform_head_seeds": [0],
        },
    )
    monkeypatch.setattr(xd, "SCOPE_FILE_SHA256", _sha(scope))
    monkeypatch.setattr(xd, "METHOD_FREEZE_CANONICAL_SHA256", evaluation._canonical_sha256(freeze))
    # Contract membership validation has its own data-layer tests. Supply its
    # independently captured controller records here, retaining the real
    # FeatureStore, checkpoint, training-QA and provenance verifiers below it.
    roles = {
        role: {"controller_records": load_manifest_jsonl(source / "frozen" / filename)}
        for role, filename in (("fit", "train.jsonl"), ("select", "val.jsonl"))
    }
    monkeypatch.setattr(
        xd_materialization,
        "validate_xd_head_data_contract",
        lambda *a, **k: {"roles": roles, "source_hashes": {}},
    )
    kwargs = {
        "freeze": freeze,
        "expected_encoder": "toy",
        "role_lock_path": plan,
        "head_data_contract_path": contract,
        "source_manifest_root": tmp_path,
        "scope_path": scope,
        "original_role_lock_path": original,
        "method_freeze_path": freeze_path,
        "head_data_contract_sha256": _sha(contract),
    }
    return source, kwargs


def test_xd_source_reuses_actual_checkpoint_qa_and_feature_store_verification(xd_source):
    source, kwargs = xd_source
    loaded = xd.load_frozen_xd_detector_source(source, **kwargs)
    assert loaded.training_identity.seed == 0
    loaded.verify_unchanged()
    loaded.checkpoint.write_bytes(b"tampered after verification")
    with pytest.raises(ValueError, match="changed after verification"):
        loaded.verify_unchanged()


def test_xd_source_rejects_incomplete_controller_role_before_checkpoint_qa(xd_source):
    source, kwargs = xd_source
    path = source / "frozen" / "train.jsonl"
    path.write_text(path.read_text(encoding="utf8").splitlines()[0] + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="fixed controller contract"):
        xd.load_frozen_xd_detector_source(source, **kwargs)


def test_xd_source_rejects_failed_head_qa(xd_source):
    source, kwargs = xd_source
    path = source / "head" / "training_qa.json"
    qa = json.loads(path.read_text(encoding="utf8"))
    qa["status"] = "failed"
    _write(path, qa)
    with pytest.raises(ValueError, match="QA did not pass"):
        xd.load_frozen_xd_detector_source(source, **kwargs)
