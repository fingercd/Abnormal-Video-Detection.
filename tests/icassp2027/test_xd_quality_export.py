"""Synthetic-only XD exports. No real test GT, model scores or model execution."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_quality_export as helper
import test_xd_raw_alignment as raw_helper

from vadbench.data import xd_materialization
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.paper import evaluation
from vadbench.paper import xd_evaluation as xd
from vadbench.paper import xd_quality_export as export
from vadbench.paper.compatibility import RepresentationIdentity
from vadbench.research import xd_feature_grid as grid
from vadbench.research import xd_raw_alignment as raw_audit

pipeline = raw_helper.pipeline
synthetic_dataset = helper.synthetic_dataset
_bounded_cpu_threads = helper._bounded_cpu_threads
_write, _sha, _binding = helper._write, helper._sha, helper._binding


@pytest.fixture
def pair(tmp_path, synthetic_dataset, pipeline, monkeypatch):
    # Use a small subset of an existing synthetic source registry. Only trusted
    # fixture seals/counts and training-role membership change to this domain;
    # real source QA, permits, native-window and score verification all execute.
    data = SimpleNamespace(**vars(synthetic_dataset))
    data.fit = [replace(r, duration_seconds=4.0) for r in data.fit]
    data.select = [replace(r, duration_seconds=4.0) for r in data.select]
    helper._apply_seals(monkeypatch, data)
    dense_rep = RepresentationIdentity(
        data.backbone, {"name": "identity", "calibration": "none"}, 4, "float32", {"kind": "native"}
    )
    method_rep = replace(dense_rep, reducer={"name": "paired_random", "seed": 0})
    dense_head = helper._source_head(tmp_path / "dense-head", data, dense_rep)
    method_head = helper._source_head(tmp_path / "method-head", data, method_rep)
    # Exercise the real producer -> recursively verified consumer contract.
    # The injected probe writes fixed synthetic FFprobe/OpenCV artifacts only.
    audit_request, _, _ = pipeline
    metadata_path = Path(audit_request.canonical_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    metadata["gt_reference_length"] = 192
    for i, row in enumerate(metadata["records"]):
        row.update(t=6, gt_start=96 * i, gt_end_exclusive=96 * (i + 1))
    metadata["provenance"]["ti_table_sha256"] = grid.feature_grid_table_sha256(metadata["records"])
    _write(metadata_path, metadata)
    metadata_sha = grid.feature_grid_metadata_sha256(metadata)
    gt = Path(audit_request.prior_alignment_root) / "sealed-inputs/gt.npy"
    # Fine frame boundaries intentionally are not multiples of 16.
    np.save(gt, np.r_[np.zeros(13), np.ones(29), np.zeros(54), np.zeros(96)])
    annotations = gt.with_name("annotations.txt")
    annotations.write_text(metadata["records"][0]["video_id"] + " 13 42\n", encoding="utf8")
    for module in (grid, xd):
        monkeypatch.setattr(module, "OFFICIAL_VIDEO_COUNT", 2)
        monkeypatch.setattr(module, "OFFICIAL_GT_LENGTH", 192)
    monkeypatch.setattr(raw_audit, "CANONICAL_SHA", metadata_sha)
    monkeypatch.setattr(raw_audit, "GT_SHA", _sha(gt))
    monkeypatch.setattr(raw_audit, "ANNOTATION_SHA", _sha(annotations))
    monkeypatch.setattr(raw_audit, "METHOD_SHA", evaluation._canonical_sha256(data.freeze_doc))
    monkeypatch.setattr(xd, "CANONICAL_METADATA_SHA256", metadata_sha)
    monkeypatch.setattr(xd, "GT_FILE_SHA256", _sha(gt))

    def probe(path, directory, runtime):
        directory.mkdir(parents=True)
        _write(directory / "ffprobe.json", raw_helper._timing(100))
        _write(directory / "opencv.json", raw_helper._opencv(100))
        (directory / "ffprobe.stderr.txt").write_bytes(b"")

    audit_request = replace(audit_request, method_freeze=str(data.freeze))
    audit_result = raw_audit.run_raw_alignment_audit(
        audit_request, probe=probe, runtime_factory=lambda _: {"synthetic_engineering_only": True}
    )
    assert audit_result["status"] == "sealed"
    original_data_root = data.root
    data.root = Path(audit_request.raw_root)
    # Later tests create extra heads from the same synthetic fit/select records.
    for record in [*data.fit, *data.select]:
        destination = record.resolve_path(data.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record.resolve_path(original_data_root), destination)
    data.manifest = Path(audit_result["run_dir"]) / "private/test.jsonl"
    data.test = load_manifest_jsonl(data.manifest)
    data.audit = Path(audit_result["seal"]["path"])
    monkeypatch.setattr(
        xd, "METHOD_FREEZE_CANONICAL_SHA256", evaluation._canonical_sha256(data.freeze_doc)
    )
    # Toy encoder replaces only the public production encoder whitelist for this
    # synthetic adapter fixture; no production path gains a whitelist bypass.
    monkeypatch.setattr(
        xd.FrozenXDEvaluationRequest,
        "__post_init__",
        evaluation.FrozenEvaluationRequest.__post_init__,
    )
    scope = tmp_path / "scope.json"
    _write(
        scope,
        {
            "secondary_encoders": ["toy"],
            "methods": ["identity", "paired_random", "global_uniform", "pair_linear"],
            "head_seeds": [0, 1, 2],
            "global_uniform_head_seeds": [0],
            "derived_role_plan_file_sha256": _sha(data.lock),
            "original_role_lock_file_sha256": _sha(data.lock),
        },
    )
    monkeypatch.setattr(xd, "SCOPE_FILE_SHA256", _sha(scope))
    roles = {
        "fit": {"controller_records": load_manifest_jsonl(dense_head.train_manifest)},
        "select": {"controller_records": load_manifest_jsonl(dense_head.validation_manifest)},
    }
    monkeypatch.setattr(
        xd_materialization,
        "validate_xd_head_data_contract",
        lambda *a, **k: {
            "roles": roles,
            "source_hashes": {
                str(data.source_root / name): _sha(data.source_root / name)
                for name in ("fit.jsonl", "select.jsonl")
            },
        },
    )
    request = xd.FrozenXDEvaluationRequest(
        encoder="toy",
        device="cpu",
        dataset_root=str(data.root),
        test_manifest=str(data.manifest),
        audit_report=str(data.audit),
        method_source_run=str(dense_head.root),
        output_root=str(tmp_path),
        canonical_metadata=str(metadata_path),
        ground_truth=str(gt),
        raw_coordinate_receipt_sha256=_sha(data.audit),
        head_data_contract_sha256=_sha(data.contract),
        original_role_lock_path=str(data.lock),
        scope_path=str(scope),
        freeze_path=str(data.freeze),
        role_lock_path=str(data.lock),
        head_data_contract_path=str(data.contract),
        source_manifest_root=str(data.source_root),
    )
    coordinates = xd.verify_raw_coordinates(request)
    # Fixture writer uses the real XD scorer; no expected metrics are fabricated.
    monkeypatch.setattr(
        helper,
        "evaluate_manifest_predictions",
        lambda predictions, *a, **k: SimpleNamespace(
            to_dict=lambda: xd.score_xd_predictions(predictions, coordinates, gt)
        ),
    )
    source_kwargs = dict(
        freeze=data.freeze_doc,
        expected_encoder="toy",
        role_lock_path=data.lock,
        head_data_contract_path=data.contract,
        source_manifest_root=data.source_root,
        scope_path=scope,
        original_role_lock_path=data.lock,
        method_freeze_path=data.freeze,
        head_data_contract_sha256=_sha(data.contract),
    )
    dense_head = xd.load_frozen_xd_detector_source(dense_head.root, **source_kwargs)
    method_head = xd.load_frozen_xd_detector_source(method_head.root, **source_kwargs)

    def build(name, source, dense=None):
        root = helper._evaluation_run(tmp_path / name, data, source, dense_source=dense)
        stage_path = root / "provenance/stages/evaluation.json"
        stage = json.loads(stage_path.read_text(encoding="utf8"))
        config = vars(
            replace(
                request,
                method_source_run=str(source.root),
                dense_source_run=None if dense is None else str(dense.root),
            )
        )
        stage.update(
            stage="frozen_xd_evaluation",
            config=config,
            config_sha256=evaluation._canonical_sha256(config),
        )
        for key, path in (
            ("xd_scope", scope),
            ("xd_canonical_metadata", metadata_path),
            ("xd_gt", gt),
            ("xd_original_role_lock", data.lock),
        ):
            stage["inputs"][key] = {"location": str(path.resolve()), "sha256": _sha(path)}
        _write(stage_path, stage)
        resolved = json.loads((root / "resolved.json").read_text(encoding="utf8"))
        resolved["coverage"]["videos"] = 2
        _write(root / "resolved.json", resolved)
        helper._refresh(root)
        result = json.loads((root / "result.json").read_text(encoding="utf8"))
        result["protocol"] = xd.PROTOCOL_ID
        _write(root / "result.json", result)
        return root

    return SimpleNamespace(
        dense=build("dense", dense_head),
        method=build("method", method_head, dense_head),
        data=data,
        request=request,
        coordinates=coordinates,
        gt=gt,
        build=build,
        dense_head=dense_head,
        method_head=method_head,
        source_kwargs=source_kwargs,
    )


def test_completed_same_seed_primary_and_secondary(pair):
    for mode in ("refit_head", "direct_insert"):
        result = export.compare_frozen_xd_quality(pair.dense, pair.method, mode=mode)
        assert result["n_videos"] == 2 and result["n_frames"] == 192
        assert result["positive_frames"] == 29
        assert result["valid_draws"] == 10000 and result["delta_method_minus_dense"] == 0
        assert result["noninferiority"] == "supported" and result["head_seed"] == 0
        assert result["metric"] == "frame_pr_auc"
        assert "not full official train3954" in result["training_scope"]


@pytest.mark.parametrize(
    "mutation", ["pending", "ledger", "permit", "checkpoint", "metric", "gap", "stage"]
)
def test_tampered_completed_run_is_rejected(pair, mutation):
    root = pair.method
    if mutation == "pending":
        path = root / "result.json"
        doc = json.loads(path.read_text(encoding="utf8"))
        doc["status"] = "pending"
        _write(path, doc)
    elif mutation == "stage":
        path = root / "provenance/stages/evaluation.json"
        doc = json.loads(path.read_text(encoding="utf8"))
        doc["config"]["encoder"] = "foreign"
        _write(path, doc)
    elif mutation == "metric":
        path = root / "metrics-primary-refit-head.json"
        doc = json.loads(path.read_text(encoding="utf8"))
        doc["frame_pr_auc"] += 0.01
        _write(path, doc)
        helper._refresh(root, "primary_metrics")
    else:
        path = root / "predictions-primary-refit-head.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        if mutation == "permit":
            rows[0]["metadata"]["paper_compatibility"] = {}
        elif mutation == "checkpoint":
            rows[0]["metadata"]["checkpoint_sha256"] = "0" * 64
        elif mutation == "gap":
            rows.pop(0)
        else:
            rows[0]["anomaly_score"] = 0.123
        helper._write_index(path, rows)
        if mutation != "ledger":
            helper._refresh(root, "primary_predictions")
    with pytest.raises(ValueError):
        export.compare_frozen_xd_quality(pair.dense, root)


@pytest.mark.parametrize("mutation", ["missing", "stride", "anchor", "native_frames"])
def test_actual_native_windows_not_just_union_are_verified(pair, mutation):
    root = pair.method
    index = root / "features/test/index.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
    if mutation == "missing":
        rows.pop(1)
    elif mutation == "stride":
        rows[-1]["metadata"]["sampling"]["actual_frame_stride"] = 1
    elif mutation == "anchor":
        rows[-1]["metadata"]["sampling"]["end_anchored"] = False
    else:
        rows[0]["metadata"]["sampling"]["frame_indices"] = [0] * 16
    helper._write_index(index, rows)
    status_path = root / "features/test/status.json"
    status = json.loads(status_path.read_text(encoding="utf8"))
    status["records_written_to_shards"] = len(rows)
    _write(status_path, status)
    resolved_path = root / "resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf8"))
    for name in ("index", "status"):
        resolved["artifacts"]["test_feature_store"][name] = _binding(
            root / "features/test" / ("index.jsonl" if name == "index" else "status.json")
        )
    _write(resolved_path, resolved)
    helper._refresh(root)
    with pytest.raises(ValueError, match="sampler|window|frame_indices"):
        export.compare_frozen_xd_quality(pair.dense, root)


def test_gt_seal_and_head_qa_are_required(pair):
    np.save(pair.gt, np.zeros(192))
    with pytest.raises(ValueError, match="SHA"):
        export.compare_frozen_xd_quality(pair.dense, pair.method)


def test_prefix_keeps_fine_gt_and_rejects_missing_raw_suffix(pair):
    dense = export._load_completed_run(pair.dense, "refit_head")
    videos = export.paired_xd_frame_intervals(
        pair.coordinates, pair.gt, dense.records, dense.records
    )
    positive = next(v for v in videos if v.weak_label)
    assert 13 in positive.intervals and 42 in positive.intervals
    altered = [replace(r, frame_end=96) if r.frame_end == 100 else r for r in dense.records]
    with pytest.raises(ValueError):
        export.paired_xd_frame_intervals(pair.coordinates, pair.gt, dense.records, altered)


def test_seed_summary_and_cli_keep_videos_per_seed(pair, tmp_path):
    result = export.compare_frozen_xd_quality(pair.dense, pair.method)
    values = [
        {
            **result,
            "head_seed": seed,
            "dense_head": {
                **result["dense_head"],
                "head_seed": seed,
                "checkpoint_sha256": str(seed) * 64,
            },
            "method_head": {
                **result["method_head"],
                "head_seed": seed,
                "checkpoint_sha256": str(seed + 3) * 64,
            },
        }
        for seed in (0, 1, 2)
    ]
    summary = export.summarize_xd_quality_seeds(values)
    assert summary["n_videos_per_seed"] == 2 and summary["head_seed_count"] == 3
    assert summary["best_seed_selection"] is False
    path = Path(__file__).parents[2] / "scripts/icassp2027/compare_frozen_xd_quality.py"
    spec = importlib.util.spec_from_file_location("xd_quality_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "export"
    assert (
        cli.main(
            [
                "--dense-run",
                str(pair.dense),
                "--method-run",
                str(pair.method),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        json.loads((output / "summary.json").read_text(encoding="utf8"))["n_videos_per_seed"] == 2
    )


def test_failed_source_qa_is_rejected(pair):
    path = pair.method_head.root / "head/training_qa.json"
    document = json.loads(path.read_text(encoding="utf8"))
    document["status"] = "failed"
    _write(path, document)
    with pytest.raises(ValueError, match="QA did not pass"):
        export.compare_frozen_xd_quality(pair.dense, pair.method)


def test_mismatched_seed_and_nonexact_dense_checkpoint_fail(pair, tmp_path):
    seed1 = helper._source_head(
        tmp_path / "other-seed", pair.data, pair.method_head.representation, seed=1
    )
    seed1 = xd.load_frozen_xd_detector_source(seed1.root, **pair.source_kwargs)
    with pytest.raises(ValueError, match="heads differ in seed"):
        export.compare_frozen_xd_quality(pair.dense, pair.build("wrong-seed", seed1))
    other = helper._source_head(
        tmp_path / "other-dense", pair.data, pair.dense_head.representation, weight=2.0
    )
    other = xd.load_frozen_xd_detector_source(other.root, **pair.source_kwargs)
    with pytest.raises(ValueError, match="exact dense baseline head"):
        export.compare_frozen_xd_quality(
            pair.dense,
            pair.build("wrong-checkpoint", pair.method_head, other),
            mode="direct_insert",
        )


def test_actual_repeat_loader_and_cache_routes(pair, tmp_path, monkeypatch):
    from vadbench.artifacts import PredictionRecord
    from vadbench.paper import xd_head_repeats as heads
    from vadbench.paper import xd_repeat_evaluation as repeats
    from vadbench.paper.compatibility import validate_compatibility

    def make_repeat(name, base, seed):
        root = tmp_path / name
        # Reuse only the deterministic fixture writer; then run the real XD
        # repeat loader over its checkpoint, QA, source and stage bindings.
        with monkeypatch.context() as patch:
            patch.setattr(
                helper.repeat_evaluation, "load_frozen_repeat_head_source", lambda *a, **k: None
            )
            helper._repeat_head(root, pair.data, base, seed)
        result_path = root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf8"))
        result["kind"] = "frozen_xd_head_repeat_v1"
        result["xd_scope"] = _binding(Path(pair.request.scope_path))
        result["xd_original_role_lock"] = _binding(Path(pair.request.original_role_lock_path))
        _write(result_path, result)
        stage_path = root / "provenance/stages/repeat.json"
        stage = json.loads(stage_path.read_text(encoding="utf8"))
        stage["stage"] = "frozen_xd_head_repeat"
        stage["config_sha256"] = evaluation._canonical_sha256(stage["config"])
        _write(stage_path, stage)
        return heads.load_frozen_xd_repeat_head_source(root, **pair.source_kwargs)

    dense = make_repeat("dense-repeat1", pair.dense_head, 1)
    method = make_repeat("method-repeat1", pair.method_head, 1)

    def predict(config, *, feature_store, evaluation_manifest, training, output_path, device):
        cached = (
            pair.dense
            if config.declaration.evaluation_representation.reducer["name"] == "identity"
            else pair.method
        )
        records = [
            PredictionRecord.from_dict(json.loads(line))
            for line in (cached / "predictions-primary-refit-head.jsonl")
            .read_text(encoding="utf8")
            .splitlines()
        ]
        permit = validate_compatibility(config.declaration)
        records = [
            replace(
                r,
                run_id=Path(output_path).parent.name,
                metadata={
                    **r.metadata,
                    "checkpoint_sha256": _sha(Path(training)),
                    "paper_compatibility": permit,
                },
            )
            for r in records
        ]
        helper._write_index(Path(output_path), [r.to_dict() for r in records])
        return records

    monkeypatch.setattr(repeats, "predict_detector", predict)

    def execute(name, head, source, dense_head=None):
        request = repeats.FrozenXDRepeatEvaluationRequest(
            encoder="toy",
            repeat_head_run=str(head.root),
            source_evaluation_run=str(source),
            output_root=str(tmp_path),
            dense_head_repeat_run=None if dense_head is None else str(dense_head.root),
            run_id=name,
        )
        return Path(repeats.run_frozen_xd_repeat_evaluation(request).run_dir)

    dense_run = execute("dense-seed1", dense, pair.dense)
    method_run = execute("method-seed1", method, pair.method, dense)
    for mode in ("refit_head", "direct_insert"):
        result = export.compare_frozen_xd_quality(dense_run, method_run, mode=mode)
        assert (
            result["head_seed"]
            == result["dense_head"]["head_seed"]
            == result["method_head"]["head_seed"]
            == 1
        )
        assert result["n_videos"] == 2 and result["delta_method_minus_dense"] == 0.0
        assert result["method_head"]["kind"] == "frozen_xd_head_repeat_v1"
    with pytest.raises(ValueError, match="heads differ in seed"):
        export.compare_frozen_xd_quality(pair.dense, method_run)
    path = pair.method / "provenance/stages/evaluation.json"
    stage = json.loads(path.read_text(encoding="utf8"))
    stage["status"] = "failed"
    _write(path, stage)
    with pytest.raises(ValueError, match="completed seed-0 XD stage"):
        export.compare_frozen_xd_quality(dense_run, method_run)


def test_complete_probe_artifact_tamper_is_rejected_before_quality(pair):
    seal_root = Path(pair.request.audit_report).parent
    artifact = seal_root / "private/probes/0000/ffprobe.json"
    artifact.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="SHA|seal|timing"):
        export.compare_frozen_xd_quality(pair.dense, pair.method)


def test_late_raw_byte_mutation_is_rejected_before_gt_value_read(pair, monkeypatch):
    dense = export._load_completed_run(pair.dense, "refit_head")
    raw = pair.coordinates.manifest[0].resolve_path(pair.request.dataset_root)
    raw.write_bytes(raw.read_bytes() + b"late synthetic mutation")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("GT opened after raw mutation"))
    with pytest.raises(ValueError, match="SHA"):
        export.paired_xd_frame_intervals(pair.coordinates, pair.gt, dense.records, dense.records)


def test_two_coherent_feature_stores_cannot_substitute_same_unsealed_raw_bytes(pair):
    # Both methods consistently claim the same changed bytes. Their semantic
    # fingerprints, source digests and actual row metadata are all recomputed,
    # so equality between methods or FeatureStore self-consistency is insufficient.
    for root in (pair.dense, pair.method):
        resolved_path = root / "resolved.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf8"))
        folder = root / "features/test"
        document_path = folder / "resolved.json"
        document = json.loads(document_path.read_text(encoding="utf8"))
        evidence = document["data_content_evidence"]
        video_id = evidence["videos"][0]["video_id"]
        evidence["videos"][0]["sha256"] = "f" * 64
        evidence["source_digest"] = helper.compute_encoder_fingerprint(
            {
                "canonical_manifest": evidence["canonical_manifest_sha256"],
                "video_contents": evidence["videos"],
            }
        )
        sampling = replace(
            export.SamplingIdentity.from_mapping(resolved["evaluation_sampling"]),
            source_digest=evidence["source_digest"],
        )
        representation = RepresentationIdentity.from_mapping(resolved["evaluation_representation"])
        paper_identity = helper._paper_identity(representation, sampling)
        document["spec"]["sampling"] = sampling.to_dict()
        document["paper_identity"] = paper_identity
        document["runtime"]["sampling_fingerprint"] = sampling.fingerprint
        document["runtime"]["source_data_digest"] = sampling.source_digest
        fingerprint = helper.compute_encoder_fingerprint(
            {"paper_pooled_cache": helper._semantic_runtime_identity(document["runtime"])}
        )
        document["encoder_fingerprint"] = fingerprint
        _write(document_path, document)
        rows = [
            json.loads(line)
            for line in (folder / "index.jsonl").read_text(encoding="utf8").splitlines()
        ]
        for row in rows:
            row["encoder_fingerprint"] = fingerprint
            row["metadata"]["paper_identity"] = paper_identity
            if row["video_id"] == video_id:
                row["metadata"]["source_video"]["content_sha256"] = "f" * 64
        helper._write_index(folder / "index.jsonl", rows)
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text(encoding="utf8"))
        status["encoder_fingerprint"] = fingerprint
        _write(status_path, status)
        resolved.update(
            evaluation_sampling=sampling.to_dict(),
            evaluation_sampling_fingerprint=sampling.fingerprint,
            evaluation_encoder_fingerprint=fingerprint,
        )
        for name, filename in (
            ("resolved", "resolved.json"),
            ("index", "index.jsonl"),
            ("status", "status.json"),
        ):
            resolved["artifacts"]["test_feature_store"][name] = _binding(folder / filename)
        _write(resolved_path, resolved)
        helper._refresh(root)
    with pytest.raises(
        ValueError, match="extracted video SHA/bytes differ from the raw-coordinate seal"
    ):
        export.compare_frozen_xd_quality(pair.dense, pair.method)
