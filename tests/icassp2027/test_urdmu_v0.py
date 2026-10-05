"""v0_partial_cache training and v0_quality scoring over merged feature views.

v0 is an engineering small-sample diagnostic: these tests pin its identity
semantics, explicit below-formal budget, contract chain and overlap recording,
and prove it cannot impersonate formal/development runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_feature_merge import _merge
from test_paper_extraction import (
    _CV2,
    _Adapter,
    _record,
    _sampling,
    _spec,
    _touch,
)

from vadbench.checkpoints import sha256_file
from vadbench.contracts import EncoderOutput, TokenTimeline
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.paper.extraction import extract_pooled_features
from vadbench.paper.urdmu_evaluation import URDMUEvaluationRequest, run_urdmu_evaluation
from vadbench.paper.urdmu_training import (
    V0_DATA_ROLE,
    V0_VIEW_SCHEMA,
    URDMUTrainingRequest,
)


class _Adapter768(_Adapter):
    """Fixture encoder emitting real D768 pooled vectors for the UR-DMU head."""

    def encode(self, batch, train=False):
        assert not train
        assert all(item.startswith("sample-") for item in batch.video_ids)
        values = np.asarray(batch.frame_indices, dtype=np.float32).mean(axis=1)
        pooled = np.zeros((batch.batch_size, 768), dtype=np.float32)
        pooled[:, :4] = np.stack([values + offset for offset in range(4)], axis=1)
        tokens = 3
        features = np.repeat(pooled[:, None, :], tokens, axis=1)
        starts = np.zeros((batch.batch_size, tokens), dtype=np.float32)
        ends = np.ones((batch.batch_size, tokens), dtype=np.float32)
        timeline = TokenTimeline(
            start_s=starts, end_s=ends, valid_mask=np.ones_like(starts, dtype=bool)
        )
        return EncoderOutput(features=features, pooled=pooled, timeline=timeline)


def _representation768(adapter):
    from test_paper_extraction import _verified_identity

    from vadbench.paper.extraction import representation_from_verified_encoder

    return representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": "identity"},
        output_dim=768,
        precision="float32",
        position_strategy={"kind": "native"},
    )


def _run768(tmp_path: Path, name: str, video_ids: list[str]):
    records = [
        _record(video_id, split="train", anomaly=index % 2 == 1)
        for index, video_id in enumerate(video_ids)
    ]
    _touch(tmp_path, records)
    adapter = _Adapter768()
    representation = _representation768(adapter)
    sampling = _sampling(records, root=tmp_path, regime="test_dense", clips=32, window_stride=2)
    spec = _spec(representation, sampling, kind="dense", stride=2)
    return extract_pooled_features(
        spec,
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id=name,
        backend=_CV2(64),
    )


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def _records(tmp_path: Path, video_ids: list[str]) -> list[VideoManifestRecord]:
    records = []
    for index, video_id in enumerate(video_ids):
        path = tmp_path / f"{video_id}.mp4"
        # Keep the bytes identical to the extraction runs (empty fixture files)
        # so the manifest content evidence matches the merge provenance.
        path.touch(exist_ok=True)
        records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=path.name,
                split="train",
                category="Anomaly" if index % 2 else "Normal",
                is_anomaly=index % 2 == 1,
                num_frames=64,
                fps=8.0,
                duration_seconds=8.0,
                metadata={
                    "content_sha256": sha256_file(path),
                    "content_size_bytes": path.stat().st_size,
                },
            )
        )
    return records


@pytest.fixture
def view(tmp_path):
    """Two completed shard runs merged into one v0 view with its contracts."""
    first = _run768(tmp_path, "run-a", ["sample-a", "sample-b"])
    second = _run768(tmp_path, "run-b", ["sample-c", "sample-d"])
    assert first.completed and second.completed
    merged = _merge(
        tmp_path,
        (first, second),
        ["sample-a", "sample-b", "sample-c", "sample-d"],
        run_id="v0-view",
    )
    root = Path(merged["run_dir"])
    records = _records(tmp_path, ["sample-a", "sample-b", "sample-c", "sample-d"])
    manifest = write_manifest_jsonl(records, tmp_path / "v0-train.jsonl")
    role = _write(tmp_path / "v0-roles.json", {record.video_id: "engineering_train" for record in records})
    base_contract = {
        "schema": V0_VIEW_SCHEMA,
        "status": "ready",
        "data_role": V0_DATA_ROLE,
        "dataset": "ucf_crime",
        "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "source_role_contract": {"path": str(role), "sha256": sha256_file(role)},
        "feature_view": {
            "path": str(root / "merge-contract.json"),
            "sha256": sha256_file(root / "merge-contract.json"),
        },
        "evaluation_overlap": {
            "video_ids": ["sample-b"],
            "note": "diagnostic overlap with the development select view",
        },
    }
    # The training contract is what the head binds; the score (held-out)
    # contract references it through training_view_contract.
    contract = _write(tmp_path / "v0-view-contract.json", base_contract)
    score_contract = _write(
        tmp_path / "v0-score-contract.json",
        {
            **base_contract,
            "training_view_contract": {"path": str(contract), "sha256": sha256_file(contract)},
        },
    )
    return records, manifest, root, contract, merged, score_contract


def _request(tmp_path, view, **overrides):
    _records, manifest, root, contract, _merged, _score = view
    values = {
        "dataset": "ucf_crime",
        "feature_store": str(root),
        "train_manifest": str(manifest),
        "source_contract_path": str(contract),
        "source_contract_sha256": sha256_file(contract),
        "upstream_dir": "not-accessed-until-training",
        "output_root": str(tmp_path / "runs"),
        "device": "cpu",
        "run_mode": "v0_partial_cache",
        "steps": 2,
        "bags_per_class": 1,
        "run_id": "v0-train",
    }
    values.update(overrides)
    return URDMUTrainingRequest(**values)


def test_v0_budget_must_stay_below_formal_contract(tmp_path, view):
    base = _request(tmp_path, view)
    with pytest.raises(ValueError, match="below the formal"):
        replace(base, steps=3000)
    with pytest.raises(ValueError, match="below the formal"):
        replace(base, bags_per_class=64)


def test_v0_cannot_take_a_dense_extraction_contract(tmp_path, view):
    base = _request(tmp_path, view)
    with pytest.raises(ValueError, match="merged feature view"):
        replace(base, extraction_contract_path="contract.json", extraction_contract_sha256="a" * 64)


def test_v0_allows_explicit_seed_and_any_smaller_budget(tmp_path, view):
    request = _request(tmp_path, view, seed=7, steps=3, bags_per_class=2)
    assert request.seed == 7 and request.steps == 3


def test_v0_rejects_wrong_contract_schema(tmp_path, view):
    _records, manifest, root, contract, _merged, _score = view
    wrong = _write(
        tmp_path / "wrong.json",
        {
            "schema": "urdmu.engineering-training-view/v1",
            "status": "ready",
            "data_role": "engineering_only",
            "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "source_role_contract": {"path": str(tmp_path / "v0-roles.json"), "sha256": sha256_file(tmp_path / "v0-roles.json")},
        },
    )
    with pytest.raises(ValueError, match="v0 partial-cache view contract"):
        from vadbench.paper import urdmu_training as training

        training._source(_request(tmp_path, view, source_contract_path=str(wrong), source_contract_sha256=sha256_file(wrong)))


def test_v0_requires_both_real_classes(tmp_path, view):
    records, manifest, root, contract, _merged, _score = view
    normal_only = [replace(record, is_anomaly=False, category="Normal") for record in records]
    manifest = write_manifest_jsonl(normal_only, tmp_path / "v0-normal-only.jsonl")
    role = tmp_path / "v0-roles.json"
    bound = _write(
        tmp_path / "v0-normal-only-contract.json",
        {
            "schema": V0_VIEW_SCHEMA,
            "status": "ready",
            "data_role": V0_DATA_ROLE,
            "dataset": "ucf_crime",
            "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "source_role_contract": {"path": str(role), "sha256": sha256_file(role)},
            "feature_view": {
                "path": str(root / "merge-contract.json"),
                "sha256": sha256_file(root / "merge-contract.json"),
            },
            "evaluation_overlap": {"video_ids": []},
        },
    )
    from vadbench.paper import urdmu_training as training

    with pytest.raises(ValueError, match="real normal and anomalous"):
        training._source(
            _request(
                tmp_path,
                view,
                train_manifest=str(manifest),
                source_contract_path=str(bound),
                source_contract_sha256=sha256_file(bound),
            )
        )


@pytest.fixture
def upstream():
    root = Path(
        os.environ.get(
            "URDMU_UPSTREAM_DIR", "outputs/icassp2027/research/author-recipes-20260918/UR-DMU"
        )
    ).resolve()
    if not root.is_dir():
        pytest.skip("pinned external UR-DMU checkout unavailable")
    extra = None
    if importlib.util.find_spec("einops") is None:
        site = Path("C:/Users/lenovo/anaconda3/envs/pytorch/Lib/site-packages")
        if not (site / "einops").is_dir():
            pytest.skip("an existing real einops installation is required")
        extra = str(site)
        sys.path.append(extra)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield root
    finally:
        torch.set_num_threads(threads)
        if extra is not None:
            sys.path.remove(extra)


def test_v0_training_end_to_end_records_identity_and_overlap(tmp_path, monkeypatch, view, upstream):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    from vadbench.paper.urdmu_training import run_urdmu_training

    result = run_urdmu_training(
        _request(tmp_path, view, upstream_dir=str(upstream), steps=2, bags_per_class=1)
    )
    assert result["status"] == "completed"
    assert result["run_mode"] == "v0_partial_cache"
    assert result["data_role"] == V0_DATA_ROLE
    assert result["v0_diagnostic"] is True
    assert result["video_counts"] == {"total": 4, "normal": 2, "anomaly": 2}
    assert result["evaluation_overlap"]["video_ids"] == ["sample-b"]
    assert result["optimizer_steps"] == 2

    run = Path(result["run_dir"])
    payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    assert metadata["run_mode"] == "v0_partial_cache"
    assert metadata["data_role"] == V0_DATA_ROLE
    assert metadata["steps"] == 2 and metadata["bags_per_class"] == 1
    assert metadata["v0_diagnostic"] is True
    assert metadata["merged_view_digest"]
    receipt = json.loads((run / "result.json").read_text())
    assert receipt["v0_diagnostic"] is True


def _fake_v0_head(tmp_path, view) -> Path:
    """A v0-trained-head receipt pair without running the optimizer."""
    import vadbench.paper.urdmu_evaluation as evaluation
    from vadbench.paper.urdmu_backend import UPSTREAM_COMMIT
    from vadbench.paper.urdmu_training import COMPONENTS

    _records, manifest, root, contract, _merged, _score = view
    resolved = json.loads((root / "resolved.json").read_text())
    identity = resolved["identity"]
    run = tmp_path / "v0-head"
    checkpoint = run / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    metadata = {
        "schema": "urdmu.training/v1",
        "status": "completed_training",
        "run_mode": "v0_partial_cache",
        "data_role": V0_DATA_ROLE,
        "development_role": None,
        "dataset": "ucf_crime",
        "checkpoint_role": "dense_reference",
        "representation": identity["spec"]["representation"],
        "representation_fingerprint": evaluation.RepresentationIdentity.from_mapping(
            identity["spec"]["representation"]
        ).fingerprint,
        "sampling": identity["spec"]["sampling"],
        "sampling_fingerprint": evaluation.SamplingIdentity.from_mapping(
            identity["spec"]["sampling"]
        ).fingerprint,
        "encoder_fingerprint": identity["merged_view_digest"],
        "merged_view_digest": identity["merged_view_digest"],
        "v0_diagnostic": True,
        "video_counts": {"total": 4, "normal": 2, "anomaly": 2},
        "evaluation_overlap": {"video_ids": ["sample-b"]},
        "source_training_view": {"view_contract": str(contract)},
        "source_sha256": {
            str(contract): sha256_file(contract),
            str(manifest): sha256_file(manifest),
        },
        "backend": {"upstream_commit": UPSTREAM_COMMIT},
        "checkpoint_selection": "fixed_final_step_no_test_selection",
        "steps": 2,
        "bags_per_class": 1,
        "seed": 0,
        "nonzero_gradient_steps": 2,
    }
    torch.save({"model_state_dict": {"weight": torch.ones(1)}, "metadata": metadata}, checkpoint)
    digest = sha256_file(checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": digest, "size_bytes": checkpoint.stat().st_size})
    _write(
        run / "training_qa.json",
        {
            "status": "passed",
            "checkpoint": {"path": str(checkpoint), "sha256": digest, "step": 2},
            "nonzero_gradient_steps": 2,
            "changed_parameters_by_component": {component: 1 for component in COMPONENTS},
            "reload_parity": {"exact_equal": True, "flag": "Test", "training": False},
        },
    )
    _write(
        run / "result.json",
        {"status": "completed", "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest},
    )
    _write(
        run / "provenance" / "stages" / "training.json",
        {
            "stage": "urdmu_training",
            "status": "completed",
            "config": {"run_mode": "v0_partial_cache", "train_manifest": str(manifest)},
            "inputs": {
                "train_manifest": {"location": str(manifest), "sha256": sha256_file(manifest)}
            },
        },
    )
    return run


def _retamper_checkpoint(head: Path, mutate) -> None:
    """Rewrite a fake head checkpoint and refresh every SHA receipt around it."""
    checkpoint = head / "checkpoints" / "final.pt"
    payload = torch.load(checkpoint, weights_only=False)
    mutate(payload["metadata"])
    torch.save(payload, checkpoint)
    digest = sha256_file(checkpoint)
    _write(
        checkpoint.with_suffix(".pt.json"),
        {"sha256": digest, "size_bytes": checkpoint.stat().st_size},
    )
    result = json.loads((head / "result.json").read_text())
    result["checkpoint_sha256"] = digest
    _write(head / "result.json", result)
    qa = json.loads((head / "training_qa.json").read_text())
    qa["checkpoint"]["sha256"] = digest
    _write(head / "training_qa.json", qa)


def _eval_request(tmp_path, view, head: Path, **overrides):
    _records, manifest, root, contract, _merged, _score = view
    values = {
        "dataset": "ucf_crime",
        "phase": "v0_quality",
        "mode": "direct_insert",
        "trained_run": str(head),
        "feature_store": str(root),
        "evaluation_manifest": str(manifest),
        "upstream_dir": "must-not-load-in-unit-test",
        "output_root": str(tmp_path / "scores"),
        "device": "cpu",
        "feature_contract_path": str(_score),
        "feature_contract_sha256": sha256_file(_score),
        "run_id": "v0-scores",
    }
    values.update(overrides)
    return URDMUEvaluationRequest(**values)


def test_v0_quality_scoring_writes_explicit_diagnostic_records(tmp_path, monkeypatch, view):
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view

    def fake_scores(source, target, request):
        return (
            {
                record.video_id: np.full(
                    len(target.rows_by_video[record.video_id]), float(record.is_anomaly)
                )
                for record in target.records
            },
            {},
        )

    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", fake_scores)
    result = run_urdmu_evaluation(_eval_request(tmp_path, view, head))
    assert result.completed
    assert result.video_metrics["scope"] == "v0_diagnostic_video_level_only"
    assert result.video_metrics["video_roc_auc"] == 1.0
    assert result.video_metrics["official_frame_scores_read"] is False
    assert result.video_metrics["v0_diagnostic"] is True
    assert result.video_metrics["evaluation_overlap"]["video_ids"] == ["sample-b"]

    rows = [json.loads(line) for line in Path(result.predictions).read_text(encoding="utf8").splitlines()]
    assert rows
    assert all(row["metadata"]["score_level"] == "frame" for row in rows)
    assert all("frame_coordinate_mapping" in row["metadata"] for row in rows)
    run = Path(result.run_dir)
    resolved = json.loads((run / "resolved.json").read_text())
    assert resolved["v0_diagnostic"] is True
    assert resolved["official_frame_scores_read"] is False
    assert resolved["evaluation_overlap"]["video_ids"] == ["sample-b"]
    assert (run / "v0-video-metrics.json").is_file()


def test_v0_quality_rejects_development_head(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)

    def mutate(metadata):
        metadata["run_mode"] = "development"
        metadata["data_role"] = "development-fit"
        metadata["steps"] = 3000
        metadata["bags_per_class"] = 64

    _retamper_checkpoint(head, mutate)
    with pytest.raises(ValueError, match="v0_partial_cache engineering head"):
        run_urdmu_evaluation(_eval_request(tmp_path, view, head))


def test_development_phase_rejects_v0_head(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)
    with pytest.raises(ValueError, match="not a native dense extraction"):
        run_urdmu_evaluation(
            _eval_request(tmp_path, view, head, phase="development_video")
        )


def test_v0_head_must_be_bound_to_the_view_contract(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view

    def unbind_contract(metadata):
        metadata["source_sha256"] = {
            key: value for key, value in metadata["source_sha256"].items() if key != str(contract)
        }

    _retamper_checkpoint(head, unbind_contract)
    with pytest.raises(ValueError, match="not bound to the training view contract"):
        run_urdmu_evaluation(_eval_request(tmp_path, view, head))


def test_v0_view_contract_must_record_evaluation_overlap(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view
    document = json.loads(_score.read_text())
    del document["evaluation_overlap"]
    _score.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation overlap"):
        run_urdmu_evaluation(
            _eval_request(
                tmp_path,
                view,
                head,
                feature_contract_sha256=sha256_file(_score),
            )
        )


def test_v0_score_contract_must_reference_the_heads_training_contract(tmp_path, view):
    """A score contract pointing at an unknown training contract fails closed."""
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view
    decoy = _write(tmp_path / "decoy-train-contract.json", {"schema": "decoy"})
    document = json.loads(_score.read_text())
    document["training_view_contract"] = {
        "path": str(decoy),
        "sha256": sha256_file(decoy),
    }
    _score.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to the training view contract"):
        run_urdmu_evaluation(
            _eval_request(
                tmp_path,
                view,
                head,
                feature_contract_sha256=sha256_file(_score),
            )
        )


def test_v0_score_contract_without_training_view_binding_rejected(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view
    document = json.loads(_score.read_text())
    del document["training_view_contract"]
    _score.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks its training view contract binding"):
        run_urdmu_evaluation(
            _eval_request(
                tmp_path,
                view,
                head,
                feature_contract_sha256=sha256_file(_score),
            )
        )


def test_v0_scoring_rejects_manifest_outside_the_merged_view(tmp_path, view):
    head = _fake_v0_head(tmp_path, view)
    _records, manifest, root, contract, _merged, _score = view
    outsider = write_manifest_jsonl(
        [replace(_records[0], video_id="sample-outsider")], tmp_path / "outsider.jsonl"
    )
    with pytest.raises(ValueError, match="target list differs"):
        run_urdmu_evaluation(
            _eval_request(tmp_path, view, head, evaluation_manifest=str(outsider))
        )


def test_v0_training_consumes_subset_view(tmp_path, monkeypatch, upstream):
    """A subset view is a first-class v0 feature view, not a native run."""
    from vadbench.paper.feature_merge import FeatureSubsetRequest, run_feature_subset
    from vadbench.paper.urdmu_training import run_urdmu_training

    source = _run768(tmp_path, "run-a", ["sample-a", "sample-b"])
    assert source.completed
    subset = run_feature_subset(
        FeatureSubsetRequest(
            source_run_root=source.run_dir,
            target_video_ids=("sample-a", "sample-b"),
            output_root=str(tmp_path / "subsets"),
            run_id="v0-subset",
            transport="copy",
        )
    )
    subset_root = Path(subset["run_dir"])
    records = _records(tmp_path, ["sample-a", "sample-b"])
    manifest = write_manifest_jsonl(records, tmp_path / "v0-subset-train.jsonl")
    role = _write(tmp_path / "v0-subset-roles.json", {r.video_id: "engineering_train" for r in records})
    contract = _write(
        tmp_path / "v0-subset-contract.json",
        {
            "schema": V0_VIEW_SCHEMA,
            "status": "ready",
            "data_role": V0_DATA_ROLE,
            "dataset": "ucf_crime",
            "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "source_role_contract": {"path": str(role), "sha256": sha256_file(role)},
            "feature_view": {
                "path": str(subset_root / "subset-contract.json"),
                "sha256": sha256_file(subset_root / "subset-contract.json"),
            },
            "evaluation_overlap": {"video_ids": []},
        },
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = run_urdmu_training(
        URDMUTrainingRequest(
            dataset="ucf_crime",
            feature_store=str(subset_root),
            train_manifest=str(manifest),
            source_contract_path=str(contract),
            source_contract_sha256=sha256_file(contract),
            upstream_dir=str(upstream),
            output_root=str(tmp_path / "runs"),
            device="cpu",
            run_mode="v0_partial_cache",
            steps=2,
            bags_per_class=1,
            run_id="v0-subset-train",
        )
    )
    assert result["status"] == "completed"
    assert result["run_mode"] == "v0_partial_cache"
    assert result["video_counts"] == {"total": 2, "normal": 1, "anomaly": 1}
    payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["metadata"]["merged_view_digest"] == subset["subset_view_digest"]
