"""Decision-support evidence: engineering-shard merged views in formal training.

Q1/Q2: constituent-run data_role is never inspected — an engineering-mode
extraction run merged into a complete view passes every gate of
``_merged_official_dense_source`` when identity/target/SHA bindings match the
official view. The only data_role check in the formal path is on the *native
dense extraction contract the request binds*, never on the merged runs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_official_extraction import _CV, _Adapter
from test_urdmu_merged_official import _records

from vadbench.checkpoints import sha256_file
from vadbench.data import official_training
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.paper import urdmu_training as training
from vadbench.paper.feature_merge import (
    FeatureRunMergeRequest,
    run_feature_merge,
)
from vadbench.paper.official_extraction import (
    OfficialDenseExtractionRequest,
    run_official_dense_extraction,
)


def _authority(tmp_path, monkeypatch, video_ids):
    """Monkeypatched official training view + protocol + source contract."""
    records = []
    for index, video_id in enumerate(video_ids):
        path = tmp_path / f"{video_id}.mp4"
        path.touch()
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
                    "original_role": "fit" if index % 2 == 0 else "select",
                    "training_role": "official-fulltrain-final",
                    "content_sha256": sha256_file(path),
                    "content_size_bytes": path.stat().st_size,
                },
            )
        )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "icassp2027.official-detector-protocol/v2",
                "datasets": {"ucf_crime": {"encoders": ["videomaev2"]}},
            }
        )
    )
    view_contract = tmp_path / "view-contract.json"
    view_contract.write_text("{}", encoding="utf-8")

    def load(path, expected_sha256, **_kwargs):
        assert Path(path) == view_contract
        return tuple(records), {
            "dataset": "ucf_crime",
            "dataset_root": str(tmp_path),
            "inputs": {},
        }

    monkeypatch.setattr(official_training, "load_official_training_view", load)
    return records, protocol, view_contract


def _engineering_run(tmp_path, protocol, view_contract, video_ids, run_id):
    adapter = _Adapter()
    request = OfficialDenseExtractionRequest(
        encoder="videomaev2",
        device="cpu",
        training_contract_path=str(view_contract),
        training_contract_sha256=sha256_file(view_contract),
        protocol_path=str(protocol),
        protocol_sha256=sha256_file(protocol),
        output_root=str(tmp_path / "eng-runs"),
        run_id=run_id,
        engineering_video_ids=tuple(video_ids),
    )

    def factory(_request):
        return adapter, {
            "identity": {
                "adapter": "videomaev2",
                "constructor": {"clip_frames": 4},
                "checkpoint": {"sha256": {"fixture.bin": "fixture"}},
            }
        }

    result = run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    assert result["status"] == "completed"
    assert result["data_role"] == "engineering-subset-of-official-fulltrain"
    document = json.loads(Path(result["contract"]["path"]).read_text())
    return document["feature_store"]["root"]


def _native_fulltrain_contract(tmp_path, anchor_root, view_contract, manifest, protocol):
    anchor = Path(anchor_root)
    contract = tmp_path / "native-fulltrain-contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "icassp2027.official-dense-extraction/v1",
                "status": "ready",
                "data_role": "official-fulltrain-final",
                "dataset": "ucf_crime",
                "training_view_contract": {
                    "path": str(view_contract.resolve()),
                    "sha256": sha256_file(view_contract),
                },
                "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
                "feature_store": {
                    "root": str(anchor),
                    **{
                        name: {"path": str(anchor / filename), "sha256": sha256_file(anchor / filename)}
                        for name, filename in (
                            ("resolved", "resolved.json"),
                            ("status", "status.json"),
                            ("index", "index.jsonl"),
                        )
                    },
                },
                "protocol": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
            }
        ),
        encoding="utf-8",
    )
    return contract


def test_engineering_runs_merged_view_passes_formal_dense_source(tmp_path, monkeypatch):
    """Q1/Q2 evidence: engineering-role runs carry no gate in the formal path."""
    video_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
    _authority_records, protocol, view_contract = _authority(tmp_path, monkeypatch, video_ids)
    first = _engineering_run(tmp_path, protocol, view_contract, video_ids[:2], "eng-a")
    second = _engineering_run(tmp_path, protocol, view_contract, video_ids[2:], "eng-b")
    merged = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=tuple(video_ids),
            run_roots=(first, second),
            output_root=str(tmp_path / "merged"),
            run_id="eng-merged",
            transport="copy",
        )
    )
    merged_root = Path(merged["run_dir"])
    records = _records(tmp_path, video_ids)
    manifest = write_manifest_jsonl(records, tmp_path / "train.jsonl")
    native_contract = _native_fulltrain_contract(
        tmp_path, first, view_contract, manifest, protocol
    )
    request = training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=str(merged_root),
        train_manifest=str(manifest),
        source_contract_path=str(view_contract),
        source_contract_sha256=sha256_file(view_contract),
        upstream_dir="not-accessed",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="formal",
        steps=3000,
        bags_per_class=64,
        protocol_path=str(protocol),
        extraction_contract_path=str(native_contract),
        extraction_contract_sha256=sha256_file(native_contract),
        feature_view_contract_path=str(merged_root / "merge-contract.json"),
        feature_view_contract_sha256=sha256_file(merged_root / "merge-contract.json"),
    )
    document, representation, sampling, receipt, sequences = training._dense_source(
        request, records, {}, None
    )
    assert document["merged_view"] is True
    assert receipt["merged_view"] is True
    assert len(sequences) == 4
    assert {s["video_id"] for s in sequences} == set(video_ids)


def test_formal_without_any_contract_rejected():
    with pytest.raises(ValueError, match="dense extraction contract or a bound feature view"):
        training.URDMUTrainingRequest(
            dataset="ucf_crime",
            feature_store="x",
            train_manifest="y",
            source_contract_path="z",
            source_contract_sha256="a" * 64,
            upstream_dir="u",
            output_root="o",
            device="cpu",
            run_mode="formal",
        )


def _equivalence_merge(tmp_path, monkeypatch, role_equivalence=True, authority=None, declared=None):
    video_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
    _records_authority, protocol, view_contract = _authority(tmp_path, monkeypatch, video_ids)
    first = _engineering_run(tmp_path, protocol, view_contract, video_ids[:2], "eq-a")
    second = _engineering_run(tmp_path, protocol, view_contract, video_ids[2:], "eq-b")
    selected_authority = authority if authority is not None else view_contract
    merged = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=tuple(video_ids),
            run_roots=(first, second),
            output_root=str(tmp_path / "merged"),
            run_id="eq-view",
            transport="copy",
            role_equivalence=(
                declared
                if declared is not None
                else ("engineering-shards-of-official-fit" if role_equivalence else None)
            ),
            authority_contract_path=str(selected_authority) if (role_equivalence or declared) else None,
            authority_contract_sha256=(
                sha256_file(selected_authority) if (role_equivalence or declared) else None
            ),
        )
    )
    return merged, view_contract, protocol, video_ids


def _formal_equivalence_request(tmp_path, merged, view_contract, protocol, video_ids, **overrides):
    merged_root = Path(merged["run_dir"])
    records = _records(tmp_path, video_ids)
    manifest = write_manifest_jsonl(records, tmp_path / "train.jsonl")
    values = {
        "dataset": "ucf_crime",
        "feature_store": str(merged_root),
        "train_manifest": str(manifest),
        "source_contract_path": str(view_contract),
        "source_contract_sha256": sha256_file(view_contract),
        "upstream_dir": "not-accessed",
        "output_root": str(tmp_path / "runs"),
        "device": "cpu",
        "run_mode": "formal",
        "steps": 3000,
        "bags_per_class": 64,
        "protocol_path": str(protocol),
        "feature_view_contract_path": str(merged_root / "merge-contract.json"),
        "feature_view_contract_sha256": sha256_file(merged_root / "merge-contract.json"),
    }
    values.update(overrides)
    return training.URDMUTrainingRequest(**values), records


def test_role_equivalence_view_backs_formal_training_without_native_contract(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(tmp_path, monkeypatch)
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    assert document["role_equivalence"]["declared"] == "engineering-shards-of-official-fit"
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    result = training._dense_source(request, records, {}, None)
    assert result[0]["merged_view"] is True
    assert result[3]["role_equivalence"] is not None
    assert result[3]["native_contract"] is None
    assert len(result[4]) == 4


def test_role_equivalence_requires_the_declaration(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, role_equivalence=False
    )
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    with pytest.raises(ValueError, match="role equivalence declaration"):
        training._dense_source(request, records, {}, None)


def test_role_equivalence_authority_must_match_the_requested_source(tmp_path, monkeypatch):
    decoy = tmp_path / "decoy-authority.json"
    decoy.write_text("{}", encoding="utf-8")
    merged, _view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, authority=decoy
    )
    request, records = _formal_equivalence_request(
        tmp_path, merged, _view_contract, protocol, video_ids
    )
    with pytest.raises(ValueError, match="role equivalence authority differs"):
        training._dense_source(request, records, {}, None)


def test_role_equivalence_rejects_moved_source_runs(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(tmp_path, monkeypatch)
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    source_run = Path(document["source_runs"][0]["root"])
    tampered = json.loads((source_run / "resolved.json").read_text())
    tampered["spec"]["micro_batch_size"] = 999
    (source_run / "resolved.json").write_text(json.dumps(tampered), encoding="utf-8")
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    with pytest.raises(ValueError, match="receipt changed"):
        training._dense_source(request, records, {}, None)


def _role_lock_setup(tmp_path):
    authority_dir = tmp_path / "authority"
    lock_dir = authority_dir / "sources"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / "role_lock.json"
    lock.write_text(json.dumps({"partitions": {}}), encoding="utf-8")
    authority = authority_dir / "view-contract.json"
    authority.write_text("{}", encoding="utf-8")
    return authority, lock


def _equivalence_merge_with_lock(
    tmp_path, monkeypatch, lock=None, lock_sha="auto", lock_setup=True
):
    video_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
    _auth_records, protocol, _view = _authority(tmp_path, monkeypatch, video_ids)
    if lock_setup:
        authority, authority_lock = _role_lock_setup(tmp_path)
    else:
        authority = tmp_path / "authority" / "view-contract.json"
        authority.parent.mkdir(parents=True, exist_ok=True)
        authority.write_text("{}", encoding="utf-8")
        authority_lock = authority.parent / "sources" / "role_lock.json"
    first = _engineering_run(tmp_path, protocol, _view, video_ids[:2], "lock-a")
    second = _engineering_run(tmp_path, protocol, _view, video_ids[2:], "lock-b")
    selected_lock = lock if lock is not None else authority_lock
    request_kwargs = {
        "target_video_ids": tuple(video_ids),
        "run_roots": (first, second),
        "output_root": str(tmp_path / "merged"),
        "run_id": "lock-view",
        "transport": "copy",
        "role_equivalence": "engineering-shards-of-official-fit",
        "authority_contract_path": str(authority),
        "authority_contract_sha256": sha256_file(authority),
    }
    if lock_sha == "auto":
        request_kwargs["original_role_lock_path"] = str(selected_lock)
        request_kwargs["original_role_lock_sha256"] = sha256_file(selected_lock)
    elif lock_sha is not None:
        request_kwargs["original_role_lock_path"] = str(selected_lock)
        request_kwargs["original_role_lock_sha256"] = lock_sha
    merged = run_feature_merge(FeatureRunMergeRequest(**request_kwargs))
    return merged, authority, selected_lock, protocol, video_ids


def test_merge_records_original_role_lock_and_development_gate_passes(tmp_path, monkeypatch):
    merged, _authority, lock, protocol, video_ids = _equivalence_merge_with_lock(
        tmp_path, monkeypatch
    )
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    expected_lock = {"path": str(lock.resolve()), "sha256": sha256_file(lock)}
    assert document["original_role_lock"] == expected_lock
    records = _records(tmp_path, video_ids)
    manifest = write_manifest_jsonl(records, tmp_path / "train.jsonl")
    request = training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=merged["run_dir"],
        train_manifest=str(manifest),
        source_contract_path=str(_authority),
        source_contract_sha256=sha256_file(_authority),
        upstream_dir="not-accessed",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="development",
        steps=3000,
        bags_per_class=64,
        protocol_path=str(protocol),
        feature_view_contract_path=str(Path(merged["run_dir"]) / "merge-contract.json"),
        feature_view_contract_sha256=sha256_file(Path(merged["run_dir"]) / "merge-contract.json"),
    )
    result = training._dense_source(request, records, {}, {"development_role_lock": expected_lock})
    assert result[0]["merged_view"] is True


def test_development_gate_rejects_missing_lock(tmp_path, monkeypatch):
    merged, authority, _lock, protocol, video_ids = _equivalence_merge_with_lock(
        tmp_path, monkeypatch, lock_sha=None, lock_setup=False
    )
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    assert document["original_role_lock"] is None
    records = _records(tmp_path, video_ids)
    request = training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=merged["run_dir"],
        train_manifest=str(write_manifest_jsonl(records, tmp_path / "train.jsonl")),
        source_contract_path=str(authority),
        source_contract_sha256=sha256_file(authority),
        upstream_dir="not-accessed",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="development",
        steps=3000,
        bags_per_class=64,
        protocol_path=str(protocol),
        feature_view_contract_path=str(Path(merged["run_dir"]) / "merge-contract.json"),
        feature_view_contract_sha256=sha256_file(Path(merged["run_dir"]) / "merge-contract.json"),
    )
    with pytest.raises(ValueError, match="development role lock differs"):
        training._dense_source(
            request, records, {}, {"development_role_lock": {"path": "x", "sha256": "y"}}
        )


def test_development_gate_rejects_wrong_lock(tmp_path, monkeypatch):
    merged, authority, lock, protocol, video_ids = _equivalence_merge_with_lock(
        tmp_path, monkeypatch
    )
    records = _records(tmp_path, video_ids)
    request = training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=merged["run_dir"],
        train_manifest=str(write_manifest_jsonl(records, tmp_path / "train.jsonl")),
        source_contract_path=str(authority),
        source_contract_sha256=sha256_file(authority),
        upstream_dir="not-accessed",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="development",
        steps=3000,
        bags_per_class=64,
        protocol_path=str(protocol),
        feature_view_contract_path=str(Path(merged["run_dir"]) / "merge-contract.json"),
        feature_view_contract_sha256=sha256_file(Path(merged["run_dir"]) / "merge-contract.json"),
    )
    wrong_lock = {"path": str(lock.resolve()), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="development role lock differs"):
        training._dense_source(request, records, {}, {"development_role_lock": wrong_lock})


def test_merge_lock_sha_mismatch_and_default_resolution(tmp_path, monkeypatch):
    from vadbench.paper.feature_merge import FeatureMergeError

    with pytest.raises(FeatureMergeError, match="role lock SHA-256 differs"):
        _equivalence_merge_with_lock(tmp_path, monkeypatch, lock_sha="1" * 64)
    # default resolution: no --path, sha given, sources/role_lock.json exists
    merged, _authority2, lock2, _protocol2, _videos = _default_lock_merge(tmp_path, monkeypatch)
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    assert document["original_role_lock"] == {
        "path": str(lock2.resolve()),
        "sha256": sha256_file(lock2),
    }


def _default_lock_merge(tmp_path, monkeypatch):
    video_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]
    _auth_records, protocol, _view = _authority(tmp_path, monkeypatch, video_ids)
    authority, authority_lock = _role_lock_setup(tmp_path)
    first = _engineering_run(tmp_path, protocol, _view, video_ids[:2], "def-a")
    second = _engineering_run(tmp_path, protocol, _view, video_ids[2:], "def-b")
    merged = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=tuple(video_ids),
            run_roots=(first, second),
            output_root=str(tmp_path / "merged"),
            run_id="default-lock-view",
            transport="copy",
            role_equivalence="engineering-shards-of-official-fit",
            authority_contract_path=str(authority),
            authority_contract_sha256=sha256_file(authority),
            original_role_lock_sha256=sha256_file(authority_lock),
        )
    )
    return merged, authority, authority_lock, protocol, video_ids


def test_patch_tool_adds_lock_preserving_original(tmp_path, monkeypatch):
    from vadbench.paper.feature_merge import patch_view_contract

    merged, authority, lock, _protocol, video_ids = _equivalence_merge_with_lock(
        tmp_path, monkeypatch
    )
    view_root = Path(merged["run_dir"])
    original_contract = view_root / "merge-contract.json"
    original_sha = sha256_file(original_contract)
    result = patch_view_contract(
        view_root=view_root,
        original_role_lock_path=lock,
        original_role_lock_sha256=sha256_file(lock),
    )
    assert sha256_file(original_contract) == original_sha
    patched_path = Path(result["contract"]["path"])
    assert patched_path.name == "merge-contract.v2.json"
    patched = json.loads(patched_path.read_text())
    assert patched["original_role_lock"] == {
        "path": str(lock.resolve()),
        "sha256": sha256_file(lock),
    }
    assert patched["patched_from"]["sha256"] == original_sha
    # the development gate accepts the patched contract
    records = _records(tmp_path, video_ids)
    request = training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=str(view_root),
        train_manifest=str(write_manifest_jsonl(records, tmp_path / "train.jsonl")),
        source_contract_path=str(authority),
        source_contract_sha256=sha256_file(authority),
        upstream_dir="not-accessed",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="development",
        steps=3000,
        bags_per_class=64,
        protocol_path=str(_protocol),
        feature_view_contract_path=str(patched_path),
        feature_view_contract_sha256=sha256_file(patched_path),
    )
    expected_lock = {"path": str(lock.resolve()), "sha256": sha256_file(lock)}
    result = training._dense_source(
        request, records, {}, {"development_role_lock": expected_lock}
    )
    assert result[0]["merged_view"] is True


def test_patch_tool_rejects_bad_inputs(tmp_path, monkeypatch):
    from vadbench.paper.feature_merge import FeatureMergeError, patch_view_contract

    merged, _authority, lock, _protocol, _videos = _equivalence_merge_with_lock(
        tmp_path, monkeypatch
    )
    view_root = Path(merged["run_dir"])
    with pytest.raises(FeatureMergeError, match="SHA-256 differs"):
        patch_view_contract(
            view_root=view_root,
            original_role_lock_path=lock,
            original_role_lock_sha256="2" * 64,
        )
    with pytest.raises(FeatureMergeError, match="refuses to overwrite"):
        patch_view_contract(
            view_root=view_root,
            original_role_lock_path=lock,
            original_role_lock_sha256=sha256_file(lock),
            output_name="merge-contract.json",
        )
    # a view without the role-equivalence declaration cannot be patched
    source_run = json.loads((view_root / "merge-contract.json").read_text())["source_runs"][0]["root"]
    plain = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=("sample-a", "sample-b"),
            run_roots=(source_run,),
            output_root=str(tmp_path / "plain-merged"),
            run_id="plain-view",
            transport="copy",
        )
    )
    with pytest.raises(FeatureMergeError, match="recognized role-equivalence"):
        patch_view_contract(
            view_root=plain["run_dir"],
            original_role_lock_path=lock,
            original_role_lock_sha256=sha256_file(lock),
        )


FULLTRAIN_DECLARED = "engineering-shards-of-official-fulltrain"


def test_fulltrain_declaration_accepted_on_formal_complete_view(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, declared=FULLTRAIN_DECLARED
    )
    document = json.loads((Path(merged["run_dir"]) / "merge-contract.json").read_text())
    assert document["role_equivalence"]["declared"] == FULLTRAIN_DECLARED
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    # quarantine-aware accepted count supplied by the authority receipt
    result = training._dense_source(
        request, records, {}, {"accepted_members": len(records)}
    )
    assert result[0]["merged_view"] is True


def test_fulltrain_declaration_accepts_quarantine_nested_count(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, declared=FULLTRAIN_DECLARED
    )
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    result = training._dense_source(
        request,
        records,
        {},
        {"quarantine_exemption": {"accepted_members": len(records)}},
    )
    assert result[0]["merged_view"] is True


def test_fulltrain_declaration_rejected_without_complete_coverage(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, declared=FULLTRAIN_DECLARED
    )
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    with pytest.raises(ValueError, match="complete official training view"):
        training._dense_source(request, records, {}, {"accepted_members": len(records) + 1})
    with pytest.raises(ValueError, match="target list differs"):
        training._dense_source(
            request,
            records[: len(records) - 1],
            {},
            {"accepted_members": len(records) - 1},
        )


def test_fulltrain_declaration_is_formal_only(tmp_path, monkeypatch):
    merged, view_contract, protocol, video_ids = _equivalence_merge(
        tmp_path, monkeypatch, declared=FULLTRAIN_DECLARED
    )
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids, run_mode="development"
    )
    with pytest.raises(ValueError, match="formal path only"):
        training._dense_source(
            request, records, {}, {"accepted_members": len(records)}
        )


def test_patch_tool_rewrites_declaration_to_fulltrain(tmp_path, monkeypatch):
    from vadbench.paper.feature_merge import FeatureMergeError, patch_view_contract

    merged, view_contract, protocol, video_ids = _equivalence_merge(tmp_path, monkeypatch)
    view_root = Path(merged["run_dir"])
    original_sha = sha256_file(view_root / "merge-contract.json")
    result = patch_view_contract(
        view_root=view_root,
        role_equivalence=FULLTRAIN_DECLARED,
        authority_contract_path=view_contract,
        authority_contract_sha256=sha256_file(view_contract),
        output_name="merge-contract.fulltrain.json",
    )
    assert sha256_file(view_root / "merge-contract.json") == original_sha
    patched_path = Path(result["contract"]["path"])
    patched = json.loads(patched_path.read_text())
    assert patched["role_equivalence"]["declared"] == FULLTRAIN_DECLARED
    assert patched["patched_from"]["role_equivalence_before"] == (
        "engineering-shards-of-official-fit"
    )
    assert patched["role_equivalence"]["authority_contract"]["sha256"] == sha256_file(view_contract)
    # the patched contract is consumable on the formal path
    request, records = _formal_equivalence_request(
        tmp_path, merged, view_contract, protocol, video_ids
    )
    request = replace(
        request,
        feature_view_contract_path=str(patched_path),
        feature_view_contract_sha256=sha256_file(patched_path),
    )
    result = training._dense_source(
        request, records, {}, {"accepted_members": len(records)}
    )
    assert result[0]["merged_view"] is True
    # guard rails
    with pytest.raises(FeatureMergeError, match="at least one patch"):
        patch_view_contract(view_root=view_root, output_name="nothing.json")
    with pytest.raises(FeatureMergeError, match="fulltrain role-equivalence target"):
        patch_view_contract(
            view_root=view_root,
            role_equivalence="engineering-shards-of-official-fit",
            authority_contract_path=view_contract,
            authority_contract_sha256=sha256_file(view_contract),
        )
