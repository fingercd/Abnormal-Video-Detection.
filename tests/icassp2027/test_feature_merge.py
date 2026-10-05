"""Multi-run FeatureStore merge: disjoint coverage, identity, integrity, transport."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_paper_extraction import _CV2, _Adapter, _record, _representation, _sampling, _spec, _touch

from vadbench.checkpoints import sha256_file
from vadbench.features import FeatureStore
from vadbench.paper.extraction import extract_pooled_features
from vadbench.paper import feature_merge as feature_merge_module
from vadbench.paper.feature_merge import (
    MERGE_SCHEMA,
    FeatureMergeError,
    FeatureRunMergeRequest,
    run_feature_merge,
)


def _run(tmp_path: Path, name: str, video_ids: list[str], *, reducer: str = "identity"):
    records = [
        _record(video_id, split="train", anomaly=index % 2 == 1)
        for index, video_id in enumerate(video_ids)
    ]
    _touch(tmp_path, records)
    adapter = _Adapter()
    representation = _representation(adapter, reducer=reducer)
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


def _merge(tmp_path: Path, runs, videos, *, transport="copy", run_id="view"):
    return run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=tuple(videos),
            run_roots=tuple(run.run_dir for run in runs),
            output_root=str(tmp_path / "merged"),
            run_id=run_id,
            transport=transport,
        )
    )


def test_merge_disjoint_runs_copy_roundtrip(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a", "sample-b"])
    second = _run(tmp_path, "run-b", ["sample-c"])
    assert first.completed and second.completed

    result = _merge(tmp_path, (first, second), ["sample-a", "sample-b", "sample-c"])
    assert result["status"] == "completed" and result["complete"] is True
    destination = Path(result["run_dir"])
    store = FeatureStore(destination)
    rows = store.records()
    assert len(rows) == first.records_written + second.records_written
    assert {row.video_id for row in rows} == {"sample-a", "sample-b", "sample-c"}
    for row in rows:
        bundle = store.load_bundle(row)
        assert np.isfinite(bundle["features"]).all() and np.isfinite(bundle["pooled"]).all()

    contract = json.loads((destination / "merge-contract.json").read_text())
    assert contract["schema"] == MERGE_SCHEMA and contract["complete"] is True
    assert contract["transport"]["shared_blob_inodes"] is False
    provenance = contract["video_provenance"]
    assert provenance["sample-c"]["source_run"] == second.run_dir
    assert provenance["sample-a"]["source_run"] == first.run_dir
    assert provenance["sample-a"]["record_count"] > 0
    assert provenance["sample-a"]["content"]["sha256"]

    status = json.loads((destination / "status.json").read_text())
    assert status["completed"] is True and status["videos"] == 3
    resolved = json.loads((destination / "resolved.json").read_text())
    assert resolved["merged_view"] is True and resolved["complete"] is True
    # Per-run subset fingerprints stay distinct inside one honest merged view.
    assert len(resolved["identity"]["encoder_fingerprints"]) == 2

    # Transported shard bytes are identical to the completed source shards.
    for video in ("sample-a", "sample-b"):
        source_shard = Path(first.run_dir) / "shards" / video
        target_shard = destination / "shards" / video
        source_files = {p.relative_to(source_shard).as_posix(): p for p in source_shard.rglob("*") if p.is_file()}
        target_files = {p.relative_to(target_shard).as_posix(): p for p in target_shard.rglob("*") if p.is_file()}
        assert source_files.keys() == target_files.keys()
        assert all(source_files[name].read_bytes() == target_files[name].read_bytes() for name in source_files)


def test_merge_rejects_cross_run_conflict_and_lists_it(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    second = _run(tmp_path, "run-b", ["sample-a", "sample-c"])
    with pytest.raises(FeatureMergeError, match="more than one run"):
        _merge(tmp_path, (first, second), ["sample-a", "sample-c"])
    assert not (tmp_path / "merged" / "view").exists()


def test_merge_rejects_missing_target_and_leaves_no_view(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    with pytest.raises(FeatureMergeError, match="missing from every run"):
        _merge(tmp_path, (first,), ["sample-a", "sample-missing"])
    assert not (tmp_path / "merged" / "view").exists()


def test_merge_rejects_videos_outside_the_target_list(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a", "sample-b"])
    with pytest.raises(FeatureMergeError, match="outside the target list"):
        _merge(tmp_path, (first,), ["sample-a"])


def test_merge_rejects_partial_runs(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    status_path = Path(first.run_dir) / "status.json"
    status = json.loads(status_path.read_text())
    status["completed"] = False
    status["status"] = "partial_failed"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(FeatureMergeError, match="not complete"):
        _merge(tmp_path, (first,), ["sample-a"])


def test_merge_rejects_run_identity_drift(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    second = _run(tmp_path, "run-b", ["sample-b"])
    resolved_path = Path(second.run_dir) / "resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["spec"]["representation"]["reducer"] = {"name": "global_uniform"}
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    with pytest.raises(FeatureMergeError, match="identities differ"):
        _merge(tmp_path, (first, second), ["sample-a", "sample-b"])


def test_merge_hardlink_shares_blob_inodes_on_one_filesystem(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    second = _run(tmp_path, "run-b", ["sample-b"])
    result = _merge(tmp_path, (first, second), ["sample-a", "sample-b"], transport="hardlink_npz")
    destination = Path(result["run_dir"])
    assert result["transport"] == "hardlink_npz"
    for video, run in (("sample-a", first), ("sample-b", second)):
        source_shard = Path(run.run_dir) / "shards" / video
        target_shard = destination / "shards" / video
        source_blobs = sorted(p for p in source_shard.rglob("*.npz"))
        assert source_blobs
        for blob in source_blobs:
            target = target_shard / blob.relative_to(source_shard)
            assert target.exists()
            assert target.stat().st_ino == blob.stat().st_ino
            assert target.stat().st_nlink == blob.stat().st_nlink
    store = FeatureStore(destination)
    for row in store.records():
        store.load_bundle(row)


def test_merge_request_rejects_duplicates_and_bad_transport(tmp_path):
    with pytest.raises(ValueError, match="unique video IDs"):
        FeatureRunMergeRequest(("a", "a"), ("x",), str(tmp_path))
    with pytest.raises(ValueError, match="transport"):
        FeatureRunMergeRequest(("a",), ("x",), str(tmp_path), transport="symlink")
    with pytest.raises(ValueError, match="requires an explicit prefer_run"):
        FeatureRunMergeRequest(("a",), ("x",), str(tmp_path), duplicate_policy="prefer_run")


def test_merge_preflights_default_role_lock_before_loading_source_runs(tmp_path, monkeypatch):
    authority = tmp_path / "authority"
    lock = authority / "sources" / "role_lock.json"
    lock.parent.mkdir(parents=True)
    authority.mkdir(exist_ok=True)
    (authority / "contract.json").write_text("{}", encoding="utf-8")
    lock.write_text(json.dumps({"partitions": {}}), encoding="utf-8")
    contract = authority / "contract.json"
    monkeypatch.setattr(
        feature_merge_module,
        "_load_run",
        lambda *_args, **_kwargs: pytest.fail("source run loading must not start before lock preflight"),
    )
    with pytest.raises(FeatureMergeError, match="must be supplied before merging"):
        run_feature_merge(
            FeatureRunMergeRequest(
                target_video_ids=("sample-a",),
                run_roots=(str(tmp_path / "missing-run"),),
                output_root=str(tmp_path / "merged"),
                role_equivalence="engineering-shards-of-official-fulltrain",
                authority_contract_path=str(contract),
                authority_contract_sha256=sha256_file(contract),
            )
        )


def test_merge_cli_accepts_fulltrain_role_equivalence(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("sample-a\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts/icassp2027/merge_feature_runs.py"
    main = runpy.run_path(str(script))["main"]
    with pytest.raises(FeatureMergeError, match="authority contract SHA"):
        main(
            [
                "--video-list",
                str(ids),
                "--run-root",
                str(tmp_path / "missing-run"),
                "--output-root",
                str(tmp_path / "merged"),
                "--role-equivalence",
                "engineering-shards-of-official-fulltrain",
                "--authority-contract-path",
                str(tmp_path / "authority.json"),
                "--authority-contract-sha256",
                "0" * 64,
            ]
        )


def _clone_run(run_dir: Path, destination: Path) -> Path:
    shutil.copytree(run_dir, destination)
    status_path = destination / "status.json"
    status = json.loads(status_path.read_text())
    status["feature_root"] = str(destination)
    status_path.write_text(json.dumps(status), encoding="utf-8")
    return destination


def test_merge_prefer_run_keeps_identical_duplicate_with_audit(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    clone = _clone_run(Path(first.run_dir), tmp_path / "runs" / "run-b")
    result = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=("sample-a",),
            run_roots=(first.run_dir, str(clone)),
            output_root=str(tmp_path / "merged"),
            run_id="prefer-view",
            transport="copy",
            duplicate_policy="prefer_run",
            prefer_run=str(clone),
        )
    )
    assert result["status"] == "completed"
    audit = result["duplicate_audit"]
    assert audit["sample-a"]["verdict"] == "duplicate_identical"
    assert audit["sample-a"]["kept_run"] == str(clone)
    assert audit["sample-a"]["dropped_run"] == first.run_dir
    contract = json.loads((Path(result["run_dir"]) / "merge-contract.json").read_text())
    assert contract["duplicate_policy"] == "prefer_run"
    assert contract["duplicate_audit"] == audit
    provenance = contract["video_provenance"]
    assert provenance["sample-a"]["source_run"] == str(clone)
    store = FeatureStore(result["run_dir"])
    rows = store.records()
    assert len(rows) == first.records_written
    for row in rows:
        store.load_bundle(row)


def test_merge_prefer_run_refuses_divergent_duplicate_bytes(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    # Different source bytes before the second run: same adapter identity,
    # different features for the same video id.
    (tmp_path / "sample-a.mp4").write_bytes(b"divergent-source-bytes")
    second = _run(tmp_path, "run-b", ["sample-a"])
    assert first.completed and second.completed
    with pytest.raises(FeatureMergeError, match="bytes differ"):
        run_feature_merge(
            FeatureRunMergeRequest(
                target_video_ids=("sample-a",),
                run_roots=(first.run_dir, second.run_dir),
                output_root=str(tmp_path / "merged"),
                run_id="divergent",
                transport="copy",
                duplicate_policy="prefer_run",
                prefer_run=second.run_dir,
            )
        )


def test_merge_prefer_run_must_cover_every_duplicate(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    clone = _clone_run(Path(first.run_dir), tmp_path / "runs" / "run-b")
    other = _run(tmp_path, "run-c", ["sample-b"])
    with pytest.raises(FeatureMergeError, match="does not cover duplicated video"):
        run_feature_merge(
            FeatureRunMergeRequest(
                target_video_ids=("sample-a", "sample-b"),
                run_roots=(first.run_dir, str(clone), other.run_dir),
                output_root=str(tmp_path / "merged"),
                run_id="uncovered",
                transport="copy",
                duplicate_policy="prefer_run",
                prefer_run=other.run_dir,
            )
        )


def test_subset_view_copy_roundtrip(tmp_path):
    from vadbench.paper.feature_merge import (
        SUBSET_CONTRACT_SCHEMA,
        SUBSET_VIEW_SCHEMA,
        FeatureSubsetRequest,
        run_feature_subset,
    )

    source = _run(tmp_path, "run-a", ["sample-a", "sample-b", "sample-c"])
    result = run_feature_subset(
        FeatureSubsetRequest(
            source_run_root=source.run_dir,
            target_video_ids=("sample-a", "sample-c"),
            output_root=str(tmp_path / "subsets"),
            run_id="subset-view",
            transport="copy",
        )
    )
    assert result["status"] == "completed" and result["subset_view"] is True
    destination = Path(result["run_dir"])
    resolved = json.loads((destination / "resolved.json").read_text())
    assert resolved["schema"] == SUBSET_VIEW_SCHEMA and resolved["subset_view"] is True
    contract = json.loads((destination / "subset-contract.json").read_text())
    assert contract["schema"] == SUBSET_CONTRACT_SCHEMA
    assert contract["subset_of"]["root"] == source.run_dir
    assert contract["target_video_ids"] == ["sample-a", "sample-c"]
    assert "not a native extraction run" in contract["note"]
    store = FeatureStore(destination)
    rows = store.records()
    assert {row.video_id for row in rows} == {"sample-a", "sample-c"}
    assert len(rows) == result["clips"]
    for row in rows:
        store.load_bundle(row)


def test_subset_view_rejects_videos_outside_the_source_run(tmp_path):
    from vadbench.paper.feature_merge import FeatureSubsetRequest, run_feature_subset

    source = _run(tmp_path, "run-a", ["sample-a"])
    with pytest.raises(FeatureMergeError, match="missing from the source run"):
        run_feature_subset(
            FeatureSubsetRequest(
                source_run_root=source.run_dir,
                target_video_ids=("sample-a", "sample-missing"),
                output_root=str(tmp_path / "subsets"),
                run_id="bad-subset",
                transport="copy",
            )
        )


def _mark_partial(run_dir: Path, failures=True) -> Path:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["completed"] = False
    status["status"] = "partial_failed"
    if failures:
        status["failures"] = [
            {"video_id": "sample-zz", "type": "RuntimeError", "message": "fixture abort"}
        ]
    status_path.write_text(json.dumps(status), encoding="utf-8")
    (run_dir / "index.jsonl").unlink()
    return run_dir


def _merge_with_partial(tmp_path, runs, videos, partial_roots, run_id="rescue-view"):
    return run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=tuple(videos),
            run_roots=tuple(run.run_dir for run in runs),
            output_root=str(tmp_path / "merged"),
            run_id=run_id,
            transport="copy",
            allow_partial=tuple(str(root) for root in partial_roots),
        )
    )


def test_merge_rescues_complete_shards_from_partial_run_with_audit(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    second = _mark_partial(Path(_run(tmp_path, "run-b", ["sample-b", "sample-c"]).run_dir))
    result = _merge_with_partial(tmp_path, (first, _Partial(second)), ["sample-a", "sample-b", "sample-c"], [second])
    assert result["status"] == "completed"
    audit = json.loads((Path(result["run_dir"]) / "merge-contract.json").read_text())[
        "partial_rescue"
    ]
    assert sorted(audit) == ["sample-b", "sample-c"]
    assert audit["sample-b"]["source_run"] == str(second)
    assert audit["sample-b"]["source_status"] == "partial_failed"
    assert audit["sample-b"]["failures"]
    store = FeatureStore(result["run_dir"])
    for row in store.records():
        store.load_bundle(row)


def test_merge_rescue_skips_incomplete_shard_and_reports_missing(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    partial = _mark_partial(Path(_run(tmp_path, "run-b", ["sample-b", "sample-c"]).run_dir))
    # sample-c's shard is destroyed: the partial run may only rescue sample-b.
    shutil.rmtree(partial / "shards" / "sample-c")
    with pytest.raises(FeatureMergeError, match="missing from every run"):
        _merge_with_partial(tmp_path, (first, _Partial(partial)), ["sample-a", "sample-b", "sample-c"], [partial])
    cover = _run(tmp_path, "run-c", ["sample-c"])
    result = _merge_with_partial(
        tmp_path, (first, _Partial(partial), cover), ["sample-a", "sample-b", "sample-c"], [partial]
    )
    audit = json.loads((Path(result["run_dir"]) / "merge-contract.json").read_text())[
        "partial_rescue"
    ]
    assert sorted(audit) == ["sample-b"]


def test_merge_rejects_partial_root_not_listed_in_run_roots(tmp_path):
    first = _run(tmp_path, "run-a", ["sample-a"])
    with pytest.raises(FeatureMergeError, match="must also be listed in run_roots"):
        _merge_with_partial(tmp_path, (first,), ["sample-a"], [tmp_path / "elsewhere"])


class _Partial:
    """Wraps a path so _merge_with_partial can take plain run objects or paths."""

    def __init__(self, root: Path):
        self.run_dir = str(root)
