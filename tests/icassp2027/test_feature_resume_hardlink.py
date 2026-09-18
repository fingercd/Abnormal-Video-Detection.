"""Hardlink transport tests use only newly created pytest temporary data."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_feature_resume import _run, _setup, _snapshot
from test_paper_extraction import _CV2, _Adapter, _record, _representation, _sampling, _spec, _touch

from vadbench.features import FeatureStore, atomic_write_jsonl
from vadbench.paper.extraction import extract_pooled_features


def _source(tmp_path):
    records, adapter, spec = _setup(tmp_path)
    old = _run(tmp_path, records, adapter, spec, "original")
    assert old.completed
    return records, spec, Path(old.run_dir)


def _first_blob(source):
    shard = source / "shards" / "one"
    store = FeatureStore(shard)
    row = store.records()[0]
    return shard, row, shard / row.arrays["features"].path


def _assert_failed_without_publication(result):
    root = Path(result.run_dir)
    assert not result.completed
    assert not (root / "index.jsonl").exists()
    assert not (root / "shards" / "one" / "index.jsonl").exists()
    assert json.loads((root / "status.json").read_text())["completed"] is False


def test_hardlink_shared_inode_preserves_bytes_and_execution_identity(tmp_path):
    records, spec, source = _source(tmp_path)
    before = _snapshot(source)
    source_blobs = {p: p.stat() for p in source.rglob("*.npz")}
    adapter = _Adapter()
    linked = _run(tmp_path, records, adapter, spec, "linked", source, transport="hardlink_npz")
    assert linked.completed, linked.failures
    assert adapter.seen_batches == []
    target = Path(linked.run_dir)
    old_resolved = json.loads((source / "resolved.json").read_text())
    new_resolved = json.loads((target / "resolved.json").read_text())
    assert new_resolved["resume_transport"] == "hardlink_npz"
    for field in ("spec", "encoder_fingerprint", "paper_identity"):
        assert old_resolved[field] == new_resolved[field]
    assert _snapshot(source) == before
    assert all(p.stat().st_nlink == value.st_nlink + 1 for p, value in source_blobs.items())
    source_rows = FeatureStore(source).records()
    target_rows = FeatureStore(target).records()
    for old, new in zip(source_rows, target_rows, strict=True):
        a, b = source / old.arrays["features"].path, target / new.arrays["features"].path
        assert os.path.samefile(a, b)
        assert old.arrays["features"].sha256 == new.arrays["features"].sha256
        assert new.arrays["features"].path == new.arrays["pooled"].path
        assert new.metadata["reused_from"]["transport"] == "hardlink_npz"
        assert new.metadata["runtime_reference"]["sha256"] == hashlib.sha256((target / "resolved.json").read_bytes()).hexdigest()
    receipts = [json.loads(line) for p in (target / "resume_lineage/hardlinks").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert len(receipts) == len(source_rows)
    for row in receipts:
        assert row["source_after"]["nlink"] == row["source_before"]["nlink"] + 1
        assert row["source_after"]["inode"] == row["target_after"]["inode"]
        assert "stat_ctime_ns" in row["source_after"]  # Same clock granularity is allowed.
    transport = json.loads((target / "resume_lineage/transport.json").read_text())
    assert transport["source_nlink_and_ctime_may_change"] and not transport["independent_blob_backup"]
    assert not os.path.samefile(source / "index.jsonl", target / "index.jsonl")
    assert not os.path.samefile(source / "resolved.json", target / "resume_lineage/source_resolved.json")
    # A further strict resume accepts the linked run without identity relaxation.
    again = _run(tmp_path, records, adapter, spec, "linked-again", target, transport="hardlink_npz")
    assert again.completed and adapter.seen_batches == []


def test_default_copy_uses_distinct_inode_and_no_hardlink_journal(tmp_path):
    records, spec, source = _source(tmp_path)
    result = _run(tmp_path, records, _Adapter(), spec, "copy", source)
    assert result.completed
    target = Path(result.run_dir)
    old = FeatureStore(source).records()[0]
    new = FeatureStore(target).records()[0]
    assert not os.path.samefile(source / old.arrays["features"].path, target / new.arrays["features"].path)
    assert not (target / "resume_lineage/hardlinks").exists()
    assert json.loads((target / "resolved.json").read_text())["resume_transport"] == "copy"


def test_normal_writer_never_modifies_shared_blob_inode(tmp_path):
    records, spec, source = _source(tmp_path)
    result = _run(tmp_path, records, _Adapter(), spec, "linked", source, transport="hardlink_npz")
    assert result.completed
    before = _snapshot(source)
    store = FeatureStore(Path(result.run_dir) / "shards/one/blocks/000000")
    row = store.records()[0]
    bundle = store.load_bundle(row)
    linked_path = store.root / row.arrays["features"].path
    original_inode = linked_path.stat().st_ino
    args = dict(video_id=row.video_id, clip_id=row.clip_id, clip_index=row.clip_index,
                encoder_fingerprint=row.encoder_fingerprint, start_s=row.start_s, end_s=row.end_s,
                frame_start=row.frame_start, frame_end=row.frame_end, metadata=row.metadata)
    store.write(**args, **bundle)  # Existing hash path: discard only the new tempfile.
    changed = store.write(**args, features=bundle["features"] + 123, pooled=bundle["pooled"] + 123)
    assert changed.arrays["features"].path != row.arrays["features"].path
    assert linked_path.stat().st_ino == original_inode and _snapshot(source) == before
    with pytest.raises(FileExistsError):
        store.write(**args, **bundle, overwrite=False)
    assert _snapshot(source) == before


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "npy", "wrong_key"])
def test_physical_archive_contract_rejected_before_any_link(tmp_path, monkeypatch, mutation):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    shard, record, blob = _first_blob(source)
    rows = [r.to_dict() for r in FeatureStore(shard).records()]
    if mutation in {"extra", "duplicate"}:
        with zipfile.ZipFile(blob, "a") as archive:
            if mutation == "duplicate":
                data = archive.read("features.npy")
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("features.npy", data)
            else:
                archive.writestr("unexpected.npy", b"not a permitted member")
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        for ref in rows[0]["arrays"].values():
            ref["sha256"] = digest
    elif mutation == "npy":
        bundle = FeatureStore(shard).load_bundle(record)
        rows[0]["storage_format"] = "npy"
        for name, array in bundle.items():
            path = blob.with_suffix(f".{name}.npy")
            np.save(path, array)
            rows[0]["arrays"][name].update(path=path.relative_to(shard).as_posix(), key=None,
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        rows[0]["arrays"]["features"]["key"] = "pooled"
    atomic_write_jsonl(shard / "index.jsonl", rows)
    before = _snapshot(source)
    monkeypatch.setattr(resume.os, "link", lambda *_args, **_kwargs: pytest.fail("must validate before linking"))
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert adapter.seen_batches == [] and _snapshot(source) == before
    assert "hardlink_npz" in result.failures[0]["message"]


def test_symlink_blob_rejected(tmp_path):
    records, spec, source = _source(tmp_path)
    _shard, _row, blob = _first_blob(source)
    real = blob.with_suffix(".original")
    blob.rename(real)
    try:
        blob.symlink_to(real)
    except OSError as error:
        pytest.skip(f"local account cannot create symlinks: {error}")
    before = _snapshot(source)
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert "symlink" in result.failures[0]["message"] and _snapshot(source) == before


@pytest.mark.parametrize("alias_side", ["source_blob", "target_parent"])
def test_symlink_gate_precedes_any_link_even_without_os_symlink_privilege(tmp_path, monkeypatch, alias_side):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    _shard, _record, blob = _first_blob(source)
    target_parent = tmp_path / "runs/rejected/shards"
    aliased = blob if alias_side == "source_blob" else target_parent
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == aliased or original(path))
    monkeypatch.setattr(resume.os, "link", lambda *_args, **_kwargs: pytest.fail("symlink gate must precede link"))
    before = _snapshot(source)
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert "symlink" in result.failures[0]["message"]
    if alias_side == "target_parent":
        assert not target_parent.exists()
    assert _snapshot(source) == before


def test_source_inode_swap_after_validation_rejected_before_link(tmp_path, monkeypatch):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    _shard, _row, blob = _first_blob(source)
    before = _snapshot(source)
    publish = resume.FeatureResumeSource._publish_hardlinks

    def swapped(self, destination, prepared, video_id):
        replacement = blob.with_suffix(".swap")
        replacement.write_bytes(blob.read_bytes())
        replacement.replace(blob)
        return publish(self, destination, prepared, video_id)

    monkeypatch.setattr(resume.FeatureResumeSource, "_publish_hardlinks", swapped)
    monkeypatch.setattr(resume.os, "link", lambda *_args, **_kwargs: pytest.fail("changed inode must precede link"))
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert "source changed" in result.failures[0]["message"]
    assert _snapshot(source) == before  # The injected swap changed inode, not data.


@pytest.mark.parametrize("failure", ["existing_target", "permission", "cross_device", "copy_instead_of_link", "second_link"])
def test_link_errors_never_fallback_or_publish_video(tmp_path, monkeypatch, failure):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    before = _snapshot(source)
    link = os.link
    calls = []

    def injected(src, dst, **kwargs):
        calls.append(Path(dst))
        if failure == "existing_target":
            Path(dst).write_bytes(b"preexisting-target-must-survive")
            return link(src, dst, **kwargs)
        if failure == "copy_instead_of_link":
            return shutil.copyfile(src, dst)
        if failure == "second_link" and len(calls) == 1:
            return link(src, dst, **kwargs)
        raise OSError(errno.EXDEV if failure == "cross_device" else errno.EPERM, "injected link failure")

    monkeypatch.setattr(resume.os, "link", injected)
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert adapter.seen_batches == [] and _snapshot(source) == before
    assert len(calls) == (2 if failure == "second_link" else 1)
    if failure == "existing_target":
        assert calls[0].read_bytes() == b"preexisting-target-must-survive"
    if failure == "second_link":
        journal = Path(result.run_dir) / "resume_lineage/hardlinks/one.jsonl"
        assert len(journal.read_text().splitlines()) == 1


def test_filesystem_mismatch_rejected_before_link(tmp_path, monkeypatch):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    state = resume._blob_state

    def changed_device(path, root):
        return {**state(path, root), "device": -1}

    monkeypatch.setattr(resume, "_blob_state", changed_device)
    monkeypatch.setattr(resume.os, "link", lambda *_args, **_kwargs: pytest.fail("filesystem gate must precede link"))
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert "filesystem" in result.failures[0]["message"]


def test_existing_video_destination_rejected(tmp_path, monkeypatch):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    publish = resume.FeatureResumeSource._publish_hardlinks

    def occupied(self, destination, prepared, video_id):
        destination.mkdir(parents=True)
        (destination / "keep.txt").write_text("existing")
        return publish(self, destination, prepared, video_id)

    monkeypatch.setattr(resume.FeatureResumeSource, "_publish_hardlinks", occupied)
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert (Path(result.run_dir) / "shards/one/keep.txt").read_text() == "existing"


def test_destination_parent_symlink_rejected_before_source_mkdir_or_link(tmp_path, monkeypatch):
    import vadbench.paper.feature_resume as resume

    records, spec, source = _source(tmp_path)
    before = _snapshot(source)
    publish = resume.FeatureResumeSource._publish_hardlinks

    def aliased(self, destination, prepared, video_id):
        try:
            destination.parent.symlink_to(source, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"local account cannot create directory symlinks: {error}")
        return publish(self, destination, prepared, video_id)

    monkeypatch.setattr(resume.FeatureResumeSource, "_publish_hardlinks", aliased)
    monkeypatch.setattr(resume.os, "link", lambda *_args, **_kwargs: pytest.fail("must reject target alias before link"))
    result = _run(tmp_path, records, _Adapter(), spec, "rejected", source, transport="hardlink_npz")
    _assert_failed_without_publication(result)
    assert "symlink" in result.failures[0]["message"]
    assert not (source / "one").exists() and _snapshot(source) == before


@pytest.mark.parametrize("alias", [False, True])
def test_output_tree_source_overlap_rejected_before_any_output(tmp_path, alias):
    records, spec, source = _source(tmp_path)
    output = source
    if alias:
        output = tmp_path / "aliased-output"
        try:
            output.symlink_to(source, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"local account cannot create directory symlinks: {error}")
    before = _snapshot(source)
    with pytest.raises(ValueError, match="overlap|symlink"):
        extract_pooled_features(spec, adapter=_Adapter(), manifest=records, dataset_root=tmp_path,
            output_root=output, run_id="must-not-exist", backend=_CV2(64),
            resume_source=source, resume_transport="hardlink_npz")
    assert not (source / "must-not-exist").exists() and _snapshot(source) == before


def test_hardlink_keeps_64_row_blocks_without_reencoding(tmp_path):
    records = [_record("long", split="train", anomaly=False, frames=140)]
    _touch(tmp_path, records)
    adapter = _Adapter()
    spec = _spec(_representation(adapter, reducer="identity"),
                 _sampling(records, root=tmp_path, regime="test_dense", clips=32, window_stride=2),
                 kind="dense", stride=2)

    def run(name, source=None, transport="copy"):
        return extract_pooled_features(spec, adapter=adapter, manifest=records, dataset_root=tmp_path,
            output_root=tmp_path / "runs", run_id=name, backend=_CV2(140),
            resume_source=source, resume_transport=transport)

    original = run("original")
    adapter.seen_batches.clear()
    result = run("linked", original.run_dir, "hardlink_npz")
    assert result.completed and result.records_written == 69 and adapter.seen_batches == []
    blocks = Path(result.run_dir) / "shards/long/blocks"
    assert [len(FeatureStore(p).records()) for p in sorted(blocks.iterdir())] == [64, 5]


@pytest.mark.parametrize("transport,source,reducer", [("unknown", "source", "identity"), ("hardlink_npz", None, "identity"), ("hardlink_npz", "source", "global_uniform")])
def test_transport_api_rejects_unsupported_scope_before_output(tmp_path, transport, source, reducer):
    records, adapter, spec = _setup(tmp_path)
    spec = replace(spec, representation=_representation(adapter, reducer=reducer))
    with pytest.raises(ValueError, match="resume_transport|identity resume source"):
        _run(tmp_path, records, adapter, spec, "rejected", source, transport=transport)
    assert not (tmp_path / "runs").exists()


def test_official_request_forwards_transport_and_rejects_invalid_scope(tmp_path, monkeypatch):
    from test_official_extraction import _CV, _case

    from vadbench.paper.official_extraction import run_official_dense_extraction

    request, adapter, factory, _calls = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identity resume source"):
        replace(request, resume_transport="hardlink_npz")
    with pytest.raises(ValueError, match="identity resume source"):
        replace(request, resume_source="source", resume_transport="hardlink_npz", reducer="global_uniform")
    with pytest.raises(ValueError, match="resume_transport"):
        replace(request, resume_transport="unknown")
    result = run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    source = json.loads(Path(result["contract"]["path"]).read_text())["feature_store"]["root"]
    adapter.calls = 0
    linked = run_official_dense_extraction(replace(request, run_id="linked", resume_source=source, resume_transport="hardlink_npz"),
                                         adapter_factory=factory, video_backend=_CV())
    assert linked["status"] == "completed" and adapter.calls == 0
    target = json.loads(Path(linked["contract"]["path"]).read_text())["feature_store"]["root"]
    assert json.loads((Path(target) / "resolved.json").read_text())["resume_transport"] == "hardlink_npz"
