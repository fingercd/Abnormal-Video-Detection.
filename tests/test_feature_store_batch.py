from __future__ import annotations

import numpy as np
import pytest

from vadbench.features import FeatureStore


def _item(index: int, *, value: float = 0.0, clip_id: str | None = None):
    return {
        "video_id": "video",
        "clip_id": clip_id or f"clip-{index}",
        "clip_index": index,
        "encoder_fingerprint": "sha256:" + "a" * 64,
        "features": np.full((1, 3), value, dtype=np.float32),
        "pooled": np.full(3, value, dtype=np.float32),
        "start_s": float(index),
        "end_s": float(index + 1),
        "frame_start": index,
        "frame_end": index + 1,
        "metadata": {"kind": "fixture", "index": index},
    }


@pytest.mark.parametrize("storage_format", ["npz", "npy"])
def test_write_many_persists_each_input_in_order(storage_format, tmp_path):
    store = FeatureStore(tmp_path / storage_format, storage_format=storage_format)
    items = [_item(2, value=2.0), _item(1, value=1.0)]
    written = store.write_many(items)

    assert [record.clip_id for record in written] == ["clip-2", "clip-1"]
    persisted = {record.clip_id: record for record in store.records()}
    for item, record in zip(items, written, strict=True):
        loaded = store.load_bundle(persisted[record.clip_id])
        assert np.array_equal(loaded["features"], item["features"])
        assert np.array_equal(loaded["pooled"], item["pooled"])
        assert persisted[record.clip_id].metadata == item["metadata"]


def test_write_many_upserts_existing_key_without_losing_other_rows(tmp_path):
    store = FeatureStore(tmp_path / "store")
    store.write_many([_item(0), _item(1)])
    replacement = _item(0, value=9.0)
    added = _item(2, value=2.0)
    store.write_many([replacement, added])

    records = {record.clip_id: record for record in store.records()}
    assert set(records) == {"clip-0", "clip-1", "clip-2"}
    assert np.array_equal(store.load_bundle(records["clip-0"])["pooled"], replacement["pooled"])


def test_write_many_rejects_duplicate_batch_key_without_changing_index(tmp_path):
    store = FeatureStore(tmp_path / "store")
    store.write(**_item(0))
    before = store.index_path.read_bytes()
    with pytest.raises(ValueError, match="duplicate batch"):
        store.write_many([_item(1), _item(2, clip_id="clip-1")])
    assert store.index_path.read_bytes() == before


def test_write_many_later_existing_key_with_overwrite_disabled_keeps_index(tmp_path):
    store = FeatureStore(tmp_path / "store")
    store.write(**_item(0))
    before = store.index_path.read_bytes()
    existing = {**_item(0, value=9.0), "overwrite": False}
    with pytest.raises(FileExistsError):
        store.write_many([_item(1), existing])
    assert store.index_path.read_bytes() == before


def test_write_many_later_invalid_interval_does_not_publish_prior_blob(tmp_path):
    store = FeatureStore(tmp_path / "store")
    store.write(**_item(0))
    before = store.index_path.read_bytes()
    invalid = {**_item(2), "end_s": -1.0}
    with pytest.raises(ValueError, match="time interval"):
        store.write_many([_item(1), invalid])
    assert store.index_path.read_bytes() == before


def test_write_many_rejects_later_invalid_item_before_index_or_blob_write(tmp_path):
    store = FeatureStore(tmp_path / "store")
    store.write(**_item(0))
    before = store.index_path.read_bytes()
    invalid = _item(2)
    invalid["features"] = object()
    with pytest.raises(TypeError):
        store.write_many([_item(1), invalid])
    assert store.index_path.read_bytes() == before
    assert len(store.records()) == 1


def test_write_many_blob_or_index_failure_never_publishes_partial_index(tmp_path, monkeypatch):
    from vadbench import features as module

    store = FeatureStore(tmp_path / "store")
    store.write(**_item(0))
    before = store.index_path.read_bytes()
    original_blob = store._write_npz
    calls = 0

    def fail_second_blob(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected blob failure")
        return original_blob(*args, **kwargs)

    monkeypatch.setattr(store, "_write_npz", fail_second_blob)
    with pytest.raises(OSError, match="blob"):
        store.write_many([_item(1), _item(2)])
    assert store.index_path.read_bytes() == before

    monkeypatch.setattr(store, "_write_npz", original_blob)
    monkeypatch.setattr(module, "atomic_write_jsonl", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("index failure")))
    with pytest.raises(OSError, match="index"):
        store.write_many([_item(1), _item(2)])
    assert store.index_path.read_bytes() == before


def test_write_many_publishes_one_index_for_the_batch(tmp_path, monkeypatch):
    from vadbench import features as module

    store = FeatureStore(tmp_path / "store")
    original = module.atomic_write_jsonl
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_write_jsonl", counted)
    store.write_many([_item(0), _item(1), _item(2)])
    assert calls == 1


def test_write_many_holds_one_lock_for_the_whole_batch(tmp_path, monkeypatch):
    from vadbench import features as module

    class _Lock:
        entered = exited = 0

        def __init__(self, _path):
            pass

        def __enter__(self):
            type(self).entered += 1
            return self

        def __exit__(self, *_args):
            type(self).exited += 1

    monkeypatch.setattr(module, "_InterProcessLock", _Lock)
    FeatureStore(tmp_path / "store").write_many([_item(0), _item(1)])
    assert (_Lock.entered, _Lock.exited) == (1, 1)
