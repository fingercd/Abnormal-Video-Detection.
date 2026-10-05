from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_paper_extraction import _CV2, _Adapter, _record, _representation, _sampling, _spec, _touch

from vadbench.features import FeatureStore, atomic_write_jsonl
from vadbench.paper.detection import _verify_feature_identity
from vadbench.paper.extraction import _merge_shards, extract_pooled_features


class _NativeResumeAdapter:
    """Actual CPU bridge forwards, wrapped in the pooled extraction contract."""

    pooling = "mean"

    def __init__(self, native, frames):
        from vadbench.contracts import EncoderCapabilities

        self.native = native
        self.model = native.model
        self.seen_batches = []
        self.capabilities = EncoderCapabilities(
            supports_fixed_clip=True,
            supports_training=False,
            fixed_num_frames=frames,
            min_frames=frames,
            max_frames=frames,
        )

    def encode(self, batch, train=False):
        from vadbench.contracts import EncoderOutput, TokenTimeline

        assert not train
        self.seen_batches.append(batch.batch_size)
        features = self.native.encode(batch)
        shape = tuple(features.shape[:2])
        return EncoderOutput(
            features=features,
            pooled=features.mean(dim=1),
            timeline=TokenTimeline(
                start_s=np.zeros(shape), end_s=np.ones(shape), valid_mask=np.ones(shape, dtype=bool)
            ),
        )


def _native_resume_source(tmp_path, encoder_id, strategy):
    from test_pair_deployment import _case
    from test_paper_extraction import _Capture

    from vadbench.paper.extraction import (
        PooledExtractionSpec,
        make_sampling_identity,
        representation_from_verified_encoder,
    )
    from vadbench.token_reduction.deployment import PairMergeDeployment
    from vadbench.token_reduction.pair_merge import PairLinearGate

    native, bridge, layout, frames, dim = _case(encoder_id)
    adapter = _NativeResumeAdapter(native, frames)
    gate = PairLinearGate(dim) if strategy == "pair_linear" else None
    deployment = PairMergeDeployment(
        bridge,
        layout,
        depth=0,
        dim=dim,
        batch_sizes=range(1, 9),
        gate=gate,
        calibration_manifest_digest="a" * 64 if gate is not None else None,
        strategy="mean" if gate is not None else strategy,
    )
    verified = {
        "adapter": encoder_id,
        "constructor": {"clip_frames": frames},
        "checkpoint": {"sha256": {"fixture": "a" * 64}},
        "code": {"commit": "old"},
    }
    representation = representation_from_verified_encoder(
        runtime_id=encoder_id,
        adapter=adapter,
        verified_encoder_identity=verified,
        preprocessing={"profile": "native-fixture"},
        readout={"kind": "mean"},
        reducer=dict(deployment.reducer_identity),
        output_dim=dim,
        precision="float32",
        position_strategy={"kind": "native"},
    )
    records = [_record("native", split="train", anomaly=False)]
    _touch(tmp_path, records)
    sampling = make_sampling_identity(
        records,
        dataset_root=tmp_path,
        sampling_kind="uniform_full",
        clip_frames=frames,
        frame_stride=1,
        num_segments=32,
    )
    spec = PooledExtractionSpec(
        runtime_id=encoder_id,
        verified_encoder_identity=verified,
        representation=representation,
        sampling=sampling,
        sampling_kind="uniform_full",
        clip_frames=frames,
        frame_stride=1,
        micro_batch_size=7,
    )

    class Capture(_Capture):
        def get(self, property_id):
            if property_id in (
                self.backend.CAP_PROP_FRAME_WIDTH,
                self.backend.CAP_PROP_FRAME_HEIGHT,
            ):
                return 4.0
            return super().get(property_id)

    class Backend(_CV2):
        def __init__(self):
            super().__init__(64)
            self.frames = [
                np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1) for frame in self.frames
            ]

        def VideoCapture(self, path):
            return Capture(self, self.frames)

    def run(name, source=None, selected_spec=None):
        return extract_pooled_features(
            spec if selected_spec is None else selected_spec,
            adapter=adapter,
            manifest=records,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id=name,
            backend=Backend(),
            encode_context_factory=deployment,
            resume_source=source,
        )

    old = run("native_source")
    assert old.completed, old.failures
    return adapter, spec, old, run


def _setup(tmp_path, kind="uniform_full"):
    records = [
        _record("one", split="train", anomaly=False),
        _record("two", split="train", anomaly=True),
    ]
    _touch(tmp_path, records)
    adapter = _Adapter()
    representation = _representation(adapter, reducer="identity")
    sampling = _sampling(
        records,
        root=tmp_path,
        regime="train_32" if kind == "uniform_full" else "test_dense",
        clips=32,
        window_stride=2,
    )
    spec = _spec(representation, sampling, kind=kind, stride=2)
    return records, adapter, spec


def _run(tmp_path, records, adapter, spec, name, source=None, *, transport="copy"):
    return extract_pooled_features(
        spec,
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id=name,
        backend=_CV2(64),
        resume_source=source,
        resume_transport=transport,
    )


def _snapshot(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def _legacy_rows(source, videos):
    runtime = json.loads((source / "resolved.json").read_text())["runtime"]
    for video in videos:
        index = source / "shards" / video / "index.jsonl"
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        for row in rows:
            row["metadata"].pop("runtime_reference")
            row["metadata"]["runtime"] = runtime
        atomic_write_jsonl(index, rows)


@pytest.mark.parametrize("kind", ["uniform_full", "dense"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("transport", ["copy", "hardlink_npz"])
def test_complete_video_reused_partial_video_reexecuted_source_unchanged(tmp_path, kind, legacy, transport):
    records, adapter, spec = _setup(tmp_path, kind)
    encode = adapter.encode

    def interrupted(batch, train=False):
        if len(adapter.seen_batches) == 6:
            raise RuntimeError("GPU guard yielded to foreign user")
        return encode(batch, train=train)

    adapter.encode = interrupted
    old = _run(tmp_path, records, adapter, spec, "interrupted")
    assert not old.completed
    source = Path(old.run_dir)
    assert not (source / "index.jsonl").exists()
    # Recreate the legacy format's published partial video index. Old jobs
    # published each clip individually; this must never qualify as complete.
    partial = source / "shards" / "two"
    _merge_shards(partial, sorted((partial / "blocks").iterdir()))
    if legacy:
        _legacy_rows(source, ["one", "two"])
    before = _snapshot(source)
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "resumed", source, transport=transport)
    assert result.completed, result.failures
    assert len(adapter.seen_batches) == 5  # only the incomplete second video
    status = json.loads(Path(result.status_path).read_text())
    expected = 32 if kind == "uniform_full" else 31
    assert status["resume"] == {
        "source_run": str(source),
        "reused_videos": 1,
        "reused_clips": expected,
        "decoded_videos": 1,
        "decoded_clips": expected,
    }
    _verify_feature_identity(
        result.feature_root,
        tuple(records),
        encoder_fingerprint=result.encoder_fingerprint,
        representation=spec.representation,
        sampling=spec.sampling,
    )
    old_store = FeatureStore(source / "shards" / "one")
    new_store = FeatureStore(result.feature_root)
    new_rows = new_store.records()
    assert len(new_rows) == 2 * expected
    for old_row, new_row in zip(old_store.records(), new_rows[:expected], strict=True):
        assert np.array_equal(
            old_store.load_array(old_row, "pooled"), new_store.load_array(new_row, "pooled")
        )
        assert "reused_from" in new_row.metadata
        assert (
            new_row.metadata["runtime_reference"]["sha256"]
            == hashlib.sha256((Path(result.run_dir) / "resolved.json").read_bytes()).hexdigest()
        )
    assert _snapshot(source) == before
    assert (Path(result.run_dir) / "resume_lineage" / "source_resolved.json").read_bytes() == (
        source / "resolved.json"
    ).read_bytes()


def test_complete_mixed_legacy_reference_store_uses_zero_encoder_calls(tmp_path):
    records, adapter, spec = _setup(tmp_path)
    old = _run(tmp_path, records, adapter, spec, "complete")
    source = Path(old.run_dir)
    _legacy_rows(source, ["one"])
    before = _snapshot(source)
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "copy", source)
    assert result.completed, result.failures
    assert adapter.seen_batches == []
    _verify_feature_identity(
        result.feature_root,
        tuple(records),
        encoder_fingerprint=result.encoder_fingerprint,
        representation=spec.representation,
        sampling=spec.sampling,
    )
    assert _snapshot(source) == before


@pytest.mark.parametrize(
    "tamper",
    [
        "blob",
        "shape",
        "dtype",
        "finite",
        "sampling",
        "runtime_reference",
        "source",
        "runtime",
        "duplicate",
    ],
)
@pytest.mark.parametrize("transport", ["copy", "hardlink_npz"])
def test_tampered_source_rejected_without_root_index(tmp_path, tamper, transport):
    records, adapter, spec = _setup(tmp_path)
    old = _run(tmp_path, records, adapter, spec, "complete")
    source = Path(old.run_dir)
    shard = source / "shards" / "one"
    index = shard / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    row = rows[0]
    if tamper == "blob":
        (shard / row["arrays"]["pooled"]["path"]).write_bytes(b"corrupted")
    elif tamper == "shape":
        row["arrays"]["pooled"]["shape"] = [99]
    elif tamper == "dtype":
        row["arrays"]["pooled"]["dtype"] = "<f8"
    elif tamper == "finite":
        store = FeatureStore(shard)
        original = store.records()[0]
        store.write(
            video_id=original.video_id,
            clip_id=original.clip_id,
            clip_index=original.clip_index,
            encoder_fingerprint=original.encoder_fingerprint,
            features=np.full((1, 4), np.nan, dtype=np.float32),
            pooled=np.full(4, np.nan, dtype=np.float32),
            start_s=original.start_s,
            end_s=original.end_s,
            frame_start=original.frame_start,
            frame_end=original.frame_end,
            metadata=original.metadata,
        )
        rows = [r.to_dict() for r in store.records()]
    elif tamper == "sampling":
        row["metadata"]["sampling"]["input_start_frame"] += 1
    elif tamper == "runtime_reference":
        row["metadata"]["runtime_reference"]["sha256"] = "0" * 64
    elif tamper == "source":
        row["metadata"]["source"]["is_anomaly"] = True
    elif tamper == "runtime":
        row["metadata"]["runtime"]["adapter_type"] = "other.Adapter"
    else:
        rows.append(row)
    atomic_write_jsonl(index, rows)
    before = _snapshot(source)
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "rejected", source, transport=transport)
    assert not result.completed
    assert not (Path(result.run_dir) / "index.jsonl").exists()
    assert adapter.seen_batches == []
    assert _snapshot(source) == before


def test_changed_data_and_native_precision_rejected(tmp_path):
    records, adapter, spec = _setup(tmp_path)
    old = _run(tmp_path, records, adapter, spec, "complete")
    changed = replace(spec, representation=replace(spec.representation, precision="float16"))
    result = _run(tmp_path, records, _Adapter(), changed, "precision", old.run_dir)
    assert not result.completed
    assert "differs" in result.failures[0]["message"]
    (tmp_path / records[0].path).write_bytes(b"different video bytes")
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)
    result = _run(
        tmp_path, records, _Adapter(), replace(spec, sampling=sampling), "data", old.run_dir
    )
    assert not result.completed
    assert "differs" in result.failures[0]["message"]


def test_long_video_writes_bounded_indexes_and_publishes_once(tmp_path, monkeypatch):
    import vadbench.paper.extraction as extraction

    records = [_record("long", split="train", anomaly=False, frames=140)]
    _touch(tmp_path, records)
    adapter = _Adapter()
    sampling = _sampling(records, root=tmp_path, regime="test_dense", clips=32, window_stride=2)
    spec = replace(
        _spec(_representation(adapter, reducer="identity"), sampling, kind="dense", stride=2),
        micro_batch_size=7,
    )
    original_write = FeatureStore.write_many
    original_publish = extraction.atomic_write_jsonl
    sizes, publications, batch_sizes = [], [], []

    def checked_write(store, items):
        assert store.root.parent.name == "blocks"
        result = original_write(store, items)
        batch_sizes.append(len(items))
        sizes.append(len(store.records()))
        return result

    def checked_publish(path, rows):
        publications.append(Path(path))
        return original_publish(path, rows)

    monkeypatch.setattr(FeatureStore, "write_many", checked_write)
    monkeypatch.setattr(extraction, "atomic_write_jsonl", checked_publish)

    def run(name, resume=None):
        return extract_pooled_features(
            spec,
            adapter=adapter,
            manifest=records,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id=name,
            backend=_CV2(140),
            resume_source=resume,
        )

    old = run("long")
    assert old.completed and old.records_written == 69
    assert max(sizes) == 64
    assert len(batch_sizes) == 11 and sum(batch_sizes) == 69 and max(batch_sizes) == 7
    assert publications.count(Path(old.run_dir) / "shards" / "long" / "index.jsonl") == 1
    adapter.seen_batches.clear()
    result = run("copy", old.run_dir)
    assert result.completed and not adapter.seen_batches
    assert publications.count(Path(result.run_dir) / "shards" / "long" / "index.jsonl") == 1
    assert max(sizes) == 64


def test_cross_block_batch_failure_keeps_only_completed_block_unpublished(tmp_path, monkeypatch):
    records = [_record("long", split="train", anomaly=False, frames=140)]
    _touch(tmp_path, records)
    adapter = _Adapter()
    sampling = _sampling(records, root=tmp_path, regime="test_dense", clips=32, window_stride=2)
    spec = replace(
        _spec(_representation(adapter, reducer="identity"), sampling, kind="dense", stride=2),
        micro_batch_size=7,
    )
    original_write = FeatureStore.write_many

    def fail_second_block(store, items):
        if store.root.name == "000001":
            raise OSError("injected second-block write failure")
        return original_write(store, items)

    monkeypatch.setattr(FeatureStore, "write_many", fail_second_block)
    result = extract_pooled_features(
        spec, adapter=adapter, manifest=records, dataset_root=tmp_path,
        output_root=tmp_path / "runs", run_id="cross-block-failure", backend=_CV2(140),
    )
    assert not result.completed and result.records_written == 64
    root = Path(result.run_dir)
    shard = root / "shards" / "long"
    assert len(FeatureStore(shard / "blocks" / "000000").records()) == 64
    assert not (shard / "blocks" / "000001" / "index.jsonl").exists()
    assert not (shard / "index.jsonl").exists()
    assert not (root / "index.jsonl").exists()


def test_interrupted_microshards_do_not_publish_video_index(tmp_path):
    records, adapter, spec = _setup(tmp_path)
    encode = adapter.encode

    def interrupted(batch, train=False):
        if adapter.seen_batches:
            raise RuntimeError("foreign GPU contention")
        return encode(batch, train=train)

    adapter.encode = interrupted
    old = _run(tmp_path, records, adapter, spec, "partial")
    assert not old.completed
    source = Path(old.run_dir)
    assert (source / "shards" / "one" / "blocks" / "000000" / "index.jsonl").is_file()
    assert not (source / "shards" / "one" / "index.jsonl").exists()
    adapter = _Adapter()
    result = _run(tmp_path, records, adapter, spec, "fresh", source)
    assert result.completed and len(adapter.seen_batches) == 10
    assert json.loads(Path(result.status_path).read_text())["resume"]["reused_videos"] == 0


def test_controller_reuses_all_roles_and_trains_new_head(tmp_path):
    from test_paper_extraction import _verified_identity

    from vadbench.data.manifest import write_manifest_jsonl
    from vadbench.paper.controller import DetectionExperimentRequest, run_detection_experiment

    train = [
        _record("normal", split="train", anomaly=False),
        _record("abnormal", split="train", anomaly=True),
    ]
    validation = [_record("validation", split="val", anomaly=False)]
    evaluation = [_record("evaluation", split="val", anomaly=True)]
    _touch(tmp_path, [*train, *validation, *evaluation])
    request = DetectionExperimentRequest(
        encoder="fixture",
        device="cpu",
        dataset_root=str(tmp_path),
        train_manifest=str(write_manifest_jsonl(train, tmp_path / "train.jsonl")),
        validation_manifest=str(write_manifest_jsonl(validation, tmp_path / "val.jsonl")),
        evaluation_manifest=str(write_manifest_jsonl(evaluation, tmp_path / "eval.jsonl")),
        output_root=str(tmp_path / "runs"),
        output_dim=4,
        frame_stride=1,
        epochs=1,
        run_id="original",
    )
    adapter = _Adapter()

    def factory(_summary):
        return adapter, {"constructor": {"clip_frames": 4}, "identity": _verified_identity()}

    original = run_detection_experiment(request, adapter_factory=factory, video_backend=_CV2(64))
    before = _snapshot(Path(original.run_dir))
    adapter.seen_batches.clear()
    resumed = run_detection_experiment(
        replace(request, run_id="resumed", resume_source_run=original.run_dir),
        adapter_factory=factory,
        video_backend=_CV2(64),
    )
    assert resumed.completed and not adapter.seen_batches
    assert resumed.training_checkpoint != original.training_checkpoint
    assert Path(resumed.training_checkpoint).is_file()
    for role in ("train", "validation", "evaluation"):
        receipt = json.loads(
            (Path(resumed.run_dir) / "features" / role / "status.json").read_text()
        )
        assert receipt["resume"]["decoded_videos"] == 0
        assert receipt["resume"]["reused_videos"] > 0
    assert _snapshot(Path(original.run_dir)) == before


@pytest.mark.parametrize(
    "encoder_id,strategy",
    [
        ("videomaev2", "global_uniform"),
        ("videomaev2", "pair_linear"),
        ("vjepa2", "global_uniform"),
    ],
)
def test_native_reducer_receipts_and_tampering(tmp_path, encoder_id, strategy):
    import copy

    adapter, spec, old, run = _native_resume_source(tmp_path, encoder_id, strategy)
    source = Path(old.run_dir)
    before = _snapshot(source)
    adapter.seen_batches.clear()
    good = run("valid_resume", source)
    assert good.completed and adapter.seen_batches == [], good.failures
    assert _snapshot(source) == before
    # A second resume also retains exact original tail-batch execution evidence.
    repeated = run("repeated_resume", good.run_dir)
    assert repeated.completed and adapter.seen_batches == [], repeated.failures
    index = source / "shards" / "native" / "index.jsonl"
    original = index.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    valid = rows[0]["metadata"]["reduction_execution"]
    mutations = {
        "encoder_id": "other_encoder",
        "intervention_depth": 99,
        "native_input_tokens": 1,
        "gathered_tokens": valid["gathered_tokens"] + 1,
        "gathered_shape": [7, 123, 88],
        "suffix_shapes": {},
        "vjepa2_original_rope_positions": encoder_id != "vjepa2",
        "vjepa2_position_injections": 99,
        "recorded_position_masks": True,
        "per_layer_token_counts": {},
        "plugin_overhead_ms": {"gather_ms": -1.0, "transform_ms": 0.0},
    }
    for name, value in mutations.items():
        changed = copy.deepcopy(rows)
        changed[0]["metadata"]["reduction_execution"][name] = value
        atomic_write_jsonl(index, changed)
        tampered = _snapshot(source)
        rejected = run(f"invalid_{name}", source)
        assert not rejected.completed and adapter.seen_batches == [], name
        assert "reduction execution" in rejected.failures[0]["message"], rejected.failures
        assert not (Path(rejected.run_dir) / "index.jsonl").exists()
        assert _snapshot(source) == tampered
    for name, target, mutate in (
        ("batch", 0, lambda r: r["gathered_shape"].__setitem__(0, 1)),
        (
            "hidden",
            0,
            lambda r: r["gathered_shape"].__setitem__(2, spec.representation.output_dim + 1),
        ),
        ("tail", -1, lambda r: r["gathered_shape"].__setitem__(0, 7)),
        ("suffix_gap", 0, lambda r: r["suffix_shapes"].update({"5": r["gathered_shape"]})),
        ("suffix_hidden", 0, lambda r: r["suffix_shapes"]["1"].__setitem__(2, 999)),
        ("timing_boolean", 0, lambda r: r["plugin_overhead_ms"].__setitem__("gather_ms", True)),
        ("timing_missing", 0, lambda r: r.pop("plugin_overhead_ms")),
    ):
        changed = copy.deepcopy(rows)
        mutate(changed[target]["metadata"]["reduction_execution"])
        atomic_write_jsonl(index, changed)
        rejected = run(f"invalid_{name}", source)
        assert not rejected.completed and adapter.seen_batches == [], name
        assert "reduction execution" in rejected.failures[0]["message"]
    index.write_bytes(original)


def test_git_diagnostic_changes_preserve_native_identity_but_code_and_weights_do_not(tmp_path):
    records, adapter, spec = _setup(tmp_path)
    old = _run(tmp_path, records, adapter, spec, "original")
    source = Path(old.run_dir)
    changed = replace(
        spec,
        verified_encoder_identity={
            **spec.verified_encoder_identity,
            "code": {"commit": "new-commit", "source_sha256": "new-broad-tree"},
        },
    )
    adapter.seen_batches.clear()
    good = _run(tmp_path, records, adapter, changed, "new_git", source)
    assert good.completed and adapter.seen_batches == []
    resolved_path = source / "resolved.json"
    original = resolved_path.read_bytes()
    for name in ("native_code", "weights"):
        resolved = json.loads(original)
        if name == "native_code":
            first = next(iter(resolved["runtime"]["implementation_files"].values()))
            first["sha256"] = "0" * 64
        else:
            resolved["runtime"]["verified_encoder_identity"]["checkpoint"]["sha256"] = {
                "model.bin": "0" * 64
            }
        resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
        rejected = _run(tmp_path, records, adapter, spec, name, source)
        assert not rejected.completed and adapter.seen_batches == []
    resolved_path.write_bytes(original)
