from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.manifest import VideoManifestRecord
from vadbench.features import FeatureStore
from vadbench.paper.compatibility import (
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
)
from vadbench.paper.detection import DetectionConfig, predict_detector, train_detector
from vadbench.paper.extraction import (
    PooledExtractionSpec,
    adapter_runtime_summary,
    extract_pooled_features,
    make_sampling_identity,
    representation_from_verified_encoder,
)


class _Capture:
    def __init__(self, backend: _CV2, frames: list[np.ndarray]) -> None:
        self.backend = backend
        self.frames = frames
        self.position = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, property_id: int) -> float:
        values = {
            self.backend.CAP_PROP_FRAME_COUNT: len(self.frames),
            self.backend.CAP_PROP_FPS: self.backend.fps,
            self.backend.CAP_PROP_FRAME_WIDTH: 2,
            self.backend.CAP_PROP_FRAME_HEIGHT: 2,
        }
        return float(values[property_id])

    def set(self, _property_id: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self):
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self) -> None:
        pass


class _CV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FPS = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(
        self, count: int, fps: float = 8.0, counts_by_name: dict[str, int] | None = None
    ) -> None:
        self.fps = fps
        self.frames = []
        for index in range(count):
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            frame[..., 0] = index  # B; reader converts it to RGB's final channel.
            self.frames.append(frame)
        self.counts_by_name = dict(counts_by_name or {})

    def VideoCapture(self, path: str) -> _Capture:  # noqa: N802
        count = self.counts_by_name.get(Path(path).name, len(self.frames))
        return _Capture(self, self.frames[:count])


class _Adapter:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_training=False,
        fixed_num_frames=4,
        min_frames=4,
        max_frames=4,
    )
    backend = "fixture"
    implementation_source = "fixture-native"
    preprocess_profile = "fixture-rgb"
    pooling = "mean"
    revision = "fixture-r1"

    def __init__(self) -> None:
        self.seen_batches = []

    def encode(self, batch, train=False):
        assert not train
        assert all(item.startswith("sample-") for item in batch.video_ids)
        assert set(batch.metadata) <= {"source_num_frames", "source_fps"}
        self.seen_batches.append(batch)
        values = np.asarray(batch.frame_indices, dtype=np.float32).mean(axis=1)
        pooled = np.stack([values + offset for offset in range(4)], axis=1)
        features = np.repeat(pooled[:, None, :], 3, axis=1)
        starts = np.zeros((batch.batch_size, 3), dtype=np.float32)
        ends = np.ones((batch.batch_size, 3), dtype=np.float32)
        timeline = TokenTimeline(
            start_s=starts, end_s=ends, valid_mask=np.ones_like(starts, dtype=bool)
        )
        return EncoderOutput(features=features, pooled=pooled, timeline=timeline)


def _record(video_id: str, *, split: str, anomaly: bool, frames: int = 64) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=frames,
        fps=8.0,
        duration_seconds=frames / 8.0,
    )


def _touch(root: Path, records) -> None:
    for record in records:
        (root / record.path).touch()


def _verified_identity(
    *, code: str = "fixture-code", constructor: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "adapter": "fixture",
        "constructor": {"clip_frames": 4} if constructor is None else constructor,
        "checkpoint": {"id": "fixture", "sha256": {"model.bin": "fixture"}},
        "code": {"source_sha256": code},
    }


def _representation(adapter: _Adapter, *, reducer: str) -> RepresentationIdentity:
    return representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": reducer},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )


def _sampling(
    records,
    *,
    root: Path,
    regime: str,
    clips: int,
    window_stride: int = 1,
    short_policy: str = "strict",
) -> SamplingIdentity:
    return make_sampling_identity(
        records,
        dataset_root=root,
        sampling_kind="uniform_full" if regime == "train_32" else "dense",
        clip_frames=4,
        frame_stride=1,
        num_segments=clips,
        window_stride=window_stride,
        short_policy=short_policy,  # type: ignore[arg-type]
    )


def _spec(
    representation,
    sampling,
    *,
    kind: str,
    stride: int = 1,
    identity: dict[str, object] | None = None,
    short_policy: str = "strict",
) -> PooledExtractionSpec:
    return PooledExtractionSpec(
        runtime_id="fixture",
        verified_encoder_identity=_verified_identity() if identity is None else identity,
        representation=representation,
        sampling=sampling,
        sampling_kind=kind,  # type: ignore[arg-type]
        clip_frames=4,
        frame_stride=1,
        num_segments=32,
        window_stride=stride,
        short_policy=short_policy,  # type: ignore[arg-type]
        micro_batch_size=7,
    )


def test_pooled_uniform_and_dense_extraction_feed_detector(tmp_path: Path) -> None:
    train = [
        _record("normal", split="train", anomaly=False),
        _record("abnormal", split="train", anomaly=True),
    ]
    test = [_record("evaluation", split="test", anomaly=True)]
    _touch(tmp_path, [*train, *test])
    train_adapter = _Adapter()
    train_representation = _representation(train_adapter, reducer="identity")
    eval_representation = _representation(_Adapter(), reducer="token_merge")
    train_sampling = _sampling(train, root=tmp_path, regime="train_32", clips=32)
    eval_sampling = _sampling(test, root=tmp_path, regime="test_dense", clips=31, window_stride=2)

    train_result = extract_pooled_features(
        _spec(train_representation, train_sampling, kind="uniform_full"),
        adapter=train_adapter,
        manifest=train,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="train",
        backend=_CV2(64),
    )
    eval_result = extract_pooled_features(
        _spec(eval_representation, eval_sampling, kind="dense", stride=2),
        adapter=_Adapter(),
        manifest=test,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="dense",
        backend=_CV2(64, counts_by_name={"short.mp4": 3}),
    )

    assert train_result.completed and eval_result.completed
    assert train_result.records_written == 64
    assert eval_result.records_written == 31
    assert train_result.encoder_fingerprint != eval_result.encoder_fingerprint
    store = FeatureStore(train_result.feature_root)
    row = store.records()[0]
    bundle = store.load_bundle(row)
    assert set(bundle) == {"features", "pooled"}
    np.testing.assert_allclose(bundle["features"][0], bundle["pooled"])
    assert row.metadata["storage"]["intermediate_tokens_stored"] is False
    assert row.metadata["encoder_output"]["token_count"] == 3
    assert all("is_anomaly" not in item.metadata for item in train_adapter.seen_batches)

    declaration = CompatibilityDeclaration(
        mode="direct_insert",
        training_representation=train_representation,
        evaluation_representation=eval_representation,
        training_sampling=train_sampling,
        evaluation_sampling=eval_sampling,
        baseline_evaluation_sampling=eval_sampling,
        training_identity=TrainingIdentity(
            head={"kind": "topk_mil", "k": 1},
            fit_split_digest="sha256:" + compute_manifest_sha256(train),
            seed=3,
            optimization={"epochs": 1, "lr": 0.01},
        ),
        sampling_change="train32_to_testdense",
    )
    config = DetectionConfig(
        declaration=declaration,
        training_encoder_fingerprint=train_result.encoder_fingerprint,
        evaluation_encoder_fingerprint=eval_result.encoder_fingerprint,
        head="topk",
        head_kwargs={"k": 1},
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        max_steps=1,
        seed=3,
    )
    training = train_detector(
        config,
        feature_store=train_result.feature_root,
        train_manifest=train,
        output_dir=tmp_path / "detector",
        device="cpu",
    )
    predictions = predict_detector(
        config,
        feature_store=eval_result.feature_root,
        evaluation_manifest=test,
        training=training,
        output_path=tmp_path / "predictions.jsonl",
        device="cpu",
    )
    assert [
        frame for item in predictions for frame in range(item.frame_start, item.frame_end)
    ] == list(range(64))


def test_short_video_leaves_only_shard_and_partial_status(tmp_path: Path) -> None:
    records = [
        _record("good", split="train", anomaly=False, frames=64),
        _record("short", split="train", anomaly=True, frames=3),
    ]
    _touch(tmp_path, records)
    representation = _representation(_Adapter(), reducer="identity")
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)

    result = extract_pooled_features(
        _spec(representation, sampling, kind="uniform_full"),
        adapter=_Adapter(),
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="partial",
        backend=_CV2(64, counts_by_name={"short.mp4": 3}),
    )

    assert not result.completed
    assert result.feature_root is None
    assert result.records_written == 32
    assert result.failures[0]["video_id"] == "short"
    assert "短视频" in result.failures[0]["message"]
    assert not (Path(result.run_dir) / "index.jsonl").exists()
    assert (Path(result.run_dir) / "shards" / "good" / "index.jsonl").is_file()


def test_spec_binds_uniform_segments_and_dense_window_stride_to_sampling_identity(
    tmp_path: Path,
) -> None:
    adapter = _Adapter()
    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    representation = _representation(adapter, reducer="identity")
    uniform = _sampling(records, root=tmp_path, regime="train_32", clips=32)
    dense = _sampling(records, root=tmp_path, regime="test_dense", clips=31, window_stride=2)

    assert "video_reader_sha256" in uniform.frame_selection["implementation"]
    with pytest.raises(ValueError, match="num_segments"):
        replace(_spec(representation, uniform, kind="uniform_full"), num_segments=16)
    with pytest.raises(ValueError, match="window_stride"):
        replace(_spec(representation, dense, kind="dense", stride=2), window_stride=1)
    with pytest.raises(ValueError, match="implementation evidence"):
        replace(
            _spec(representation, uniform, kind="uniform_full"),
            sampling=replace(
                uniform,
                frame_selection={
                    **uniform.frame_selection,
                    "implementation": {"dense_sampling_sha256": "stale"},
                },
            ),
        )


def test_explicit_short_policy_binds_identity_resolved_config_and_actual_frame_stride(
    tmp_path: Path,
) -> None:
    class _ShortAdapter(_Adapter):
        capabilities = EncoderCapabilities(
            supports_fixed_clip=True,
            supports_training=False,
            fixed_num_frames=64,
            min_frames=64,
            max_frames=64,
        )

    records = [_record("short-vjepa", split="train", anomaly=False, frames=104)]
    _touch(tmp_path, records)
    adapter = _ShortAdapter()
    representation = _representation(adapter, reducer="identity")
    fallback = make_sampling_identity(
        records,
        dataset_root=tmp_path,
        sampling_kind="uniform_full",
        clip_frames=64,
        frame_stride=2,
        num_segments=32,
        short_policy="stride1_if_needed",
    )
    strict = make_sampling_identity(
        records,
        dataset_root=tmp_path,
        sampling_kind="uniform_full",
        clip_frames=64,
        frame_stride=2,
        num_segments=32,
    )
    assert fallback.fingerprint != strict.fingerprint
    assert fallback.frame_selection["short_video_policy"] == "stride1_if_needed"
    spec = PooledExtractionSpec(
        runtime_id="fixture",
        verified_encoder_identity=_verified_identity(),
        representation=representation,
        sampling=fallback,
        sampling_kind="uniform_full",
        clip_frames=64,
        frame_stride=2,
        num_segments=32,
        short_policy="stride1_if_needed",
    )
    with pytest.raises(ValueError, match="short_video_policy"):
        replace(spec, short_policy="strict")

    result = extract_pooled_features(
        spec,
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="short-policy",
        backend=_CV2(104),
    )

    assert result.completed
    resolved = json.loads((Path(result.run_dir) / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["spec"]["short_policy"] == "stride1_if_needed"
    row = FeatureStore(result.feature_root).records()[0]
    assert row.metadata["sampling"]["actual_frame_stride"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dimension", "representation.output_dim"),
        ("precision", "dtype does not match"),
    ],
)
def test_extraction_rejects_actual_output_dimension_or_precision_mismatch(
    tmp_path: Path, change: str, message: str
) -> None:
    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    adapter = _Adapter()
    representation = _representation(adapter, reducer="identity")
    if change == "dimension":
        representation = replace(representation, output_dim=5)
    else:
        representation = replace(representation, precision="float16")
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)

    result = extract_pooled_features(
        _spec(representation, sampling, kind="uniform_full"),
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id=change,
        backend=_CV2(64),
    )

    assert not result.completed
    assert message in result.failures[0]["message"]
    assert not (Path(result.run_dir) / "index.jsonl").exists()


def test_extraction_refuses_representation_without_matching_verified_weight_evidence(
    tmp_path: Path,
) -> None:
    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    adapter = _Adapter()
    representation = _representation(adapter, reducer="identity")
    representation = replace(
        representation,
        backbone=replace(representation.backbone, weights_digest="sha256:unverified"),
    )
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)

    with pytest.raises(ValueError, match="weights_digest"):
        extract_pooled_features(
            _spec(representation, sampling, kind="uniform_full"),
            adapter=adapter,
            manifest=records,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id="unverified",
            backend=_CV2(64),
        )


def test_backbone_code_identity_ignores_broad_orchestration_git_evidence() -> None:
    adapter = _Adapter()
    first = representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(code="whole-repository-state-a"),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": "identity"},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )
    second = representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(code="whole-repository-state-b"),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": "identity"},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )

    assert first.backbone.code_digest == second.backbone.code_digest
    assert first.backbone.weights_digest == second.backbone.weights_digest


@dataclass(frozen=True)
class _V2RuntimeConfig:
    image_size: int = 224
    pooling: str = "auto"
    model_name: str = r"D:\assets\videomaev2"
    device_str: str = "cuda:0"


class _Processor:
    def __init__(self, *, do_rescale: bool = True, size: int = 224) -> None:
        self.do_rescale = do_rescale
        self.size = size

    def to_dict(self) -> dict[str, object]:
        return {
            "do_rescale": self.do_rescale,
            "size": {"height": self.size, "width": self.size},
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "device": "cuda:0",
        }


class _NestedModel:
    config = {"hidden_size": 4, "patch_size": 16}


class _Backbone:
    def __init__(self) -> None:
        self.config = {"image_size": 224, "num_frames": 4}
        self.model = _NestedModel()


class _V2StyleEncoder:
    def __init__(self) -> None:
        self.cfg = _V2RuntimeConfig()
        self.backbone = _Backbone()
        self.processor = _Processor()


class _V2StyleAdapter:
    capabilities = _Adapter.capabilities
    backend = "fixture-v2"
    implementation_source = "fixture-v2-native"
    preprocess_profile = "fixture-rgb"
    encode = _Adapter.encode

    def __init__(self) -> None:
        self.encoder = _V2StyleEncoder()
        self.seen_batches = []


def _v2_representation(
    adapter: _V2StyleAdapter, identity: dict[str, object]
) -> RepresentationIdentity:
    return representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=identity,
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "pooled"},
        reducer={"name": "identity"},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )


def test_verified_constructor_and_loaded_v2_runtime_configs_change_representation_and_cache(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    identity = _verified_identity(
        constructor={
            "model_name": {"checkpoint_sha256": {"model.bin": "fixture"}},
            "image_size": 224,
            "num_frames": 4,
            "pooling": "auto",
        }
    )
    adapter = _V2StyleAdapter()
    adapter.encoder.backbone.config["torch_dtype"] = torch.float32
    first = _v2_representation(adapter, identity)
    assert (
        adapter_runtime_summary(adapter)["runtime_configurations"]["backbone_config"]["torch_dtype"]
        == "torch.float32"
    )

    adapter.encoder.backbone.config["torch_dtype"] = torch.float16
    changed_dtype = _v2_representation(adapter, identity)
    assert changed_dtype.fingerprint != first.fingerprint
    adapter.encoder.backbone.config["torch_dtype"] = torch.float32

    changed_constructor = _v2_representation(
        adapter,
        _verified_identity(
            constructor={
                "model_name": {"checkpoint_sha256": {"model.bin": "fixture"}},
                "image_size": 336,
                "num_frames": 4,
                "pooling": "auto",
            }
        ),
    )
    assert changed_constructor.fingerprint != first.fingerprint

    adapter.encoder.cfg = replace(adapter.encoder.cfg, pooling="cls")
    changed_pooling = _v2_representation(adapter, identity)
    assert changed_pooling.fingerprint != first.fingerprint

    adapter.encoder.processor.do_rescale = False
    changed_rescale = _v2_representation(adapter, identity)
    assert changed_rescale.fingerprint != changed_pooling.fingerprint

    adapter.encoder.processor.size = 112
    changed_size = _v2_representation(adapter, identity)
    assert changed_size.fingerprint != changed_rescale.fingerprint

    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)
    with pytest.raises(ValueError, match="code_digest"):
        extract_pooled_features(
            _spec(first, sampling, kind="uniform_full", identity=identity),
            adapter=adapter,
            manifest=records,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id="frozen-runtime-mismatch",
            backend=_CV2(64),
        )

    result = extract_pooled_features(
        _spec(changed_size, sampling, kind="uniform_full", identity=identity),
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="changed-runtime",
        backend=_CV2(64),
    )
    assert result.completed

    adapter.encoder.cfg = replace(adapter.encoder.cfg, pooling="mean")
    current = _v2_representation(adapter, identity)
    changed_result = extract_pooled_features(
        _spec(current, sampling, kind="uniform_full", identity=identity),
        adapter=adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id="changed-runtime-cache",
        backend=_CV2(64),
    )
    assert changed_result.completed
    assert changed_result.encoder_fingerprint != result.encoder_fingerprint


def test_constructor_identity_ignores_deployment_paths_and_device() -> None:
    adapter = _Adapter()
    common = {"image_size": 224, "model_name": {"checkpoint_sha256": {"model.bin": "fixture"}}}
    first = representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(
            constructor=common | {"checkpoint_path": r"D:\weights\a", "device": "cuda:0"}
        ),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": "identity"},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )
    second = representation_from_verified_encoder(
        runtime_id="fixture",
        adapter=adapter,
        verified_encoder_identity=_verified_identity(
            constructor=common | {"checkpoint_path": r"E:\weights\b", "device": "cpu"}
        ),
        preprocessing={"profile": "fixture-rgb"},
        readout={"kind": "mean"},
        reducer={"name": "identity"},
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )
    assert first.fingerprint == second.fingerprint


def test_same_manifest_with_changed_video_bytes_rejects_sampling_identity(tmp_path: Path) -> None:
    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    path = tmp_path / "video.mp4"
    path.write_bytes(b"first-content")
    adapter = _Adapter()
    representation = _representation(adapter, reducer="identity")
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)
    path.write_bytes(b"changed-content")

    with pytest.raises(ValueError, match="video content evidence"):
        extract_pooled_features(
            _spec(representation, sampling, kind="uniform_full"),
            adapter=adapter,
            manifest=records,
            dataset_root=tmp_path,
            output_root=tmp_path / "runs",
            run_id="changed-bytes",
            backend=_CV2(64),
        )
