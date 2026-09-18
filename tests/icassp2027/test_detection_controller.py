from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.paper.controller import (
    DetectionExperimentRequest,
    _disjoint,
    run_detection_experiment,
)


class _Capture:
    def __init__(self, backend):
        self.backend, self.position = backend, 0

    def isOpened(self):
        return True  # noqa: N802

    def get(self, value):
        return {
            self.backend.CAP_PROP_FRAME_COUNT: 64,
            self.backend.CAP_PROP_FPS: 8,
            self.backend.CAP_PROP_FRAME_WIDTH: 2,
            self.backend.CAP_PROP_FRAME_HEIGHT: 2,
        }[value]

    def set(self, _property, value):
        self.position = int(value)
        return True

    def read(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[..., 0] = self.position
        self.position += 1
        return True, frame

    def release(self):
        pass


class _CV:
    (
        CAP_PROP_FRAME_COUNT,
        CAP_PROP_FPS,
        CAP_PROP_FRAME_WIDTH,
        CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_POS_FRAMES,
    ) = range(1, 6)

    def VideoCapture(self, _path):
        return _Capture(self)  # noqa: N802


class _Adapter:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_training=False,
        fixed_num_frames=4,
        min_frames=4,
        max_frames=4,
    )
    preprocess_profile, pooling, backend, implementation_source = (
        "toy-rgb",
        "mean",
        "toy",
        "toy-native",
    )

    def encode(self, batch, train=False):
        assert not train and set(batch.metadata) <= {"source_num_frames", "source_fps"}
        pooled = np.repeat(np.arange(4, dtype=np.float32)[None, :], batch.batch_size, axis=0)
        tokens = getattr(self, "context_tokens", 2)
        return EncoderOutput(
            features=np.repeat(pooled[:, None, :], tokens, axis=1),
            pooled=pooled,
            timeline=TokenTimeline(
                start_s=np.zeros((batch.batch_size, tokens), dtype=np.float32),
                end_s=np.ones((batch.batch_size, tokens), dtype=np.float32),
                valid_mask=np.ones((batch.batch_size, tokens), dtype=bool),
            ),
        )


def _record(video_id, split, anomaly):
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
    )


@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("short_policy", ["strict", "stride1_if_needed"])
def test_engineering_controller_freezes_extracts_and_predicts(
    tmp_path: Path, with_validation: bool, short_policy: str
):
    train = [_record("normal", "train", False), _record("abnormal", "train", True)]
    validation = [_record("holdout", "val", False)] if with_validation else []
    evaluation = [_record("evaluation", "val", True)]
    for record in [*train, *validation, *evaluation]:
        (tmp_path / record.path).touch()
    train_path = write_manifest_jsonl(train, tmp_path / "train.jsonl")
    evaluation_path = write_manifest_jsonl(evaluation, tmp_path / "evaluation.jsonl")
    validation_path = (
        write_manifest_jsonl(validation, tmp_path / "validation.jsonl") if validation else None
    )
    identity = {
        "adapter": "toy",
        "constructor": {"clip_frames": 4},
        "checkpoint": {"id": "toy", "sha256": {"toy.bin": "fixture"}},
        "code": {"source_sha256": "broad"},
    }
    request = DetectionExperimentRequest(
        encoder="toy",
        device="cpu",
        dataset_root=str(tmp_path),
        train_manifest=str(train_path),
        validation_manifest=None if validation_path is None else str(validation_path),
        evaluation_manifest=str(evaluation_path),
        output_root=str(tmp_path / "runs"),
        frame_stride=2,
        short_policy=short_policy,
        dense_window_stride=None,
        output_dim=4,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        run_id="toy",
    )
    result = run_detection_experiment(
        request,
        adapter_factory=lambda _summary: (
            _Adapter(),
            {"constructor": {"clip_frames": 4}, "identity": identity},
        ),
        video_backend=_CV(),
    )
    run = Path(result.run_dir)
    assert (
        result.completed
        and Path(result.training_checkpoint).is_file()
        and Path(result.predictions).is_file()
    )
    assert (
        result.metrics is None
        and (run / "frozen" / "train.jsonl").is_file()
        and (run / "result.json").is_file()
    )
    assert json.loads((run / "resolved_sampling.json").read_text())["dense_window_stride"] == 4
    assert json.loads((run / "resolved_sampling.json").read_text())["short_policy"] == short_policy
    for role in ("train", "evaluation", *(["validation"] if with_validation else [])):
        extracted = json.loads((run / "features" / role / "resolved.json").read_text())
        assert extracted["spec"]["short_policy"] == short_policy
        assert extracted["spec"]["sampling"]["frame_selection"]["short_video_policy"] == short_policy
    role_checks = json.loads((run / "frozen" / "role_checks.json").read_text())
    assert role_checks["near_duplicate_exclusion"] == "not_established_by_video_identity_checks"
    if with_validation:
        assert (run / "features" / "validation" / "index.jsonl").is_file()


def test_disjoint_rejects_same_source_under_different_video_ids():
    fit = replace(_record("a", "train", False), metadata={"source_id": "camera-event-1"})
    holdout = replace(_record("b", "val", True), metadata={"source_group": "camera-event-1"})
    with pytest.raises(ValueError, match="source:camera-event-1"):
        _disjoint([fit], [], [holdout])


@pytest.mark.parametrize("encoder", ["videomaev2", "timesformer", "videomae", "clip"])
def test_real_profile_uses_external_assets_and_rejects_inactive(
    tmp_path: Path, monkeypatch, encoder
):
    from vadbench import orchestration
    from vadbench.config import ConfigError
    from vadbench.registry import ENCODER_REGISTRY

    repository = Path(__file__).resolve().parents[2]
    project_files = [
        "projects/icassp2027/profile.yaml",
        "projects/icassp2027/protocol.yaml",
        "configs/encoders/videomaev2-base.yaml",
        "configs/encoders/timesformer.yaml",
        "configs/encoders/vjepa2.yaml",
        "configs/encoders/videomae.yaml",
        "registry/checkpoints.yaml",
    ]
    for name in project_files:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, destination)
    external_weights = tmp_path / "existing-assets" / "verified-v2"
    configured_encoder = "videomaev2" if encoder == "clip" else encoder
    frames = 8 if encoder == "timesformer" else 16
    conversion = "np" if encoder in {"timesformer", "videomae"} else None
    (tmp_path / "projects/icassp2027/assets.local.yaml").write_text(
        f"checkpoint_paths:\n  {configured_encoder}: " + external_weights.as_posix() + "\n",
        encoding="utf-8",
    )
    train = [_record("a", "train", False), _record("b", "train", True)]
    evaluation = [_record("c", "val", True)]
    for record in [*train, *evaluation]:
        (tmp_path / record.path).touch()
    train_path = write_manifest_jsonl(train, tmp_path / "train.jsonl")
    eval_path = write_manifest_jsonl(evaluation, tmp_path / "eval.jsonl")
    received = []

    def verified_identity(definition, *, project_root):
        assert Path(project_root) == tmp_path
        assert Path(definition["checkpoint"]["local_path"]) == external_weights
        if conversion is not None:
            assert definition["constructor"]["processor_tensor_type"] == conversion
        return {
            "adapter": encoder,
            "constructor": {
                "model_name": {"checkpoint_sha256": {"toy.bin": "fixture"}},
                "num_frames": frames,
                **({"processor_tensor_type": conversion} if conversion is not None else {}),
            },
            "checkpoint": {"id": "toy", "sha256": {"toy.bin": "fixture"}},
            "code": {"source_sha256": "fixture"},
        }

    def create(name, **constructor):
        received.append((name, constructor))
        asset_key = "model_name" if encoder == "videomaev2" else "model_path"
        assert Path(constructor[asset_key]) == external_weights
        assert constructor["device"] == "cpu"
        if conversion is not None:
            assert constructor["processor_tensor_type"] == conversion
        adapter = _Adapter()
        adapter.capabilities = replace(
            adapter.capabilities, fixed_num_frames=frames, min_frames=frames, max_frames=frames
        )
        return adapter

    monkeypatch.setattr(orchestration, "encoder_identity", verified_identity)
    monkeypatch.setattr(ENCODER_REGISTRY, "create", create)
    request = DetectionExperimentRequest(
        encoder=encoder,
        project=str(tmp_path / "projects/icassp2027/profile.yaml"),
        device="cpu",
        dataset_root=str(tmp_path),
        train_manifest=str(train_path),
        validation_manifest=None,
        evaluation_manifest=str(eval_path),
        output_root=str(tmp_path / "runs"),
        output_dim=4,
        processor_tensor_type=conversion,
    )
    if encoder == "clip":
        with pytest.raises(ConfigError, match="not active"):
            run_detection_experiment(request, video_backend=_CV())
        assert not received
    else:
        assert run_detection_experiment(request, video_backend=_CV()).completed
        assert len(received) == 1


@pytest.mark.parametrize("reducer", ["global_uniform", "paired_random", "pair_linear"])
def test_reduced_controller_binds_context_across_all_roles(tmp_path, monkeypatch, reducer):
    from vadbench.features import FeatureStore
    from vadbench.paper import reduction_setup

    train = [_record("fit-normal", "train", False), _record("fit-positive", "train", True)]
    validation = [_record("select", "val", False)]
    evaluation = [_record("canary", "val", True)]
    paths = {}
    for name, records in (("train", train), ("validation", validation), ("evaluation", evaluation)):
        for record in records:
            (tmp_path / record.path).touch()
        paths[name] = str(write_manifest_jsonl(records, tmp_path / f"{name}.jsonl"))
    adapter = _Adapter()
    identity = {"adapter": "toy", "constructor": {"clip_frames": 4},
                "checkpoint": {"id": "toy", "sha256": {"toy.bin": "fixture"}}}

    class Context:
        reducer_identity = {"name": reducer, "seed": 0}
        calls = 0
        exits = 0

        def __call__(self, batch):
            assert all(value.startswith("sample-") for value in batch.video_ids)
            self.calls += 1
            return self

        def __enter__(self):
            adapter.context_tokens = 1
            return self

        def __exit__(self, *_args):
            del adapter.context_tokens
            self.exits += 1

        def validate_execution(self):
            return {"native_input_tokens": 2, "gathered_tokens": 1}

    context = Context()
    setup_calls = []

    def prepare(actual, encoder, batch, **kwargs):
        assert actual is adapter and encoder == "toy"
        assert batch.batch_size == 1 and batch.num_frames == 4
        assert kwargs["reducer"] == reducer and kwargs["output_dim"] == 4
        assert kwargs["verified_encoder_identity"] == identity
        setup_calls.append(kwargs)
        return context, {"status": "synthetic_fixture_only", "reducer": reducer}

    monkeypatch.setattr(reduction_setup, "prepare_reduction", prepare)
    request = DetectionExperimentRequest(
        encoder="toy", device="cpu", dataset_root=str(tmp_path),
        train_manifest=paths["train"], validation_manifest=paths["validation"],
        evaluation_manifest=paths["evaluation"], output_root=str(tmp_path / "runs"),
        output_dim=4, reducer=reducer,
        calibration_run=str(tmp_path / "calibration") if reducer == "pair_linear" else None,
    )
    result = run_detection_experiment(
        request, adapter_factory=lambda _summary: (adapter, {"constructor": {"clip_frames": 4}, "identity": identity}),
        video_backend=_CV(),
    )
    assert result.completed and len(setup_calls) == 1
    assert context.calls == context.exits and context.calls > 3
    assert not hasattr(adapter, "context_tokens")
    for role in ("train", "validation", "evaluation"):
        store = FeatureStore(Path(result.run_dir) / "features" / role)
        assert all(row.metadata["reduction_execution"]["gathered_tokens"] == 1 for row in store.records())
        resolved = json.loads((store.root / "resolved.json").read_text())
        assert resolved["spec"]["representation"]["reducer"]["name"] == reducer
