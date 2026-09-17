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
        return EncoderOutput(
            features=np.repeat(pooled[:, None, :], 2, axis=1),
            pooled=pooled,
            timeline=TokenTimeline(
                start_s=np.zeros((batch.batch_size, 2), dtype=np.float32),
                end_s=np.ones((batch.batch_size, 2), dtype=np.float32),
                valid_mask=np.ones((batch.batch_size, 2), dtype=bool),
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
def test_engineering_controller_freezes_extracts_and_predicts(
    tmp_path: Path, with_validation: bool
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
    role_checks = json.loads((run / "frozen" / "role_checks.json").read_text())
    assert role_checks["near_duplicate_exclusion"] == "not_established_by_video_identity_checks"
    if with_validation:
        assert (run / "features" / "validation" / "index.jsonl").is_file()


def test_disjoint_rejects_same_source_under_different_video_ids():
    fit = replace(_record("a", "train", False), metadata={"source_id": "camera-event-1"})
    holdout = replace(_record("b", "val", True), metadata={"source_group": "camera-event-1"})
    with pytest.raises(ValueError, match="source:camera-event-1"):
        _disjoint([fit], [], [holdout])


@pytest.mark.parametrize("encoder", ["videomaev2", "clip"])
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
    (tmp_path / "projects/icassp2027/assets.local.yaml").write_text(
        "checkpoint_paths:\n  videomaev2: " + external_weights.as_posix() + "\n",
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
        return {
            "adapter": "videomaev2",
            "checkpoint": {"id": "toy", "sha256": {"toy.bin": "fixture"}},
            "code": {"source_sha256": "fixture"},
        }

    def create(name, **constructor):
        received.append((name, constructor))
        assert Path(constructor["model_name"]) == external_weights
        assert constructor["device"] == "cpu"
        adapter = _Adapter()
        adapter.capabilities = replace(
            adapter.capabilities, fixed_num_frames=16, min_frames=16, max_frames=16
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
    )
    if encoder == "clip":
        with pytest.raises(ConfigError, match="not active"):
            run_detection_experiment(request, video_backend=_CV())
        assert not received
    else:
        assert run_detection_experiment(request, video_backend=_CV()).completed
        assert len(received) == 1
