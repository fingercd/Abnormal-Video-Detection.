from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vadbench.artifacts import PredictionRecord
from vadbench.data.manifest import VideoManifestRecord
from vadbench.engine.predict import TORCH_AVAILABLE, predict_feature_head
from vadbench.engine.train import save_checkpoint
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.tasks import build_task

if TORCH_AVAILABLE:
    import torch


def _config(*, task: str = "weak_mil", clips: int = 2) -> dict:
    task_section = {
        "kind": task,
        "head_kwargs": {"hidden_dim": 5},
    }
    if task == "weak_mil":
        task_section["pooling"] = "attention"
    return {
        "task": task_section,
        "training": {"batch_size": 2, "num_workers": 0},
        "sampler": {"segments_per_video": clips},
        "encoder": {"trainable": False},
        "output": {"root": "outputs", "run_name": "predict-test"},
    }


def _manifest(
    video_id: str = "video",
    *,
    anomaly: bool = True,
    num_frames: int | None = 4,
    fps: float | None = 2.0,
) -> VideoManifestRecord:
    duration = None if num_frames is None or fps is None else num_frames / fps
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split="test",
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=num_frames,
        fps=fps,
        duration_seconds=duration,
    )


def _features(
    tmp_path: Path,
    intervals: list[tuple[int, int]],
    *,
    video_id: str = "video",
    clip_indices: list[int] | None = None,
) -> tuple[FeatureStore, str]:
    store = FeatureStore(tmp_path / "features")
    fingerprint = compute_encoder_fingerprint({"adapter": "predict-test"})
    indices = list(range(len(intervals))) if clip_indices is None else clip_indices
    for row, ((start, end), clip_index) in enumerate(zip(intervals, indices, strict=True)):
        store.write(
            video_id=video_id,
            clip_id=f"{video_id}:clip-{row}",
            clip_index=clip_index,
            encoder_fingerprint=fingerprint,
            features=np.full((2, 3), row + 1, dtype=np.float32),
            pooled=np.full(3, row + 1, dtype=np.float32),
            start_s=start / 2,
            end_s=end / 2,
            frame_start=start,
            frame_end=end,
            metadata={
                "source": {
                    "sampling_kind": "uniform_segments",
                    "segment_start_frames": start,
                    "segment_end_frames": end,
                    "split": "test",
                    "is_anomaly": True,
                }
            },
        )
    return store, fingerprint


def _checkpoint(
    tmp_path: Path,
    config: dict,
    fingerprint: str,
    *,
    task: str = "wsvad",
) -> Path:
    model = build_task(
        task,
        None,
        feature_dim=3,
        head="attention" if task == "wsvad" else None,
        head_kwargs={"hidden_dim": 5},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / "run" / "checkpoints" / "final.pt"
    save_checkpoint(
        path,
        model,
        step=2,
        epoch=1,
        metadata={
            "task": task,
            "encoder_fingerprint": fingerprint,
            "feature_dim": 3,
            "feature_level": "clip",
            "config": config,
        },
    )
    return path


def test_predict_module_imports_without_requiring_torch_execution() -> None:
    assert isinstance(TORCH_AVAILABLE, bool)
    assert callable(predict_feature_head)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_weak_checkpoint_is_rebuilt_and_writes_full_coverage_jsonl(
    tmp_path: Path,
) -> None:
    config = _config()
    store, fingerprint = _features(tmp_path, [(0, 2), (2, 4)])
    checkpoint = _checkpoint(tmp_path, config, fingerprint)
    output = tmp_path / "predictions.jsonl"
    records = predict_feature_head(
        config,
        store,
        [_manifest()],
        checkpoint,
        output,
        device="cpu",
    )
    assert len(records) == 2
    assert all(isinstance(item, PredictionRecord) for item in records)
    assert [item.clip_index for item in records] == [0, 1]
    assert [(item.frame_start, item.frame_end) for item in records] == [(0, 2), (2, 4)]
    assert [item.anomaly_score for item in records] == pytest.approx([0.5, 0.5])
    assert all(item.encoder_fingerprint == fingerprint for item in records)
    assert all(item.metadata["ground_truth_scope"] == "video" for item in records)
    rows = [
        PredictionRecord.from_dict(json.loads(line))
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == records


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_temporal_checkpoint_predicts_video_without_temporal_training_labels(
    tmp_path: Path,
) -> None:
    config = _config(task="temporal_supervised")
    store, fingerprint = _features(tmp_path, [(0, 2), (2, 4)])
    checkpoint = _checkpoint(
        tmp_path,
        config,
        fingerprint,
        task="temporal",
    )
    records = predict_feature_head(
        config,
        store,
        [_manifest()],
        checkpoint,
        tmp_path / "temporal.jsonl",
        device="cpu",
    )
    assert len(records) == 2
    assert all(item.metadata["task"] == "temporal" for item in records)
    assert [item.anomaly_score for item in records] == pytest.approx([0.5, 0.5])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
@pytest.mark.parametrize(
    ("intervals", "message"),
    [
        ([(0, 1), (2, 4)], "gap"),
        ([(0, 3), (2, 4)], "overlaps"),
    ],
)
def test_strict_coverage_rejects_gap_and_overlap(
    tmp_path: Path,
    intervals: list[tuple[int, int]],
    message: str,
) -> None:
    config = _config()
    store, fingerprint = _features(tmp_path, intervals)
    checkpoint = _checkpoint(tmp_path, config, fingerprint)
    with pytest.raises(ValueError, match=message):
        predict_feature_head(
            config,
            store,
            [_manifest()],
            checkpoint,
            tmp_path / "invalid.jsonl",
            device="cpu",
        )
    assert not (tmp_path / "invalid.jsonl").exists()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
@pytest.mark.parametrize(
    ("num_frames", "fps", "message"),
    [
        (4, None, "requires manifest fps"),
        (None, 2.0, "requires manifest num_frames"),
    ],
)
def test_strict_coverage_requires_manifest_geometry(
    tmp_path: Path,
    num_frames: int | None,
    fps: float | None,
    message: str,
) -> None:
    config = _config()
    store, fingerprint = _features(tmp_path, [(0, 2), (2, 4)])
    checkpoint = _checkpoint(tmp_path, config, fingerprint)
    with pytest.raises(ValueError, match=message):
        predict_feature_head(
            config,
            store,
            [_manifest(num_frames=num_frames, fps=fps)],
            checkpoint,
            tmp_path / "missing-geometry.jsonl",
            device="cpu",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_predict_rejects_duplicate_clip_indices(tmp_path: Path) -> None:
    config = _config()
    store, fingerprint = _features(
        tmp_path,
        [(0, 2), (2, 4)],
        clip_indices=[0, 0],
    )
    checkpoint = _checkpoint(tmp_path, config, fingerprint)
    with pytest.raises(ValueError, match="duplicate clip_index"):
        predict_feature_head(
            config,
            store,
            [_manifest()],
            checkpoint,
            tmp_path / "duplicate.jsonl",
            device="cpu",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_predict_requires_real_frame_ranges_and_verified_sidecar(tmp_path: Path) -> None:
    config = _config(clips=1)
    store = FeatureStore(tmp_path / "features")
    fingerprint = compute_encoder_fingerprint({"adapter": "predict-test"})
    store.write(
        video_id="video",
        clip_id="video:clip-0",
        clip_index=0,
        encoder_fingerprint=fingerprint,
        features=np.ones((1, 3), dtype=np.float32),
        pooled=np.ones(3, dtype=np.float32),
        start_s=0,
        end_s=2,
        metadata={"source": {"split": "test", "is_anomaly": True}},
    )
    checkpoint = _checkpoint(tmp_path, config, fingerprint)
    with pytest.raises(ValueError, match="real source frame ranges"):
        predict_feature_head(
            config,
            store,
            [_manifest()],
            checkpoint,
            tmp_path / "no-frames.jsonl",
            device="cpu",
        )

    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="checksum manifest"):
        predict_feature_head(
            config,
            store,
            [_manifest()],
            checkpoint,
            tmp_path / "no-sidecar.jsonl",
            device="cpu",
            strict_coverage=False,
        )
