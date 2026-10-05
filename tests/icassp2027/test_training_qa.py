from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vadbench.data.manifest import VideoManifestRecord
from vadbench.engine import runner
from vadbench.engine.runner import TORCH_AVAILABLE, HeadOnlyTrainingConfig, train_feature_head
from vadbench.features import FeatureStore, compute_encoder_fingerprint

if TORCH_AVAILABLE:
    import torch


def _manifest(video_id: str, anomaly: bool) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split="train",
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=8,
        fps=4.0,
        duration_seconds=2.0,
    )


def _inputs(tmp_path: Path) -> tuple[FeatureStore, list[VideoManifestRecord], str]:
    store = FeatureStore(tmp_path / "features")
    fingerprint = compute_encoder_fingerprint({"training-qa": "fixture"})
    records = [_manifest("normal", False), _manifest("positive", True)]
    for record in records:
        for clip_index in range(2):
            value = float(record.is_anomaly) * 3.0 + clip_index
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:{clip_index}",
                clip_index=clip_index,
                encoder_fingerprint=fingerprint,
                features=np.full((2, 4), value, dtype=np.float32),
                pooled=np.full(4, value, dtype=np.float32),
                start_s=float(clip_index),
                end_s=float(clip_index + 1),
                metadata={"source": {"split": "train", "is_anomaly": record.is_anomaly}},
            )
    return store, records, fingerprint


def _config(*, verify_training: bool) -> HeadOnlyTrainingConfig:
    return HeadOnlyTrainingConfig(
        task="weak_mil",
        head="topk",
        head_kwargs={"k": 1, "dropout": 0.0},
        batch_size=2,
        epochs=2,
        learning_rate=0.01,
        expected_clips=2,
        seed=7,
        verify_training=verify_training,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_training_qa_writes_gradient_update_and_reload_parity(tmp_path: Path) -> None:
    store, records, fingerprint = _inputs(tmp_path)
    result = train_feature_head(
        _config(verify_training=True),
        feature_store=store,
        train_manifest=records,
        output_dir=tmp_path / "run",
        encoder_fingerprint=fingerprint,
        device="cpu",
    )
    qa = json.loads((tmp_path / "run" / "training_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "passed"
    assert qa["nonzero_gradient_steps"] >= 1
    assert qa["changed_parameter_count"] >= 1
    assert qa["parameter_delta_l2"] > 0
    assert all(item["exact_equal"] for item in qa["reload_parity"]["outputs"].values())
    sidecar = json.loads(Path(result.checkpoint_manifest_path).read_text(encoding="utf-8"))
    assert sidecar["metadata"]["status"] == "completed"
    assert sidecar["metadata"]["training_qa"]["status"] == "passed"
    assert qa["checkpoint"]["sha256"] == sidecar["sha256"] == result.history["checkpoint"]["sha256"]
    assert result.history["training_qa"]["status"] == "passed"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_training_qa_default_keeps_legacy_artifact_shape(tmp_path: Path) -> None:
    store, records, fingerprint = _inputs(tmp_path)
    result = train_feature_head(
        _config(verify_training=False),
        feature_store=store,
        train_manifest=records,
        output_dir=tmp_path / "run",
        encoder_fingerprint=fingerprint,
        device="cpu",
    )
    assert "training_qa" not in result.history
    assert not (tmp_path / "run" / "training_qa.json").exists()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_training_qa_rejects_zero_observed_gradient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, records, fingerprint = _inputs(tmp_path)
    original = runner.train_one_step

    def zero_grad(*args, **kwargs):
        result = original(*args, **kwargs)
        for parameter in args[0].parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        return result

    monkeypatch.setattr(runner, "train_one_step", zero_grad)
    with pytest.raises(RuntimeError, match="no finite nonzero gradient"):
        train_feature_head(
            _config(verify_training=True),
            feature_store=store,
            train_manifest=records,
            output_dir=tmp_path / "run",
            encoder_fingerprint=fingerprint,
            device="cpu",
        )
    qa = json.loads((tmp_path / "run" / "training_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "failed"
    pending = json.loads((tmp_path / "run" / "checkpoints" / "final.pt.json").read_text(encoding="utf-8"))
    assert pending["metadata"]["status"] == "trained_pending_qa"
    assert "training_qa" not in pending["metadata"]
    assert not (tmp_path / "run" / "history.json").exists()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_training_qa_rejects_zero_parameter_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, records, fingerprint = _inputs(tmp_path)
    original = runner.train_one_step
    initial: dict[str, torch.Tensor] = {}

    def restore_parameters(*args, **kwargs):
        model = args[0]
        if not initial:
            initial.update({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
        result = original(*args, **kwargs)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(initial[name])
        return result

    monkeypatch.setattr(runner, "train_one_step", restore_parameters)
    with pytest.raises(RuntimeError, match="no trainable parameter change"):
        train_feature_head(
            _config(verify_training=True),
            feature_store=store,
            train_manifest=records,
            output_dir=tmp_path / "run",
            encoder_fingerprint=fingerprint,
            device="cpu",
        )
    assert json.loads((tmp_path / "run" / "training_qa.json").read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_training_qa_rejects_reload_prediction_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, records, fingerprint = _inputs(tmp_path)
    original = runner.load_checkpoint

    def corrupt_reloaded(*args, **kwargs):
        metadata = original(*args, **kwargs)
        with torch.no_grad():
            next(args[1].parameters()).add_(1.0)
        return metadata

    monkeypatch.setattr(runner, "load_checkpoint", corrupt_reloaded)
    with pytest.raises(RuntimeError, match="checkpoint reload parity failed"):
        train_feature_head(
            _config(verify_training=True),
            feature_store=store,
            train_manifest=records,
            output_dir=tmp_path / "run",
            encoder_fingerprint=fingerprint,
            device="cpu",
        )
    assert json.loads((tmp_path / "run" / "training_qa.json").read_text(encoding="utf-8"))["status"] == "failed"
