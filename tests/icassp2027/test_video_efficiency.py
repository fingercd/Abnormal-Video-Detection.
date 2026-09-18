"""CPU fixtures for full-video frozen-detector timing boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.icassp2027 import benchmark_frozen_video_detector as video_cli
from vadbench.contracts import ClipBatch
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.engine.runner import HeadOnlyTrainingConfig
from vadbench.engine.train import save_checkpoint
from vadbench.paper import video_efficiency
from vadbench.paper.video_efficiency import (
    VideoTimingSettings,
    measure_full_video_detector,
    predictor_path_qa,
    run_full_video_detector,
)
from vadbench.tasks import build_task


class _Adapter:
    def encode(self, batch, train=False):
        assert train is False
        values = torch.arange(batch.batch_size * 4, dtype=torch.float32).reshape(batch.batch_size, 4)
        return SimpleNamespace(features=values.unsqueeze(1), pooled=values)


class _Context:
    def __init__(self, batch):
        self.batch = batch

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def validate_execution(self):
        return {"gathered_tokens": 1, "suffix_shapes": {"1": [self.batch.batch_size, 1, 4]}}


def _batch(size: int) -> ClipBatch:
    indices = np.tile(np.asarray([[0, 1]], dtype=np.int64), (size, 1))
    return ClipBatch(
        frames=np.zeros((size, 2, 2, 2, 3), dtype=np.uint8),
        timestamps_s=indices.astype(np.float64),
        video_ids=("fit-video",) * size,
        frame_indices=indices,
        valid_mask=np.ones((size, 2), dtype=bool),
    )


def test_full_video_pass_uses_all_windows_b8_tail_and_existing_aggregation(monkeypatch):
    # The one-frame overlap exercises legal input overlap before the existing
    # aggregator creates final non-overlapping frame intervals.
    samples = DenseSamplingPlan(clip_frames=3, frame_stride=1, window_stride=2).sample(20)
    monkeypatch.setattr(video_efficiency, "build_clip_batch", lambda _path, _id, clips: _batch(len(clips)))
    task = build_task("weak_mil", None, feature_dim=4, head="topk", head_kwargs={"k": 3}, task_kwargs={"ranking_weight": 0.0})
    task.eval()
    original_prediction_step = task.prediction_step
    observed_modes = []

    def observed_prediction_step(batch):
        observed_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        return original_prediction_step(batch)

    monkeypatch.setattr(task, "prediction_step", observed_prediction_step)
    receipt = run_full_video_detector(
        adapter=_Adapter(), detector_task=task, video_path="unused.mp4", video_id="fit-video", samples=samples,
        num_frames=20, fps=30.0, context_factory=_Context, batch_size=8,
    )
    assert receipt["windows"] == 10
    assert receipt["encoder_batches"] == [8, 2]
    assert receipt["final_batch_size"] == 2
    assert receipt["frame_scores_finite"] is True
    assert receipt["coverage"]["complete"] is True
    assert receipt["aggregation"]["overlap_before_aggregation"] is True
    assert receipt["reduction_executions"]
    assert observed_modes and all(inference and not grad for inference, grad in observed_modes)


def test_existing_task_predictor_qa_uses_no_labels():
    task = build_task("weak_mil", None, feature_dim=4, head="topk", head_kwargs={"k": 3}, task_kwargs={"ranking_weight": 0.0})
    task.eval()
    receipt = predictor_path_qa(task, feature_dim=4, device="cpu")
    assert receipt["status"] == "passed"
    assert receipt["synthetic_labels_used"] is False
    assert receipt["existing_interval_aggregation"]["scores_finite"] is True
    assert receipt["coverage"]["complete"] is True


def test_head_loader_accepts_real_flat_runner_config_and_freezes_loaded_task(tmp_path):
    settings = HeadOnlyTrainingConfig(
        task="weak_mil", feature_level="clip", head="topk", head_kwargs={"k": 3},
        task_kwargs={"ranking_weight": 0.0}, batch_size=16, epochs=20,
        learning_rate=1e-3, weight_decay=0.0, seed=0, expected_clips=32,
    )
    original = build_task(settings.task, None, feature_dim=4, head=settings.head,
                          head_kwargs=settings.head_kwargs, task_kwargs=settings.task_kwargs)
    checkpoint = tmp_path / "final.pt"
    save_checkpoint(checkpoint, original, step=1, epoch=20, metadata={})
    source = SimpleNamespace(checkpoint=checkpoint, checkpoint_metadata={"config": asdict(settings)},
                             representation=SimpleNamespace(output_dim=4))
    loaded, receipt = video_cli._head(source, device="cpu")
    features = torch.randn(1, 5, 4)
    with torch.inference_mode():
        expected = original.prediction_step({"features": features, "valid_mask": torch.ones(1, 5, dtype=torch.bool)}).predictions.snippet_scores
        actual = loaded.prediction_step({"features": features, "valid_mask": torch.ones(1, 5, dtype=torch.bool)}).predictions.snippet_scores
    torch.testing.assert_close(actual, expected)
    assert receipt["feature_dim"] == 4
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


class _Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self):
        self.cuda.records += 1

    def elapsed_time(self, _other):
        return 3.0


class _Cuda:
    def __init__(self, clock):
        self.clock = clock
        self.pending = False
        self.records = 0
        self.resets = 0

    def is_available(self):
        return True

    def synchronize(self, _device=None):
        if self.pending:
            self.clock[0] += 0.4
            self.pending = False

    def empty_cache(self):
        return None

    def reset_peak_memory_stats(self, _device=None):
        self.resets += 1

    def memory_allocated(self, _device=None):
        return 100

    def memory_reserved(self, _device=None):
        return 200

    def max_memory_allocated(self, _device=None):
        return 150

    def max_memory_reserved(self, _device=None):
        return 250

    def Event(self, *, enable_timing):
        assert enable_timing
        return _Event(self)


def test_full_video_timing_stops_wall_clock_after_cuda_sync():
    clock = [0.0]
    cuda = _Cuda(clock)
    fake_torch = SimpleNamespace(cuda=cuda)

    def operation():
        cuda.pending = True
        return {"coverage": {"complete": True}}

    receipt = measure_full_video_detector(operation, torch_module=fake_torch, device="cuda:0",
                                          settings=VideoTimingSettings(warmup=1, repeat=2), clock=lambda: clock[0])
    assert receipt["summary"]["wall_seconds"]["median"] == pytest.approx(0.4)
    assert receipt["summary"]["cuda_event_seconds"]["median"] == pytest.approx(0.003)
    assert cuda.resets == 3
    assert receipt["samples"][0]["peak_reserved_bytes"] == 250


def test_full_video_cli_help_and_dry_run_are_executable():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/icassp2027/benchmark_frozen_video_detector.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), environment.get("PYTHONPATH", "")))
    help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=root, env=environment,
                                 capture_output=True, text=True, check=True)
    assert "--method-sources" in help_result.stdout
    plan_result = subprocess.run([sys.executable, str(script), "--encoder", "videomaev2"], cwd=root, env=environment,
                                 capture_output=True, text=True, check=True)
    assert '"repeat": 10' in plan_result.stdout
    assert '"video_id": "Abuse005_x264"' in plan_result.stdout
