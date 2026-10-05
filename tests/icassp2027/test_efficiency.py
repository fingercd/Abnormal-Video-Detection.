"""Boundary tests for the frozen-reducer CUDA timing receipt."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.icassp2027 import benchmark_frozen_reducers as timing_cli
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord
from vadbench.paper.efficiency import (
    CudaTimingRuntime,
    FrozenTimingSettings,
    failed_scope,
    measure_scope,
    summary,
)


class _Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self):
        self.cuda.event_records += 1

    def elapsed_time(self, _other):
        return 2.5


class _Cuda:
    def __init__(self):
        self.syncs = 0
        self.empty = 0
        self.resets = 0
        self.event_records = 0

    def is_available(self):
        return True

    def synchronize(self, _device=None):
        self.syncs += 1

    def empty_cache(self):
        self.empty += 1

    def reset_peak_memory_stats(self, _device=None):
        self.resets += 1

    def memory_allocated(self, _device=None):
        return 100

    def memory_reserved(self, _device=None):
        return 200

    def max_memory_allocated(self, _device=None):
        return 160

    def max_memory_reserved(self, _device=None):
        return 260

    def Event(self, *, enable_timing):
        assert enable_timing is True
        return _Event(self)


class _Torch:
    def __init__(self):
        self.cuda = _Cuda()


class _AsyncClock:
    def __init__(self):
        self.value = 0.0
        self.pending = False

    def __call__(self):
        return self.value


class _AsyncCuda(_Cuda):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def synchronize(self, _device=None):
        super().synchronize(_device)
        if self.clock.pending:
            self.clock.value += 0.25
            self.clock.pending = False


def test_scope_wall_clock_stops_only_after_async_cuda_completion():
    clock = _AsyncClock()
    fake = _Torch()
    fake.cuda = _AsyncCuda(clock)
    runtime = CudaTimingRuntime(fake, device="cuda:0")

    def operation():
        # Model work has only been queued.  It becomes observable exactly when
        # synchronize() runs after end-event recording.
        clock.pending = True
        return {"actual_feature_tokens": 9}

    receipt = measure_scope(
        operation, runtime=runtime, settings=FrozenTimingSettings(warmup=0, repeat=1),
        scope="adapter", batch_size=1, sampled_frames=8, clock=clock,
    )

    assert receipt["samples"][0]["wall_seconds"] == pytest.approx(0.25)
    assert receipt["samples"][0]["throughput"]["clips_per_second"] == pytest.approx(4.0)


def test_scope_uses_sync_wall_time_events_and_peak_reset():
    fake = _Torch()
    runtime = CudaTimingRuntime(fake, device="cuda:0")
    calls = 0
    ticks = iter(value / 10 for value in range(20))

    def operation():
        nonlocal calls
        calls += 1
        return {"actual_feature_tokens": 17}

    receipt = measure_scope(
        operation,
        runtime=runtime,
        settings=FrozenTimingSettings(warmup=1, repeat=2),
        scope="pure_model",
        batch_size=1,
        sampled_frames=16,
        clock=lambda: next(ticks),
    )

    assert calls == 3  # one excluded warmup plus two formal repeats
    assert fake.cuda.resets == 3
    assert fake.cuda.empty == 1
    assert fake.cuda.syncs >= 8
    assert fake.cuda.event_records == 6
    assert receipt["summary"]["wall_seconds"]["median"] == pytest.approx(0.1)
    assert receipt["summary"]["cuda_event_seconds"]["median"] == pytest.approx(0.0025)
    assert receipt["model_resident_baseline_gpu_memory_bytes"] == {"allocated": 100, "reserved": 200}
    assert receipt["samples"][0]["gpu_memory_bytes"]["after"]["peak_reserved"] == 260
    assert receipt["samples"][0]["operation_receipt"] == {"actual_feature_tokens": 17}


def test_failed_scope_keeps_null_statistics_instead_of_fake_zero():
    receipt = failed_scope(scope="adapter", error=RuntimeError("native suffix failed"))
    assert receipt["status"] == "failed"
    assert receipt["samples"] is None
    assert receipt["summary"] is None
    assert receipt["error"]["message"] == "native suffix failed"


def test_native_videomaev2_route_accepts_its_real_pooled_backbone_contract():
    import torch

    from vadbench.integrations.videomaev2_encoder import VideoMAEv2Encoder

    script = Path(__file__).resolve().parents[2] / "scripts/icassp2027/benchmark_frozen_reducers.py"
    namespace = runpy.run_path(str(script))

    class Backbone(torch.nn.Module):
        def forward(self, *, pixel_values):
            return pixel_values.mean(dim=(2, 3, 4))

    class Worker(torch.nn.Module):
        _pool = VideoMAEv2Encoder._pool

        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

        def _tensor_from_rgb_lists(self, clips):
            raise AssertionError("preparation must be outside this native forward test")

    worker = Worker()
    adapter = type("Adapter", (), {"encoder": worker})()
    route = namespace["_native_route"](adapter, "videomaev2", worker)
    inputs = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 2, 2)
    result = route.run(inputs, None)
    assert result.shape == (2, 3)
    assert torch.equal(result, worker._pool(worker.backbone(pixel_values=inputs)))


def test_statistic_requires_observations_and_reports_requested_quantiles():
    values = summary([1.0, 2.0, 3.0])
    assert values["median"] == 2.0
    assert values["p10"] == pytest.approx(1.2)
    assert values["p90"] == pytest.approx(2.8)
    with pytest.raises(ValueError, match="empty"):
        summary([])


def test_frozen_timing_cli_help_and_dry_run_are_executable():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/icassp2027/benchmark_frozen_reducers.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), environment.get("PYTHONPATH", "")))
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=root, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert "--fit-manifest" in help_result.stdout
    assert "--role-lock" in help_result.stdout
    plan = subprocess.run(
        [sys.executable, str(script), "--encoder", "vjepa2"], cwd=root, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert '"warmup": 5' in plan.stdout
    assert '"repeat": 30' in plan.stdout
    assert '"expected_output_dim": 1024' in plan.stdout
    assert timing_cli.FROZEN_FIT128_MANIFEST_SHA256 in plan.stdout


def test_fit_selector_requires_the_exact_train_only_contract():
    train = VideoManifestRecord("fit-a", "fit-a.mp4", DatasetSplit.TRAIN, "normal", False)
    selected = timing_cli._choose_fit_video((train,), "fit-a")
    assert selected.video_id == "fit-a"
    validation = VideoManifestRecord("val-a", "val-a.mp4", DatasetSplit.VAL, "normal", False)
    with pytest.raises(ValueError, match="TRAIN manifest"):
        timing_cli._choose_fit_video((train, validation), "fit-a")
    with pytest.raises(ValueError, match="explicit --fit-video-id"):
        timing_cli._choose_fit_video((train,), None)
