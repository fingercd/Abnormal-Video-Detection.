from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
import yaml

from vadbench.benchmark import (
    BenchmarkCase,
    BenchmarkSettings,
    BenchmarkWorkload,
    run_benchmark_suite,
    write_performance_result,
)
from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    StreamState,
    StreamStep,
    TokenTimeline,
)


class _Clock:
    def __init__(self, step: float = 0.01) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def _batch(indices: list[int], *, video_id: str = "video") -> ClipBatch:
    frame_indices = np.asarray([indices], dtype=np.int64)
    timestamps = frame_indices.astype(np.float32) * 0.5
    return ClipBatch(
        frames=np.zeros((1, len(indices), 4, 4, 3), dtype=np.uint8),
        timestamps_s=timestamps,
        video_ids=(video_id,),
        frame_indices=frame_indices,
    )


def _output(batch: ClipBatch, *, feature_stage: str | None = None) -> EncoderOutput:
    timestamps = np.asarray(batch.timestamps_s, dtype=np.float32)
    lengths = batch.valid_lengths
    valid = np.zeros_like(timestamps, dtype=bool)
    for row, length in enumerate(lengths):
        valid[row, : int(length)] = True
    timeline = TokenTimeline(
        start_s=timestamps,
        end_s=timestamps + 0.5,
        valid_mask=valid,
        source_frame_start=(
            None if batch.frame_indices is None else np.asarray(batch.frame_indices, dtype=np.int64)
        ),
        source_frame_end=(
            None
            if batch.frame_indices is None
            else np.asarray(batch.frame_indices, dtype=np.int64) + 1
        ),
    )
    return EncoderOutput(
        features=np.zeros((batch.batch_size, batch.num_frames, 3), dtype=np.float32),
        timeline=timeline,
        aux={} if feature_stage is None else {"feature_stage": feature_stage},
    )


class _FixedAdapter:
    capabilities = EncoderCapabilities(supports_fixed_clip=True)

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        assert train is False
        self.calls += 1
        return _output(batch)


class _StreamingAdapter:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=True,
    )

    def __init__(self) -> None:
        self.init_calls = 0
        self.step_calls = 0

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        raise AssertionError("streaming benchmark 不应调用 encode")

    def init_state(self, video_id: str) -> StreamState:
        self.init_calls += 1
        return StreamState(video_id=video_id)

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression=None,
    ) -> StreamStep:
        assert train is False
        assert compression is None
        self.step_calls += 1
        before = state.step_index * chunk.num_frames
        after = before + chunk.num_frames
        return StreamStep(
            output=_output(chunk, feature_stage="projected_visual"),
            state=state.replace(step_index=state.step_index + 1),
            telemetry={
                "decoder_kv_tokens_before_max": before,
                "decoder_kv_tokens_after_max": after,
                "input_tokens": after,
                "reused_tokens": before,
                "cache_bytes": after * 16,
                "cache_hit": state.step_index > 0,
                "feature_stage": "projected_visual",
                "feature_cache_conditioned": False,
                "cache_owner": "language_model_decoder",
                "is_vision_encoder_kv": False,
                "decoder_kv_layers": 2,
                "native_hermes_compression_ms": 2.0,
                "native_hermes_compression_enabled": True,
                "native_hermes_compression_mode": "predict",
                "native_hermes_compression_called": True,
                "native_hermes_compression_applied": state.step_index > 0,
                "native_hermes_visual_budget_tokens": 2,
                "native_hermes_protected_prefix_tokens": 1,
                "native_hermes_effective_total_budget_tokens": 3,
                "native_hermes_tokens_before_min": after + 2,
                "native_hermes_tokens_before_max": after + 2,
                "native_hermes_tokens_before_total": after + 2,
                "native_hermes_tokens_after_min": after,
                "native_hermes_tokens_after_max": after,
                "native_hermes_tokens_after_total": after,
                "native_hermes_tokens_evicted_total": 2,
            },
        )

    def finalize(self, state: StreamState) -> None:
        return None


class _ContextualStreamingAdapter(_StreamingAdapter):
    """Fake only the public adapter contract, without importing torch."""

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression=None,
    ) -> StreamStep:
        step = super().encode_step(chunk, state, train=train, compression=compression)
        return StreamStep(
            output=_output(chunk, feature_stage="decoder_contextual"),
            state=step.state,
            telemetry={
                **step.telemetry,
                "feature_stage": "decoder_contextual",
                "feature_cache_conditioned": True,
            },
        )


def _case(
    name: str,
    adapter,
    mode: str,
    batches: tuple[ClipBatch, ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        adapter=adapter,
        workload=BenchmarkWorkload(
            name="same-smoke-video",
            mode=mode,
            decode=lambda: "decoded",
            preprocess=lambda _decoded: batches,
            sampling={"sample_fps": 2.0, "frames_per_unit": 2},
        ),
        config={"encoder": {"name": name}, "precision": "fake32"},
    )


def test_fixed_cpu_benchmark_runs_warmup_repeats_and_aggregates() -> None:
    adapter = _FixedAdapter()
    result = run_benchmark_suite(
        [_case("fixed", adapter, "fixed", (_batch([0, 1, 2, 3]),))],
        BenchmarkSettings(warmup=1, repeat=3, synchronize_cuda=True, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    case = result["cases"][0]
    assert result["comparison"]["comparable"] is True
    assert result["comparison"]["accuracy_comparable"] is True
    assert adapter.calls == 4
    assert len(case["repeats"]) == 3
    repeat = case["repeats"][0]
    assert repeat["counts"] == {
        "frames": 4,
        "source_video_seconds": 2.0,
        "output_tokens": 4,
        "decoder_kv_input_tokens": 0,
    }
    assert repeat["timings_seconds"]["decode"] > 0
    assert repeat["timings_seconds"]["preprocess"] > 0
    assert repeat["timings_seconds"]["encoder"] > 0
    assert repeat["timings_seconds"]["native_compression"] == 0
    assert repeat["throughput"]["end_to_end_frames_per_second"] > 0
    assert repeat["throughput"]["encoder_tokens_per_second"] > 0
    assert repeat["peak_gpu_memory_bytes"] is None
    assert repeat["cache"]["steps"] == []
    assert case["aggregate"]["timings_seconds"]["wall"].keys() == {
        "mean",
        "std",
        "p50",
        "p90",
        "p95",
        "min",
        "max",
    }
    assert case["provenance"]["torch_device"]["torch_available"] is False
    assert len(case["provenance"]["config_sha256"]) == 64


def test_streaming_benchmark_resets_state_and_collects_native_kv_telemetry() -> None:
    adapter = _StreamingAdapter()
    case = _case(
        "streaming",
        adapter,
        "streaming",
        (_batch([0, 1]), _batch([2, 3])),
    )
    result = run_benchmark_suite(
        [case],
        BenchmarkSettings(warmup=1, repeat=2, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    assert adapter.init_calls == 3
    assert adapter.step_calls == 6
    first = result["cases"][0]["repeats"][0]
    assert first["counts"]["frames"] == 4
    assert first["counts"]["source_video_seconds"] == 2.0
    assert first["counts"]["output_tokens"] == 4
    assert first["counts"]["decoder_kv_input_tokens"] == 6
    assert first["cache"]["kv_tokens_before_max"] == 2
    assert first["cache"]["kv_tokens_after_max"] == 4
    assert first["cache"]["reused_tokens_total"] == 2
    assert first["cache"]["cache_bytes_peak"] == 64
    assert first["cache"]["native_compression_calls"] == 2
    assert first["cache"]["native_compression_applied_steps"] == 1
    assert first["timings_seconds"]["native_compression"] == 0.004
    assert first["timings_seconds"]["encoder_excluding_native_compression"] >= 0
    assert result["comparison"]["accuracy_comparable"] is False
    assert "projected_visual" in result["comparison"]["accuracy_reasons"][0]
    assert result["cases"][0]["accuracy_eligibility"]["performance_only"] is True
    assert result["cases"][0]["accuracy_eligibility"]["cache_conditioned"] is False
    assert [step["kv_tokens_after"] for step in first["cache"]["steps"]] == [2, 4]
    assert first["cache"]["steps"][1]["cache_owner"] == "language_model_decoder"
    assert first["cache"]["steps"][1]["is_vision_encoder_kv"] is False
    assert first["cache"]["steps"][1]["native_compression_mode"] == "predict"
    assert first["cache"]["steps"][1]["native_visual_budget_tokens"] == 2
    assert first["cache"]["steps"][1]["feature_stage"] == "projected_visual"
    assert first["cache"]["steps"][1]["cache_hit"] is True
    assert first["cache"]["steps"][1]["native_tokens_before_total"] == 6
    assert first["cache"]["steps"][1]["native_tokens_after_total"] == 4
    assert first["cache"]["steps"][1]["native_tokens_evicted_total"] == 2
    assert result["cases"][0]["timing_semantics"] == {
        "encoder_includes_native_compression": True,
        "native_compression_source": "adapter_telemetry_host_wall",
        "encoder_excluding_native_compression_is_approximate": True,
        "throughput_wall_time_source": "cuda_synchronized_perf_counter",
    }


def test_fixed_and_streaming_same_frames_are_comparable_despite_chunk_grouping() -> None:
    fixed = _case("fixed", _FixedAdapter(), "fixed", (_batch([0, 1, 2, 3]),))
    streaming = _case(
        "streaming",
        _StreamingAdapter(),
        "streaming",
        (_batch([0, 1]), _batch([2, 3])),
    )
    result = run_benchmark_suite(
        [fixed, streaming],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    assert result["comparison"]["comparable"] is False
    assert result["comparison"]["accuracy_comparable"] is False
    assert any("feature_stage" in reason for reason in result["comparison"]["reasons"])
    assert result["comparison"]["pairwise"][0]["comparable"] is False


def test_decoder_contextual_is_the_only_streaming_accuracy_conditioned_stage() -> None:
    result = run_benchmark_suite(
        [
            _case(
                "contextual",
                _ContextualStreamingAdapter(),
                "streaming",
                (_batch([0, 1]),),
            )
        ],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    eligibility = result["cases"][0]["accuracy_eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["performance_only"] is False
    assert eligibility["cache_conditioned"] is True


def test_different_sampling_is_reported_as_not_comparable_with_reasons() -> None:
    fixed = _case("fixed", _FixedAdapter(), "fixed", (_batch([0, 1, 2, 3]),))
    streaming = _case(
        "streaming",
        _StreamingAdapter(),
        "streaming",
        (_batch([0, 2]), _batch([4, 6])),
    )
    result = run_benchmark_suite(
        [fixed, streaming],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    comparison = result["comparison"]
    assert comparison["comparable"] is False
    assert comparison["reasons"]
    assert any("coordinate_sha256" in reason for reason in comparison["reasons"])
    assert comparison["pairwise"][0]["comparable"] is False


def test_different_task_or_declared_sampling_is_not_comparable() -> None:
    batch = _batch([0, 1])
    fixed = _case("fixed", _FixedAdapter(), "fixed", (batch,))
    streaming = BenchmarkCase(
        name="streaming",
        adapter=_StreamingAdapter(),
        workload=BenchmarkWorkload(
            name="same-smoke-video",
            mode="streaming",
            decode=lambda: None,
            preprocess=lambda _decoded: (batch,),
            sampling={"sample_fps": 1.0, "frames_per_unit": 2},
            task="temporal_supervised",
        ),
    )
    result = run_benchmark_suite(
        [fixed, streaming],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )
    reasons = result["comparison"]["reasons"]
    assert result["comparison"]["comparable"] is False
    assert any("benchmark task" in reason for reason in reasons)
    assert any("声明采样协议" in reason for reason in reasons)


def test_missing_frame_indices_fails_closed_for_cross_encoder_comparison() -> None:
    batch = _batch([0, 1])
    without_indices = ClipBatch(
        frames=batch.frames,
        timestamps_s=batch.timestamps_s,
        video_ids=batch.video_ids,
    )
    result = run_benchmark_suite(
        [
            _case("fixed-a", _FixedAdapter(), "fixed", (without_indices,)),
            _case("fixed-b", _FixedAdapter(), "fixed", (without_indices,)),
        ],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )

    assert result["comparison"]["comparable"] is False
    assert any("缺少 frame_indices" in reason for reason in result["comparison"]["reasons"])


def test_capability_mismatch_fails_before_benchmarking() -> None:
    with pytest.raises(ValueError, match="supports_fixed_clip"):
        run_benchmark_suite(
            [
                _case(
                    "misdeclared",
                    _StreamingAdapter(),
                    "fixed",
                    (_batch([0, 1]),),
                )
            ],
            BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
            torch_module=None,
            clock=_Clock(),
        )


def test_generator_materialization_is_included_in_preprocess_timing() -> None:
    clock = _Clock()
    adapter = _FixedAdapter()

    def preprocess(_decoded):
        def batches():
            clock()  # synthetic work performed only while consuming the generator
            yield _batch([0, 1])

        return batches()

    case = BenchmarkCase(
        name="generator",
        adapter=adapter,
        workload=BenchmarkWorkload(
            name="generator",
            mode="fixed",
            decode=lambda: None,
            preprocess=preprocess,
            sampling={"sample_fps": 2.0},
        ),
    )
    result = run_benchmark_suite(
        [case],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=clock,
    )
    assert result["cases"][0]["repeats"][0]["timings_seconds"]["preprocess"] > 0.01


def test_adapter_eval_and_torch_inference_mode_are_used_and_restored() -> None:
    active = {"inference": False}

    class InferenceContext:
        def __enter__(self):
            active["inference"] = True

        def __exit__(self, *_args):
            active["inference"] = False

    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class EncoderOwner:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool) -> None:
            self.training = mode

    class EvalAdapter(_FixedAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = EncoderOwner()

        def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
            assert self.encoder.training is False
            assert active["inference"] is True
            return super().encode(batch, train=train)

    adapter = EvalAdapter()
    torch_module = SimpleNamespace(
        __version__="fake",
        cuda=NoCuda(),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(),
        inference_mode=InferenceContext,
    )
    run_benchmark_suite(
        [_case("eval", adapter, "fixed", (_batch([0, 1]),))],
        BenchmarkSettings(warmup=1, repeat=1, device="cpu"),
        torch_module=torch_module,
        clock=_Clock(),
    )
    assert adapter.encoder.training is True
    assert active["inference"] is False


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronize_calls = 0
        self.reset_calls = 0
        self.memory_calls = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self, device=None) -> None:
        assert device == "cuda:0"
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self, device=None) -> None:
        assert device == "cuda:0"
        self.reset_calls += 1

    def max_memory_allocated(self, device=None) -> int:
        assert device == "cuda:0"
        self.memory_calls += 1
        return 2048

    def max_memory_reserved(self, device=None) -> int:
        assert device == "cuda:0"
        return 4096

    def memory_allocated(self, device=None) -> int:
        assert device == "cuda:0"
        return 1024

    def memory_reserved(self, device=None) -> int:
        assert device == "cuda:0"
        return 3072

    def get_device_name(self, device=None) -> str:
        return "Fake A100"

    def get_device_capability(self, device=None) -> tuple[int, int]:
        return 8, 0

    def get_device_properties(self, device=None):
        return SimpleNamespace(total_memory=80 * 1024**3)


def test_cuda_synchronization_peak_memory_and_device_provenance_are_recorded() -> None:
    cuda = _FakeCuda()
    torch_module = SimpleNamespace(
        __version__="fake-2.9",
        cuda=cuda,
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 91002)),
    )
    result = run_benchmark_suite(
        [_case("fixed", _FixedAdapter(), "fixed", (_batch([0, 1]),))],
        BenchmarkSettings(warmup=0, repeat=2, device="cuda:0"),
        torch_module=torch_module,
        clock=_Clock(),
    )

    case = result["cases"][0]
    assert cuda.synchronize_calls > 0
    assert cuda.reset_calls == 2
    assert cuda.memory_calls == 2
    assert [repeat["peak_gpu_memory_bytes"] for repeat in case["repeats"]] == [2048, 2048]
    assert case["repeats"][0]["gpu_memory_bytes"] == {
        "allocated": {"baseline": 1024, "peak": 2048, "steady": 1024},
        "reserved": {"baseline": 3072, "peak": 4096, "steady": 3072},
    }
    assert case["aggregate"]["memory"]["allocated_peak_bytes"]["p50"] == 2048
    device = case["provenance"]["torch_device"]
    assert device["torch_version"] == "fake-2.9"
    assert device["cuda_version"] == "12.8"
    assert device["device_name"] == "Fake A100"
    assert device["device_capability"] == [8, 0]
    assert device["device_total_memory_bytes"] == 80 * 1024**3


def test_result_validates_against_schema_and_writes_utf8(tmp_path: Path) -> None:
    result = run_benchmark_suite(
        [
            _case("固定编码器", _FixedAdapter(), "fixed", (_batch([0, 1]),)),
            _case("流式编码器", _StreamingAdapter(), "streaming", (_batch([0, 1]),)),
        ],
        BenchmarkSettings(warmup=0, repeat=1, device="cpu"),
        torch_module=None,
        clock=_Clock(),
    )
    schema_path = Path("schemas/performance-result-v1.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result)

    output = write_performance_result(result, tmp_path / "性能.json")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["cases"][0]["name"] == "固定编码器"
    assert not output.with_suffix(".json.tmp").exists()


def test_benchmark_yaml_declares_fixed_identity_and_native_cases() -> None:
    config = yaml.safe_load(
        Path("configs/benchmarks/video-encoder-smoke.yaml").read_text(encoding="utf-8")
    )
    assert config["benchmark"]["warmup"] >= 1
    assert config["benchmark"]["repeat"] >= 3
    assert config["input"]["sampled_frames"] == 32
    cases = {case["name"]: case for case in config["cases"]}
    assert cases["videomaev2-fixed"]["mode"] == "fixed"
    assert cases["hermes-stream-identity"]["track"] == "raw_decoder_kv"
    assert cases["hermes-stream-identity"]["compression"]["policy"] == "identity"
    assert cases["hermes-stream-identity"]["encoder"]["params"]["native_compression_mode"] == "off"
    assert (
        cases["hermes-stream-identity"]["encoder"]["params"]["feature_stage"] == "projected_visual"
    )
    assert (
        cases["hermes-stream-native-predict"]["encoder"]["params"]["native_compression_mode"]
        == "predict"
    )
    native = cases["hermes-stream-native-predict"]
    assert native["track"] == "native_decoder_kv_predict"
    assert native["compression"]["policy"] == "identity"
    assert native["encoder"]["params"]["feature_stage"] == "projected_visual"
    assert native["encoder"]["params"]["kv_size"] < config["input"]["sampled_frames"]
    assert native["result_requirements"]["native_compression_applied_steps_min"] == 1
