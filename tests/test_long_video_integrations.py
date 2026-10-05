from __future__ import annotations

import builtins
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vadbench.contracts import ClipBatch, EncoderOutput, StreamState
from vadbench.environment_registry import load_encoder_candidates
from vadbench.integrations.catalog import load_integration_catalog
from vadbench.integrations.long_video.base import (
    DEFAULT_NEUTRAL_PROMPT,
    LongVideoAssetError,
    LongVideoWorkerError,
)
from vadbench.integrations.long_video.longvu import LongVUAdapter
from vadbench.integrations.long_video.ma_lmm import MALMMAdapter
from vadbench.integrations.long_video.moviechat import MovieChatAdapter
from vadbench.integrations.long_video.streaming_vlm import StreamingVLMAdapter
from vadbench.integrations.long_video.videochat import VideoChatAdapter
from vadbench.integrations.long_video.videochat_flash import VideoChatFlashAdapter
from vadbench.integrations.long_video.videochat_online import VideoChatOnlineAdapter
from vadbench.registry import EncoderRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXED_TARGETS = (
    (
        "longvu",
        LongVUAdapter,
        "vadbench.integrations.long_video.longvu:LongVUAdapter",
        "projected_visual",
    ),
    (
        "videochat",
        VideoChatAdapter,
        "vadbench.integrations.long_video.videochat:VideoChatAdapter",
        "projected_visual",
    ),
    (
        "videochat_flash",
        VideoChatFlashAdapter,
        "vadbench.integrations.long_video.videochat_flash:VideoChatFlashAdapter",
        "projected_visual",
    ),
)

STREAM_TARGETS = (
    (
        "videochat_online",
        VideoChatOnlineAdapter,
        "vadbench.integrations.long_video.videochat_online:VideoChatOnlineAdapter",
        "visual_memory",
    ),
    (
        "ma_lmm",
        MALMMAdapter,
        "vadbench.integrations.long_video.ma_lmm:MALMMAdapter",
        "visual_memory",
    ),
    (
        "moviechat",
        MovieChatAdapter,
        "vadbench.integrations.long_video.moviechat:MovieChatAdapter",
        "visual_memory",
    ),
    (
        "streaming_vlm",
        StreamingVLMAdapter,
        "vadbench.integrations.long_video.streaming_vlm:StreamingVLMAdapter",
        "decoder_contextual",
    ),
)

CANDIDATE_ONLY_CONFIGS = {
    "uniformerv2": "uniformerv2.yaml",
    "umt": "umt.yaml",
    "infinipot_v": "infinipot-v.yaml",
    "mukv": "mukv.yaml",
}


def _batch(*, start: int = 0, video_id: str = "surveillance") -> ClipBatch:
    frames = np.arange(1 * 4 * 6 * 8 * 3, dtype=np.uint8).reshape(1, 4, 6, 8, 3)
    return ClipBatch(
        frames=frames,
        timestamps_s=np.array(
            [[start * 0.25, (start + 1) * 0.25, (start + 2) * 0.25, (start + 3) * 0.25]],
            dtype=np.float64,
        ),
        frame_indices=np.array([[start, start + 1, start + 2, start + 3]], dtype=np.int64),
        video_ids=(video_id,),
        metadata={"source": "test"},
    )


class _FixedWorker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        batch: ClipBatch,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "shape": tuple(batch.frames.shape),
                "dtype": str(batch.frames.dtype),
                "prompt": prompt,
                "feature_stage": feature_stage,
            }
        )
        return {
            "features": np.arange(1 * 3 * 5, dtype=np.float32).reshape(1, 3, 5),
            "aux": {"worker": "fake-fixed"},
        }


class _StreamWorker:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def init_state(
        self,
        video_id: str,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        self.operations.append("init_state")
        return {
            "video_id": video_id,
            "seen": 0,
            "prompt": prompt,
            "feature_stage": feature_stage,
        }

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        *,
        prompt: str,
        feature_stage: str,
        compression: Any = None,
    ) -> dict[str, Any]:
        self.operations.append("encode_step")
        previous = state.opaque or {"seen": 0}
        seen = int(previous["seen"]) + int(chunk.valid_lengths[0])
        return {
            "features": np.full((1, 2, 6), float(seen), dtype=np.float32),
            "state": {
                "seen": seen,
                "prompt": prompt,
                "feature_stage": feature_stage,
            },
            "telemetry": {
                "worker": "fake-stream",
                "compression_is_none": compression is None,
            },
        }

    def finalize(self, state: StreamState) -> None:
        self.operations.append("finalize")
        return None


class _CacheWorker(_StreamWorker):
    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        *,
        prompt: str,
        feature_stage: str,
        compression: Any = None,
    ) -> dict[str, Any]:
        result = super().encode_step(
            chunk,
            state,
            prompt=prompt,
            feature_stage=feature_stage,
            compression=compression,
        )
        result["cache"] = np.ones((1, 2, 3), dtype=np.float32)
        return result


class _NonFiniteWorker(_FixedWorker):
    def encode(
        self,
        batch: ClipBatch,
        *,
        prompt: str,
        feature_stage: str,
    ) -> dict[str, Any]:
        return {"features": np.array([[[np.nan, 0.0]]], dtype=np.float32)}


@pytest.mark.parametrize(("encoder_id", "adapter_cls", "target", "stage"), FIXED_TARGETS)
def test_fixed_long_video_targets_register_construct_and_normalize_bthwc(
    encoder_id: str,
    adapter_cls: type[Any],
    target: str,
    stage: str,
) -> None:
    worker = _FixedWorker()
    registry = EncoderRegistry()
    registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)

    adapter = registry.create(encoder_id, worker=worker)
    output = adapter.encode(_batch())

    assert isinstance(output, EncoderOutput)
    assert output.features.shape == (1, 3, 5)
    assert output.pooled.shape == (1, 5)
    assert output.timeline.num_tokens == 3
    np.testing.assert_array_equal(output.timeline.start_s, [[0.0, 0.25, 0.5]])
    np.testing.assert_array_equal(output.timeline.source_frame_start, [[0, 1, 2]])
    assert output.aux["integration_id"] == encoder_id
    assert output.aux["feature_stage"] == stage
    assert output.aux["prompt"] == DEFAULT_NEUTRAL_PROMPT
    assert output.aux["implementation_source"] == "external_worker_facade"
    assert np.isfinite(np.asarray(output.features)).all()
    assert worker.calls == [
        {
            "shape": (1, 4, 6, 8, 3),
            "dtype": "uint8",
            "prompt": DEFAULT_NEUTRAL_PROMPT,
            "feature_stage": stage,
        }
    ]


def test_json_worker_preserves_supplied_pooled_and_timeline() -> None:
    class Worker:
        def encode(self, batch: ClipBatch, **_: Any) -> dict[str, Any]:
            return {
                "output": {
                    "features": [[[1.0, 2.0], [3.0, 4.0]]],
                    "pooled": [[9.0, 8.0]],
                },
                "timeline": {
                    "start_s": [[0.0, 0.5]],
                    "end_s": [[0.25, 0.75]],
                    "source_frame_start": [[0, 2]],
                    "source_frame_end": [[1, 3]],
                },
                "aux": {"worker": "json"},
            }

    output = LongVUAdapter(worker=Worker()).encode(_batch())

    assert isinstance(output.features, np.ndarray)
    assert isinstance(output.pooled, np.ndarray)
    np.testing.assert_array_equal(output.pooled, [[9.0, 8.0]])
    np.testing.assert_array_equal(output.timeline.source_frame_start, [[0, 2]])
    assert output.aux["worker"] == "json"


def test_worker_parameter_names_do_not_change_canonical_objects() -> None:
    batch = _batch()
    fixed_inputs: list[Any] = []

    class FixedWorker:
        def encode(self, frames: Any) -> dict[str, Any]:
            fixed_inputs.append(frames)
            return {"features": np.ones((1, 1, 2), dtype=np.float32)}

    LongVUAdapter(worker=FixedWorker()).encode(batch)
    assert len(fixed_inputs) == 1
    assert fixed_inputs[0] is batch

    stream_inputs: list[tuple[Any, Any]] = []

    class StreamWorker:
        def init_state(self, identifier: str) -> dict[str, Any]:
            return {"video_id": identifier}

        def encode_step(self, frames: Any, worker_state: Any) -> dict[str, Any]:
            stream_inputs.append((frames, worker_state))
            return {
                "features": np.ones((1, 1, 2), dtype=np.float32),
                "state": {"seen": 4},
            }

        def finalize(self, worker_state: Any) -> None:
            stream_inputs.append((None, worker_state))

    adapter = StreamingVLMAdapter(worker=StreamWorker())
    state = adapter.init_state("surveillance")
    step = adapter.encode_step(batch, state)
    adapter.finalize(step.state)

    assert len(stream_inputs) == 2
    assert stream_inputs[0][0] is batch
    assert stream_inputs[0][1] is state
    assert stream_inputs[1][0] is None
    assert stream_inputs[1][1] is step.state


@pytest.mark.parametrize(("encoder_id", "adapter_cls", "target", "stage"), STREAM_TARGETS)
def test_streaming_targets_register_construct_and_advance_explicit_state(
    encoder_id: str,
    adapter_cls: type[Any],
    target: str,
    stage: str,
) -> None:
    registry = EncoderRegistry()
    registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)
    worker = _StreamWorker()
    adapter = registry.create(encoder_id, worker=worker, cache_mode="identity")

    state = adapter.init_state("surveillance")
    first = adapter.encode_step(_batch(start=0), state)
    second = adapter.encode_step(_batch(start=4), first.state)

    assert first.state.step_index == 1
    assert second.state.step_index == 2
    assert first.state.opaque["seen"] == 4
    assert second.state.opaque["seen"] == 8
    assert first.output is not None and first.output.features.shape == (1, 2, 6)
    assert second.output is not None and second.output.features.shape == (1, 2, 6)
    assert second.output.aux["feature_stage"] == stage
    assert second.output.aux["prompt"] == DEFAULT_NEUTRAL_PROMPT
    assert second.telemetry["cache_mode"] == "off"
    assert second.telemetry["cache_compression"] == "disabled"
    assert second.state.next_timestamp_s is not None
    assert second.state.next_timestamp_s > first.state.next_timestamp_s
    assert adapter.finalize(second.state) is None
    assert worker.operations == ["init_state", "encode_step", "encode_step", "finalize"]


def test_non_identity_cache_policy_is_rejected_without_calling_worker() -> None:
    worker = _StreamWorker()
    adapter = StreamingVLMAdapter(worker=worker)
    state = adapter.init_state("surveillance")
    with pytest.raises(LongVideoWorkerError) as captured:
        adapter.encode_step(_batch(), state, compression="keep_recent")
    assert captured.value.code == "compression_disabled"
    assert "不支持 KV 压缩" in str(captured.value)
    assert worker.operations == ["init_state"]


def test_nonfinite_worker_output_fails_closed() -> None:
    adapter = LongVUAdapter(worker=_NonFiniteWorker())
    with pytest.raises(LongVideoWorkerError) as captured:
        adapter.encode(_batch())
    assert captured.value.code == "invalid_output_health"


def test_declared_decoder_cache_is_normalized_when_worker_exposes_it() -> None:
    adapter = StreamingVLMAdapter(worker=_CacheWorker())
    state = adapter.init_state("surveillance")
    step = adapter.encode_step(_batch(), state)
    assert set(step.state.caches) == {"default"}
    cache = step.state.caches["default"]
    assert cache.kind.value == "decoder_kv"
    assert cache.sequence_length == 2
    assert step.cache_updates == {}


def test_missing_checkout_and_checkpoint_fail_closed_with_structured_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(LongVideoAssetError) as captured:
        VideoChatAdapter(
            project_root=tmp_path,
            checkout_path="external/missing-videochat",
            model_path="weights/missing-videochat",
        )

    error = captured.value.to_dict()
    assert error["code"] == "missing_asset"
    assert error["integration_id"] == "videochat"
    missing = error["details"]["missing"]
    assert {entry["kind"] for entry in missing} == {"checkout", "checkpoint"}
    assert all(str(tmp_path) in entry["path"] for entry in missing)


def test_explicit_loader_hook_uses_only_local_assets(tmp_path: Path) -> None:
    checkout = tmp_path / "external" / "longvu"
    model_path = tmp_path / "weights" / "longvu"
    checkout.mkdir(parents=True)
    model_path.mkdir(parents=True)
    calls: list[dict[str, Any]] = []
    processor = object()
    processor_calls: list[Any] = []

    class LoadedWorker:
        def encode(self, batch: ClipBatch, *, processor: Any) -> dict[str, Any]:
            processor_calls.append(processor)
            return {"features": np.ones((batch.batch_size, 3, 5), dtype=np.float32)}

    def load_model(
        *, checkout_path: str, model_path: str, device: str
    ) -> tuple[LoadedWorker, object]:
        calls.append({"checkout_path": checkout_path, "model_path": model_path, "device": device})
        return LoadedWorker(), processor

    adapter = LongVUAdapter(
        project_root=tmp_path,
        checkout_path=checkout,
        model_path=model_path,
        load_model_fn=load_model,
        device="cpu",
    )
    output = adapter.encode(_batch())

    assert output.features.shape == (1, 3, 5)
    assert calls == [
        {
            "checkout_path": str(checkout.resolve()),
            "model_path": str(model_path.resolve()),
            "device": "cpu",
        }
    ]
    assert adapter.processor is processor
    assert processor_calls == [processor]
    assert adapter.implementation_source == "explicit_loader"


@pytest.mark.parametrize(
    "loaded",
    [
        {"model": _FixedWorker()},
        (_FixedWorker(), object(), object()),
    ],
)
def test_loader_rejects_ambiguous_returns_without_retry(tmp_path: Path, loaded: Any) -> None:
    model_path = tmp_path / "weights" / "longvu"
    model_path.mkdir(parents=True)
    calls: list[int] = []

    def load_model(**_: Any) -> Any:
        calls.append(1)
        return loaded

    with pytest.raises(LongVideoWorkerError) as captured:
        LongVUAdapter(project_root=tmp_path, model_path=model_path, load_model_fn=load_model)

    assert captured.value.code == "model_load_failed"
    assert calls == [1]


@pytest.mark.parametrize(
    ("route", "error_code"),
    [("fixed", "forward_failed"), ("stream", "forward_failed"), ("loader", "model_load_failed")],
)
def test_upstream_type_error_is_not_retried(tmp_path: Path, route: str, error_code: str) -> None:
    calls: list[str] = []
    sentinel = TypeError("upstream bug")

    if route == "fixed":

        class Worker:
            def encode(self, batch: ClipBatch) -> Any:
                calls.append(route)
                raise sentinel

        def exercise() -> None:
            LongVUAdapter(worker=Worker()).encode(_batch())

    elif route == "stream":

        class Worker:
            def encode_step(self, chunk: ClipBatch, state: StreamState) -> Any:
                calls.append(route)
                raise sentinel

        adapter = StreamingVLMAdapter(worker=Worker())

        def exercise() -> None:
            adapter.encode_step(_batch(), adapter.init_state("surveillance"))

    else:
        model_path = tmp_path / "weights" / "longvu"
        model_path.mkdir(parents=True)

        def load_model(**_: Any) -> Any:
            calls.append(route)
            raise sentinel

        def exercise() -> None:
            LongVUAdapter(project_root=tmp_path, model_path=model_path, load_model_fn=load_model)

    with pytest.raises(LongVideoWorkerError) as captured:
        exercise()

    assert captured.value.code == error_code
    assert captured.value.__cause__ is sentinel
    assert calls == [route]


def test_long_video_modules_remain_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] in {"torch", "transformers", "decord", "timm"}:
            raise AssertionError(f"long-video adapter imported heavy dependency {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    registry = EncoderRegistry()
    for encoder_id, adapter_cls, target, _ in (*FIXED_TARGETS, *STREAM_TARGETS):
        registry.register_lazy(encoder_id, target, capabilities=adapter_cls.capabilities)
        registry.load_factory(encoder_id)
    assert set(registry.names()) == {
        *(item[0] for item in FIXED_TARGETS),
        *(item[0] for item in STREAM_TARGETS),
    }


def test_registered_long_video_configs_and_locks_are_pinned_and_consistent() -> None:
    config_names = {
        "longvu": "longvu.yaml",
        "videochat": "videochat.yaml",
        "videochat_online": "videochat-online.yaml",
        "videochat_flash": "videochat-flash.yaml",
        "ma_lmm": "ma-lmm.yaml",
        "moviechat": "moviechat.yaml",
        "streaming_vlm": "streaming-vlm.yaml",
    }
    stages = {item[0]: item[3] for item in (*FIXED_TARGETS, *STREAM_TARGETS)}
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")

    for encoder_id, file_name in config_names.items():
        config_path = PROJECT_ROOT / "configs" / "encoders" / file_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["adapter"] == encoder_id
        assert config["constructor"]["prompt"] == DEFAULT_NEUTRAL_PROMPT
        assert config["constructor"]["feature_stage"] == stages[encoder_id]
        expected_source = (
            "native_upstream"
            if encoder_id in {"longvu", "streaming_vlm", "videochat_flash", "videochat_online"}
            else "external_worker_facade"
        )
        assert config["output"]["implementation_source"] == expected_source
        assert config["cache_semantics"]["compression"] == "disabled"

        lock_path = PROJECT_ROOT / config["upstream_lock"]
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        assert lock["integration"] == encoder_id
        assert commit_pattern.fullmatch(lock["source"]["commit"])
        assert lock["source"]["commit"] in lock["source"]["commit_url"]
        expected_weight_status = (
            "verified"
            if encoder_id in {"longvu", "streaming_vlm", "videochat_flash", "videochat_online"}
            else "planned"
        )
        assert lock["weights"]["status"] == expected_weight_status


def test_candidate_only_configs_and_locks_remain_research_records() -> None:
    candidates = {item["id"]: item for item in load_encoder_candidates(PROJECT_ROOT)}
    catalog = load_integration_catalog(
        PROJECT_ROOT / "registry/encoder-integrations.yaml",
        project_root=PROJECT_ROOT,
    )

    assert CANDIDATE_ONLY_CONFIGS.keys().isdisjoint(catalog.ids)
    for encoder_id, file_name in CANDIDATE_ONLY_CONFIGS.items():
        candidate = candidates[encoder_id]
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "encoders" / file_name).read_text(encoding="utf-8")
        )
        lock = yaml.safe_load(
            (PROJECT_ROOT / candidate["upstream_lock"]).read_text(encoding="utf-8")
        )

        assert candidate["registration_state"] == "candidate_only"
        assert config["adapter"] == encoder_id
        assert config["upstream_lock"] == candidate["upstream_lock"]
        assert lock["integration"] == encoder_id
        assert lock["source"]["commit"] == candidate["checkpoint"]["revision"]
        assert lock["source"]["commit"] in lock["source"]["commit_url"]
