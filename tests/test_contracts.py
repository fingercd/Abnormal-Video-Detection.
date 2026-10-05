from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from vadbench.contracts import (
    CacheKind,
    CacheUpdate,
    CacheView,
    CapabilityError,
    ClipBatch,
    ContractError,
    EncoderCapabilities,
    EncoderOutput,
    StreamState,
    StreamStep,
    TokenTimeline,
    validate_clip_for_capabilities,
    validate_encoder_adapter,
    validate_stream_step,
)


def make_clip(*, batch: int = 1, frames: int = 4) -> ClipBatch:
    pixels = np.zeros((batch, frames, 8, 10, 3), dtype=np.uint8)
    timestamps = np.repeat(
        np.arange(frames, dtype=np.float64)[None, :] / 4.0,
        batch,
        axis=0,
    )
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        video_ids=tuple(f"video-{index}" for index in range(batch)),
        frame_indices=np.repeat(np.arange(frames, dtype=np.int64)[None, :], batch, axis=0),
    )


def make_timeline(*, batch: int = 1, tokens: int = 3) -> TokenTimeline:
    starts = np.repeat(np.arange(tokens, dtype=np.float64)[None, :], batch, axis=0)
    return TokenTimeline(
        start_s=starts,
        end_s=starts + 0.5,
        source_frame_start=np.repeat(np.arange(tokens, dtype=np.int64)[None, :], batch, axis=0),
        source_frame_end=np.repeat((np.arange(tokens, dtype=np.int64) + 1)[None, :], batch, axis=0),
    )


def test_clip_batch_enforces_canonical_layout_dtype_and_time_axis() -> None:
    clip = make_clip(batch=2, frames=5)

    assert clip.layout == "BTHWC"
    assert clip.dtype == "uint8"
    assert clip.batch_size == 2
    assert clip.num_frames == 5
    assert clip.spatial_size == (8, 10)
    np.testing.assert_array_equal(clip.valid_lengths, [5, 5])
    assert isinstance(clip.metadata, MappingProxyType)

    with pytest.raises(ContractError, match="BTHWC"):
        ClipBatch(
            frames=np.zeros((1, 3, 4, 8), dtype=np.uint8),
            timestamps_s=np.zeros((1, 4)),
            video_ids=("video",),
        )
    with pytest.raises(ContractError, match="C=3"):
        ClipBatch(
            frames=np.zeros((1, 4, 8, 8, 1), dtype=np.uint8),
            timestamps_s=np.zeros((1, 4)),
            video_ids=("video",),
        )
    with pytest.raises(ContractError, match="uint8"):
        ClipBatch(
            frames=np.zeros((1, 4, 8, 8, 3), dtype=np.float32),
            timestamps_s=np.zeros((1, 4)),
            video_ids=("video",),
        )
    with pytest.raises(ContractError, match="单调不减"):
        ClipBatch(
            frames=np.zeros((1, 4, 8, 8, 3), dtype=np.uint8),
            timestamps_s=np.array([[0.0, 0.2, 0.1, 0.3]]),
            video_ids=("video",),
        )


def test_clip_batch_accepts_trailing_padding_but_rejects_mask_holes() -> None:
    frames = np.zeros((1, 4, 8, 8, 3), dtype=np.uint8)
    clip = ClipBatch(
        frames=frames,
        timestamps_s=np.array([[0.0, 0.1, np.nan, np.nan]]),
        video_ids=("video",),
        valid_mask=np.array([[True, True, False, False]]),
        frame_indices=np.array([[0, 1, -1, -1]]),
    )
    np.testing.assert_array_equal(clip.valid_lengths, [2])

    with pytest.raises(ContractError, match="连续前缀"):
        ClipBatch(
            frames=frames,
            timestamps_s=np.array([[0.0, 0.1, 0.2, 0.3]]),
            video_ids=("video",),
            valid_mask=np.array([[True, False, True, False]]),
        )


def test_timeline_and_encoder_output_enforce_matching_bsd_shape() -> None:
    timeline = make_timeline(batch=2, tokens=3)
    output = EncoderOutput(
        features=np.zeros((2, 3, 7), dtype=np.float32),
        timeline=timeline,
        pooled=np.zeros((2, 7), dtype=np.float32),
        aux={"backend": "unit"},
    )

    assert output.tokens is output.features
    assert output.batch_size == 2
    assert output.num_tokens == 3
    assert output.feature_dim == 7
    assert timeline.valid is None

    with pytest.raises(ContractError, match="timeline 一致"):
        EncoderOutput(features=np.zeros((2, 4, 7)), timeline=timeline)
    with pytest.raises(ContractError, match="pooled"):
        EncoderOutput(
            features=np.zeros((2, 3, 7)),
            timeline=timeline,
            pooled=np.zeros((2, 8)),
        )
    with pytest.raises(ContractError, match="end<start"):
        TokenTimeline(start_s=np.array([[1.0]]), end_s=np.array([[0.5]]))


def test_capabilities_reject_fake_cache_and_validate_frame_limits() -> None:
    with pytest.raises(CapabilityError, match="streaming"):
        EncoderCapabilities(supports_fixed_clip=True, supports_kv_cache=True)
    with pytest.raises(CapabilityError, match="cache policy"):
        EncoderCapabilities(
            supports_fixed_clip=False,
            supports_streaming=True,
            supports_external_cache_policy=True,
        )

    memory_capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_visual_memory_cache=True,
    )
    assert {kind.value for kind in memory_capabilities.cache_kinds} == {"visual_memory"}
    assert memory_capabilities.cache_access == "read"

    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        fixed_num_frames=4,
        supports_training=False,
    )
    assert capabilities.supports_grad is False
    assert capabilities.cache_access == "none"
    validate_clip_for_capabilities(make_clip(frames=4), capabilities)
    with pytest.raises(CapabilityError, match="fixed_num_frames"):
        validate_clip_for_capabilities(make_clip(frames=3), capabilities)
    with pytest.raises(CapabilityError, match="训练"):
        validate_clip_for_capabilities(make_clip(frames=4), capabilities, train=True)


def test_cache_view_validates_axis_timeline_and_slices_all_tensors() -> None:
    timeline = make_timeline(tokens=4)
    cache = CacheView(
        kind=CacheKind.KV,
        tensors={
            "layer.0.key": np.arange(24).reshape(1, 2, 4, 3),
            "layer.0.value": np.arange(24, 48).reshape(1, 2, 4, 3),
        },
        sequence_axis=-2,
        timeline=timeline,
    )

    assert cache.kind.value == "decoder_kv"
    assert cache.sequence_axis == 2
    assert cache.sequence_length == 4
    assert cache.batch_size == 1
    assert cache.nbytes == 48 * np.dtype(np.int64).itemsize

    recent = cache.slice_sequence(2, 4)
    assert recent.sequence_length == 2
    np.testing.assert_array_equal(
        recent.tensors["layer.0.key"], cache.tensors["layer.0.key"][:, :, 2:]
    )
    np.testing.assert_array_equal(recent.timeline.start_s, [[2.0, 3.0]])

    with pytest.raises(ContractError, match="timeline"):
        CacheView(
            kind=CacheKind.KV,
            tensors={"key": np.zeros((1, 2, 4, 3))},
            sequence_axis=2,
            timeline=make_timeline(tokens=3),
        )
    with pytest.raises(ContractError, match="batch 轴"):
        CacheView(
            kind=CacheKind.TOKEN,
            tensors={"tokens": np.zeros((1, 4, 3))},
            sequence_axis=0,
            timeline=timeline,
        )


def test_stream_state_replace_and_stream_step_capability_audit() -> None:
    timeline = make_timeline(tokens=2)
    cache = CacheView(
        kind=CacheKind.TOKEN,
        tensors={"tokens": np.zeros((1, 2, 3), dtype=np.float32)},
        sequence_axis=1,
        timeline=timeline,
    )
    previous = StreamState(video_id="video", step_index=0)
    state = previous.replace(step_index=1, caches={"vision": cache})
    output = EncoderOutput(features=np.zeros((1, 2, 3)), timeline=timeline)
    step = StreamStep(
        output=output,
        state=state,
        cache_updates={"vision": CacheUpdate.append(cache)},
        telemetry={"latency_ms": 2.0},
    )
    clip = make_clip(frames=4)
    clip = ClipBatch(
        frames=clip.frames,
        timestamps_s=clip.timestamps_s,
        video_ids=("video",),
        frame_indices=clip.frame_indices,
    )
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_streaming=True,
        supports_token_cache=True,
        supports_external_cache_policy=True,
    )
    validate_stream_step(
        step,
        previous_state=previous,
        chunk=clip,
        capabilities=capabilities,
    )

    with pytest.raises(CapabilityError, match="未声明"):
        validate_stream_step(
            step,
            previous_state=previous,
            chunk=clip,
            capabilities=EncoderCapabilities(
                supports_fixed_clip=True,
                supports_streaming=True,
            ),
        )


def test_adapter_validation_requires_real_stream_methods_and_compression_parameter() -> None:
    class MissingStreamMethods:
        capabilities = EncoderCapabilities(
            supports_fixed_clip=False,
            supports_streaming=True,
            supports_kv_cache=True,
        )

    with pytest.raises(CapabilityError, match="init_state"):
        validate_encoder_adapter(MissingStreamMethods())

    class MissingCompressionParameter:
        capabilities = EncoderCapabilities(
            supports_fixed_clip=False,
            supports_streaming=True,
            supports_kv_cache=True,
            supports_external_cache_policy=True,
        )

        def init_state(self, video_id: str) -> StreamState:
            return StreamState(video_id=video_id)

        def encode_step(self, chunk: ClipBatch, state: StreamState) -> StreamStep:
            raise NotImplementedError

        def finalize(self, state: StreamState) -> None:
            return None

    with pytest.raises(CapabilityError, match="compression"):
        validate_encoder_adapter(MissingCompressionParameter())
