from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vadbench.compression import IdentityCachePolicy, KeepRecentCachePolicy
from vadbench.contracts import CacheKind, CacheUpdateMode, ClipBatch, ContractError
from vadbench.integrations import (
    HERMES_LLAVA_OV_CAPABILITIES,
    VIDEOMAEV2_CAPABILITIES,
)
from vadbench.integrations.hermes import (
    HermesLlavaOVAdapter,
    import_hermes_load_model,
)
from vadbench.integrations.videomaev2 import VideoMAEv2Adapter
from vadbench.integrations.videomaev2_encoder import (
    VideoMAEv2Encoder as CanonicalVideoMAEv2Encoder,
)
from vadbench.registry import ENCODER_REGISTRY


def make_clip(*, frames: int, video_id: str = "video", start_s: float = 0.0) -> ClipBatch:
    pixels = np.arange(frames * 4 * 6 * 3, dtype=np.uint8).reshape(1, frames, 4, 6, 3)
    timestamps = (start_s + np.arange(frames, dtype=np.float64) / 2.0)[None, :]
    indices = np.arange(frames, dtype=np.int64)[None, :]
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        video_ids=(video_id,),
        frame_indices=indices,
    )


class _HookHandle:
    def __init__(self, hooks: list[Any], hook: Any) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class _HookModule:
    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> _HookHandle:
        self.hooks.append(hook)
        return _HookHandle(self.hooks, hook)

    def emit(self, output: Any) -> None:
        for hook in tuple(self.hooks):
            hook(self, (), output)


class _FakeVideoMAEEncoder:
    def __init__(self) -> None:
        self.backbone = _HookModule()
        self.last_block = _HookModule()
        self.last_clips: Any = None

    def _get_encoder_layers(self) -> list[_HookModule]:
        return [self.last_block]

    def __call__(self, clips: Any) -> np.ndarray:
        self.last_clips = clips
        batch = len(clips)
        sequence = np.arange(batch * 4 * 3, dtype=np.float32).reshape(batch, 4, 3)
        pooled = sequence.mean(axis=1)
        # Child hook runs before the root and exposes the richer sequence.
        self.last_block.emit(sequence)
        self.backbone.emit(pooled)
        return pooled


class _FakeLanguageModelOutput:
    def __init__(
        self,
        *,
        past_key_values: Any,
        hidden_states: tuple[np.ndarray, ...] | None,
    ) -> None:
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states


class _FakeLanguageModel:
    def __init__(self, owner: _FakeHermesModel) -> None:
        self.owner = owner

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeLanguageModelOutput:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        *args: Any,
        inputs_embeds: np.ndarray,
        past_key_values: Any,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> _FakeLanguageModelOutput:
        del args, kwargs
        token_count = inputs_embeds.shape[1]
        past_length = 0 if not past_key_values else past_key_values[0][0].shape[2]
        if past_key_values is None:
            past_key_values = []
            for _ in range(self.owner.num_layers):
                empty = np.zeros((1, 2, 0, 3), dtype=np.float32)
                past_key_values.append((empty, empty.copy()))
        updated = []
        for layer_index, (key, value) in enumerate(past_key_values):
            key_delta = np.full((1, 2, token_count, 3), layer_index + 1, dtype=np.float32)
            value_delta = np.full((1, 2, token_count, 3), layer_index + 11, dtype=np.float32)
            updated.append(
                (
                    np.concatenate((key, key_delta), axis=2),
                    np.concatenate((value, value_delta), axis=2),
                )
            )

        hidden_states = None
        if output_hidden_states and self.owner.expose_decoder_hidden:
            # Include synthetic past positions; the adapter must expose only
            # the current visual q_len, never these sentinel past values.
            past_hidden = np.full(
                (1, past_length, inputs_embeds.shape[-1]),
                -999.0,
                dtype=np.float32,
            )
            current_hidden = np.asarray(inputs_embeds) + float(past_length)
            hidden_states = (np.concatenate((past_hidden, current_hidden), axis=1),)
        return _FakeLanguageModelOutput(
            past_key_values=updated,
            hidden_states=hidden_states,
        )


class _FakeHermesModel:
    def __init__(
        self,
        *,
        layers: int = 2,
        native_summary: bool = False,
        expose_decoder_hidden: bool = True,
    ) -> None:
        self.num_layers = layers
        self.kv_cache: Any = None
        self._position_ids_cache = [None for _ in range(layers)]
        self.token_activity_cache = [None for _ in range(layers)]
        self._layer_position_ids: dict[int, Any] = {}
        self.total_processed_frames = 0
        self.last_encoded_frames = 0
        self.visual_start_idx = 0
        self.conv_history: list[Any] = []
        self.native_predict_calls = 0
        self.native_pseudo_calls = 0
        self.last_pseudo_questions: tuple[str, str] | None = None
        self.native_summary = native_summary
        self.expose_decoder_hidden = expose_decoder_hidden
        self.language_model = _FakeLanguageModel(self)

    def encode_init_prompt(self) -> None:
        self.kv_cache = [
            (
                np.zeros((1, 2, 2, 3), dtype=np.float32),
                np.zeros((1, 2, 2, 3), dtype=np.float32),
            )
            for _ in range(self.num_layers)
        ]
        self._position_ids_cache = [np.arange(2) for _ in range(self.num_layers)]
        self.visual_start_idx = 2

    def get_video_features(self, video_chunk: Any) -> np.ndarray:
        frames = int(video_chunk.shape[0])
        return np.arange(frames * 2 * 3, dtype=np.float32).reshape(1, frames * 2, 3)

    def encode_video_chunk(self, video_chunk: Any) -> None:
        features = self.get_video_features(video_chunk)
        token_count = features.shape[1]
        if self.kv_cache is None:
            self._position_ids_cache = [
                np.asarray([], dtype=np.int64) for _ in range(self.num_layers)
            ]
        output = self.language_model(
            inputs_embeds=features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        updated_positions = []
        for layer_index in range(self.num_layers):
            current_positions = self._position_ids_cache[layer_index]
            start = int(current_positions[-1]) + 1 if len(current_positions) else 0
            updated_positions.append(
                np.concatenate((current_positions, np.arange(start, start + token_count)))
            )
        self.kv_cache = output.past_key_values
        self.last_encoded_frames = int(video_chunk.shape[0])
        self.total_processed_frames += self.last_encoded_frames
        self._position_ids_cache = updated_positions

    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers: Any) -> None:
        compressed = []
        compressed_positions = []
        for layer_index, (key, value) in enumerate(self.kv_cache):
            indices = np.asarray(keep_indices_all_layers[layer_index], dtype=np.int64)
            key_kept = key[:, :, indices, :]
            value_kept = value[:, :, indices, :]
            positions_kept = self._position_ids_cache[layer_index][indices]
            if self.native_summary and layer_index == self.num_layers - 1:
                evicted = np.ones(key.shape[2], dtype=bool)
                evicted[indices] = False
                key_kept = np.concatenate(
                    (key_kept, key[:, :, evicted, :].mean(axis=2, keepdims=True)),
                    axis=2,
                )
                value_kept = np.concatenate(
                    (value_kept, value[:, :, evicted, :].mean(axis=2, keepdims=True)),
                    axis=2,
                )
                positions_kept = np.concatenate(
                    (positions_kept, np.asarray([positions_kept[-1] + 1]))
                )
            compressed.append((key_kept, value_kept))
            compressed_positions.append(positions_kept)
        self.kv_cache = compressed
        self._position_ids_cache = compressed_positions

    def _native_keep_indices(self) -> list[list[int]]:
        result = []
        for key, _value in self.kv_cache:
            length = key.shape[2]
            prefix = list(range(min(self.visual_start_idx, length)))
            visual = list(range(min(self.visual_start_idx, length), length))
            budget = int(self.kv_size)
            if self.native_summary and len(result) == self.num_layers - 1:
                budget -= 1
            result.append(prefix + ([] if budget <= 0 else visual[-budget:]))
        return result

    def predict_and_compress(self) -> None:
        self.native_predict_calls += 1
        self.apply_kv_cache_pruning_strict(self._native_keep_indices())

    def pseudo_forward(self, local_question: str, global_question: str) -> None:
        self.native_pseudo_calls += 1
        self.last_pseudo_questions = (local_question, global_question)
        self.apply_kv_cache_pruning_strict(self._native_keep_indices())


def test_integrations_register_lazily_without_model_dependencies() -> None:
    video_spec = ENCODER_REGISTRY.get_spec("videomaev2")
    hermes_spec = ENCODER_REGISTRY.get_spec("hermes_llava_ov")

    assert video_spec.is_lazy
    assert hermes_spec.is_lazy
    assert video_spec.capabilities == VIDEOMAEV2_CAPABILITIES
    assert hermes_spec.capabilities == HERMES_LLAVA_OV_CAPABILITIES
    assert hermes_spec.metadata["cache_owner"] == "language_model_decoder"

    video_adapter = ENCODER_REGISTRY.create("videomaev2", encoder=_FakeVideoMAEEncoder())
    hermes_adapter = ENCODER_REGISTRY.create("hermes_llava_ov", model=_FakeHermesModel())
    assert video_adapter.capabilities == VIDEOMAEV2_CAPABILITIES
    assert hermes_adapter.capabilities == HERMES_LLAVA_OV_CAPABILITIES


def test_videomaev2_exposes_sequence_and_pooling() -> None:
    fake = _FakeVideoMAEEncoder()
    adapter = VideoMAEv2Adapter(encoder=fake, num_frames=16)
    batch = make_clip(frames=16)

    output = adapter.encode(batch, train=True)

    assert output.features.shape == (1, 4, 3)
    assert output.pooled.shape == (1, 3)
    np.testing.assert_allclose(output.pooled, output.features.mean(axis=1))
    assert output.timeline.start_s.shape == (1, 4)
    assert output.timeline.source_frame_start.shape == (1, 4)
    assert output.aux["sequence_source"] == "observed_backbone"
    assert output.aux["timeline_policy"].endswith("approximation")
    assert len(fake.last_clips) == 1
    assert len(fake.last_clips[0]) == 16
    assert fake.last_clips[0][0].dtype == np.uint8
    assert adapter.capabilities.supports_training
    assert not adapter.capabilities.supports_streaming
    assert not adapter.capabilities.supports_kv_cache
    assert not adapter.capabilities.supports_token_cache

    padded = ClipBatch(
        frames=np.zeros((1, 18, 4, 6, 3), dtype=np.uint8),
        timestamps_s=np.array([[*(np.arange(16, dtype=np.float64) / 2.0), np.nan, np.nan]]),
        video_ids=("padded",),
        valid_mask=np.array([[*[True] * 16, False, False]]),
        frame_indices=np.array([[*range(16), -1, -1]], dtype=np.int64),
    )
    padded_output = adapter.encode(padded)
    assert len(fake.last_clips[0]) == 16
    assert padded_output.timeline.start_s.shape == (1, 4)


def test_legacy_videomaev2_import_reexports_canonical_implementation() -> None:
    from lab_anomaly.models.vit_video_encoder import VideoMAEv2Encoder

    assert VideoMAEv2Encoder is CanonicalVideoMAEv2Encoder


def test_hermes_stream_step_captures_visual_tokens_and_decoder_kv() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(model=model, sample_fps=2.0)
    state = adapter.init_state("video")

    assert len(state.caches) == 2
    assert all(view.kind is CacheKind.DECODER_KV for view in state.caches.values())
    assert all(view.sequence_length == 2 for view in state.caches.values())
    assert all("position_ids" in view.tensors for view in state.caches.values())
    assert all(view.metadata["is_vision_encoder_kv"] is False for view in state.caches.values())

    padded_chunk = ClipBatch(
        frames=np.zeros((1, 6, 4, 6, 3), dtype=np.uint8),
        timestamps_s=np.array([[0.0, 0.5, 1.0, 1.5, np.nan, np.nan]]),
        video_ids=("video",),
        valid_mask=np.array([[True, True, True, True, False, False]]),
        frame_indices=np.array([[0, 1, 2, 3, -1, -1]], dtype=np.int64),
    )
    step = adapter.encode_step(
        padded_chunk,
        state,
        compression=IdentityCachePolicy(),
    )

    assert step.output is not None
    assert step.output.features.shape == (1, 8, 3)
    np.testing.assert_allclose(step.output.pooled, step.output.features.mean(axis=1))
    assert step.output.aux["feature_stage"] == "projected_visual"
    assert step.output.aux["cache_conditioned"] is False
    assert step.output.aux["comparison_scope"] == "performance_only"
    assert step.output.aux["cache_owner"] == "language_model_decoder"
    assert step.state.step_index == 1
    assert all(view.sequence_length == 10 for view in step.state.caches.values())
    assert all(update.view.sequence_length == 8 for update in step.cache_updates.values())
    assert step.telemetry["projected_visual_tokens"] == 8
    assert step.telemetry["input_tokens"] == 10
    assert step.telemetry["reused_tokens"] == 0
    assert step.telemetry["output_tokens"] == 8
    assert step.telemetry["cache_hit"] is False
    assert step.telemetry["frames_encoded"] == 4
    assert step.telemetry["decoder_kv_tokens_before_max"] == 2
    assert step.telemetry["decoder_kv_tokens_after_max"] == 10
    assert step.telemetry["decoder_kv_replaced_by_policy"] is False
    assert step.telemetry["is_vision_encoder_kv"] is False
    assert not adapter.capabilities.supports_token_cache


def test_hermes_feature_stages_expose_distinct_and_explicit_semantics() -> None:
    clip = make_clip(frames=2)
    projected_adapter = HermesLlavaOVAdapter(model=_FakeHermesModel())
    projected = projected_adapter.encode_step(clip, projected_adapter.init_state("video")).output
    assert projected is not None

    contextual_adapter = HermesLlavaOVAdapter(
        model=_FakeHermesModel(),
        feature_stage="decoder_contextual",
    )
    contextual = contextual_adapter.encode_step(clip, contextual_adapter.init_state("video")).output
    assert contextual is not None

    assert projected.features.shape == contextual.features.shape == (1, 4, 3)
    np.testing.assert_allclose(contextual.features, projected.features + 2.0)
    np.testing.assert_allclose(contextual.timeline.start_s, projected.timeline.start_s)
    np.testing.assert_allclose(contextual.timeline.end_s, projected.timeline.end_s)
    assert np.all(np.asarray(contextual.features) != -999.0)
    assert projected.aux["feature_stage"] == "projected_visual"
    assert projected.aux["comparison_scope"] == "performance_only"
    assert projected.aux["cache_conditioned"] is False
    assert contextual.aux["feature_stage"] == "decoder_contextual"
    assert contextual.aux["comparison_scope"] == "accuracy_and_performance"
    assert contextual.aux["cache_conditioned"] is True
    assert contextual.aux["decoder_context_scope"] == "current_q_len_only"


def test_hermes_decoder_contextual_fails_when_upstream_hidden_state_is_missing() -> None:
    model = _FakeHermesModel(expose_decoder_hidden=False)
    adapter = HermesLlavaOVAdapter(model=model, feature_stage="decoder_contextual")

    with pytest.raises(RuntimeError, match="未返回 last_hidden_state/hidden_states"):
        adapter.encode_step(make_clip(frames=2), adapter.init_state("video"))
    assert callable(model.language_model.forward)


def test_hermes_decoder_contextual_features_reflect_compressed_past_cache() -> None:
    identity_adapter = HermesLlavaOVAdapter(
        model=_FakeHermesModel(),
        kv_size=5,
        feature_stage="decoder_contextual",
    )
    identity_first = identity_adapter.encode_step(
        make_clip(frames=4), identity_adapter.init_state("video")
    )
    identity_second = identity_adapter.encode_step(
        make_clip(frames=1, start_s=2.0), identity_first.state
    )

    native_adapter = HermesLlavaOVAdapter(
        model=_FakeHermesModel(),
        kv_size=5,
        feature_stage="decoder_contextual",
        native_compression_mode="predict",
    )
    native_first = native_adapter.encode_step(
        make_clip(frames=4), native_adapter.init_state("video")
    )
    native_second = native_adapter.encode_step(make_clip(frames=1, start_s=2.0), native_first.state)

    assert identity_second.output is not None and native_second.output is not None
    assert max(view.sequence_length for view in identity_first.state.caches.values()) == 10
    assert max(view.sequence_length for view in native_first.state.caches.values()) == 7
    # The fake decoder adds past cache length to current hidden states.  A
    # three-token cache difference must therefore affect accuracy features.
    np.testing.assert_allclose(
        identity_second.output.features,
        native_second.output.features + 3.0,
    )
    assert identity_second.output.aux["cache_conditioned"] is True
    assert native_second.output.aux["cache_conditioned"] is True


def test_hermes_external_policy_replaces_decoder_cache_and_reports_telemetry() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(model=model)
    state = adapter.init_state("video")

    step = adapter.encode_step(
        make_clip(frames=4),
        state,
        compression=KeepRecentCachePolicy(max_tokens=5),
    )

    assert all(view.sequence_length == 5 for view in step.state.caches.values())
    assert all(layer[0].shape[2] == 5 for layer in model.kv_cache)
    assert all(len(position_ids) == 5 for position_ids in model._position_ids_cache)
    np.testing.assert_array_equal(model._position_ids_cache[0], [5, 6, 7, 8, 9])
    assert step.telemetry["decoder_kv_tokens_after_min"] == 5
    assert step.telemetry["decoder_kv_tokens_after_max"] == 5
    assert step.telemetry["external_cache_policy"] == "keep_recent"
    assert step.telemetry["external_cache_policy_applied"] is True
    assert step.telemetry["native_hermes_compression_enabled"] is False
    assert step.telemetry["native_hermes_compression_applied"] is False
    assert step.state.metadata["last_policy"] == "keep_recent"
    second = adapter.encode_step(
        make_clip(frames=2, start_s=2.0),
        step.state,
        compression=KeepRecentCachePolicy(max_tokens=5),
    )
    assert all(view.sequence_length == 5 for view in second.state.caches.values())
    assert second.telemetry["cache_hit"] is True
    assert second.telemetry["reused_tokens"] == 5
    np.testing.assert_array_equal(model._position_ids_cache[0], [9, 10, 11, 12, 13])
    assert adapter.finalize(step.state) is None


def test_hermes_native_predict_compression_calls_upstream_and_replaces_cache() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(
        model=model,
        kv_size=5,
        native_compression_mode="predict",
    )
    state = adapter.init_state("video")

    step = adapter.encode_step(
        make_clip(frames=4),
        state,
        compression=IdentityCachePolicy(),
    )

    assert model.native_predict_calls == 1
    assert all(view.sequence_length == 7 for view in step.state.caches.values())
    assert all(update.mode is CacheUpdateMode.REPLACE for update in step.cache_updates.values())
    assert step.telemetry["native_hermes_compression_called"] is True
    assert step.telemetry["native_hermes_compression_applied"] is True
    assert step.telemetry["native_hermes_compression_mode"] == "predict"
    assert step.telemetry["native_hermes_tokens_before_max"] == 10
    assert step.telemetry["native_hermes_tokens_after_max"] == 7
    assert step.telemetry["native_hermes_visual_budget_tokens"] == 5
    assert (
        step.telemetry["native_hermes_tokens_after_max"]
        - step.telemetry["native_hermes_protected_prefix_tokens"]
        == step.telemetry["native_hermes_visual_budget_tokens"]
    )
    assert step.telemetry["native_hermes_effective_total_budget_tokens"] == 7
    assert step.telemetry["native_hermes_compression_ms"] >= 0.0
    assert step.telemetry["external_cache_policy_applied"] is False
    assert step.state.metadata["last_policy"] == "native_hermes:predict"

    second = adapter.encode_step(make_clip(frames=1, start_s=2.0), step.state)
    assert model.native_predict_calls == 2
    assert all(view.sequence_length == 7 for view in second.state.caches.values())
    assert second.telemetry["native_hermes_tokens_before_max"] == 9
    assert second.telemetry["native_hermes_tokens_after_max"] == 7


def test_hermes_native_compression_waits_until_effective_budget_is_exceeded() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(
        model=model,
        kv_size=100,
        native_compression_mode="predict",
    )
    step = adapter.encode_step(make_clip(frames=2), adapter.init_state("video"))

    assert model.native_predict_calls == 0
    assert step.telemetry["native_hermes_compression_enabled"] is True
    assert step.telemetry["native_hermes_compression_called"] is False
    assert step.telemetry["native_hermes_compression_applied"] is False
    assert step.telemetry["native_hermes_tokens_before_max"] == 6
    assert step.telemetry["native_hermes_tokens_after_max"] == 6


def test_hermes_native_replace_may_be_shorter_than_previous_stream_state() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(model=model, kv_size=5)
    raw_step = adapter.encode_step(make_clip(frames=4), adapter.init_state("video"))
    assert all(view.sequence_length == 10 for view in raw_step.state.caches.values())

    # Simulate enabling the registered native policy for a later ablation step:
    # post-compression state is shorter than the previous explicit state, which
    # must be represented as REPLACE rather than rejected as a bad append.
    adapter.native_compression_mode = "predict"
    compressed = adapter.encode_step(
        make_clip(frames=1, start_s=2.0),
        raw_step.state,
    )

    assert all(view.sequence_length == 7 for view in compressed.state.caches.values())
    assert min(view.sequence_length for view in raw_step.state.caches.values()) > 7
    assert all(
        update.mode is CacheUpdateMode.REPLACE for update in compressed.cache_updates.values()
    )
    assert compressed.telemetry["native_hermes_compression_applied"] is True


def test_hermes_native_static_pseudo_uses_fixed_questions_and_rejects_double_policy() -> None:
    model = _FakeHermesModel()
    adapter = HermesLlavaOVAdapter(
        model=model,
        kv_size=3,
        native_compression_mode="static_pseudo",
        native_local_question="local neutral",
        native_global_question="global neutral",
    )
    state = adapter.init_state("video")

    with pytest.raises(ContractError, match="双重压缩"):
        adapter.encode_step(
            make_clip(frames=4),
            state,
            compression=KeepRecentCachePolicy(max_tokens=2),
        )
    assert model.native_pseudo_calls == 0

    step = adapter.encode_step(make_clip(frames=4), state)
    assert model.native_pseudo_calls == 1
    assert model.last_pseudo_questions == ("local neutral", "global neutral")
    assert step.telemetry["native_hermes_compression_mode"] == "static_pseudo"
    assert step.telemetry["native_hermes_compression_applied"] is True
    assert step.telemetry["native_hermes_tokens_after_max"] == 5


def test_hermes_native_summary_keeps_position_and_timeline_shapes_aligned() -> None:
    model = _FakeHermesModel(native_summary=True)
    adapter = HermesLlavaOVAdapter(
        model=model,
        kv_size=5,
        native_compression_mode="predict",
    )
    step = adapter.encode_step(make_clip(frames=4), adapter.init_state("video"))

    assert all(view.sequence_length == 7 for view in step.state.caches.values())
    last = step.state.caches["decoder_kv.layer.1"]
    assert last.metadata["native_summary_tokens"] == 1
    assert (
        last.metadata["native_summary_source_end_s"]
        >= last.metadata["native_summary_source_start_s"]
    )
    assert last.tensors["position_ids"].shape[-2] == last.sequence_length
    assert last.timeline.num_tokens == last.sequence_length
    starts = np.asarray(last.timeline.start_s)[0]
    ends = np.asarray(last.timeline.end_s)[0]
    assert np.all(np.diff(starts) >= 0)
    assert np.all(np.diff(ends) >= 0)


def test_hermes_load_model_is_imported_from_explicit_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "hermes"
    inference = checkout / "inference"
    inference.mkdir(parents=True)
    (inference / "__init__.py").write_text("", encoding="utf-8")
    (inference / "llavaov_hermes.py").write_text(
        "def load_model(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )

    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "inference" or name.startswith("inference.")
    }
    for name in tuple(previous_modules):
        sys.modules.pop(name, None)
    try:
        loader = import_hermes_load_model(checkout)
        assert loader(model_path="fake")["model_path"] == "fake"
        assert Path(sys.modules["inference.llavaov_hermes"].__file__).is_relative_to(checkout)
    finally:
        for name in tuple(sys.modules):
            if name == "inference" or name.startswith("inference."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def test_hermes_rejects_fixed_clip_shortcut() -> None:
    adapter = HermesLlavaOVAdapter(model=_FakeHermesModel())
    with pytest.raises(Exception, match="只声明 streaming"):
        adapter.encode(make_clip(frames=2))
