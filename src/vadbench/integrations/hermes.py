"""Streaming adapter for the upstream HERMES LLaVA-OneVision implementation.

HERMES stores projected visual tokens in the *language-model decoder* KV cache.
The cache views exposed here are therefore deliberately named ``decoder_kv``;
they are not vision-encoder KV activations.  The classifier-facing output is
captured separately from ``get_video_features`` immediately before those visual
tokens are fed to the decoder.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    CacheKind,
    CachePolicy,
    CacheUpdate,
    CacheView,
    CapabilityError,
    ClipBatch,
    ContractError,
    EncoderCapabilities,
    EncoderOutput,
    StreamingVideoEncoderAdapter,
    StreamState,
    StreamStep,
    TokenTimeline,
    validate_clip_for_capabilities,
    validate_stream_step,
)

DEFAULT_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=False,
    supports_streaming=True,
    supports_kv_cache=True,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=True,
    supports_training=False,
    min_frames=1,
)

_IMPORT_LOCK = threading.RLock()
_INTERNAL_STATE_FIELDS = (
    "_position_ids_cache",
    "token_activity_cache",
    "total_processed_frames",
    "last_encoded_frames",
    "visual_start_idx",
    "conv_history",
)
_NATIVE_COMPRESSION_MODES = frozenset({"off", "predict", "static_pseudo"})
_FEATURE_STAGES = frozenset({"projected_visual", "decoder_contextual"})


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _to_numpy(value: Any) -> np.ndarray:
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_metadata(value: Any) -> Any:
    """Clone small HERMES bookkeeping state without cloning the large KV cache."""

    if isinstance(value, dict):
        return {key: _clone_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_metadata(item) for item in value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except Exception:  # pragma: no cover - exotic upstream state
            return value
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def import_hermes_load_model(checkout_path: str | os.PathLike[str]) -> Any:
    """Load HERMES' non-packaged ``load_model`` from an explicit checkout.

    Upstream uses absolute imports such as ``inference.abstract_hermes``.  The
    checkout root must consequently remain on ``sys.path`` for the model's
    later lazy imports.  We validate module provenance to avoid silently using
    an unrelated top-level package also named ``inference``.
    """

    checkout = Path(checkout_path).expanduser().resolve()
    entrypoint = checkout / "inference" / "llavaov_hermes.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"HERMES checkout 缺少 inference/llavaov_hermes.py：{checkout}")

    with _IMPORT_LOCK:
        existing = sys.modules.get("inference.llavaov_hermes")
        if existing is not None:
            origin = getattr(existing, "__file__", None)
            if origin is None or not _is_within(Path(origin), checkout):
                raise ImportError(
                    "进程已导入另一个 inference.llavaov_hermes；请在干净进程中加载指定 "
                    f"HERMES checkout：{checkout}"
                )
            loader = getattr(existing, "load_model", None)
            if not callable(loader):
                raise ImportError("HERMES inference.llavaov_hermes 缺少 callable load_model")
            return loader

        checkout_text = str(checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module("inference.llavaov_hermes")
        except Exception as exc:
            raise ImportError(
                "无法从 external checkout 导入 HERMES；请按 upstream lock 安装其隔离依赖"
            ) from exc
        origin = getattr(module, "__file__", None)
        if origin is None or not _is_within(Path(origin), checkout):
            raise ImportError(
                "导入的 inference.llavaov_hermes 不属于指定 HERMES checkout："
                f"origin={origin!r}, checkout={checkout}"
            )
        loader = getattr(module, "load_model", None)
        if not callable(loader):
            raise ImportError("HERMES inference.llavaov_hermes 缺少 callable load_model")
        return loader


def _cache_layers(raw_cache: Any) -> list[tuple[Any, Any]]:
    if raw_cache is None:
        return []
    if hasattr(raw_cache, "to_legacy_cache"):
        raw_cache = raw_cache.to_legacy_cache()
    elif hasattr(raw_cache, "key_cache") and hasattr(raw_cache, "value_cache"):
        raw_cache = tuple(zip(raw_cache.key_cache, raw_cache.value_cache, strict=True))
    try:
        layers = list(raw_cache)
    except TypeError as exc:
        raise ContractError("HERMES decoder kv_cache 不是可迭代的逐层缓存") from exc
    normalized: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, Sequence) or len(layer) < 2:
            raise ContractError(f"HERMES decoder KV 第 {layer_index} 层缺少 key/value")
        key, value = layer[0], layer[1]
        key_shape, value_shape = _shape(key), _shape(value)
        if len(key_shape) < 3 or key_shape != value_shape:
            raise ContractError(
                f"HERMES decoder KV 第 {layer_index} 层 key/value shape 非法："
                f"{key_shape}/{value_shape}"
            )
        normalized.append((key, value))
    return normalized


def _cache_name(layer_index: int) -> str:
    return f"decoder_kv.layer.{layer_index}"


def _flat_timeline(
    *, batch_size: int, token_count: int, start_s: float = 0.0, end_s: float = 0.0
) -> TokenTimeline:
    starts = np.full((batch_size, token_count), start_s, dtype=np.float64)
    ends = np.full((batch_size, token_count), end_s, dtype=np.float64)
    return TokenTimeline(start_s=starts, end_s=ends)


def _chunk_token_timeline(
    chunk: ClipBatch, token_count: int, *, source_frames: bool
) -> TokenTimeline:
    valid_frames = int(chunk.valid_lengths[0])
    times = _to_numpy(chunk.timestamps_s)[:, :valid_frames].astype(np.float64, copy=False)
    batch_size, num_frames = times.shape
    frame_slots = np.minimum(
        (np.arange(token_count, dtype=np.int64) * num_frames) // token_count,
        num_frames - 1,
    )
    starts = times[:, frame_slots]
    frame_ends = np.empty_like(times)
    if num_frames > 1:
        frame_ends[:, :-1] = times[:, 1:]
        diffs = np.diff(times, axis=1)
        duration = np.asarray(
            [np.median(row[row > 0]) if np.any(row > 0) else 0.0 for row in diffs]
        )
    else:
        duration = np.zeros(batch_size, dtype=np.float64)
    frame_ends[:, -1] = times[:, -1] + duration
    ends = frame_ends[:, frame_slots]
    if not source_frames:
        return TokenTimeline(start_s=starts, end_s=ends)

    if chunk.frame_indices is None:
        indices = np.broadcast_to(np.arange(num_frames), (batch_size, num_frames))
    else:
        indices = _to_numpy(chunk.frame_indices)[:, :valid_frames]
    token_indices = indices[:, frame_slots].astype(np.int64, copy=False)
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=token_indices,
        source_frame_end=token_indices + 1,
    )


def _slice_sequence(value: Any, start: int, stop: int) -> Any:
    slices = [slice(None)] * len(_shape(value))
    slices[-2] = slice(start, stop)
    return value[tuple(slices)]


def _mean_tokens(features: Any) -> Any:
    if type(features).__module__.split(".", 1)[0] == "torch":
        return features.mean(dim=1)
    return np.asarray(features).mean(axis=1)


def _ensure_single_batch_bsd(value: Any, label: str) -> Any:
    shape = _shape(value)
    if len(shape) == 2:
        if hasattr(value, "unsqueeze"):
            value = value.unsqueeze(0)
        else:
            value = np.expand_dims(np.asarray(value), axis=0)
        shape = _shape(value)
    if len(shape) != 3 or shape[0] != 1 or min(shape) <= 0:
        raise RuntimeError(f"{label} 必须是非空 [1,S,D]，实际为 {shape}")
    return value


def _decoder_hidden_from_output(output: Any) -> Any | None:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    if hidden is not None:
        return hidden
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None and isinstance(output, Mapping):
        hidden_states = output.get("hidden_states")
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (tuple, list)):
        return hidden_states[-1] if hidden_states else None
    return hidden_states


def _as_model_chunk(frames: Any) -> Any:
    """Use a torch tensor for real upstream code, while keeping fake tests light."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return _to_numpy(frames)
    if isinstance(frames, torch.Tensor):
        return frames
    return torch.as_tensor(np.asarray(frames), dtype=torch.uint8)


def _cache_summary(caches: Mapping[str, CacheView]) -> dict[str, int]:
    lengths = [cache.sequence_length for cache in caches.values()]
    return {
        "layers": len(lengths),
        "tokens_min": min(lengths, default=0),
        "tokens_max": max(lengths, default=0),
        "tokens_total": sum(lengths),
        "bytes": sum(cache.nbytes for cache in caches.values()),
    }


def _normalize_native_compression_mode(value: str | bool | None) -> str:
    if value is None or value is False:
        return "off"
    if value is True:
        return "predict"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"none": "off", "disabled": "off", "false": "off", "true": "predict"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _NATIVE_COMPRESSION_MODES:
        raise ValueError(
            f"native_compression_mode 必须是 off、predict 或 static_pseudo，实际为 {value!r}"
        )
    return normalized


def _normalize_feature_stage(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "projected": "projected_visual",
        "visual": "projected_visual",
        "decoder": "decoder_contextual",
        "contextual": "decoder_contextual",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _FEATURE_STAGES:
        raise ValueError(
            f"feature_stage 必须是 projected_visual 或 decoder_contextual，实际为 {value!r}"
        )
    return normalized


class HermesLlavaOVAdapter(StreamingVideoEncoderAdapter):
    """Expose HERMES projected tokens and replaceable decoder KV state."""

    capabilities = DEFAULT_CAPABILITIES

    def __init__(
        self,
        *,
        checkout_path: str | os.PathLike[str] = "external/hermes",
        model_path: str | os.PathLike[str] = "weights/hermes-llava-ov-0.5b",
        kv_size: int | None = 2000,
        sample_fps: float = 0.5,
        streaming: bool = True,
        device: str = "cuda",
        initialize_prompt: bool = True,
        feature_stage: str = "projected_visual",
        native_compression_mode: str | bool | None = "off",
        native_local_question: str = (
            "Describe recent visible actions, objects, people, and scene changes in concrete terms."
        ),
        native_global_question: str = (
            "Summarize the video timeline, recurring entities, locations, and interactions."
        ),
        model: Any | None = None,
        processor: Any | None = None,
        load_model_fn: Any | None = None,
    ) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps 必须大于 0")
        if kv_size is not None and (type(kv_size) is not int or kv_size <= 0):
            raise ValueError("kv_size 必须是正整数或 None")
        normalized_feature_stage = _normalize_feature_stage(feature_stage)
        native_mode = _normalize_native_compression_mode(native_compression_mode)
        if native_mode != "off" and kv_size is None:
            raise ValueError("启用 HERMES 原生压缩时 kv_size 不能为空")
        if not native_local_question.strip() or not native_global_question.strip():
            raise ValueError("HERMES native pseudo questions 不能为空")
        if model is None:
            loader = load_model_fn or import_hermes_load_model(checkout_path)
            loaded = loader(
                model_path=str(model_path),
                kv_size=kv_size,
                streaming=streaming,
                device=device,
                sample_fps=sample_fps,
            )
            if not isinstance(loaded, Sequence) or len(loaded) < 1:
                raise RuntimeError("HERMES load_model 必须返回 (model, processor)")
            model = loaded[0]
            processor = loaded[1] if len(loaded) > 1 else processor
        if not callable(getattr(model, "encode_video_chunk", None)):
            raise TypeError("HERMES model 必须实现 encode_video_chunk")
        if not callable(getattr(model, "get_video_features", None)):
            raise TypeError("HERMES model 必须实现 get_video_features 以捕获投影后视觉 token")
        if normalized_feature_stage == "decoder_contextual":
            language_model = getattr(model, "language_model", None)
            if language_model is None or not callable(getattr(language_model, "forward", None)):
                raise TypeError(
                    "feature_stage=decoder_contextual 需要 callable model.language_model.forward"
                )
        if native_mode != "off" and not callable(
            getattr(model, "apply_kv_cache_pruning_strict", None)
        ):
            raise TypeError("启用 HERMES 原生压缩需要 apply_kv_cache_pruning_strict")
        required_native_method = (
            "predict_and_compress" if native_mode == "predict" else "pseudo_forward"
        )
        if native_mode != "off" and not callable(getattr(model, required_native_method, None)):
            raise TypeError(
                f"native_compression_mode={native_mode} 需要 model.{required_native_method}"
            )

        self.model = model
        if kv_size is not None:
            self.model.kv_size = kv_size
        self.processor = processor or getattr(model, "processor", None)
        self.kv_size = kv_size
        self.sample_fps = float(sample_fps)
        self.initialize_prompt = bool(initialize_prompt)
        self.feature_stage = normalized_feature_stage
        self.native_compression_mode = native_mode
        self.native_local_question = native_local_question
        self.native_global_question = native_global_question
        self._lock = threading.RLock()

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        raise CapabilityError(
            "HERMES 适配器只声明 streaming；请使用 init_state/encode_step，"
            "避免把有状态 decoder KV 路径当作固定 clip encoder"
        )

    def _reset_model(self) -> None:
        self.model.kv_cache = None
        if hasattr(self.model, "conv_history"):
            self.model.conv_history = []
        if hasattr(self.model, "_position_ids_cache"):
            count = len(self.model._position_ids_cache)
            self.model._position_ids_cache = [None for _ in range(count)]
        if hasattr(self.model, "token_activity_cache"):
            count = len(self.model.token_activity_cache)
            self.model.token_activity_cache = [None for _ in range(count)]
        if hasattr(self.model, "_layer_position_ids"):
            self.model._layer_position_ids = {}
        if hasattr(self.model, "total_processed_frames"):
            self.model.total_processed_frames = 0
        if hasattr(self.model, "last_encoded_frames"):
            self.model.last_encoded_frames = 0
        if hasattr(self.model, "visual_start_idx"):
            self.model.visual_start_idx = 0
        reset_rotary = getattr(self.model, "_set_rotary_required_len", None)
        if callable(reset_rotary):
            reset_rotary(0)

    def _snapshot_internal_state(self) -> dict[str, Any]:
        snapshot = {
            name: _clone_metadata(getattr(self.model, name))
            for name in _INTERNAL_STATE_FIELDS
            if hasattr(self.model, name)
        }
        if getattr(self.model, "kv_cache", None) is not None:
            # Preserve only the lightweight container class, never a hidden
            # second copy of the cache tensors.  The first HERMES step expects
            # the DynamicCache created by encode_init_prompt.
            snapshot["decoder_kv_container_type"] = type(self.model.kv_cache)
        return snapshot

    def _restore_internal_state(self, opaque: Any) -> None:
        if not isinstance(opaque, Mapping):
            return
        for name in _INTERNAL_STATE_FIELDS:
            if name in opaque:
                setattr(self.model, name, _clone_metadata(opaque[name]))

    @staticmethod
    def _view_for_layer(
        layer_index: int,
        key: Any,
        value: Any,
        position_ids: Any,
        timeline: TokenTimeline,
        metadata: Mapping[str, Any] | None = None,
    ) -> CacheView:
        view_metadata = {
            "owner": "language_model_decoder",
            "layer_index": layer_index,
            "is_vision_encoder_kv": False,
        }
        if metadata is not None:
            view_metadata.update(metadata)
        return CacheView(
            kind=CacheKind.DECODER_KV,
            tensors={"key": key, "value": value, "position_ids": position_ids},
            sequence_axis=-2,
            timeline=timeline,
            metadata=view_metadata,
        )

    def _position_ids_tensor(
        self,
        layer_index: int,
        key: Any,
        *,
        require_existing: bool = False,
    ) -> Any:
        """Normalize one HERMES layer's decoder positions to [B,1,S,1]."""

        key_shape = _shape(key)
        batch_size, length = key_shape[0], key_shape[-2]
        raw: Any | None = None
        caches = getattr(self.model, "_position_ids_cache", None)
        if caches is not None and layer_index < len(caches):
            candidate = caches[layer_index]
            if (
                candidate is not None
                and int(
                    candidate.numel() if hasattr(candidate, "numel") else np.asarray(candidate).size
                )
                == length
            ):
                raw = candidate
        if require_existing and raw is None:
            raise ContractError(f"HERMES 第 {layer_index} 层压缩后的 position_ids 与 KV 长度不一致")

        if type(key).__module__.split(".", 1)[0] == "torch":
            torch = importlib.import_module("torch")
            if raw is None:
                positions = torch.arange(length, device=key.device, dtype=torch.long)
            else:
                positions = torch.as_tensor(raw, device=key.device, dtype=torch.long).reshape(-1)
            return positions.reshape(1, 1, length, 1).expand(batch_size, 1, length, 1)

        if raw is None:
            positions_np = np.arange(length, dtype=np.int64)
        else:
            positions_np = _to_numpy(raw).astype(np.int64, copy=False).reshape(-1)
        return np.broadcast_to(
            positions_np.reshape(1, 1, length, 1),
            (batch_size, 1, length, 1),
        )

    def _initial_cache_views(self) -> dict[str, CacheView]:
        views: dict[str, CacheView] = {}
        for layer_index, (key, value) in enumerate(_cache_layers(self.model.kv_cache)):
            shape = _shape(key)
            views[_cache_name(layer_index)] = self._view_for_layer(
                layer_index,
                key,
                value,
                self._position_ids_tensor(layer_index, key),
                _flat_timeline(batch_size=shape[0], token_count=shape[-2]),
            )
        return views

    def _install_cache_views(
        self,
        caches: Mapping[str, CacheView],
        *,
        container_type: type[Any] | None = None,
    ) -> None:
        if not caches:
            self.model.kv_cache = None
            return
        ordered = sorted(
            caches.values(), key=lambda view: int(view.metadata.get("layer_index", -1))
        )
        expected = list(range(len(ordered)))
        actual = [int(view.metadata.get("layer_index", -1)) for view in ordered]
        if actual != expected:
            raise ContractError(f"decoder KV layer_index 必须连续，实际为 {actual}")
        if any("position_ids" not in view.tensors for view in ordered):
            raise ContractError("decoder KV replacement 必须同时保留 position_ids")
        legacy_cache = [(view.tensors["key"], view.tensors["value"]) for view in ordered]
        from_legacy = getattr(container_type, "from_legacy_cache", None)
        if callable(from_legacy):
            self.model.kv_cache = from_legacy(tuple(legacy_cache))
        else:
            self.model.kv_cache = legacy_cache

        # A policy can shorten or reorder cache tensors.  Restore the exact
        # decoder positions that travelled through the same policy instead of
        # silently relabelling already-RoPE-rotated keys with arange positions.
        if hasattr(self.model, "_position_ids_cache"):
            positions: list[Any] = []
            for view in ordered:
                position_ids = view.tensors["position_ids"]
                # B=1 is enforced for streaming; HERMES stores one 1-D decoder
                # position vector per layer.
                positions.append(
                    position_ids[0, 0, :, 0].clone()
                    if hasattr(position_ids, "clone")
                    else np.asarray(position_ids[0, 0, :, 0]).copy()
                )
            self.model._position_ids_cache = positions
        if hasattr(self.model, "token_activity_cache"):
            self.model.token_activity_cache = [None for _ in ordered]
        if hasattr(self.model, "_layer_position_ids"):
            self.model._layer_position_ids = {}

    def init_state(self, video_id: str) -> StreamState:
        if not isinstance(video_id, str) or not video_id:
            raise ContractError("video_id 必须是非空字符串")
        with self._lock:
            self._reset_model()
            if self.initialize_prompt:
                encode_prompt = getattr(self.model, "encode_init_prompt", None)
                if not callable(encode_prompt):
                    raise RuntimeError(
                        "initialize_prompt=true，但 HERMES model 缺少 encode_init_prompt"
                    )
                encode_prompt()
            caches = self._initial_cache_views()
            return StreamState(
                video_id=video_id,
                caches=caches,
                opaque=self._snapshot_internal_state(),
                metadata={
                    "adapter": "hermes_llava_ov",
                    "cache_owner": "language_model_decoder",
                },
            )

    def _capture_chunk_features(self, model_chunk: Any) -> tuple[Any, Any | None]:
        projected_outputs: list[Any] = []
        contextual_outputs: list[Any] = []
        original_get_video_features = self.model.get_video_features
        language_model = getattr(self.model, "language_model", None)
        original_language_forward = (
            getattr(language_model, "forward", None)
            if self.feature_stage == "decoder_contextual"
            else None
        )

        def recording_get_video_features(*args: Any, **kwargs: Any) -> Any:
            features = original_get_video_features(*args, **kwargs)
            projected_outputs.append(features)
            return features

        def recording_language_forward(*args: Any, **kwargs: Any) -> Any:
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None or len(_shape(inputs_embeds)) != 3:
                raise RuntimeError(
                    "HERMES current-chunk decoder forward 缺少 [B,q_len,D] inputs_embeds"
                )
            q_len = _shape(inputs_embeds)[1]
            forwarded = dict(kwargs)
            forwarded["output_hidden_states"] = True
            assert callable(original_language_forward)
            output = original_language_forward(*args, **forwarded)
            hidden = _decoder_hidden_from_output(output)
            if hidden is None:
                raise RuntimeError(
                    "feature_stage=decoder_contextual，但 language_model 当前 chunk forward "
                    "未返回 last_hidden_state/hidden_states"
                )
            hidden = _ensure_single_batch_bsd(hidden, "decoder contextual hidden state")
            if _shape(hidden)[1] < q_len:
                raise RuntimeError(
                    "decoder contextual hidden state 短于当前 q_len："
                    f"hidden={_shape(hidden)[1]}, q_len={q_len}"
                )
            # Some model wrappers may expose past+current hidden states.  Only
            # the current visual q_len belongs to this chunk's EncoderOutput.
            contextual_outputs.append(hidden[:, -q_len:, :])
            return output

        self.model.get_video_features = recording_get_video_features
        if self.feature_stage == "decoder_contextual":
            assert language_model is not None and callable(original_language_forward)
            language_model.forward = recording_language_forward
        try:
            # This is the real HERMES streaming path.  We intentionally do not
            # call get_video_features separately and duplicate vision compute.
            self.model.encode_video_chunk(model_chunk)
        finally:
            self.model.get_video_features = original_get_video_features
            if self.feature_stage == "decoder_contextual":
                language_model.forward = original_language_forward
        if not projected_outputs:
            raise RuntimeError(
                "HERMES encode_video_chunk 未调用 get_video_features，无法无重复计算地捕获视觉 token"
            )
        projected = _ensure_single_batch_bsd(projected_outputs[-1], "HERMES 投影后视觉 token")
        contextual = None
        if self.feature_stage == "decoder_contextual":
            if not contextual_outputs:
                raise RuntimeError(
                    "feature_stage=decoder_contextual，但未捕获当前 chunk decoder hidden state"
                )
            contextual = _ensure_single_batch_bsd(
                contextual_outputs[-1], "decoder contextual hidden state"
            )
            if _shape(contextual)[1] != _shape(projected)[1]:
                raise RuntimeError(
                    "decoder contextual current q_len 与 projected visual token 数不一致："
                    f"contextual={_shape(contextual)[1]}, projected={_shape(projected)[1]}"
                )
        return projected, contextual

    def _build_raw_after_views(
        self,
        chunk: ClipBatch,
        state: StreamState,
    ) -> tuple[dict[str, CacheView], dict[str, CacheUpdate]]:
        """Describe the append-only cache immediately after encode_video_chunk."""

        layers = _cache_layers(self.model.kv_cache)
        if state.caches and len(layers) != len(state.caches):
            raise ContractError(f"HERMES decoder KV 层数改变：{len(state.caches)} -> {len(layers)}")
        raw_views: dict[str, CacheView] = {}
        updates: dict[str, CacheUpdate] = {}

        for layer_index, (key, value) in enumerate(layers):
            name = _cache_name(layer_index)
            current = state.caches.get(name)
            after_length = _shape(key)[-2]
            before_length = current.sequence_length if current is not None else 0
            if after_length < before_length:
                raise ContractError(
                    f"HERMES encode_video_chunk 意外缩短 {name}：{before_length}->{after_length}"
                )
            delta_length = after_length - before_length
            if delta_length == 0:
                if current is not None:
                    raw_views[name] = current
                continue

            delta_timeline = _chunk_token_timeline(chunk, delta_length, source_frames=False)
            delta_view = self._view_for_layer(
                layer_index,
                _slice_sequence(key, before_length, after_length),
                _slice_sequence(value, before_length, after_length),
                _slice_sequence(
                    self._position_ids_tensor(layer_index, key),
                    before_length,
                    after_length,
                ),
                delta_timeline,
            )
            update = CacheUpdate.append(delta_view, owner="language_model_decoder")
            updates[name] = update
            # The raw HERMES cache already contains current + delta.  Build a
            # matching view without copying its large key/value tensors.
            if current is None:
                full_timeline = delta_timeline
            else:
                full_timeline = TokenTimeline(
                    start_s=np.concatenate(
                        (
                            _to_numpy(current.timeline.start_s),
                            _to_numpy(delta_timeline.start_s),
                        ),
                        axis=1,
                    ),
                    end_s=np.concatenate(
                        (_to_numpy(current.timeline.end_s), _to_numpy(delta_timeline.end_s)),
                        axis=1,
                    ),
                )
            raw_views[name] = self._view_for_layer(
                layer_index,
                key,
                value,
                self._position_ids_tensor(layer_index, key),
                full_timeline,
            )
        return raw_views, updates

    def _apply_external_policy(
        self,
        raw_views: Mapping[str, CacheView],
        raw_updates: Mapping[str, CacheUpdate],
        state: StreamState,
        compression: CachePolicy | None,
    ) -> tuple[dict[str, CacheView], dict[str, CacheUpdate], bool]:
        policy_name = getattr(compression, "name", None)
        if compression is None or policy_name == "identity":
            return dict(raw_views), dict(raw_updates), False

        final_views: dict[str, CacheView] = {}
        final_updates: dict[str, CacheUpdate] = {}
        for name, raw_view in raw_views.items():
            update = raw_updates.get(name)
            current = state.caches.get(name)
            if update is None:
                final_views[name] = raw_view
                continue
            result = compression.apply(current, update)
            if not isinstance(result, CacheView):
                raise ContractError("CachePolicy.apply 必须返回 CacheView")
            if result.kind is not CacheKind.DECODER_KV:
                raise ContractError("HERMES 外部 policy 只能返回 decoder_kv CacheView")
            final_views[name] = result
            final_updates[name] = CacheUpdate.replace(
                result,
                owner="language_model_decoder",
                reason="external_cache_policy",
                policy=str(policy_name or type(compression).__name__),
            )
        self._install_cache_views(final_views)
        return final_views, final_updates, True

    @staticmethod
    def _normalized_keep_indices(indices: Any, length: int) -> np.ndarray:
        values = _to_numpy(indices).astype(np.int64, copy=False).reshape(-1)
        values = values[(values >= 0) & (values < length)]
        if values.size == 0:
            return np.asarray([0], dtype=np.int64)
        return np.unique(values)

    def _native_timeline_for_layer(
        self,
        raw_view: CacheView,
        keep_indices: Any,
        after_length: int,
    ) -> tuple[TokenTimeline, dict[str, Any]]:
        before_length = raw_view.sequence_length
        kept = self._normalized_keep_indices(keep_indices, before_length)
        starts = _to_numpy(raw_view.timeline.start_s)[:, kept]
        ends = _to_numpy(raw_view.timeline.end_s)[:, kept]
        summary_count = after_length - int(kept.size)
        if summary_count not in {0, 1}:
            raise ContractError(
                "HERMES 原生压缩后的 cache 长度无法由 keep indices 解释："
                f"kept={kept.size}, after={after_length}"
            )

        metadata: dict[str, Any] = {
            "native_hermes_compressed": True,
            "native_keep_tokens": int(kept.size),
            "native_summary_tokens": summary_count,
        }
        if summary_count:
            evicted_mask = np.ones(before_length, dtype=bool)
            evicted_mask[kept] = False
            evicted_starts = _to_numpy(raw_view.timeline.start_s)[:, evicted_mask]
            evicted_ends = _to_numpy(raw_view.timeline.end_s)[:, evicted_mask]
            if evicted_starts.size == 0:
                raise ContractError("HERMES 声明 summary token，但没有可聚合的淘汰 token")
            metadata.update(
                {
                    "native_summary_source_start_s": float(np.min(evicted_starts)),
                    "native_summary_source_end_s": float(np.max(evicted_ends)),
                }
            )
            # Upstream appends the summary after kept tokens.  Use a monotonic
            # proxy timestamp while retaining its original coverage in metadata.
            proxy_time = float(np.max(_to_numpy(raw_view.timeline.end_s)))
            starts = np.concatenate((starts, np.asarray([[proxy_time]])), axis=1)
            ends = np.concatenate((ends, np.asarray([[proxy_time]])), axis=1)
        return TokenTimeline(start_s=starts, end_s=ends), metadata

    def _run_native_compression(
        self,
        raw_views: Mapping[str, CacheView],
    ) -> tuple[dict[str, CacheView], dict[str, CacheUpdate], dict[str, Any]]:
        raw_summary = _cache_summary(raw_views)
        prefix_tokens = max(0, int(getattr(self.model, "visual_start_idx", 0)))
        visual_budget = int(self.kv_size or 0)
        effective_total_budget = prefix_tokens + visual_budget
        telemetry: dict[str, Any] = {
            "native_hermes_compression_enabled": self.native_compression_mode != "off",
            "native_hermes_compression_mode": self.native_compression_mode,
            "native_hermes_compression_called": False,
            "native_hermes_compression_applied": False,
            "native_hermes_compression_ms": 0.0,
            "native_hermes_visual_budget_tokens": visual_budget,
            "native_hermes_protected_prefix_tokens": prefix_tokens,
            "native_hermes_effective_total_budget_tokens": effective_total_budget,
            "native_hermes_tokens_before_min": raw_summary["tokens_min"],
            "native_hermes_tokens_before_max": raw_summary["tokens_max"],
            "native_hermes_tokens_before_total": raw_summary["tokens_total"],
        }
        should_run = (
            self.native_compression_mode != "off"
            and raw_summary["tokens_max"] > effective_total_budget
        )
        if not should_run:
            telemetry.update(
                {
                    "native_hermes_tokens_after_min": raw_summary["tokens_min"],
                    "native_hermes_tokens_after_max": raw_summary["tokens_max"],
                    "native_hermes_tokens_after_total": raw_summary["tokens_total"],
                    "native_hermes_tokens_evicted_total": 0,
                }
            )
            return dict(raw_views), {}, telemetry

        prune = getattr(self.model, "apply_kv_cache_pruning_strict", None)
        if not callable(prune):
            raise RuntimeError("启用 HERMES 原生压缩，但 model 缺少 apply_kv_cache_pruning_strict")
        captured: dict[str, Any] = {}

        def recording_prune(keep_indices_all_layers: Any) -> Any:
            captured["keep_indices"] = keep_indices_all_layers
            return prune(keep_indices_all_layers)

        self.model.apply_kv_cache_pruning_strict = recording_prune
        started = time.perf_counter()
        try:
            if self.native_compression_mode == "predict":
                native_call = getattr(self.model, "predict_and_compress", None)
                if not callable(native_call):
                    raise RuntimeError(
                        "native_compression_mode=predict，但 model 缺少 predict_and_compress"
                    )
                native_call()
            else:
                native_call = getattr(self.model, "pseudo_forward", None)
                if not callable(native_call):
                    raise RuntimeError(
                        "native_compression_mode=static_pseudo，但 model 缺少 pseudo_forward"
                    )
                native_call(self.native_local_question, self.native_global_question)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.model.apply_kv_cache_pruning_strict = prune

        keep_indices = captured.get("keep_indices")
        if keep_indices is None:
            raise RuntimeError("HERMES 原生压缩未调用 apply_kv_cache_pruning_strict")
        layers = _cache_layers(self.model.kv_cache)
        if len(layers) != len(raw_views) or len(keep_indices) != len(layers):
            raise ContractError("HERMES 原生压缩返回的层数与 decoder KV 不一致")

        final_views: dict[str, CacheView] = {}
        updates: dict[str, CacheUpdate] = {}
        for layer_index, (key, value) in enumerate(layers):
            name = _cache_name(layer_index)
            raw_view = raw_views[name]
            after_length = _shape(key)[-2]
            timeline, metadata = self._native_timeline_for_layer(
                raw_view, keep_indices[layer_index], after_length
            )
            view = self._view_for_layer(
                layer_index,
                key,
                value,
                self._position_ids_tensor(layer_index, key, require_existing=True),
                timeline,
                metadata=metadata,
            )
            final_views[name] = view
            updates[name] = CacheUpdate.replace(
                view,
                owner="language_model_decoder",
                reason="native_hermes_hierarchical_compression",
                native_mode=self.native_compression_mode,
                before_tokens=raw_view.sequence_length,
                after_tokens=view.sequence_length,
            )

        final_summary = _cache_summary(final_views)
        if final_summary["tokens_total"] >= raw_summary["tokens_total"]:
            raise RuntimeError("HERMES 原生压缩被调用，但 decoder KV token 数未减少")
        if final_summary["tokens_max"] > effective_total_budget:
            raise RuntimeError(
                "HERMES 原生压缩后超过有效总预算："
                f"after={final_summary['tokens_max']}, budget={effective_total_budget}"
            )
        telemetry.update(
            {
                "native_hermes_compression_called": True,
                "native_hermes_compression_applied": True,
                "native_hermes_compression_ms": elapsed_ms,
                "native_hermes_tokens_after_min": final_summary["tokens_min"],
                "native_hermes_tokens_after_max": final_summary["tokens_max"],
                "native_hermes_tokens_after_total": final_summary["tokens_total"],
                "native_hermes_tokens_evicted_total": (
                    raw_summary["tokens_total"] - final_summary["tokens_total"]
                ),
            }
        )
        return final_views, updates, telemetry

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: CachePolicy | None = None,
    ) -> StreamStep:
        validate_clip_for_capabilities(chunk, self.capabilities, streaming=True, train=train)
        if chunk.video_ids != (state.video_id,):
            raise ContractError("chunk.video_ids 必须与 StreamState.video_id 一致")
        external_policy_name = getattr(compression, "name", None)
        if (
            self.native_compression_mode != "off"
            and compression is not None
            and external_policy_name != "identity"
        ):
            raise ContractError(
                "HERMES 原生层次压缩与外部非 identity CachePolicy 不能在同一步叠加；"
                "请分别运行消融，避免把双重压缩冒充 HERMES"
            )
        with self._lock:
            container_type = (
                state.opaque.get("decoder_kv_container_type")
                if isinstance(state.opaque, Mapping)
                else None
            )
            self._install_cache_views(state.caches, container_type=container_type)
            self._restore_internal_state(state.opaque)
            before_summary = _cache_summary(state.caches)

            started = time.perf_counter()
            valid_frames = int(chunk.valid_lengths[0])
            projected_visual, decoder_contextual = self._capture_chunk_features(
                _as_model_chunk(chunk.frames[0, :valid_frames])
            )
            selected_features = (
                decoder_contextual
                if self.feature_stage == "decoder_contextual"
                else projected_visual
            )
            assert selected_features is not None
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raw_views, raw_updates = self._build_raw_after_views(chunk, state)
            native_views, native_updates, native_telemetry = self._run_native_compression(raw_views)
            native_applied = bool(native_telemetry["native_hermes_compression_applied"])
            if native_applied:
                final_views = native_views
                updates = native_updates
                external_policy_applied = False
                # Do not keep the pre-compression full KV tensors alive while
                # emitting artifacts for the compressed state.
                del raw_views, raw_updates
            else:
                final_views, updates, external_policy_applied = self._apply_external_policy(
                    raw_views,
                    raw_updates,
                    state,
                    compression,
                )

            output = EncoderOutput(
                features=selected_features,
                pooled=_mean_tokens(selected_features),
                timeline=_chunk_token_timeline(
                    chunk, _shape(selected_features)[1], source_frames=True
                ),
                aux={
                    "adapter": "hermes_llava_ov",
                    "feature_stage": self.feature_stage,
                    "cache_conditioned": self.feature_stage == "decoder_contextual",
                    "comparison_scope": (
                        "accuracy_and_performance"
                        if self.feature_stage == "decoder_contextual"
                        else "performance_only"
                    ),
                    "decoder_context_scope": (
                        "current_q_len_only" if self.feature_stage == "decoder_contextual" else None
                    ),
                    "projected_visual_shape": list(_shape(projected_visual)),
                    "decoder_contextual_shape": (
                        list(_shape(decoder_contextual)) if decoder_contextual is not None else None
                    ),
                    "cache_owner": "language_model_decoder",
                    "timeline_policy": "uniform_visual_token_to_frame_approximation",
                },
            )
            after_summary = _cache_summary(final_views)
            times = _to_numpy(chunk.timestamps_s)[0, :valid_frames]
            if len(times) > 1:
                next_timestamp = float(times[-1] + max(0.0, times[-1] - times[-2]))
            else:
                next_timestamp = float(times[-1] + 1.0 / self.sample_fps)
            next_state = state.replace(
                step_index=state.step_index + 1,
                caches=final_views,
                opaque=self._snapshot_internal_state(),
                next_timestamp_s=next_timestamp,
                metadata={
                    **dict(state.metadata),
                    "cache_owner": "language_model_decoder",
                    "last_policy": (
                        f"native_hermes:{self.native_compression_mode}"
                        if native_applied
                        else external_policy_name
                    ),
                    "native_hermes_compression_mode": self.native_compression_mode,
                },
            )
            telemetry: dict[str, Any] = {
                "encode_ms": elapsed_ms,
                "input_tokens": before_summary["tokens_max"] + _shape(projected_visual)[1],
                # The system prompt prepared by init_state is not a video-cache
                # hit.  Reuse begins once a prior video chunk exists.
                "reused_tokens": (before_summary["tokens_max"] if state.step_index > 0 else 0),
                "output_tokens": _shape(selected_features)[1],
                "cache_bytes": after_summary["bytes"],
                "cache_hit": state.step_index > 0 and before_summary["tokens_max"] > 0,
                "frames_encoded": valid_frames,
                "projected_visual_tokens": _shape(projected_visual)[1],
                "feature_stage": self.feature_stage,
                "feature_cache_conditioned": self.feature_stage == "decoder_contextual",
                "decoder_kv_layers": after_summary["layers"],
                "decoder_kv_tokens_before_min": before_summary["tokens_min"],
                "decoder_kv_tokens_before_max": before_summary["tokens_max"],
                "decoder_kv_tokens_after_min": after_summary["tokens_min"],
                "decoder_kv_tokens_after_max": after_summary["tokens_max"],
                "decoder_kv_tokens_after_total": after_summary["tokens_total"],
                "decoder_kv_bytes_after": after_summary["bytes"],
                "decoder_kv_replaced_by_policy": external_policy_applied,
                "decoder_kv_replaced": native_applied or external_policy_applied,
                "external_cache_policy": external_policy_name,
                "external_cache_policy_applied": external_policy_applied,
                "cache_owner": "language_model_decoder",
                "is_vision_encoder_kv": False,
                **native_telemetry,
            }
            get_memory = getattr(self.model, "get_gpu_memory_usage_gb", None)
            if callable(get_memory):
                with suppress(Exception):  # pragma: no cover - driver/GPU dependent
                    telemetry["gpu_memory_gb"] = float(get_memory())
            step = StreamStep(
                output=output,
                state=next_state,
                cache_updates=updates,
                telemetry=telemetry,
            )
            validate_stream_step(
                step,
                previous_state=state,
                chunk=chunk,
                capabilities=self.capabilities,
            )
            return step

    def finalize(self, state: StreamState) -> EncoderOutput | None:
        # Every chunk emits its classifier-facing tokens immediately; no hidden
        # buffered frames remain to flush.
        return None


__all__ = [
    "DEFAULT_CAPABILITIES",
    "HermesLlavaOVAdapter",
    "import_hermes_load_model",
]
