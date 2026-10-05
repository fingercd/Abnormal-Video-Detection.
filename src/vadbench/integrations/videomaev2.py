"""Optional-dependency adapter for the shared VideoMAEv2 implementation.

The VADBench integration owns preprocessing and weight loading. This adapter only
normalizes the framework boundary and observes the already-running forward pass
to expose a token sequence when the backbone makes one available.  It never
advertises streaming or cache support: attention activations inside a ViT are
not reusable KV cache state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    TokenTimeline,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
)

DEFAULT_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=True,
    supports_streaming=False,
    supports_kv_cache=False,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=False,
    supports_training=True,
    fixed_num_frames=16,
    min_frames=16,
    max_frames=16,
)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _to_numpy(value: Any) -> np.ndarray:
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unsqueeze_token_axis(value: Any) -> Any:
    if hasattr(value, "unsqueeze"):
        return value.unsqueeze(1)
    return np.expand_dims(np.asarray(value), axis=1)


def _reshape_tokens(value: Any, batch_size: int) -> Any:
    shape = _shape(value)
    if len(shape) == 3:
        return value
    if len(shape) == 2:
        return _unsqueeze_token_axis(value)
    if len(shape) > 3 and shape[0] == batch_size:
        return value.reshape(batch_size, -1, shape[-1])
    raise ValueError(f"无法把 backbone 激活 shape={shape} 规范化为 [B,S,D]")


def _find_sequence(value: Any, batch_size: int) -> Any | None:
    """Find a BxSxD activation without importing torch."""

    if value is None:
        return None
    if hasattr(value, "last_hidden_state"):
        candidate = value.last_hidden_state
        if candidate is not None:
            try:
                return _reshape_tokens(candidate, batch_size)
            except (AttributeError, TypeError, ValueError):
                pass
    if isinstance(value, dict):
        preferred = ("last_hidden_state", "hidden_states", "features", "x")
        for key in preferred:
            if key in value:
                found = _find_sequence(value[key], batch_size)
                if found is not None:
                    return found
        for candidate in value.values():
            found = _find_sequence(candidate, batch_size)
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        # Some backbones return a tuple of layer activations.  Prefer the last
        # activation because it is closest to the existing legacy pooling.
        for candidate in reversed(value):
            found = _find_sequence(candidate, batch_size)
            if found is not None:
                return found
        return None
    if not hasattr(value, "shape"):
        return None
    try:
        shape = _shape(value)
    except (TypeError, ValueError):
        return None
    if not shape or shape[0] != batch_size or len(shape) < 2:
        return None
    try:
        return _reshape_tokens(value, batch_size)
    except ValueError:
        return None


def _mean_tokens(sequence: Any) -> Any:
    if type(sequence).__module__.split(".", 1)[0] == "torch":
        return sequence.mean(dim=1)
    return np.asarray(sequence).mean(axis=1)


def _timeline_for_sequence(batch: ClipBatch, token_count: int) -> TokenTimeline:
    """Approximate patch-token provenance by a monotonic uniform frame map."""

    valid_lengths = batch.valid_lengths
    if np.any(valid_lengths != valid_lengths[0]):  # pragma: no cover - fixed caps prevent it
        raise ValueError("VideoMAEv2 batch 的有效帧数必须一致")
    valid_frames = int(valid_lengths[0])
    times = _to_numpy(batch.timestamps_s)[:, :valid_frames].astype(np.float64, copy=False)
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

    if batch.frame_indices is None:
        source_indices = np.broadcast_to(np.arange(num_frames), (batch_size, num_frames))
    else:
        source_indices = _to_numpy(batch.frame_indices)[:, :valid_frames]
    token_indices = source_indices[:, frame_slots].astype(np.int64, copy=False)
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=token_indices,
        source_frame_end=token_indices + 1,
    )


class VideoMAEv2Adapter(VideoEncoderAdapter):
    """Adapt VideoMAE v2 to the canonical BTHWC contract."""

    capabilities = DEFAULT_CAPABILITIES

    def __init__(
        self,
        *,
        model_name: str = "weights/videomaev2-base-hf",
        image_size: int = 224,
        num_frames: int = 16,
        use_half: bool = True,
        pooling: str = "auto",
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        if type(num_frames) is not int or num_frames <= 0:
            raise ValueError("num_frames 必须是正整数")
        self.capabilities = EncoderCapabilities(
            supports_fixed_clip=True,
            supports_streaming=False,
            supports_kv_cache=False,
            supports_token_cache=False,
            supports_visual_memory_cache=False,
            supports_external_cache_policy=False,
            supports_training=True,
            fixed_num_frames=num_frames,
            min_frames=num_frames,
            max_frames=num_frames,
        )
        self.num_frames = num_frames
        if encoder is None:
            try:
                from vadbench.integrations.videomaev2_encoder import (
                    VideoMAEv2Encoder,
                    VideoMAEv2EncoderConfig,
                )
            except ImportError as exc:  # pragma: no cover - depends on optional torch
                raise ImportError(
                    "VideoMAEv2 适配器需要可选依赖；请安装 `vadbench[videomaev2]`"
                ) from exc
            config = VideoMAEv2EncoderConfig(
                model_name=model_name,
                image_size=image_size,
                num_frames=num_frames,
                use_half=use_half,
                pooling=pooling,
            )
            encoder = VideoMAEv2Encoder(config, device=device)
        self.encoder = encoder

    def _observation_modules(self) -> Sequence[Any]:
        modules: list[Any] = []
        backbone = getattr(self.encoder, "backbone", None)
        if backbone is not None:
            modules.append(backbone)
        get_layers = getattr(self.encoder, "_get_encoder_layers", None)
        if callable(get_layers):
            try:
                layers = get_layers()
            except Exception:  # pragma: no cover - optional upstream internals
                layers = None
            if layers is not None and len(layers) > 0 and layers[-1] not in modules:
                modules.append(layers[-1])
        return tuple(modules)

    def _sync_encoder_device(self) -> None:
        """Keep the wrapped encoder device aligned after an outer ``.to`` call."""

        parameters = getattr(self.encoder, "parameters", None)
        if not callable(parameters):
            return
        try:
            parameter = next(parameters())
        except (StopIteration, TypeError):
            return
        device = getattr(parameter, "device", None)
        if device is None or not hasattr(self.encoder, "device_str"):
            return
        self.encoder.device_str = str(device)
        config = getattr(self.encoder, "cfg", None)
        if hasattr(self.encoder, "use_half") and config is not None:
            self.encoder.use_half = bool(
                getattr(config, "use_half", False) and str(device).startswith("cuda")
            )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        self._sync_encoder_device()
        observed: list[Any] = []
        handles: list[Any] = []

        def observe(_module: Any, _inputs: Any, output: Any) -> None:
            observed.append(output)

        for module in self._observation_modules():
            register_hook = getattr(module, "register_forward_hook", None)
            if callable(register_hook):
                handles.append(register_hook(observe))

        frames = _to_numpy(batch.frames)
        valid_lengths = batch.valid_lengths
        clips = [
            [
                np.asarray(frame, dtype=np.uint8)
                for frame in frames[index, : int(valid_lengths[index])]
            ]
            for index in range(batch.batch_size)
        ]
        try:
            pooled = self.encoder(clips)
        finally:
            for handle in handles:
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()

        if not hasattr(pooled, "shape") or len(_shape(pooled)) != 2:
            raise RuntimeError(
                "VideoMAEv2Encoder 必须返回 [B,D] pooled embedding，"
                f"实际为 {getattr(pooled, 'shape', None)!r}"
            )
        sequence = None
        sequence_source = "pooled_singleton"
        # PyTorch executes child hooks before the root hook.  Score every
        # observed candidate and retain the richest compatible token sequence;
        # otherwise a root [B,D] output could hide the last block [B,S,D].
        candidates: list[Any] = []
        for candidate in observed:
            found = _find_sequence(candidate, batch.batch_size)
            if found is not None and _shape(found)[-1] == _shape(pooled)[-1]:
                candidates.append(found)
        if candidates:
            sequence = max(candidates, key=lambda value: _shape(value)[1])
            sequence_source = "observed_backbone"
        if sequence is None:
            sequence = _unsqueeze_token_axis(pooled)

        # Preserve the legacy pooling result.  A second mean is only a defensive
        # fallback for injected test doubles whose pooled output is absent.
        if pooled is None:  # pragma: no cover - guarded above
            pooled = _mean_tokens(sequence)
        output = EncoderOutput(
            features=sequence,
            pooled=pooled,
            timeline=_timeline_for_sequence(batch, _shape(sequence)[1]),
            aux={
                "adapter": "videomaev2",
                "sequence_source": sequence_source,
                "timeline_policy": "uniform_token_to_frame_approximation",
            },
        )
        validate_encoder_output(output, batch)
        return output


__all__ = [
    "DEFAULT_CAPABILITIES",
    "VideoMAEv2Adapter",
]
