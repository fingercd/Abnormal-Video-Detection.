"""Model-agnostic fixed-clip and streaming feature extraction engine."""

from __future__ import annotations

import dataclasses
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactStore, CacheTelemetryRecord
from ..contracts import (
    ClipBatch,
    EncoderOutput,
    StreamState,
    validate_clip_for_capabilities,
    validate_encoder_adapter,
    validate_encoder_output,
    validate_stream_step,
)
from ..features import (
    FeatureRecord,
    FeatureStore,
    compute_encoder_fingerprint,
    ensure_json_metadata,
)


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} has object dtype")
    return array


def _manifest_metadata(manifest: Any) -> dict[str, Any]:
    """Take a small, JSON-only manifest projection for per-record diagnostics."""

    if dataclasses.is_dataclass(manifest):
        manifest = dataclasses.asdict(manifest)
    elif hasattr(manifest, "model_dump") and callable(manifest.model_dump):
        manifest = manifest.model_dump(mode="json")
    elif not isinstance(manifest, Mapping) and hasattr(manifest, "__dict__"):
        manifest = vars(manifest)
    if not isinstance(manifest, Mapping):
        return {"name": str(manifest)}
    selected: dict[str, Any] = {}
    # Keep identifying fields but never duplicate full configs in every index row.
    for key in ("name", "adapter", "family", "model_id", "revision", "source", "mode"):
        if key in manifest:
            selected[key] = manifest[key]
    try:
        return ensure_json_metadata(selected)
    except (TypeError, ValueError):
        return {}


def _sequence_metadata(value: Any, *, batch_size: int, name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"batch.metadata[{name!r}] must be a sequence of length {batch_size}")
    result = list(value)
    if len(result) != batch_size:
        raise ValueError(f"batch.metadata[{name!r}] must have length {batch_size}")
    return result


def _slice_aux_arrays(
    aux: Mapping[str, Any], row: int, batch_size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    skipped: list[str] = []
    used_names: set[str] = set()
    for raw_name, value in aux.items():
        source_name = str(raw_name)
        name = re.sub(r"[^A-Za-z0-9_]", "_", source_name).strip("_")
        if not name or not name[0].isalpha():
            name = f"value_{name}" if name else "value"
        name = name[:64]
        if name in used_names:
            skipped.append(source_name)
            continue
        used_names.add(name)
        try:
            array = _to_numpy(value, name=f"aux.{name}")
        except (TypeError, ValueError):
            try:
                metadata[name] = ensure_json_metadata(value, max_items=128)
            except (TypeError, ValueError):
                skipped.append(name)
            continue
        if array.dtype.kind not in "biufc":
            try:
                metadata[name] = ensure_json_metadata(value, max_items=128)
            except (TypeError, ValueError):
                skipped.append(source_name)
            continue
        # Scalars are metadata; arrays remain binary.  A leading B axis is split.
        if array.ndim == 0:
            metadata[name] = array.item()
        elif array.shape[0] == batch_size:
            arrays[name] = array[row]
        else:
            arrays[name] = array
    if skipped:
        metadata["skipped_non_serializable_keys"] = sorted(skipped)
    return arrays, metadata


def _valid_row(mask: Any | None, row: int, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    valid = _to_numpy(mask, name="valid_mask")[row].astype(bool, copy=False)
    if valid.shape != (length,):
        raise ValueError(f"valid mask row must have shape {(length,)}, got {valid.shape}")
    return valid


def _batch_metadata_for_row(
    metadata: Mapping[str, Any], row: int, batch_size: int
) -> dict[str, Any]:
    """Project JSON-safe dataset metadata onto one clip index row."""

    projected: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key)
        if key in {"clip_ids", "clip_indices"}:
            continue
        value = raw_value
        is_per_item_array = (
            isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size
        )
        is_per_item_sequence = (
            not isinstance(value, (str, bytes, Mapping))
            and isinstance(value, Sequence)
            and len(value) == batch_size
        )
        if is_per_item_array or is_per_item_sequence:
            value = value[row]
        try:
            projected[key] = ensure_json_metadata(value, max_items=256)
        except (TypeError, ValueError):
            # Dense or opaque dataset values must not leak into the JSON index.
            continue
    return projected


def _cache_type(state: StreamState, adapter: Any) -> str:
    kinds = {getattr(view.kind, "value", str(view.kind)) for view in state.caches.values()}
    if not kinds:
        capabilities = adapter.capabilities
        declared = {
            "kv": capabilities.supports_kv_cache,
            "token": capabilities.supports_token_cache,
            "visual_memory": getattr(capabilities, "supports_visual_memory_cache", False),
        }
        active = [name for name, enabled in declared.items() if enabled]
        if len(active) > 1:
            return "hybrid"
        if active:
            return active[0]
        return "state" if capabilities.supports_streaming else "none"
    has_kv = any("kv" in kind for kind in kinds)
    has_token = any("token" in kind for kind in kinds)
    has_visual_memory = any("visual_memory" in kind for kind in kinds)
    if sum((has_kv, has_token, has_visual_memory)) > 1:
        return "hybrid"
    if has_kv:
        return "kv"
    if has_token:
        return "token"
    if has_visual_memory:
        return "visual_memory"
    return "state"


class FeatureExtractionEngine:
    """Persist canonical :class:`EncoderOutput` objects without model imports.

    ``adapter`` is checked only against the contracts module.  ``manifest`` may
    be a mapping, dataclass, or ordinary object; it is used for a deterministic
    encoder fingerprint and is never interpreted as a concrete model config.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        manifest: Any,
        feature_store: FeatureStore,
        artifact_store: ArtifactStore | None = None,
        encoder_fingerprint: str | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_id: str | None = None,
        train: bool = False,
    ) -> None:
        self.adapter = adapter
        self.capabilities = validate_encoder_adapter(adapter)
        self.manifest = manifest
        self.feature_store = feature_store
        self.artifact_store = artifact_store
        self.train = train
        self.encoder_fingerprint = encoder_fingerprint or compute_encoder_fingerprint(
            manifest,
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
        )
        # FeatureRecord performs the strict format check; fail at construction.
        if not (re.fullmatch(r"sha256:[0-9a-f]{64}", self.encoder_fingerprint) is not None):
            raise ValueError("encoder_fingerprint must use sha256:<64 lowercase hex chars>")
        self._manifest_metadata = _manifest_metadata(manifest)
        self._next_clip_index: dict[str, int] = {}

    def _clip_identity(
        self,
        batch: ClipBatch,
        *,
        clip_ids: Sequence[str] | None,
        clip_indices: Sequence[int] | None,
    ) -> tuple[list[str], list[int]]:
        batch_size = batch.batch_size
        if clip_ids is None:
            clip_ids = _sequence_metadata(
                batch.metadata.get("clip_ids"), batch_size=batch_size, name="clip_ids"
            )
        if clip_indices is None:
            raw_indices = _sequence_metadata(
                batch.metadata.get("clip_indices"), batch_size=batch_size, name="clip_indices"
            )
            clip_indices = None if raw_indices is None else [int(item) for item in raw_indices]

        resolved_indices: list[int] = []
        for row, video_id in enumerate(batch.video_ids):
            if clip_indices is None:
                index = self._next_clip_index.get(video_id, 0)
            else:
                index = int(clip_indices[row])
            if index < 0:
                raise ValueError("clip indices must be non-negative")
            resolved_indices.append(index)
            self._next_clip_index[video_id] = max(self._next_clip_index.get(video_id, 0), index + 1)

        if clip_ids is None:
            resolved_ids = [
                f"{video_id}:clip-{index:06d}"
                for video_id, index in zip(batch.video_ids, resolved_indices, strict=True)
            ]
        else:
            if len(clip_ids) != batch_size or any(not str(item) for item in clip_ids):
                raise ValueError(f"clip_ids must contain {batch_size} non-empty values")
            resolved_ids = [str(item) for item in clip_ids]
        return resolved_ids, resolved_indices

    def _persist_output(
        self,
        *,
        batch: ClipBatch,
        output: EncoderOutput,
        clip_ids: Sequence[str] | None = None,
        clip_indices: Sequence[int] | None = None,
        mode: str,
    ) -> list[FeatureRecord]:
        validate_encoder_output(output, batch)
        resolved_ids, resolved_indices = self._clip_identity(
            batch, clip_ids=clip_ids, clip_indices=clip_indices
        )
        features = _to_numpy(output.features, name="features")
        timeline = output.timeline
        timeline_start = _to_numpy(timeline.start_s, name="timeline.start_s")
        timeline_end = _to_numpy(timeline.end_s, name="timeline.end_s")
        pooled = None if output.pooled is None else _to_numpy(output.pooled, name="pooled")
        input_times = _to_numpy(batch.timestamps_s, name="batch.timestamps_s")
        input_indices = (
            None
            if batch.frame_indices is None
            else _to_numpy(batch.frame_indices, name="batch.frame_indices")
        )
        records: list[FeatureRecord] = []

        for row, (video_id, clip_id, clip_index) in enumerate(
            zip(batch.video_ids, resolved_ids, resolved_indices, strict=True)
        ):
            token_valid = _valid_row(timeline.valid_mask, row, output.num_tokens)
            token_positions = np.flatnonzero(token_valid)
            if token_positions.size == 0:
                raise ValueError(f"encoder emitted no valid token for {video_id}/{clip_id}")
            token_stop = int(token_positions[-1]) + 1
            # Contract masks are valid prefixes, so slicing preserves alignment.
            row_features = features[row, :token_stop]
            row_timeline_start = timeline_start[row, :token_stop]
            row_timeline_end = timeline_end[row, :token_stop]
            row_timeline_valid = token_valid[:token_stop]
            row_source_start = None
            row_source_end = None
            if timeline.source_frame_start is not None:
                row_source_start = _to_numpy(
                    timeline.source_frame_start, name="timeline.source_frame_start"
                )[row, :token_stop]
                row_source_end = _to_numpy(
                    timeline.source_frame_end, name="timeline.source_frame_end"
                )[row, :token_stop]

            input_valid = _valid_row(batch.valid_mask, row, batch.num_frames)
            valid_input_positions = np.flatnonzero(input_valid)
            start_s = float(row_timeline_start[0])
            end_s = float(row_timeline_end[-1])
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                # Defensive fallback; TokenTimeline normally rejects this earlier.
                start_s = float(input_times[row, valid_input_positions[0]])
                end_s = float(input_times[row, valid_input_positions[-1]])

            if row_source_start is not None and row_source_end is not None:
                frame_start = int(row_source_start[0])
                frame_end = int(row_source_end[-1])
            elif input_indices is not None:
                frame_start = int(input_indices[row, valid_input_positions[0]])
                frame_end = int(input_indices[row, valid_input_positions[-1]]) + 1
            else:
                frame_start = None
                frame_end = None

            aux_arrays, aux_metadata = _slice_aux_arrays(output.aux, row, output.batch_size)
            record_metadata: dict[str, Any] = {
                "extraction_mode": mode,
                "input_num_frames": int(valid_input_positions.size),
                "encoder": self._manifest_metadata,
            }
            source_metadata = _batch_metadata_for_row(batch.metadata, row, batch.batch_size)
            if source_metadata:
                record_metadata["source"] = source_metadata
            if aux_metadata:
                record_metadata["encoder_aux"] = aux_metadata
            records.append(
                self.feature_store.write(
                    video_id=video_id,
                    clip_id=clip_id,
                    clip_index=clip_index,
                    encoder_fingerprint=self.encoder_fingerprint,
                    features=row_features,
                    start_s=start_s,
                    end_s=end_s,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    timeline_start_s=row_timeline_start,
                    timeline_end_s=row_timeline_end,
                    timeline_valid=row_timeline_valid,
                    source_frame_start=row_source_start,
                    source_frame_end=row_source_end,
                    pooled=None if pooled is None else pooled[row],
                    aux_arrays=aux_arrays,
                    metadata=record_metadata,
                )
            )
        return records

    def extract_batch(
        self,
        batch: ClipBatch,
        *,
        clip_ids: Sequence[str] | None = None,
        clip_indices: Sequence[int] | None = None,
    ) -> list[FeatureRecord]:
        """Encode and persist a fixed-frame batch."""

        validate_clip_for_capabilities(batch, self.capabilities, train=self.train)
        started = time.perf_counter()
        output = self.adapter.encode(batch, train=self.train)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        records = self._persist_output(
            batch=batch,
            output=output,
            clip_ids=clip_ids,
            clip_indices=clip_indices,
            mode="fixed",
        )
        if self.artifact_store is not None:
            per_clip_ms = elapsed_ms / batch.batch_size
            valid_lengths = batch.valid_lengths
            for row, record in enumerate(records):
                self.artifact_store.append_cache_telemetry(
                    CacheTelemetryRecord(
                        run_id=self.artifact_store.run_id,
                        encoder_fingerprint=self.encoder_fingerprint,
                        video_id=record.video_id,
                        clip_id=record.clip_id,
                        mode="fixed",
                        cache_type="none",
                        cache_hit=False,
                        input_tokens=int(valid_lengths[row]),
                        reused_tokens=0,
                        output_tokens=record.token_count,
                        cache_bytes=0,
                        encode_ms=per_clip_ms,
                        metadata={"batch_size": batch.batch_size},
                    )
                )
        return records

    def extract(self, batches: Iterable[ClipBatch]) -> list[FeatureRecord]:
        """Extract an iterable of fixed batches into one index."""

        records: list[FeatureRecord] = []
        for batch in batches:
            records.extend(self.extract_batch(batch))
        return records

    def _stream_telemetry(
        self,
        *,
        chunk: ClipBatch,
        clip_id: str,
        state: StreamState,
        output: EncoderOutput | None,
        telemetry: Mapping[str, Any],
        elapsed_ms: float,
    ) -> None:
        if self.artifact_store is None:
            return

        def integer(name: str, default: int) -> int:
            value = telemetry.get(name, default)
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default

        input_tokens = integer("input_tokens", chunk.num_frames)
        reused_tokens = min(input_tokens, integer("reused_tokens", 0))
        output_tokens = integer("output_tokens", 0 if output is None else output.num_tokens)
        cache_bytes = integer(
            "cache_bytes", sum(int(getattr(view, "nbytes", 0)) for view in state.caches.values())
        )
        cache_hit = bool(telemetry.get("cache_hit", reused_tokens > 0))
        reserved = {
            "input_tokens",
            "reused_tokens",
            "output_tokens",
            "cache_bytes",
            "cache_hit",
            "encode_ms",
        }
        extras = {key: value for key, value in telemetry.items() if key not in reserved}
        try:
            extras = ensure_json_metadata(extras, max_items=256)
        except (TypeError, ValueError):
            extras = {"non_serializable_telemetry_omitted": True}
        self.artifact_store.append_cache_telemetry(
            CacheTelemetryRecord(
                run_id=self.artifact_store.run_id,
                encoder_fingerprint=self.encoder_fingerprint,
                video_id=chunk.video_ids[0],
                clip_id=clip_id,
                mode="streaming",
                cache_type=_cache_type(state, self.adapter),
                cache_hit=cache_hit,
                input_tokens=input_tokens,
                reused_tokens=reused_tokens,
                output_tokens=output_tokens,
                cache_bytes=cache_bytes,
                encode_ms=float(telemetry.get("encode_ms", elapsed_ms)),
                metadata=extras,
            )
        )

    def extract_stream(
        self,
        chunks: Iterable[ClipBatch],
        *,
        video_id: str,
        compression: Any | None = None,
    ) -> list[FeatureRecord]:
        """Consume ordered B=1 chunks using an adapter's explicit stream state."""

        self.capabilities.require("supports_streaming")
        if compression is not None:
            self.capabilities.require("supports_external_cache_policy")
        state = self.adapter.init_state(video_id)
        if not isinstance(state, StreamState) or state.video_id != video_id:
            raise ValueError("adapter.init_state must return StreamState for the requested video")
        records: list[FeatureRecord] = []
        last_chunk: ClipBatch | None = None

        for chunk in chunks:
            if chunk.batch_size != 1 or chunk.video_ids != (video_id,):
                raise ValueError("stream chunks must have B=1 and match video_id")
            validate_clip_for_capabilities(
                chunk, self.capabilities, streaming=True, train=self.train
            )
            previous_state = state
            started = time.perf_counter()
            step = self.adapter.encode_step(
                chunk,
                state,
                train=self.train,
                compression=compression,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            validate_stream_step(
                step,
                previous_state=previous_state,
                chunk=chunk,
                capabilities=self.capabilities,
            )
            state = step.state
            clip_ids, clip_indices = self._clip_identity(chunk, clip_ids=None, clip_indices=None)
            # _persist_output also resolves identity, so pass the resolved values.
            if step.output is not None:
                records.extend(
                    self._persist_output(
                        batch=chunk,
                        output=step.output,
                        clip_ids=clip_ids,
                        clip_indices=clip_indices,
                        mode="streaming",
                    )
                )
            self._stream_telemetry(
                chunk=chunk,
                clip_id=clip_ids[0],
                state=state,
                output=step.output,
                telemetry=step.telemetry,
                elapsed_ms=elapsed_ms,
            )
            last_chunk = chunk

        final_output = self.adapter.finalize(state)
        if final_output is not None:
            if last_chunk is None:
                raise ValueError(
                    "adapter.finalize emitted output although no stream chunk was consumed"
                )
            # Finalize has no new input.  Timeline remains the authoritative time
            # range; a distinct clip index avoids overwriting the last step.
            final_index = self._next_clip_index.get(video_id, 0)
            records.extend(
                self._persist_output(
                    batch=last_chunk,
                    output=final_output,
                    clip_ids=[f"{video_id}:final-{final_index:06d}"],
                    clip_indices=[final_index],
                    mode="streaming-finalize",
                )
            )
        return records


__all__ = ["FeatureExtractionEngine"]
