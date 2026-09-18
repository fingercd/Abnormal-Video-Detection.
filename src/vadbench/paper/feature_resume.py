"""Strict, read-only source validation for copying complete extraction shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.features import FeatureRecord, FeatureStore, compute_encoder_fingerprint


class FeatureResumeSource:
    """Reuse only exact per-video samples under the existing cache identity.

    Source indexes and resolved bytes are retained verbatim in the new run.
    No partial index is promoted, and no source file is ever written.
    """

    def __init__(
        self,
        source: Path,
        destination: Path,
        resolved: Mapping[str, Any],
        *,
        encode_context_factory: Any = None,
    ):
        from .extraction import _semantic_runtime_identity

        self.source = source.expanduser().resolve()
        self.destination = destination
        source_bytes = (self.source / "resolved.json").read_bytes()
        self.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        self.resolved = json.loads(source_bytes)
        for key in ("data_content_evidence", "encoder_fingerprint", "paper_identity"):
            if self.resolved.get(key) != resolved[key]:
                raise ValueError(f"resume source {key} differs from the requested extraction")
        # Batch size is a scheduling choice; every other persisted sampling and
        # representation setting must agree. Native implementation hashes stay strict.
        old_spec = {k: v for k, v in self.resolved["spec"].items() if k != "micro_batch_size"}
        new_spec = {k: v for k, v in resolved["spec"].items() if k != "micro_batch_size"}
        if old_spec != new_spec:
            raise ValueError("resume source spec differs from the requested extraction")
        self.semantic_runtime = _semantic_runtime_identity(self.resolved["runtime"])
        if self.semantic_runtime != _semantic_runtime_identity(resolved["runtime"]):
            raise ValueError("resume source native runtime differs from the requested extraction")
        if (
            compute_encoder_fingerprint({"paper_pooled_cache": self.semantic_runtime})
            != resolved["encoder_fingerprint"]
        ):
            raise ValueError("resume source runtime does not reproduce its encoder fingerprint")
        self.reduction_template = None
        representation = self.resolved["spec"]["representation"]
        reducer = representation["reducer"]
        if reducer.get("name") != "identity":
            from vadbench.token_reduction.deployment import PairMergeDeployment

            if not isinstance(encode_context_factory, PairMergeDeployment):
                raise ValueError("reduced resume requires the verified native PairMergeDeployment")
            # Reused records retain their original execution batch receipt. Keep
            # this scheduling contract exact across repeated reduced resumes.
            if self.resolved["spec"]["micro_batch_size"] != resolved["spec"]["micro_batch_size"]:
                raise ValueError("reduced resume requires the original micro_batch_size")
            if dict(encode_context_factory.reducer_identity) != reducer:
                raise ValueError("resume deployment identity differs from the frozen reducer")
            context = next(iter(encode_context_factory._contexts.values()))
            bridge = context.bridge.receipt()
            native = context.layout.token_capacity
            special = int(context.layout.special_token_mask[0].sum().item())
            kept = context.indices.shape[1]
            depth = context.depth
            dim = representation["output_dim"]
            if (
                reducer.get("encoder_id") != bridge.encoder_id
                or bridge.encoder_id != self.resolved["spec"]["runtime_id"]
                or reducer.get("depth") != depth
                or reducer.get("hidden_dim") != dim
                or reducer.get("patch_keep_ratio") != 0.5
                or not 0 <= depth < bridge.block_count - 1
                or (native - special) % 2
                or kept != special + (native - special) // 2
            ):
                raise ValueError(
                    "resume deployment native geometry differs from frozen half-budget reducer"
                )
            self.reduction_template = {
                "encoder_id": bridge.encoder_id,
                "intervention_depth": depth,
                "native_input_tokens": native,
                "gathered_tokens": kept,
                "suffix_depths": list(range(depth + 1, bridge.block_count)),
                "hidden_dim": dim,
                "vjepa2_original_rope_positions": bridge.encoder_id == "vjepa2",
                "vjepa2_position_injections": bridge.block_count - depth - 1
                if bridge.encoder_id == "vjepa2"
                else 0,
                "recorded_position_masks": context.record_position_masks,
            }
        lineage = destination / "resume_lineage"
        lineage.mkdir()
        (lineage / "source_resolved.json").write_bytes(source_bytes)
        source_status = self.source / "status.json"
        if source_status.is_file():
            (lineage / "source_status.json").write_bytes(source_status.read_bytes())
        self.lineage = lineage

    def _validate_reduction(
        self, execution: Any, *, clip_index: int, clip_count: int, token_count: int
    ) -> None:
        if self.reduction_template is None:
            if execution is not None:
                raise ValueError("identity resume source must not contain a reduction execution")
            return
        template = dict(self.reduction_template)
        dim = template.pop("hidden_dim")
        depths = template.pop("suffix_depths")
        batch_size = self.resolved["spec"]["micro_batch_size"]
        actual_batch = min(batch_size, clip_count - (clip_index // batch_size) * batch_size)
        shape = [actual_batch, template["gathered_tokens"], dim]
        template["gathered_shape"] = shape
        template["suffix_shapes"] = {str(depth): shape for depth in depths}
        if (
            token_count != template["gathered_tokens"]
            or not isinstance(execution, Mapping)
            or compute_encoder_fingerprint(execution) != compute_encoder_fingerprint(template)
        ):
            raise ValueError(
                "resume source reduction execution differs from verified native depth/layout/positions"
            )

    def copy_video(
        self,
        *,
        record: Any,
        samples: Sequence[Any],
        source_video: Mapping[str, Any],
        destination: Path,
        runtime_reference: Mapping[str, str],
    ) -> int:
        """Return copied clip count; zero means the video must be extracted."""
        from .extraction import (
            _dtype_matches_precision,
            _sample_metadata,
            _semantic_runtime_identity,
            _VideoShardWriter,
        )

        shard = (self.source / "shards" / record.video_id).resolve()
        shard.relative_to(self.source)
        index_path = shard / "index.jsonl"
        if not index_path.is_file():
            return 0
        # FeatureStore creates blobs in its constructor. Requiring the original
        # directory first keeps opening this source strictly read-only.
        if not (shard / "blobs").is_dir():
            raise ValueError("resume source shard has no blob directory")
        source_store = FeatureStore(shard)
        index_bytes = index_path.read_bytes()
        lines = [line for line in index_bytes.splitlines() if line.strip()]
        rows = [FeatureRecord.from_dict(json.loads(line)) for line in lines]
        expected = {sample.clip_index: sample for sample in samples}
        seen = set()
        validated = []
        representation = self.resolved["spec"]["representation"]
        output_dim = representation["output_dim"]
        for line, row in zip(lines, rows, strict=True):
            if row.clip_index in seen or row.clip_index not in expected:
                raise ValueError("resume shard has duplicate or unexpected clip indices")
            seen.add(row.clip_index)
            sample = expected[row.clip_index]
            clip_index, start, end, sampling = _sample_metadata(sample)
            sampling["kind"] = (
                "uniform_full" if hasattr(sample, "requested_input_start") else "dense"
            )
            if (
                row.video_id != record.video_id
                or row.clip_id != f"{record.video_id}:clip-{clip_index:06d}"
                or row.encoder_fingerprint != self.resolved["encoder_fingerprint"]
                or row.metadata.get("paper_identity") != self.resolved["paper_identity"]
                or row.metadata.get("source")
                != {"split": record.split.value, "is_anomaly": record.is_anomaly}
                or row.metadata.get("source_video") != dict(source_video)
                or row.metadata.get("sampling") != sampling
                or (row.frame_start, row.frame_end) != (start, end)
                or (row.start_s, row.end_s)
                != (start / source_video["actual_fps"], end / source_video["actual_fps"])
            ):
                raise ValueError(f"resume source clip/source identity mismatch: {row.clip_id}")
            reference = row.metadata.get("runtime_reference")
            runtime = row.metadata.get("runtime", {})
            if reference is not None:
                if reference != {
                    "base": "extraction_run",
                    "path": "resolved.json",
                    "sha256": self.source_sha256,
                }:
                    raise ValueError("resume source runtime_reference checksum/path mismatch")
                if runtime != {
                    name: self.resolved["runtime"][name]
                    for name in ("runtime_id", "adapter_type", "loaded_library_versions")
                }:
                    raise ValueError("resume source compact runtime differs from resolved runtime")
            elif _semantic_runtime_identity(runtime) != self.semantic_runtime:
                raise ValueError("resume source inline runtime differs from resolved runtime")
            if row.metadata.get("storage") != {
                "feature_level": "pooled_only",
                "intermediate_tokens_stored": False,
            }:
                raise ValueError("resume source is not a pooled-only feature shard")
            resolved_runtime = self.resolved["runtime"]
            expected_readout = {
                "declared": resolved_runtime["declared_readout"],
                "adapter": {
                    name: resolved_runtime["properties"][name]
                    for name in ("pooling", "feature_stage")
                    if name in resolved_runtime["properties"]
                },
            }
            if row.metadata.get("readout") != expected_readout:
                raise ValueError("resume source readout evidence differs from resolved runtime")
            output = row.metadata.get("encoder_output", {})
            if (
                not isinstance(output.get("token_count"), int)
                or output["token_count"] <= 0
                or not _dtype_matches_precision(
                    output.get("token_dtype", ""), representation["precision"]
                )
                or not _dtype_matches_precision(
                    output.get("pooled_dtype", ""), representation["precision"]
                )
            ):
                raise ValueError("resume source token layout/dtype evidence is invalid")
            self._validate_reduction(
                row.metadata.get("reduction_execution"),
                clip_index=clip_index,
                clip_count=len(samples),
                token_count=output["token_count"],
            )
            if set(row.arrays) != {"features", "pooled"}:
                raise ValueError("resume source arrays must be exactly features and pooled")
            # load_bundle checks the file hash; check every reference too (NPZ
            # references share bytes), and validate declared and actual layouts.
            bundle = source_store.load_bundle(row)
            for name, array in bundle.items():
                ref = row.arrays[name]
                blob_digest = hashlib.sha256(
                    source_store._resolve(ref.path).read_bytes()
                ).hexdigest()
                if (
                    ref.sha256 != blob_digest
                    or tuple(array.shape) != ref.shape
                    or array.dtype.str != ref.dtype
                    or array.nbytes != ref.nbytes
                    or array.dtype.kind != "f"
                    or not np.isfinite(array).all()
                    or array.dtype.str != output.get("stored_dtype")
                ):
                    raise ValueError(
                        "resume source array checksum/shape/dtype/finite validation failed"
                    )
                precision = representation["precision"].lower().removeprefix("torch.")
                stored_precision = "float32" if precision in {"bfloat16", "bf16"} else precision
                if not _dtype_matches_precision(array.dtype.name, stored_precision):
                    raise ValueError(
                        "resume source persisted dtype differs from representation precision"
                    )
            if (
                bundle["features"].shape != (1, output_dim)
                or bundle["pooled"].shape != (output_dim,)
                or not np.array_equal(bundle["features"][0], bundle["pooled"])
            ):
                raise ValueError("resume source pooled layout is invalid")
            validated.append((line, row, bundle))
        if len(rows) != len(samples):
            return 0  # Even valid partial clips are never copied.
        snapshot = self.lineage / "indexes" / f"{record.video_id}.jsonl"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(index_bytes)
        index_digest = hashlib.sha256(index_bytes).hexdigest()
        store = _VideoShardWriter(destination)
        for line, row, bundle in validated:
            metadata = dict(row.metadata)
            metadata["runtime_reference"] = dict(runtime_reference)
            metadata["runtime"] = {
                name: self.resolved["runtime"][name]
                for name in ("runtime_id", "adapter_type", "loaded_library_versions")
            }
            metadata["reused_from"] = {
                "source_run": str(self.source),
                "resolved_sha256": self.source_sha256,
                "index_snapshot": snapshot.relative_to(self.destination).as_posix(),
                "index_sha256": index_digest,
                "record_sha256": hashlib.sha256(line).hexdigest(),
                "arrays_sha256": {name: ref.sha256 for name, ref in row.arrays.items()},
            }
            store.write(
                video_id=row.video_id,
                clip_id=row.clip_id,
                clip_index=row.clip_index,
                encoder_fingerprint=row.encoder_fingerprint,
                features=bundle["features"],
                pooled=bundle["pooled"],
                start_s=row.start_s,
                end_s=row.end_s,
                frame_start=row.frame_start,
                frame_end=row.frame_end,
                metadata=metadata,
                overwrite=False,
            )
        store.publish(len(samples))
        return len(validated)
