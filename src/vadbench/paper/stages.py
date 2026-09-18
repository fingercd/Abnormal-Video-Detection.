"""Read-only probe execution over existing adapters and manifest/video APIs."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.artifacts import file_identity, new_run_id, record_stage
from vadbench.features import atomic_write_json, atomic_write_jsonl
from vadbench.paper.profile import PaperProject, project_path
from vadbench.paper.resolve import digest


def clean_encoder_batch(batch: Any) -> Any:
    """Preserve frame tensors while removing labels and identifying filenames."""
    from vadbench.contracts import ClipBatch

    return ClipBatch(
        frames=batch.frames,
        timestamps_s=batch.timestamps_s,
        video_ids=tuple(f"sample-{i}" for i in range(batch.batch_size)),
        valid_mask=batch.valid_mask,
        frame_indices=batch.frame_indices,
        metadata={
            k: batch.metadata[k] for k in ("source_num_frames", "source_fps") if k in batch.metadata
        },
    )


def _array(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _atomic_write_jsonl_stream(path: Path, records: Any) -> None:
    """Atomically write a JSONL iterator without materializing every record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _probe_shard_path(destination: Path, ordinal: int, encoder_id: str, clip_id: str) -> Path:
    """Build an order-preserving filename without putting external IDs in paths."""

    token = digest({"encoder_id": encoder_id, "clip_id": clip_id}).split(":", 1)[1]
    return destination / "output_shards" / f"{ordinal:08d}-{token}.json"


def _write_probe_shard(path: Path, payload: dict[str, Any]) -> None:
    """Commit one completed clip package; one writer must never overwrite it."""

    if path.exists():
        raise FileExistsError(f"probe output shard 已存在，拒绝覆盖：{path.name}")
    atomic_write_json(path, payload)


def _iter_probe_shards(shards: Path, expected_count: int):
    paths = sorted(shards.glob("*.json"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"probe shard 数量不完整：expected={expected_count}, actual={len(paths)}；拒绝生成最终 summary"
        )
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or payload.get("schema_version") != "vadbench.probe-shard.v1":
            raise RuntimeError(f"probe shard schema 非法：{path.name}")
        yield payload


def _merge_probe_shards(destination: Path, expected_count: int) -> None:
    """Create legacy-compatible final JSONL outputs from completed clip shards."""

    shards = destination / "output_shards"

    def records(field: str):
        for payload in _iter_probe_shards(shards, expected_count):
            value = payload.get(field)
            if not isinstance(value, list):
                raise RuntimeError(f"probe shard 缺少列表字段 {field!r}")
            yield from value

    _atomic_write_jsonl_stream(destination / "architecture_receipts.jsonl", records("receipts"))
    _atomic_write_jsonl_stream(destination / "input_controls.jsonl", records("controls"))
    _atomic_write_jsonl_stream(destination / "probe_summary.jsonl", records("rows"))


def observe_clip(
    adapter: Any, encoder_id: str, batch: Any, observation: dict[str, Any]
) -> dict[str, Any]:
    """Verify native/observer/identity output equivalence on one actual clip.

    This is validation, not an efficiency benchmark. It deliberately performs
    three forwards. No full attention or feature tensor is written to disk.
    """
    import torch

    from vadbench.research.collectors import ProbeCollector, ProbeLimits
    from vadbench.token_reduction import IdentityReducer, ReductionContext
    from vadbench.token_reduction.bridges import create_observation_bridge

    bridge = create_observation_bridge(encoder_id, adapter)
    model = bridge.model
    training_modes = [(module, module.training) for module in model.modules()]
    clean = clean_encoder_batch(batch)
    if clean.frame_indices is None or clean.valid_mask is None:
        raise ValueError("probe execution requires explicit source frame indices and padding mask")
    geometry = bridge.geometry(clean.frame_indices, clean.valid_mask)
    try:
        model.eval()
        with torch.no_grad(), geometry:
            baseline = adapter.encode(clean)
        geometry_verified = geometry.receipt.get("flatten_verified") or (
            geometry.receipt.get("patch_flatten_verified")
            and geometry.receipt.get("divided_layout_verified")
        )
        if geometry.layout is None or not geometry_verified:
            raise RuntimeError("patch geometry did not execute or flatten verification failed")
        layout = geometry.layout
        if baseline.features.shape[:2] != layout.valid_mask.shape:
            raise ValueError("native output token count differs from verified patch grid")
        count = bridge.receipt().block_count
        indices = sorted({max(0, math.ceil(depth * count) - 1) for depth in observation["depths"]})
        sites = bridge.observation_sites(indices)
        requested_probes = set(
            observation.get("probes", ("P01", "P02", "P04", "P07", "P10", "P11", "P13", "P16"))
        )
        token_requested = bool(requested_probes & {"P01", "P02", "P04", "P07", "P16"})
        attention_requested = bool(requested_probes & {"P10", "P11", "P13"})
        observed_sites = {
            site: module
            for site, module in sites.items()
            if (attention_requested if ".probs." in site else token_requested)
        }
        metadata = bridge.probe_token_metadata(geometry, observed_sites)
        limits = ProbeLimits(
            max_observations=observation["max_records"],
            max_sampled_tokens=observation["max_tokens"],
            max_attention_queries=observation["max_queries"],
        )
        collector = ProbeCollector(observed_sites, metadata, limits)
        with torch.no_grad(), collector:
            observed = adapter.encode(clean)
        if collector.dropped_observations:
            raise ValueError(
                "max_records truncated observation sites; increase the explicit collection budget"
            )
        identity_handles = []
        identity_shapes = []

        def identity_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = bridge.block_output_tensor(indices[0], output)
            result = IdentityReducer().reduce(
                hidden,
                layout.to(hidden.device),
                ReductionContext(layer_depth=indices[0], budget=hidden.shape[1]),
            )
            identity_shapes.append(list(result.tokens.shape))
            return bridge.replace_block_output_tensor(indices[0], output, result.tokens)

        identity_handles.append(
            sites[f"block.{indices[0]}.output"].register_forward_hook(identity_hook)
        )
        try:
            with torch.no_grad():
                identity_output = adapter.encode(clean)
        finally:
            for handle in identity_handles:
                handle.remove()
        deltas = {}
        for name, candidate in (("observer", observed), ("identity", identity_output)):
            for field in ("features", "pooled"):
                expected = _array(getattr(baseline, field))
                actual = _array(getattr(candidate, field))
                if expected.shape != actual.shape or not np.allclose(
                    expected, actual, rtol=1e-5, atol=1e-6
                ):
                    raise RuntimeError(f"{name} changed {field} relative to native forward")
                deltas[f"{name}_{field}_max_abs"] = float(np.max(np.abs(actual - expected)))
        architecture = bridge.architecture()
        architecture.update(geometry.receipt)
        architecture.update(
            input_shape=list(batch.frames.shape),
            input_layout="BTHWC",
            precision=str(baseline.features.dtype),
            device=str(baseline.features.device),
            attention_backend=architecture.get("attention_backend") or "native-unreported",
            adapter_readout="unchanged adapter pooled output",
            output_shape=list(baseline.features.shape),
            pooled_shape=list(baseline.pooled.shape),
            identity_layer_shapes=identity_shapes,
            probe_ready=not bool(collector.missing_sites),
            missing_observation_sites=list(collector.missing_sites),
            requested_probes=sorted(requested_probes),
            observed_sites=list(observed_sites),
            reduction_ready=False,
            parity=deltas,
            parity_tolerance={"rtol": 1e-5, "atol": 1e-6},
        )
        return {"architecture": architecture, "observations": collector.observations}
    finally:
        for module, mode in training_modes:
            module.training = mode


def run_probe(
    project: PaperProject, plan: dict[str, Any], *, device: str = "cpu"
) -> dict[str, Any]:
    """Run a fixed cohort through the existing manifest and model-loading stack."""
    from vadbench.data.dense_sampling import sample_uniform_full_clips
    from vadbench.data.manifest import load_manifest_jsonl
    from vadbench.data.video import OpenCVVideoReader, _batch_from_reader, _validate_record_info
    from vadbench.orchestration import encoder_identity
    from vadbench.registry import ENCODER_REGISTRY
    from vadbench.research import CohortIndex

    resolved = plan["resolved"]
    suite = resolved["suite"]
    run_id = new_run_id("probe")
    destination = project_path(project.root, project.profile["output_root"]) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    inputs = {
        "profile": project.path,
        "cohort": resolved["cohort"],
        "manifest": resolved["manifest"],
    }
    with record_stage(
        destination, "paper_probe", config=resolved, inputs=inputs, project_root=project.root
    ) as stage:
        atomic_write_json(
            destination / "resolved.json", {**plan, "plan_only": False, "run_id": run_id}
        )
        if plan["blockers"]:
            raise ValueError("; ".join(plan["blockers"]))
        policy = "W" if project.protocol["annotation_policy"] == "weak_development" else "D"
        cohort = CohortIndex.load_jsonl(resolved["cohort"], policy=policy)
        chosen = [
            row
            for row in cohort.records
            if row.partition == suite["partition"] and row.role == suite["role"]
        ]
        video_ids = list(dict.fromkeys(row.video_id for row in chosen))
        if not video_ids or len(video_ids) > suite["max_videos"]:
            raise ValueError(
                "cohort is empty or exceeds max_videos; freeze an explicitly bounded cohort"
            )
        manifest = load_manifest_jsonl(resolved["manifest"])
        if any(
            row.split.value != "train"
            or any(annotation.span is not None for annotation in row.annotations)
            for row in manifest
        ):
            raise ValueError(
                "development manifest must contain official train records without temporal truth"
            )
        by_video = {row.video_id: row for row in manifest}
        if len(by_video) != len(manifest) or any(name not in by_video for name in video_ids):
            raise ValueError("manifest has duplicate IDs or does not cover the cohort")
        by_clip = {row.clip_id: row for row in chosen}
        for row in chosen:
            if row.weak_label != int(by_video[row.video_id].is_anomaly):
                raise ValueError("cohort weak label disagrees with training manifest")
        atomic_write_json(
            destination / "data_identity.json",
            {
                name: file_identity(
                    by_video[name].resolve_path(resolved["dataset_root"]),
                    project_root=resolved["dataset_root"],
                )
                for name in video_ids
            },
        )
        windows = suite["sampling"]["windows_per_video"]
        expected_ids = {f"{name}:segment-{i:02d}" for name in video_ids for i in range(windows)}
        if set(by_clip) != expected_ids:
            raise ValueError(
                "cohort clip IDs must exactly match the frozen uniform-window sampling"
            )
        expected_clip_count = len(resolved["encoders"]) * len(video_ids) * windows
        progress = {
            "run_id": run_id,
            "status": "running",
            "expected_clips": expected_clip_count,
            "completed_clips": 0,
            "video_count": len(video_ids),
            "completed_videos": 0,
        }
        progress_path = destination / "progress.json"
        atomic_write_json(progress_path, progress)
        status_counts: Counter[str] = Counter()
        completed_by_video: Counter[str] = Counter()
        ordinal = 0
        try:
            for name, stored_definition in resolved["encoders"].items():
                definition = deepcopy(stored_definition)
                definition["constructor"]["device"] = device
                identity = encoder_identity(definition, project_root=project.root)
                adapter = ENCODER_REGISTRY.create(name, **definition["constructor"])
                constructor = definition["constructor"]
                clip_frames = constructor.get("num_frames", constructor.get("clip_frames"))
                if clip_frames is None:
                    raise ValueError("resolved encoder must specify its validated clip frame count")
                for video_id in video_ids:
                    record = by_video[video_id]
                    with OpenCVVideoReader(record.resolve_path(resolved["dataset_root"])) as reader:
                        _validate_record_info(record, reader.info, strict_manifest_info=True)
                        samples = sample_uniform_full_clips(
                            reader.info.num_frames,
                            num_segments=windows,
                            clip_frames=clip_frames,
                            frame_stride=suite["sampling"]["frame_stride"],
                        )
                        for sample in samples:
                            clip_id = f"{video_id}:segment-{sample.clip_index:02d}"
                            clip = _batch_from_reader(
                                reader,
                                video_id,
                                [sample.clip],
                                metadata={
                                    "sampling_kind": "uniform_segment_centers_full_clip",
                                    "score_frame_start": sample.score_frame_start,
                                    "score_frame_end": sample.score_frame_end,
                                    "input_window_reused": sample.input_window_reused,
                                },
                            )
                            cohort_record = by_clip[clip_id]
                            shard = _probe_shard_path(destination, ordinal, name, clip_id)
                            status_counts.update(
                                _observe_and_append(
                                    adapter,
                                    name,
                                    clip,
                                    clip_id,
                                    cohort_record,
                                    identity,
                                    run_id,
                                    suite,
                                    shard,
                                    cohort,
                                )
                            )
                            ordinal += 1
                            progress["completed_clips"] = ordinal
                            completed_by_video[video_id] += 1
                            progress["completed_videos"] = sum(
                                count == len(resolved["encoders"]) * windows
                                for count in completed_by_video.values()
                            )
                            atomic_write_json(progress_path, progress)
                del adapter
            _merge_probe_shards(destination, expected_clip_count)
        except BaseException:
            progress["status"] = "failed"
            atomic_write_json(progress_path, progress)
            raise
        summary = {
            "run_id": run_id,
            "status": "completed",
            "output": str(destination),
            "videos": len(video_ids),
            "clips": progress["completed_clips"],
            "probe_status_counts": dict(status_counts),
            "research_conclusions": None,
            "scope": "observer/identity validation and descriptive probes",
        }
        atomic_write_json(destination / "summary.json", summary)
        progress["status"] = "completed"
        atomic_write_json(progress_path, progress)
        stage["outputs"] = summary
    return summary


def _observe_and_append(
    adapter: Any,
    name: str,
    clip: Any,
    clip_id: str,
    cohort_record: Any,
    identity: dict[str, Any],
    run_id: str,
    suite: dict[str, Any],
    shard: Path,
    cohort: Any,
) -> Counter[str]:
    from vadbench.research.controls import raw_clip_controls
    from vadbench.research.labels import join_probe_rows

    if (
        cohort_record.source_frames is not None
        and tuple(clip.frame_indices[0, clip.valid_mask[0]]) != cohort_record.source_frames
    ):
        raise ValueError("actual frames differ from the cohort source-frame lock")
    result = observe_clip(adapter, name, clip, suite["observation"])
    controls = dict(raw_clip_controls(clip, clip_ids=[clip_id])[0])
    controls.update(
        encoder_id=name,
        partition=cohort_record.partition,
        weak_label=cohort_record.weak_label,
        sampling_protocol={**suite["sampling"], "clip_frames": clip.num_frames},
        timestamp_source="source frame index divided by OpenCV-reported fps",
        video_duration_s=clip.metadata["source_num_frames"] / clip.metadata["source_fps"],
        input_sampling_id=digest(
            {
                "sampling": suite["sampling"],
                "clip_frames": clip.num_frames,
                "timestamp_policy": "frame_index_over_reported_fps",
            }
        ),
    )
    receipt = {
        "encoder_id": name,
        "video_id": cohort_record.video_id,
        "clip_id": clip_id,
        "architecture": result["architecture"],
        "verified_encoder_identity": identity,
        "python_executable": sys.executable,
        "sampling": dict(clip.metadata),
        "input_controls": controls,
    }
    rows = []
    for item in result["observations"]:
        rows.extend(
            {
                **row,
                "layer_depth_fraction": 0.0
                if row["layer_index"] is None
                else (row["layer_index"] + 1) / result["architecture"]["block_count"],
            }
            for row in item.to_rows(
                run_id=run_id,
                encoder_id=name,
                checkpoint_digest=digest(identity["checkpoint"]["sha256"]),
                clip_ids=[clip_id],
                video_ids=[cohort_record.video_id],
                partitions=[suite["partition"]],
                backend=result["architecture"]["attention_backend"],
            )
            if row["probe_id"] in suite["observation"]["probes"]
        )
    joined_rows = list(join_probe_rows(rows, cohort))
    _write_probe_shard(
        shard,
        {
            "schema_version": "vadbench.probe-shard.v1",
            "receipts": [receipt],
            "controls": [controls],
            "rows": joined_rows,
        },
    )
    return Counter(str(row["status"]) for row in joined_rows)


def verify_plan(project: PaperProject, encoder_id: str, video: str, device: str) -> dict[str, Any]:
    definition = deepcopy(project.encoder(encoder_id)["definition"])
    definition["constructor"]["device"] = device
    video_path = project_path(project.root, video, external=True)
    return {
        "stage": "observer_validation",
        "plan_only": True,
        "encoder_id": encoder_id,
        "definition": definition,
        "video": str(video_path),
        "device": device,
        "label_access": "none",
        "cost_upper_bound": {"videos": 1, "forward_windows": 3},
        "observation": {
            "depths": [0.25, 0.5, 0.75, 1.0],
            "max_tokens": 64,
            "max_queries": 16,
            "max_records": 128,
        },
        "blockers": ([] if video_path.is_file() else ["missing input video"])
        + (
            []
            if Path(definition["checkpoint"]["local_path"]).exists()
            else ["missing weights path"]
        ),
    }


def run_verification(project: PaperProject, plan: dict[str, Any]) -> dict[str, Any]:
    """Verify a real video without manufacturing research labels or cohorts."""
    import vadbench
    from vadbench.data.sampling import sample_fixed_clip
    from vadbench.data.video import build_clip_batch, probe_video
    from vadbench.orchestration import encoder_identity
    from vadbench.registry import ENCODER_REGISTRY

    run_id = new_run_id("observer-validation")
    destination = project_path(project.root, project.profile["output_root"]) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    with record_stage(
        destination,
        "observer_validation",
        config=plan,
        inputs={"profile": project.path, "video": plan["video"]},
        project_root=project.root,
    ) as stage:
        atomic_write_json(
            destination / "resolved.json", {**plan, "plan_only": False, "run_id": run_id}
        )
        if plan["blockers"]:
            raise ValueError("; ".join(plan["blockers"]))
        definition = plan["definition"]
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(plan["encoder_id"], **definition["constructor"])
        constructor = definition["constructor"]
        clip_frames = constructor.get("num_frames", constructor.get("clip_frames"))
        info = probe_video(plan["video"])
        sample = sample_fixed_clip(
            info.num_frames, clip_frames=clip_frames, frame_stride=2, position="center"
        )
        batch = build_clip_batch(plan["video"], "unlabelled-validation", [sample])
        result = observe_clip(adapter, plan["encoder_id"], batch, plan["observation"])
        receipt = result["architecture"] | {
            "run_id": run_id,
            "validation_kind": "real-video-real-weights-cpu-or-explicit-device",
            "verified_encoder_identity": identity,
            "python_executable": sys.executable,
            "vadbench_file": vadbench.__file__,
            "sampled_frame_indices": batch.frame_indices.tolist(),
            "scope": "single clip engineering validation; no normal/anomaly conclusion",
        }
        atomic_write_json(destination / "architecture_receipt.json", receipt)
        rows = [
            row
            for item in result["observations"]
            for row in item.to_rows(
                run_id=run_id,
                encoder_id=plan["encoder_id"],
                checkpoint_digest=digest(identity["checkpoint"]["sha256"]),
                clip_ids=["unlabelled-validation:0"],
                video_ids=["unlabelled-validation"],
                backend=receipt["attention_backend"],
            )
        ]
        atomic_write_jsonl(destination / "probe_summary.jsonl", rows)
        summary = {
            "run_id": run_id,
            "status": "completed",
            "output": str(destination),
            "encoder_id": plan["encoder_id"],
            "device": plan["device"],
            "parity": receipt["parity"],
            "grid": receipt["grid"],
            "probe_status_counts": dict(Counter(row["status"] for row in rows)),
            "research_conclusions": None,
            "reduction_ready": False,
        }
        atomic_write_json(destination / "summary.json", summary)
        stage["outputs"] = summary
    return summary
