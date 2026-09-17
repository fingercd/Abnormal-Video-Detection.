"""Resolve a bounded probe plan without model imports, weight reads or output writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vadbench.config import ConfigError, load_yaml
from vadbench.hashing import sha256_file
from vadbench.paper.profile import PaperProject, fields, project_path


def digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def status(project: PaperProject) -> dict[str, Any]:
    return {
        "project_id": project.profile["project_id"],
        "annotation_policy": project.protocol["annotation_policy"],
        "quality_tolerance": project.protocol["quality_tolerance"],
        "encoders": [
            {
                "id": name,
                "role": item["entry"]["role"],
                "checkpoint": item["definition"]["checkpoint"]["id"],
                "revision": item["definition"]["checkpoint"]["revision"],
                "weights_path_exists": Path(
                    item["definition"]["checkpoint"]["local_path"]
                ).exists(),
                "runtime_validation": "not_checked_by_status",
                "probe_ready": None,
                "reduction_ready": False,
            }
            for name, item in project.encoders.items()
        ],
        "note": "Configuration inspection only; weight existence is not runtime verification.",
    }


def resolve_probe(project: PaperProject, suite_path: str | Path) -> dict[str, Any]:
    suite_path = project_path(project.root, str(suite_path))
    suite = load_yaml(suite_path)
    fields(
        suite,
        required={
            "schema_version",
            "stage",
            "encoder_ids",
            "method",
            "cohort",
            "manifest",
            "dataset",
            "partition",
            "role",
            "max_videos",
            "sampling",
            "observation",
        },
        optional=set(),
        context="probe suite",
    )
    if suite["schema_version"] != 1 or suite["stage"] != "probe":
        raise ConfigError("only probe suite schema_version=1 is supported")
    if suite["method"] != "identity":
        raise ConfigError(f"unimplemented method: {suite['method']!r}")
    if suite["partition"] not in {"fit", "confirm", "select"}:
        raise ConfigError("development probes cannot access official test annotations")
    if suite["role"] not in {"debug", "explore", "confirm", "select"}:
        raise ConfigError("unknown cohort role")
    expected_partition = {
        "debug": "fit",
        "explore": "fit",
        "confirm": "confirm",
        "select": "select",
    }
    if suite["partition"] != expected_partition[suite["role"]]:
        raise ConfigError("cohort role and partition disagree")
    if suite["dataset"] not in project.protocol["primary_metrics"]:
        raise ConfigError("unsupported dataset")
    ids = suite["encoder_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(x, str) for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ConfigError("encoder_ids must be a non-empty unique list")
    resolved_encoders = {name: project.encoder(name)["definition"] for name in ids}
    sampling = suite["sampling"]
    fields(
        sampling,
        required={"kind", "windows_per_video", "frame_stride", "position"},
        optional=set(),
        context="sampling",
    )
    for key, value in {
        "max_videos": suite["max_videos"],
        **{k: sampling[k] for k in ("windows_per_video", "frame_stride")},
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer")
    if sampling["position"] != "center" or sampling["kind"] != "uniform_segment_centers_full_clip":
        raise ConfigError("pilot sampling uses deterministic center windows")
    observation = suite["observation"]
    fields(
        observation,
        required={"depths", "max_tokens", "max_queries", "max_records", "probes"},
        optional=set(),
        context="observation",
    )
    depths = observation["depths"]
    if (
        not isinstance(depths, list)
        or not depths
        or any(
            isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 < d <= 1 for d in depths
        )
    ):
        raise ConfigError("observation depths must be fractions in (0, 1]")
    for key in ("max_tokens", "max_queries", "max_records"):
        if (
            isinstance(observation[key], bool)
            or not isinstance(observation[key], int)
            or observation[key] <= 0
        ):
            raise ConfigError(f"{key} must be a positive integer")
    probes = observation["probes"]
    if (
        not isinstance(probes, list)
        or not probes
        or any(not isinstance(p, str) for p in probes)
        or set(probes) - {"P01", "P02", "P04", "P07", "P10", "P11", "P13", "P16"}
    ):
        raise ConfigError("unsupported level-one probe selection")
    cohort = project_path(project.root, suite["cohort"])
    manifest = project_path(project.root, suite["manifest"])
    dataset_root = project_path(
        project.root,
        project.assets.get("dataset_roots", {}).get(
            suite["dataset"], f"data/raw/{suite['dataset']}"
        ),
        external=True,
    )
    blockers = []
    if not cohort.is_file():
        blockers.append(f"missing cohort: {cohort}")
    if not manifest.is_file():
        blockers.append(f"missing manifest: {manifest}")
    if not dataset_root.is_dir():
        blockers.append(f"missing raw video root: {dataset_root}")
    for name, definition in resolved_encoders.items():
        if not Path(definition["checkpoint"]["local_path"]).exists():
            blockers.append(f"missing weights path: {name}")
    resolved = {
        "project": project.profile,
        "protocol": project.protocol,
        "suite": suite,
        "encoders": resolved_encoders,
        "dataset_root": str(dataset_root),
        "cohort": str(cohort),
        "manifest": str(manifest),
    }
    return {
        "stage": "probe",
        "plan_only": True,
        "resolved": resolved,
        "config_digest": digest(resolved),
        "inputs": {
            "profile_sha256": sha256_file(project.path),
            "suite_sha256": sha256_file(suite_path),
            "cohort_sha256": sha256_file(cohort) if cohort.is_file() else None,
            "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
        },
        "label_access": {
            "encoder": "none",
            "collector": "none",
            "analysis": "video_level",
            "official_test": "denied",
        },
        "cost_upper_bound": {
            "videos": suite["max_videos"],
            "clip_windows": suite["max_videos"] * sampling["windows_per_video"] * len(ids),
            "forward_windows": suite["max_videos"] * sampling["windows_per_video"] * len(ids) * 3,
        },
        "output_template": str(
            project_path(project.root, project.profile["output_root"]) / "<unique-run-id>"
        ),
        "blockers": blockers,
        "validation": "Paths/configuration only; actual weights, architecture and data verified at execution.",
    }
