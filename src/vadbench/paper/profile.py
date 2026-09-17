"""Strict, lightweight project configuration; the profile owns the active list."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.config import ConfigError, load_yaml
from vadbench.orchestration import resolve_encoder_config


def fields(value: Any, *, required: set[str], optional: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a mapping")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ConfigError(f"{context}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def project_path(root: Path, value: str, *, external: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("path must be a non-empty string")
    path = (root / value).resolve()
    if not external and not path.is_relative_to(root):
        raise ConfigError(f"project configuration path escapes project root: {value}")
    return path


@dataclass(frozen=True)
class PaperProject:
    path: Path
    root: Path
    profile: dict[str, Any]
    protocol: dict[str, Any]
    assets: dict[str, Any]
    encoders: dict[str, dict[str, Any]]

    def encoder(self, name: str) -> dict[str, Any]:
        if name not in self.encoders:
            raise ConfigError(f"encoder {name!r} is not active in this project")
        return self.encoders[name]


def load_project(path: str | Path, *, root: str | Path | None = None) -> PaperProject:
    path = Path(path).resolve()
    if root is None:
        # Canonical project layout also works outside a source checkout/wheel.
        if path.parent.parent.name != "projects":
            raise ConfigError("nonstandard profile location requires --root")
        root = path.parent.parent.parent
    root = Path(root).resolve()
    profile = load_yaml(path)
    fields(
        profile,
        required={
            "schema_version",
            "project_id",
            "mode",
            "active_encoders",
            "protocol",
            "output_root",
            "paper_root",
            "annotation_policy",
        },
        optional=set(),
        context="profile",
    )
    if (
        type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
        or profile["project_id"] != "icassp2027"
    ):
        raise ConfigError("expected ICASSP 2027 profile schema_version=1")
    if profile["mode"] != "contrast_first":
        raise ConfigError("only contrast_first mode is implemented")
    entries = profile["active_encoders"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 4:
        raise ConfigError("active_encoders must contain between one and four entries")
    protocol = load_yaml(project_path(root, profile["protocol"]))
    fields(
        protocol,
        required={
            "schema_version",
            "annotation_policy",
            "partitions",
            "split_unit",
            "explore_partition",
            "normal_reference_partition",
            "test_access",
            "quality_tolerance",
            "primary_metrics",
        },
        optional=set(),
        context="protocol",
    )
    if protocol["schema_version"] != 1:
        raise ConfigError("unsupported protocol schema")
    if protocol["annotation_policy"] not in {"weak_development", "diagnostic_development"}:
        raise ConfigError("annotation_policy must declare W or D development")
    if protocol["annotation_policy"] != profile["annotation_policy"]:
        raise ConfigError("profile and protocol annotation policies disagree")
    if protocol["partitions"] != ["fit", "confirm", "select", "test"]:
        raise ConfigError("protocol must declare fit/confirm/select/test")
    if (
        protocol["split_unit"] != "video_source"
        or protocol["explore_partition"] != "fit"
        or protocol["normal_reference_partition"] != "fit"
        or protocol["test_access"] != "after_method_freeze"
    ):
        raise ConfigError("invalid development split or test access policy")
    fields(
        protocol["primary_metrics"],
        required={"ucf_crime", "xd_violence"},
        optional=set(),
        context="primary_metrics",
    )
    if protocol["primary_metrics"] != {"ucf_crime": "frame_roc_auc", "xd_violence": "frame_ap"}:
        raise ConfigError("dataset-specific primary metrics cannot be interchanged")
    tolerance = protocol["quality_tolerance"]
    if tolerance is not None and (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0 <= tolerance <= 1
    ):
        raise ConfigError("quality_tolerance must be null or a fraction in [0, 1]")
    for name in ("output_root", "paper_root"):
        project_path(root, profile[name])
    assets_path = path.with_name("assets.local.yaml")
    assets = load_yaml(assets_path) if assets_path.is_file() else {}
    fields(assets, required=set(), optional={"dataset_roots", "checkpoint_paths"}, context="assets")
    for section in assets.values():
        if not isinstance(section, dict) or any(
            not isinstance(v, str) or not v for v in section.values()
        ):
            raise ConfigError("assets mappings contain only non-empty local paths")
    encoders = {}
    for entry in entries:
        fields(
            entry,
            required={"id", "definition", "checkpoint", "role"},
            optional=set(),
            context="encoder",
        )
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise ConfigError("encoder entry fields must be non-empty strings")
        name = entry["id"]
        if not isinstance(name, str) or name in encoders:
            raise ConfigError(f"invalid or duplicate active encoder: {name!r}")
        definition = resolve_encoder_config(
            {
                "encoder": {
                    "adapter": name,
                    "definition": entry["definition"],
                    "checkpoint": entry["checkpoint"],
                    "trainable": False,
                }
            },
            project_root=root,
        )
        if not definition.get("capabilities", {}).get("supports_fixed_clip"):
            raise ConfigError(f"active encoder must support fixed clips: {name}")
        checkpoint_path = assets.get("checkpoint_paths", {}).get(name)
        if checkpoint_path is not None:
            actual = str(project_path(root, checkpoint_path, external=True))
            definition["constructor"][definition["checkpoint"]["constructor_key"]] = actual
            definition["checkpoint"]["local_path"] = actual
        encoders[name] = {"entry": entry, "definition": definition}
    if set(assets.get("checkpoint_paths", {})) - encoders.keys():
        raise ConfigError("assets.checkpoint_paths contains an inactive encoder")
    if set(assets.get("dataset_roots", {})) - {"ucf_crime", "xd_violence"}:
        raise ConfigError("unsupported dataset in assets.dataset_roots")
    return PaperProject(path, root, profile, protocol, assets, encoders)
