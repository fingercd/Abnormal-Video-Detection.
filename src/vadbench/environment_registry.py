"""Versioned encoder environment and candidate registry validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.config import load_yaml


class EncoderEnvironmentRegistryError(ValueError):
    """Raised when the v2 environment or candidate registry is inconsistent."""


@dataclass(frozen=True, slots=True)
class EncoderEnvironmentGroup:
    id: str
    prefix: Path
    seed: Path
    python_version: str
    packages: dict[str, str]
    encoders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EncoderEnvironmentRegistry:
    root: Path
    overlay_root: Path
    cache_root: Path
    new_weight_root: Path
    new_external_root: Path
    minimum_free_bytes: int
    protected_roots: tuple[Path, ...]
    groups: dict[str, EncoderEnvironmentGroup]
    overlays: dict[str, dict[str, Any]]

    def group_for(self, encoder_id: str) -> EncoderEnvironmentGroup:
        matches = [group for group in self.groups.values() if encoder_id in group.encoders]
        if len(matches) != 1:
            raise EncoderEnvironmentRegistryError(
                f"encoder {encoder_id!r} must belong to exactly one environment group"
            )
        return matches[0]

    def python_for(self, encoder_id: str) -> Path:
        return self.group_for(encoder_id).prefix / "bin" / "python"

    def overlay_for(self, encoder_id: str) -> Path | None:
        record = self.overlays.get(encoder_id)
        if record is None:
            return None
        return _inside(self.root.parent.parent, record["path"], f"overlay {encoder_id}")


@dataclass(frozen=True, slots=True)
class EncoderRuntime:
    """One encoder's declared isolated Python and import environment.

    This is deliberately only environment selection. Callers decide whether a
    missing interpreter is a blocked matrix item or a fatal benchmark error.
    """

    encoder_id: str
    group: EncoderEnvironmentGroup
    python: Path
    overlay: Path | None


def _inside(project_root: Path, value: str, label: str) -> Path:
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        selected = project_root / selected
    selected = selected.resolve()
    if selected != project_root and project_root not in selected.parents:
        raise EncoderEnvironmentRegistryError(f"{label} leaves project root: {selected}")
    return selected


def load_encoder_environment_registry(
    project_root: str | Path = ".",
    path: str | Path = "registry/encoder-environments-v2.yaml",
) -> EncoderEnvironmentRegistry:
    root = Path(project_root).resolve()
    selected = Path(path)
    if not selected.is_absolute():
        selected = root / selected
    data = load_yaml(selected)
    if data.get("schema_version") != 2:
        raise EncoderEnvironmentRegistryError("environment registry schema_version must be 2")
    environment_root = _inside(root, str(data["environment_root"]), "environment_root")
    overlay_root = _inside(root, str(data["overlay_root"]), "overlay_root")
    cache_root = _inside(root, str(data["cache_root"]), "cache_root")
    new_weight_root = _inside(root, str(data["new_weight_root"]), "new_weight_root")
    new_external_root = _inside(root, str(data["new_external_root"]), "new_external_root")
    groups_data = data.get("groups")
    if not isinstance(groups_data, dict) or len(groups_data) != 4:
        raise EncoderEnvironmentRegistryError("exactly four environment groups are required")
    protected = []
    for value in data.get("protected_roots", []):
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        protected.append(candidate.resolve())
    groups = {}
    seen = set()
    for group_id, raw in groups_data.items():
        prefix = _inside(root, str(raw["prefix"]), f"group {group_id}")
        seed = Path(str(raw["seed"])).expanduser().resolve()
        encoders = tuple(str(value) for value in raw.get("encoders", []))
        overlap = seen.intersection(encoders)
        if overlap:
            raise EncoderEnvironmentRegistryError(
                f"duplicate encoder assignments: {sorted(overlap)}"
            )
        seen.update(encoders)
        if any(prefix == item or item in prefix.parents for item in protected):
            raise EncoderEnvironmentRegistryError(f"new prefix overlaps protected root: {prefix}")
        groups[group_id] = EncoderEnvironmentGroup(
            id=group_id,
            prefix=prefix,
            seed=seed,
            python_version=str(raw["python_version"]),
            packages={str(k): str(v) for k, v in dict(raw.get("packages", {})).items()},
            encoders=encoders,
        )
    overlays = dict(data.get("overlays", {}))
    for encoder_id, raw in overlays.items():
        if encoder_id not in seen:
            raise EncoderEnvironmentRegistryError(f"overlay target is not classified: {encoder_id}")
        if (
            raw.get("group")
            != groups[next(key for key, value in groups.items() if encoder_id in value.encoders)].id
        ):
            raise EncoderEnvironmentRegistryError(f"overlay group mismatch: {encoder_id}")
        _inside(root, str(raw["path"]), f"overlay {encoder_id}")
    return EncoderEnvironmentRegistry(
        root=environment_root,
        overlay_root=overlay_root,
        cache_root=cache_root,
        new_weight_root=new_weight_root,
        new_external_root=new_external_root,
        minimum_free_bytes=int(data["minimum_free_bytes"]),
        protected_roots=tuple(protected),
        groups=groups,
        overlays=overlays,
    )


def load_encoder_candidates(
    project_root: str | Path = ".",
    path: str | Path = "registry/encoder-candidates.yaml",
) -> tuple[dict[str, Any], ...]:
    root = Path(project_root).resolve()
    selected = Path(path)
    if not selected.is_absolute():
        selected = root / selected
    data = load_yaml(selected)
    if data.get("schema_version") != 1:
        raise EncoderEnvironmentRegistryError("candidate registry schema_version must be 1")
    values = data.get("candidates")
    if not isinstance(values, list) or len(values) != int(data.get("target_count", -1)):
        raise EncoderEnvironmentRegistryError("candidate registry target_count mismatch")
    ids = [str(value.get("id")) for value in values]
    if len(ids) != len(set(ids)):
        raise EncoderEnvironmentRegistryError("candidate ids must be unique")
    states = {"registered_local", "awaiting_manual_asset", "candidate_only"}
    if any(value.get("registration_state") not in states for value in values):
        raise EncoderEnvironmentRegistryError("invalid candidate registration_state")
    return tuple(dict(value) for value in values)


def assert_new_environment_executable(
    executable: str | Path,
    registry: EncoderEnvironmentRegistry,
) -> Path:
    selected = Path(executable).expanduser().resolve()
    if selected != registry.root and registry.root not in selected.parents:
        raise EncoderEnvironmentRegistryError(
            f"python executable must be under new environment root: {selected}"
        )
    for protected in registry.protected_roots:
        if selected == protected or protected in selected.parents:
            raise EncoderEnvironmentRegistryError(
                f"python executable points into protected environment: {selected}"
            )
    return selected


def resolve_encoder_runtime(
    encoder_id: str,
    *,
    project_root: str | Path = ".",
    registry: EncoderEnvironmentRegistry | None = None,
) -> EncoderRuntime:
    """Select the declared v2 Python and optional overlay for one encoder."""

    selected_registry = (
        load_encoder_environment_registry(project_root) if registry is None else registry
    )
    group = selected_registry.group_for(encoder_id)
    python = assert_new_environment_executable(group.prefix / "bin" / "python", selected_registry)
    return EncoderRuntime(
        encoder_id=encoder_id,
        group=group,
        python=python,
        overlay=selected_registry.overlay_for(encoder_id),
    )


def build_encoder_runtime_environment(
    runtime: EncoderRuntime,
    *,
    project_root: str | Path = ".",
    registry: EncoderEnvironmentRegistry | None = None,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the reproducible process environment shared by v2 launchers.

    The selected environment is isolated from an activating shell's virtualenv
    and imports only the repository source plus its declared overlay.
    """

    root = Path(project_root).resolve()
    selected_registry = load_encoder_environment_registry(root) if registry is None else registry
    expected = resolve_encoder_runtime(
        runtime.encoder_id, project_root=root, registry=selected_registry
    )
    if expected != runtime:
        raise EncoderEnvironmentRegistryError(
            f"runtime does not match registry selection for {runtime.encoder_id!r}"
        )

    environment = dict(os.environ if base_environment is None else base_environment)
    import_paths = [str(root / "src")]
    if runtime.overlay is not None:
        import_paths.append(str(runtime.overlay))
    if runtime.encoder_id == "videomamba":
        import_paths.extend(
            [
                str(root / "external-v2" / "videomamba" / "mamba"),
                str(root / "external-v2" / "videomamba" / "causal-conv1d"),
            ]
        )
    cache_root = selected_registry.cache_root
    cache_paths = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "PIP_CACHE_DIR": cache_root / "pip",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache" / runtime.encoder_id,
        "TMPDIR": cache_root / "tmp" / runtime.encoder_id,
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment.update({key: str(value) for key, value in cache_paths.items()})
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(import_paths),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CONDA_PREFIX": str(runtime.python.parents[1]),
            "LD_LIBRARY_PATH": str(runtime.python.parents[1] / "lib"),
        }
    )
    environment.pop("VIRTUAL_ENV", None)
    return environment
