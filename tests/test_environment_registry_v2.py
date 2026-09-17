from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from vadbench.environment_registry import (
    EncoderEnvironmentRegistryError,
    assert_new_environment_executable,
    build_encoder_runtime_environment,
    load_encoder_candidates,
    load_encoder_environment_registry,
    resolve_encoder_runtime,
)
from vadbench.integrations.catalog import load_integration_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_all_25_candidates_are_classified_once_across_four_groups() -> None:
    registry = load_encoder_environment_registry(ROOT)
    candidates = load_encoder_candidates(ROOT)
    candidate_ids = {item["id"] for item in candidates}
    grouped_ids = {
        encoder_id for group in registry.groups.values() for encoder_id in group.encoders
    }
    assert len(registry.groups) == 4
    assert len(candidates) == 25
    assert len(grouped_ids) == 25
    assert grouped_ids == candidate_ids


def test_runtime_catalog_matches_registration_policy() -> None:
    candidates = load_encoder_candidates(ROOT)
    expected = {item["id"] for item in candidates if item["registration_state"] != "candidate_only"}
    catalog = load_integration_catalog(
        ROOT / "registry/encoder-integrations.yaml",
        project_root=ROOT,
    )
    assert len(expected) == 21
    assert set(catalog.ids) == expected
    assert {"uniformerv2", "umt", "infinipot_v", "mukv"}.isdisjoint(catalog.ids)
    for record in catalog.integrations:
        assert record.environment.profile.endswith("-v2")


def test_checkpoint_registry_matches_runtime_catalog() -> None:
    catalog = load_integration_catalog(
        ROOT / "registry/encoder-integrations.yaml",
        project_root=ROOT,
    )
    checkpoint_data = yaml.safe_load(
        (ROOT / "registry/checkpoints.yaml").read_text(encoding="utf-8")
    )
    expected = {record.checkpoint.registry_id for record in catalog.integrations}
    assert set(checkpoint_data["checkpoints"]) == expected
    assert len(expected) == 21


def test_new_environment_paths_never_overlap_protected_roots() -> None:
    registry = load_encoder_environment_registry(ROOT)
    for group in registry.groups.values():
        assert registry.root in group.prefix.parents
        for protected in registry.protected_roots:
            assert group.prefix != protected
            assert protected not in group.prefix.parents
    accepted = registry.root / "classic-video-v2" / "bin" / "python"
    assert assert_new_environment_executable(accepted, registry) == accepted.resolve()
    with pytest.raises(EncoderEnvironmentRegistryError, match="new environment root"):
        assert_new_environment_executable(ROOT / ".venv/bin/python", registry)


def test_runtime_selection_builds_the_same_overlay_environment_for_launchers() -> None:
    registry = load_encoder_environment_registry(ROOT)
    runtime = resolve_encoder_runtime("videomamba", project_root=ROOT, registry=registry)
    environment = build_encoder_runtime_environment(
        runtime,
        project_root=ROOT,
        registry=registry,
        base_environment={"VIRTUAL_ENV": "must-not-leak"},
    )
    assert runtime.group.id == "foundation-video-v2"
    assert runtime.overlay == registry.overlay_for("videomamba")
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")
    assert str(ROOT / "external-v2/videomamba/mamba") in environment["PYTHONPATH"]
    assert "VIRTUAL_ENV" not in environment
    assert environment["CONDA_PREFIX"] == str(runtime.python.parents[1])
    assert environment["LD_LIBRARY_PATH"] == str(runtime.python.parents[1] / "lib")


@pytest.mark.parametrize(
    ("schema_name", "registry_name"),
    [
        ("encoder-candidates-v1.schema.json", "encoder-candidates.yaml"),
        ("encoder-environments-v2.schema.json", "encoder-environments-v2.yaml"),
    ],
)
def test_v2_registries_match_json_schemas(schema_name: str, registry_name: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
    data = yaml.safe_load((ROOT / "registry" / registry_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(data)
