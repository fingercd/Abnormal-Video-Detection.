"""Tests for the runtime checkpoint provenance registry.

The registry is intentionally richer than :class:`CheckpointSpec`: the
runtime loader consumes the stable core fields while this test protects the
human/audit-facing URL, variant, size, and checksum metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from vadbench.checkpoints import load_checkpoint_registry, sha256_file
from vadbench.integrations.catalog import load_integration_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "registry" / "checkpoints.yaml"
CATALOG_PATH = PROJECT_ROOT / "registry" / "encoder-integrations.yaml"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _raw_registry() -> dict[str, Any]:
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    checkpoints = data.get("checkpoints")
    assert isinstance(checkpoints, dict)
    return data


def test_registry_covers_exactly_the_runtime_catalog_checkpoint_ids() -> None:
    raw = _raw_registry()
    entries = raw["checkpoints"]
    catalog = load_integration_catalog(CATALOG_PATH, project_root=PROJECT_ROOT)
    specs = load_checkpoint_registry(REGISTRY_PATH)

    expected_ids = {record.checkpoint.registry_id for record in catalog.integrations}
    assert len(catalog) == 21
    assert len(entries) == 21
    assert set(entries) == expected_ids
    assert set(specs) == expected_ids
    assert len({str(value.get("adapter")) for value in entries.values()}) == 21


def test_every_entry_has_traceable_asset_metadata() -> None:
    entries = _raw_registry()["checkpoints"]
    for checkpoint_id, entry in entries.items():
        assert isinstance(entry, dict), checkpoint_id
        for key in ("adapter", "source", "repo_id", "repo_url", "revision", "variant", "license"):
            assert isinstance(entry.get(key), str) and entry[key].strip(), (checkpoint_id, key)
        assert entry["repo_url"].startswith("https://"), checkpoint_id
        assert isinstance(entry.get("allow_patterns"), list), checkpoint_id
        assert isinstance(entry.get("sha256"), dict), checkpoint_id
        size = entry.get("size_bytes")
        assert size is None or (isinstance(size, int) and size > 0), checkpoint_id

        for relative, digest in entry["sha256"].items():
            assert isinstance(relative, str) and relative, checkpoint_id
            assert isinstance(digest, str) and SHA256_RE.fullmatch(digest.lower()), (
                checkpoint_id,
                relative,
            )

        files = entry.get("files")
        assert isinstance(files, list) and files, checkpoint_id
        for file_record in files:
            assert isinstance(file_record, dict), checkpoint_id
            assert "path" in file_record and "size_bytes" in file_record and "sha256" in file_record
            file_size = file_record["size_bytes"]
            assert file_size is None or (isinstance(file_size, int) and file_size > 0), (
                checkpoint_id
            )
            file_digest = file_record["sha256"]
            assert file_digest is None or (
                isinstance(file_digest, str) and SHA256_RE.fullmatch(file_digest.lower())
            ), checkpoint_id


@pytest.mark.parametrize(
    ("checkpoint_id", "relative_root"),
    [
        ("r2plus1d_18-default", "weights/r2plus1d_18"),
        ("mvitv2-default", "weights/mvitv2"),
        ("video_swin-default", "weights/video_swin"),
        ("i3d-default", "weights/i3d"),
        ("x3d-default", "weights/x3d"),
        ("slowfast-default", "weights/slowfast"),
        ("videomaev2-base-hf", "weights/videomaev2-base-hf"),
        ("hermes-llava-ov-0.5b", "weights/hermes-llava-ov-0.5b"),
    ],
)
def test_server_verified_assets_match_registered_sha256(
    checkpoint_id: str, relative_root: str
) -> None:
    """Verify files when running on the server; skip clean checkouts without assets."""

    specs = load_checkpoint_registry(REGISTRY_PATH)
    raw_entry = _raw_registry()["checkpoints"][checkpoint_id]
    spec = specs[checkpoint_id]
    root = PROJECT_ROOT / relative_root
    if not root.is_dir():
        pytest.skip(f"server-only weight directory is absent: {root}")

    expected = dict(spec.sha256)
    assert expected, checkpoint_id
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            pytest.skip(f"server-only weight file is absent: {path}")
        actual = sha256_file(path)
        assert actual == digest, (checkpoint_id, relative)
        file_size = path.stat().st_size
        assert file_size > 0
        assert file_size == raw_entry["size_bytes"]


def test_planned_entries_require_native_checkpoint_without_alias() -> None:
    entries = _raw_registry()["checkpoints"]
    planned = [entry for entry in entries.values() if entry.get("status") == "planned"]
    assert len(planned) == 5
    for entry in planned:
        assert entry.get("validation_scope") is None
        assert entry.get("local_path") != "weights/r2plus1d_18/model.pth"
        notes = str(entry.get("notes", "")).lower()
        assert "native checkpoint" in notes
        assert "no substitute" in notes
