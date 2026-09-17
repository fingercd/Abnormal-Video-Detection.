from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from vadbench.config import ConfigError
from vadbench.contracts import ClipBatch
from vadbench.features import compute_encoder_fingerprint
from vadbench.orchestration import (
    compression_from_experiment,
    encoder_identity,
    load_encoder_definition,
    resolve_encoder_config,
    slice_clip_batch,
)


def _write_definition(path: Path) -> None:
    path.write_text(
        "schema_version: 1\nadapter: videomaev2\nconstructor: {}\n",
        encoding="utf-8",
    )


def test_builtin_encoder_definition_matches_registry() -> None:
    definition = load_encoder_definition("videomaev2")
    assert definition["adapter"] == "videomaev2"
    assert definition["constructor"]["model_name"] == "weights/videomaev2-base-hf"


def test_explicit_definition_path_inside_project_root_loads(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    definition_path = project_root / "inside.yaml"
    _write_definition(definition_path)

    definition = load_encoder_definition(
        "videomaev2",
        project_root=project_root,
        path="inside.yaml",
    )
    assert definition["adapter"] == "videomaev2"


def test_explicit_definition_path_cannot_traverse_outside_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.yaml"
    _write_definition(outside)

    with pytest.raises(ValueError, match="越出 project_root"):
        load_encoder_definition(
            "videomaev2",
            project_root=project_root,
            path="../outside.yaml",
        )


def test_explicit_definition_symlink_cannot_escape_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.yaml"
    _write_definition(outside)
    linked = project_root / "linked.yaml"
    try:
        linked.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows policy can disable symlinks
        pytest.skip(f"当前平台不能创建测试软链：{exc}")

    with pytest.raises(ValueError, match="越出 project_root"):
        load_encoder_definition(
            "videomaev2",
            project_root=project_root,
            path=linked,
        )


def test_slice_clip_batch_keeps_row_metadata_aligned() -> None:
    batch = ClipBatch(
        frames=np.zeros((3, 2, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32),
        video_ids=("v", "v", "v"),
        frame_indices=np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64),
        metadata={"clip_ids": ["a", "b", "c"], "sampling": "fixed"},
    )
    sliced = slice_clip_batch(batch, 1, 3)
    assert sliced.video_ids == ("v", "v")
    assert sliced.metadata["clip_ids"] == ["b", "c"]
    assert sliced.metadata["sampling"] == "fixed"


def test_native_compression_is_owned_by_adapter() -> None:
    config = {
        "streaming": {"compression": {"policy": "hermes_native"}},
    }
    assert compression_from_experiment(config) is None


def _registered_weight_project(tmp_path: Path) -> dict:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"frozen-weight")
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "checkpoints.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoints": {
                    "tiny": {
                        "adapter": "videomaev2",
                        "source": "huggingface",
                        "repo_id": "test/tiny",
                        "revision": "0" * 40,
                        "license": "mit",
                        "local_path": "weights",
                        "sha256": {"model.bin": hashlib.sha256(b"frozen-weight").hexdigest()},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "encoder.yaml").write_text(
        yaml.safe_dump(
            {
                "adapter": "videomaev2",
                "constructor": {"model_name": "weights", "use_half": False},
                "checkpoint": {"constructor_key": "model_name", "registry_id": "tiny"},
            }
        ),
        encoding="utf-8",
    )
    return {
        "encoder": {"adapter": "videomaev2", "definition": "encoder.yaml"},
        "streaming": {"enabled": False},
    }


def test_identity_uses_loaded_weights_and_effective_precision(tmp_path: Path) -> None:
    config = _registered_weight_project(tmp_path)
    original = resolve_encoder_config(config, project_root=tmp_path)
    first = compute_encoder_fingerprint(encoder_identity(original, project_root=tmp_path))
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / "model.bin").write_bytes(b"frozen-weight")
    config["encoder"]["params"] = {"model_name": str(relocated)}
    resolved = resolve_encoder_config(config, project_root=tmp_path)
    assert resolved["checkpoint"]["local_path"] == str(relocated)
    assert compute_encoder_fingerprint(encoder_identity(resolved, project_root=tmp_path)) == first
    config["encoder"]["params"]["use_half"] = True
    changed = resolve_encoder_config(config, project_root=tmp_path)
    assert compute_encoder_fingerprint(encoder_identity(changed, project_root=tmp_path)) != first
    (relocated / "model.bin").write_bytes(b"unregistered-weight")
    from vadbench.checkpoints import CheckpointError

    with pytest.raises(CheckpointError, match="权重校验失败"):
        encoder_identity(changed, project_root=tmp_path)


def test_registry_checkpoint_controls_constructor_and_rejects_ignored_fields(
    tmp_path: Path,
) -> None:
    config = _registered_weight_project(tmp_path)
    config["encoder"]["checkpoint"] = "tiny"
    resolved = resolve_encoder_config(config, project_root=tmp_path)
    assert resolved["constructor"]["model_name"] == str(tmp_path / "weights")
    config["encoder"]["precision"] = "fp16"
    with pytest.raises(ConfigError, match="encoder.params"):
        resolve_encoder_config(config, project_root=tmp_path)


def test_explicit_checkpoint_cannot_be_silently_ignored(tmp_path: Path) -> None:
    _write_definition(tmp_path / "encoder.yaml")
    config = {
        "encoder": {"adapter": "videomaev2", "definition": "encoder.yaml", "checkpoint": "missing"}
    }
    with pytest.raises(ValueError, match="constructor_key"):
        resolve_encoder_config(config, project_root=tmp_path)
