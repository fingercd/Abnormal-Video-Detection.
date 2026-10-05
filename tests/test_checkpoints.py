from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vadbench.checkpoints import (
    CheckpointError,
    CheckpointSpec,
    load_checkpoint_registry,
    verify_checkpoint,
)


def test_registry_has_pinned_reference_models() -> None:
    specs = load_checkpoint_registry("registry/checkpoints.yaml")
    assert set(specs) >= {"videomaev2-base-hf", "hermes-llava-ov-0.5b"}
    assert len(specs["videomaev2-base-hf"].revision) == 40
    assert specs["hermes-llava-ov-0.5b"].license == "apache-2.0"


def test_verify_checkpoint_detects_digest_mismatch(tmp_path: Path) -> None:
    payload = b"vadbench"
    (tmp_path / "model.bin").write_bytes(payload)
    spec = CheckpointSpec(
        id="tiny",
        adapter="fake",
        source="huggingface",
        repo_id="example/tiny",
        revision="0" * 40,
        license="mit",
        allow_patterns=("model.bin",),
        sha256={"model.bin": hashlib.sha256(payload).hexdigest()},
    )
    assert verify_checkpoint(spec, tmp_path)["model.bin"] == hashlib.sha256(payload).hexdigest()
    broken = CheckpointSpec(**{**spec.__dict__, "sha256": {"model.bin": "0" * 64}})
    with pytest.raises(CheckpointError, match="权重校验失败"):
        verify_checkpoint(broken, tmp_path)
