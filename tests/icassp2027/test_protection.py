from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/icassp2027/check_protection.py"
SPEC = importlib.util.spec_from_file_location("check_protection", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_protection_detects_changes_and_missing_files_and_allows_only_explicit_edits(tmp_path):
    baseline = {"files": {"a.txt": hashlib.sha256(b"original").hexdigest()}}
    (tmp_path / "a.txt").write_bytes(b"original")
    assert MODULE.check(tmp_path, baseline, [])["passed"]
    (tmp_path / "a.txt").write_bytes(b"changed")
    assert MODULE.check(tmp_path, baseline, [])["changed"] == ["a.txt"]
    assert MODULE.check(tmp_path, baseline, ["a.txt"])["passed"]
    (tmp_path / "a.txt").unlink()
    assert not MODULE.check(tmp_path, baseline, ["a.txt"])["passed"]


@pytest.mark.parametrize(
    "path", ["../a", "C:/a", "/a", "src/*", "src/.env", "weights/a", "external/a", "VAD_Idea/a"]
)
def test_protection_rejects_unsafe_or_out_of_scope_paths(path):
    with pytest.raises(ValueError):
        MODULE.relative_path(path)
