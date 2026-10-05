"""Regression gates for portable workflows and historical import identities."""
from __future__ import annotations

import ast
import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "compatibility": "vadbench.data.feature_contracts",
    "detection": "vadbench.workflows.detection",
    "extraction": "vadbench.workflows.extraction",
    "feature_resume": "vadbench.workflows.feature_resume",
    "feature_merge": "vadbench.workflows.feature_merge",
}


@pytest.mark.parametrize("legacy,target", MODULES.items())
def test_legacy_import_is_the_same_module(legacy: str, target: str, monkeypatch) -> None:
    old = importlib.import_module(f"vadbench.paper.{legacy}")
    current = importlib.import_module(target)
    assert old is current
    # Old consumers that patch a module must still affect the implementation.
    monkeypatch.setattr(old, "_compatibility_probe", object(), raising=False)
    assert current._compatibility_probe is old._compatibility_probe


def test_old_serialized_permit_class_resolves_to_the_core_type() -> None:
    from vadbench.engine.compatibility import PredictionCompatibilityPermit

    assert pickle.loads(
        b"cvadbench.paper.detection\nPredictionCompatibilityPermit\n."
    ) is PredictionCompatibilityPermit


def test_extraction_uses_the_shared_deployment_contracts() -> None:
    from vadbench.token_reduction.deployment_contracts import (
        ReductionDeployment,
        ReductionExecutionContext,
    )
    from vadbench.workflows.extraction import EncodeContextFactory, ExtractionEncodeContext

    assert EncodeContextFactory is ReductionDeployment
    assert ExtractionEncodeContext is ReductionExecutionContext


def test_runtime_workflows_do_not_import_paper_in_a_fresh_process() -> None:
    names = list(MODULES.values()) + [
        "vadbench.engine.compatibility",
        "vadbench.workflows.quality_comparison",
        "vadbench.workflows.urdmu_quality_export",
        "vadbench.workflows.video_efficiency",
        "vadbench.integrations.detectors.urdmu.backend",
        "vadbench.integrations.detectors.urdmu.inference",
        "vadbench.workflows.dsanet._quality_runtime",
    ]
    code = (
        "import importlib, sys\n"
        f"for name in {names!r}: importlib.import_module(name)\n"
        "loaded = [n for n in sys.modules if n == 'vadbench.paper' or n.startswith('vadbench.paper.')]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_and_reusable_workflows_have_no_paper_imports() -> None:
    problems = []
    for directory in ("data", "engine", "integrations", "models", "research", "token_reduction", "workflows"):
        for path in (ROOT / "src/vadbench" / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                    if node.module == "vadbench":
                        names.extend(f"vadbench.{item.name}" for item in node.names)
                    if node.level and (node.module or "").split(".")[0] == "paper":
                        names.append("vadbench.paper")
                if any(name == "vadbench.paper" or name.startswith("vadbench.paper.") for name in names):
                    problems.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not problems, "Implementation imports project-specific paper code: " + ", ".join(problems)
