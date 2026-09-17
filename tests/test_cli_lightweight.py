from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
# Result validation must stay lazy: its rpds extension conflicts with OpenCV
# during model import in the pinned foundation environment.
HEAVY_ROOTS = ("torch", "transformers", "torchvision", "pytorchvideo", "jsonschema", "rpds")
MARKER = "__VADBENCH_LOADED_MODULES__="


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SOURCE_ROOT) if not existing else os.pathsep.join((str(SOURCE_ROOT), existing))
    )
    return env


def _run_probe(source: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    marker_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(MARKER)),
        None,
    )
    assert marker_line is not None, result.stdout + result.stderr
    return result, json.loads(marker_line.removeprefix(MARKER))


def test_importing_cli_does_not_import_heavy_model_stacks() -> None:
    result, loaded = _run_probe(
        f"""
        import json
        import sys

        import vadbench.cli

        roots = {HEAVY_ROOTS!r}
        loaded = sorted(
            name
            for name in sys.modules
            if any(name == root or name.startswith(root + ".") for root in roots)
        )
        print({MARKER!r} + json.dumps(loaded))
        """
    )

    assert result.returncode == 0, result.stderr
    assert loaded == []


def test_encoder_list_command_does_not_import_heavy_model_stacks() -> None:
    result, loaded = _run_probe(
        f"""
        import json
        import sys

        from vadbench.cli import main

        returncode = main(["encoders", "list"])
        roots = {HEAVY_ROOTS!r}
        loaded = sorted(
            name
            for name in sys.modules
            if any(name == root or name.startswith(root + ".") for root in roots)
        )
        print({MARKER!r} + json.dumps(loaded))
        raise SystemExit(returncode)
        """
    )

    assert result.returncode == 0, result.stderr
    assert loaded == []
