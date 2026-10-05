"""Run DSANet extract/score in the original Python 3.9 CLIP environment.

This loads the workflow under a private package name so importing the CLI does
not execute vadbench's top-level encoder registration, which requires Python
3.10+. The stage parser and runtime modules are the same files used by
``python -m vadbench.workflows.dsanet`` in modern environments.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "vadbench" / "workflows" / "dsanet"
CORE_BRIDGE = ROOT / "src" / "vadbench" / "token_reduction" / "bridges" / "clip.py"
PRIVATE_NAME = "_vadbench_dsanet_legacy"


def _cli_module():
    package = ModuleType(PRIVATE_NAME)
    package.__path__ = [str(PACKAGE)]
    sys.modules[PRIVATE_NAME] = package
    return importlib.import_module(f"{PRIVATE_NAME}.__main__")


def _stage_args(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if args and args[0] == "extract" and not any(
        arg == "--bridge" or arg.startswith("--bridge=") for arg in args[1:]
    ):
        args[1:1] = ["--bridge", str(CORE_BRIDGE)]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "quality":
        argparse.ArgumentParser(prog=str(Path(__file__).name)).error(
            "quality requires Python 3.10+; use python -m vadbench.workflows.dsanet quality"
        )
    if not args or args[0] in ("-h", "--help"):
        print("Python 3.9 launcher: extract and score. Use Python 3.10+ for quality.")
    _cli_module().main(_stage_args(args))


if __name__ == "__main__":
    main()
