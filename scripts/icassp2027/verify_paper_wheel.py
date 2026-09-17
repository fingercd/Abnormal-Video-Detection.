"""Validate paper commands from an extracted wheel, without source-checkout imports."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    with tempfile.TemporaryDirectory(prefix="vadbench-paper-wheel-") as directory:
        temporary = Path(directory)
        site = temporary / "site"
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(site)
        project = temporary / "project"
        shutil.copytree(
            root / "projects/icassp2027",
            project / "projects/icassp2027",
            ignore=shutil.ignore_patterns("assets.local.yaml", "locks"),
        )
        # No encoder YAML or registry is copied: resolution must use bundled resources.
        suite = "configs/papers/icassp2027/suites/probe-pilot.yaml"
        (project / suite).parent.mkdir(parents=True)
        shutil.copyfile(site / "vadbench/resources" / suite, project / suite)
        code = """
import json, pathlib, sys
import vadbench
from vadbench.paper.profile import load_project
from vadbench.paper.resolve import status, resolve_probe
assert pathlib.Path(vadbench.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
project = load_project('projects/icassp2027/profile.yaml')
assert len(status(project)['encoders']) == 4
plan = resolve_probe(project, 'configs/papers/icassp2027/suites/probe-pilot.yaml')
assert plan['plan_only'] and plan['blockers']
assert not pathlib.Path('outputs').exists()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps({'wheel_paper_verified': True, 'module': vadbench.__file__}))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site)
        subprocess.run(
            [sys.executable, "-c", code, str(site)], cwd=project, env=environment, check=True
        )


if __name__ == "__main__":
    main()
