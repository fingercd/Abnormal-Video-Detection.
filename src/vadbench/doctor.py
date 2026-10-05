"""本地/服务器运行环境的只读诊断。"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

OPTIONAL_MODULES = ("torch", "cv2", "transformers", "safetensors", "sklearn")


def collect_diagnostics(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    modules = {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_MODULES}
    paths = {
        name: {
            "path": str(root / name),
            "exists": (root / name).exists(),
            "writable": _is_writable(root / name),
        }
        for name in ("data", "weights", "outputs", "external")
    }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(root),
        "ffmpeg": shutil.which("ffmpeg"),
        "git": shutil.which("git"),
        "optional_modules": modules,
        "paths": paths,
    }


def _is_writable(path: Path) -> bool:
    target = path if path.exists() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    return target.is_dir() and os.access(target, os.W_OK)


def diagnostics_json(project_root: str | Path | None = None) -> str:
    return json.dumps(collect_diagnostics(project_root), ensure_ascii=False, indent=2)
