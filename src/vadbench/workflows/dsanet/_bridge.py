"""Resolve the repository CLIP bridge or a legacy replay file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def bridge_symbols(path: Path | None):
    if path is None:
        from vadbench.token_reduction.bridges import clip

        return clip.ClipVisionBridge, clip.load_openai_clip, Path(clip.__file__)
    spec = importlib.util.spec_from_file_location("_vadbench_dsanet_replay_clip_bridge", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load legacy CLIP bridge: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise
    return module.ClipVisionBridge, module.load_openai_clip, path
