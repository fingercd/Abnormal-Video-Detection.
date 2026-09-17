#!/usr/bin/env python3
"""Populate model-specific overlays by copying known-working packages read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import load_encoder_environment_registry  # noqa: E402

COPY_SPECS = {
    "videomaev2": {
        "source": Path("/users/fotile/miniconda3/envs/h3/lib/python3.11/site-packages"),
        "names": ("transformers", "transformers-4.56.1.dist-info"),
        "extra_sources": (
            (
                PROJECT_ROOT / ".venv/lib/python3.11/site-packages",
                ("easydict", "easydict-1.13.dist-info"),
            ),
        ),
    },
    "longvu": {
        "source": PROJECT_ROOT / ".venv/lib/python3.11/site-packages",
        "names": (
            "accelerate",
            "accelerate-1.11.0.dist-info",
            "easydict",
            "easydict-1.13.dist-info",
            "peft",
            "peft-0.17.1.dist-info",
            "timm",
            "timm-1.0.29.dist-info",
        ),
    },
    "videochat_online": {
        "source": PROJECT_ROOT / ".venv/lib/python3.11/site-packages",
        "names": (
            "accelerate",
            "accelerate-1.11.0.dist-info",
            "peft",
            "peft-0.17.1.dist-info",
            "timm",
            "timm-1.0.29.dist-info",
        ),
    },
    "videochat_flash": {
        "source": Path(
            "/users/fotile/miniconda3/envs/mllm-comp-internav/lib/python3.10/site-packages"
        ),
        "names": ("transformers", "transformers-4.57.3.dist-info"),
    },
    "hermes_llava_ov": {
        "source": PROJECT_ROOT / ".venv-hermes/lib/python3.11/site-packages",
        "names": (
            "accelerate",
            "accelerate-1.11.0.dist-info",
            "easydict",
            "easydict-1.13.dist-info",
            "timm",
            "timm-1.0.29.dist-info",
            "tokenizers",
            "tokenizers-0.19.1.dist-info",
            "transformers",
            "transformers-4.45.0.dev0.dist-info",
            "logzero",
            "logzero-1.7.0.dist-info",
        ),
    },
    "streaming_vlm": {
        "source": Path("/users/fotile/miniconda3/envs/h3/lib/python3.11/site-packages"),
        "names": (),
    },
    "videomamba": {
        "source": PROJECT_ROOT / "external-v2/videomamba",
        "names": (),
    },
}


def hash_overlay(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(value for value in path.rglob("*") if value.is_file()):
        if file.name == ".overlay-v2.json":
            continue
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def hash_sources(sources: list[tuple[Path, tuple[str, ...]]]) -> str:
    digest = hashlib.sha256()
    for source_root, names in sources:
        for name in names:
            source = source_root / name
            digest.update(name.encode())
            if source.is_dir():
                digest.update(hash_overlay(source).encode())
            elif source.is_file():
                with source.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
            else:
                raise FileNotFoundError(source)
    return digest.hexdigest()


def copy_item(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(source)


def populate(encoder_ids: list[str] | None = None) -> dict[str, Any]:
    registry = load_encoder_environment_registry(PROJECT_ROOT)
    selected = encoder_ids if encoder_ids else list(COPY_SPECS)
    unknown = sorted(set(selected) - set(COPY_SPECS))
    if unknown:
        raise ValueError(f"unknown overlays: {unknown}")
    items = []
    for encoder_id in selected:
        overlay = registry.overlay_for(encoder_id)
        if overlay is None:
            raise RuntimeError(f"overlay not registered: {encoder_id}")
        marker = overlay / ".overlay-v2.json"
        spec = COPY_SPECS[encoder_id]
        existing_marker = (
            json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
        )
        existing = [path for path in overlay.iterdir() if path.name != ".overlay-v2.json"]
        if existing_marker is None and existing:
            raise RuntimeError(f"refusing non-empty unmarked overlay: {overlay}")
        copied = list(existing_marker.get("copied", [])) if existing_marker else []
        sources = [(Path(spec["source"]), spec["names"])]
        sources.extend(spec.get("extra_sources", ()))
        source_fingerprint = hash_sources(sources)
        if existing_marker is not None and (
            existing_marker.get("fingerprint") != hash_overlay(overlay)
            or existing_marker.get("source_fingerprint") != source_fingerprint
        ):
            raise RuntimeError(f"refusing stale overlay: {overlay}")
        added = []
        for source_root, names in sources:
            for name in names:
                source = Path(source_root) / name
                target = overlay / name
                if target.exists() or target.is_symlink():
                    if name not in copied:
                        copied.append(name)
                    continue
                copy_item(source, target)
                copied.append(name)
                added.append(name)
        requested = registry.overlays[encoder_id].get("packages", {})
        fingerprint = hash_overlay(overlay)
        payload = {
            "schema_version": 2,
            "encoder_id": encoder_id,
            "group": registry.overlays[encoder_id]["group"],
            "source": str(spec["source"]),
            "copied": sorted(set(copied)),
            "requested_packages": requested,
            "mode": "known_working_copy",
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
        }
        marker.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        items.append(
            {
                "encoder_id": encoder_id,
                "status": "updated" if added else ("reused" if existing_marker else "created"),
                "overlay": str(overlay),
                "fingerprint": fingerprint,
            }
        )
    return {"schema_version": 2, "items": items}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", action="append")
    parser.add_argument(
        "--output",
        default="outputs/environment-migration-v2/overlay-matrix.json",
    )
    args = parser.parse_args(argv)
    payload = populate(args.id)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    if PROJECT_ROOT not in output.parents:
        raise RuntimeError("output must remain under project root")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
