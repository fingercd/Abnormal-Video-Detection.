#!/usr/bin/env python3
"""Create and audit the isolated v2 encoder environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import load_encoder_environment_registry  # noqa: E402

CONDA = Path("/users/fotile/miniconda3/bin/conda")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/environment-migration-v2"
METADATA_FILES = ("METADATA", "RECORD", "direct_url.json", "INSTALLER")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True, env=env)


def fingerprint_environment(prefix: Path) -> dict[str, Any]:
    prefix = prefix.expanduser().resolve()
    python = prefix / "bin" / "python"
    if not python.is_file():
        return {"prefix": str(prefix), "exists": False}
    info = json.loads(
        _run(
            [
                str(python),
                "-c",
                "import json,platform,sys; print(json.dumps({'executable':sys.executable,'python':platform.python_version()}))",
            ],
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout
    )
    freeze = _run(
        [str(python), "-m", "pip", "freeze", "--all"],
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    ).stdout.splitlines()
    metadata: list[dict[str, Any]] = []
    for site in sorted(prefix.glob("lib/python*/site-packages")):
        for dist in sorted(list(site.glob("*.dist-info")) + list(site.glob("*.egg-info"))):
            for name in METADATA_FILES:
                path = dist / name
                if path.is_file():
                    data = path.read_bytes()
                    metadata.append(
                        {
                            "path": path.relative_to(prefix).as_posix(),
                            "size": len(data),
                            "sha256": _sha256_bytes(data),
                        }
                    )
    history = prefix / "conda-meta" / "history"
    history_hash = _sha256_bytes(history.read_bytes()) if history.is_file() else None
    digest = hashlib.sha256()
    digest.update("\n".join(sorted(freeze)).encode())
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    digest.update(str(history_hash).encode())
    return {
        "prefix": str(prefix),
        "exists": True,
        "python": info["python"],
        "executable": info["executable"],
        "pip_freeze": sorted(freeze),
        "metadata": metadata,
        "conda_history_sha256": history_hash,
        "fingerprint_sha256": digest.hexdigest(),
    }


def protected_fingerprints(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    registry = load_encoder_environment_registry(project_root)
    return {str(path): fingerprint_environment(path) for path in registry.protected_roots}


def write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    root = PROJECT_ROOT.resolve()
    if root not in path.parents:
        raise ValueError(f"output must remain under project root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_protected(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    if set(baseline) != set(current):
        raise RuntimeError("protected environment set changed")
    changed = []
    for key in baseline:
        before = baseline[key].get("fingerprint_sha256")
        after = current[key].get("fingerprint_sha256")
        if before != after:
            changed.append({"prefix": key, "before": before, "after": after})
    if changed:
        raise RuntimeError(f"protected environment fingerprints changed: {changed}")


def disk_guard(registry: Any) -> None:
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free < registry.minimum_free_bytes:
        raise RuntimeError(
            f"free disk {free} is below required floor {registry.minimum_free_bytes}"
        )


def _repair_foundation_cuda_links(prefix: Path) -> list[dict[str, Any]]:
    source_prefix = Path("/users/fotile/miniconda3/envs/mllm-comp-internav")
    repairs = []
    for source_link in (source_prefix / "lib").iterdir():
        if not source_link.is_symlink():
            continue
        raw = os.readlink(source_link)
        marker = str(source_prefix / "lib") + "/"
        if not raw.startswith(marker) or "/site-packages/" not in raw:
            continue
        suffix = raw[len(marker) :]
        target_link = prefix / "lib" / source_link.name
        if target_link.is_symlink() and os.readlink(target_link) == suffix:
            pass
        else:
            if target_link.exists() or target_link.is_symlink():
                target_link.unlink()
            target_link.symlink_to(suffix)
        repairs.append({"path": f"lib/{source_link.name}", "target": suffix})
    return repairs


def _copy_if_missing(source: Path, target: Path) -> bool:
    if target.exists() or target.is_symlink():
        return False
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    elif source.is_file() or source.is_symlink():
        shutil.copy2(source, target, follow_symlinks=False)
    else:
        raise FileNotFoundError(source)
    return True


def _repair_h3_video_runtime(prefix: Path) -> list[dict[str, Any]]:
    repairs = []
    source_site = PROJECT_ROOT / ".venv/lib/python3.11/site-packages"
    target_site = prefix / "lib/python3.11/site-packages"
    package_names = (
        "cv2",
        "opencv_python_headless-4.11.0.86.dist-info",
        "opencv_python_headless.libs",
    )
    copied_packages = [
        name for name in package_names if _copy_if_missing(source_site / name, target_site / name)
    ]
    library_roots = (
        Path("/users/fotile/miniconda3/pkgs/libsndfile-1.2.2-hc7d488a_2/lib"),
        Path("/users/fotile/miniconda3/pkgs/libflac-1.5.0-he200343_1/lib"),
        Path("/users/fotile/miniconda3/pkgs/libopus-1.6.1-h280c20c_0/lib"),
        Path("/users/fotile/miniconda3/pkgs/libvorbis-1.3.7-h54a6638_2/lib"),
        Path("/users/fotile/miniconda3/pkgs/libogg-1.3.5-hd0c01bc_1/lib"),
        Path("/users/fotile/miniconda3/pkgs/mpg123-1.32.9-h8142553_0/lib"),
        Path("/users/fotile/miniconda3/pkgs/lame-3.100-h166bdaf_1003/lib"),
    )
    copied_libraries = []
    for source_root in library_roots:
        for source in source_root.iterdir():
            target = prefix / "lib" / source.name
            if _copy_if_missing(source, target):
                copied_libraries.append(source.name)
    repairs.append(
        {
            "packages": list(package_names),
            "copied_packages": copied_packages,
            "libraries": sorted(copied_libraries),
            "reason": "offline video and Transformers audio runtime",
        }
    )
    return repairs


def _apply_group_repairs(group_id: str, prefix: Path, marker: Path) -> None:
    if group_id == "foundation-video-v2":
        repairs = _repair_foundation_cuda_links(prefix)
    elif group_id in {"visual-vlm-v2", "stream-kv-v2"}:
        repairs = _repair_h3_video_runtime(prefix)
    else:
        repairs = []
    if not repairs:
        return
    data = json.loads(marker.read_text(encoding="utf-8"))
    data["repairs"] = repairs
    marker.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bootstrap(groups: list[str] | None, output_root: Path) -> dict[str, Any]:
    registry = load_encoder_environment_registry(PROJECT_ROOT)
    selected = groups if groups else list(registry.groups)
    unknown = sorted(set(selected) - set(registry.groups))
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    baseline_path = output_root / "old-envs-before.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["environments"]
    else:
        baseline = protected_fingerprints()
        write_json(baseline_path, {"schema_version": 1, "environments": baseline})
    disk_guard(registry)
    registry.root.mkdir(parents=True, exist_ok=True)
    registry.overlay_root.mkdir(parents=True, exist_ok=True)
    registry.cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for group_id in selected:
        group = registry.groups[group_id]
        marker = group.prefix / ".vadbench-env-v2.json"
        if group.prefix.exists():
            if not marker.is_file():
                raise RuntimeError(f"refusing unmarked existing prefix: {group.prefix}")
            status = "reused"
        else:
            group.prefix.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["PYTHONNOUSERSITE"] = "1"
            if group.seed.name == "h3":
                group.prefix.mkdir(parents=True, exist_ok=False)
                command = [
                    "cp",
                    "-a",
                    "--reflink=auto",
                    f"{group.seed}/.",
                    str(group.prefix),
                ]
                completed = _run(command, env=env)
                seed_method = "filesystem_reflink_copy"
            else:
                command = [
                    str(CONDA),
                    "create",
                    "-y",
                    "--offline",
                    "-p",
                    str(group.prefix),
                    "--clone",
                    str(group.seed),
                ]
                completed = _run(command, env=env)
                seed_method = "conda_clone_offline"
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "group": group_id,
                        "seed": str(group.seed),
                        "seed_method": seed_method,
                        "stdout_tail": completed.stdout[-4000:],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            status = "created"
        _apply_group_repairs(group_id, group.prefix, marker)
        for encoder_id in group.encoders:
            overlay = registry.overlay_for(encoder_id)
            if overlay is not None:
                overlay.mkdir(parents=True, exist_ok=True)
        results.append(
            {
                "group": group_id,
                "status": status,
                "prefix": str(group.prefix),
                "fingerprint": fingerprint_environment(group.prefix),
            }
        )
        disk_guard(registry)
    current = protected_fingerprints()
    compare_protected(baseline, current)
    write_json(
        output_root / "old-envs-after.json",
        {"schema_version": 1, "environments": current},
    )
    payload = {"schema_version": 2, "groups": results}
    write_json(output_root / "environment-matrix.json", payload)
    return payload


def verify(output_root: Path) -> dict[str, Any]:
    registry = load_encoder_environment_registry(PROJECT_ROOT)
    baseline_path = output_root / "old-envs-before.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["environments"]
    current = protected_fingerprints()
    compare_protected(baseline, current)
    groups = []
    for group_id, group in registry.groups.items():
        fingerprint = fingerprint_environment(group.prefix)
        if not fingerprint.get("exists"):
            status = "missing"
        elif fingerprint.get("python") != group.python_version:
            status = "version_mismatch"
        else:
            status = "verified"
        groups.append(
            {
                "group": group_id,
                "status": status,
                "expected_python": group.python_version,
                "fingerprint": fingerprint,
            }
        )
    payload = {"schema_version": 2, "protected_unchanged": True, "groups": groups}
    write_json(output_root / "environment-matrix.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("snapshot-old", "bootstrap", "verify"))
    parser.add_argument("--group", action="append")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if args.command == "snapshot-old":
        payload = {"schema_version": 1, "environments": protected_fingerprints()}
        write_json(output_root / "old-envs-before.json", payload)
    elif args.command == "bootstrap":
        payload = bootstrap(args.group, output_root)
    else:
        payload = verify(output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.command == "verify" and any(
        group["status"] != "verified" for group in payload["groups"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
