#!/usr/bin/env python3
"""Verify local native checkpoints and acquire permitted missing assets on node2."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.checkpoints import (  # noqa: E402
    CheckpointError,
    CheckpointSpec,
    load_checkpoint_registry,
    sha256_file,
    verify_checkpoint,
)
from vadbench.config import load_yaml  # noqa: E402
from vadbench.environment_registry import (  # noqa: E402
    load_encoder_candidates,
    load_encoder_environment_registry,
)
from vadbench.integrations.catalog import load_default_integration_catalog  # noqa: E402


def local_asset_path(entry: dict[str, Any]) -> Path:
    local = Path(str(entry["local_path"]))
    return local if local.is_absolute() else PROJECT_ROOT / local


def checkpoint_root(entry: dict[str, Any], spec: CheckpointSpec) -> Path:
    local = local_asset_path(entry)
    if len(spec.sha256) == 1 and local.name in spec.sha256:
        return local.parent
    return local


def resolve_asset_file(entry: dict[str, Any], relative: str, spec: CheckpointSpec) -> Path:
    local = local_asset_path(entry)
    if len(spec.sha256) == 1 and local.name == relative:
        return local
    return checkpoint_root(entry, spec) / relative


def verify_entry(entry: dict[str, Any], spec: CheckpointSpec) -> dict[str, Any]:
    root = checkpoint_root(entry, spec)
    try:
        actual = verify_checkpoint(spec, root)
        status = "verified"
        error = None
    except CheckpointError as exc:
        actual = {}
        status = "missing_or_mismatch"
        error = str(exc)
    files = []
    for relative, expected in spec.sha256.items():
        path = resolve_asset_file(entry, relative, spec)
        exists = path.is_file()
        digest = actual.get(relative) if exists else None
        if digest is None and exists:
            digest = sha256_file(path)
        match = exists and digest == expected
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix()
                if PROJECT_ROOT in path.parents
                else str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "expected_sha256": expected,
                "actual_sha256": digest,
                "match": match,
            }
        )
    return {"status": status, "files": files, "error": error}


def verify_checkout(candidate: dict[str, Any], definition_path: str) -> dict[str, Any]:
    definition = load_yaml(PROJECT_ROOT / definition_path)
    constructor = definition.get("constructor", {})
    checkout_value = constructor.get("checkout_path") if isinstance(constructor, dict) else None
    lock = load_yaml(PROJECT_ROOT / candidate["upstream_lock"])
    source = lock.get("source", {}) if isinstance(lock, dict) else {}
    repository = source.get("repository") if isinstance(source, dict) else None
    revision = source.get("commit") or source.get("revision") if isinstance(source, dict) else None
    if not checkout_value:
        return {
            "status": "not_required",
            "repository": repository,
            "revision": revision,
            "path": None,
        }
    checkout = Path(str(checkout_value))
    if not checkout.is_absolute():
        checkout = PROJECT_ROOT / checkout
    if not checkout.is_dir():
        return {
            "status": "missing",
            "repository": repository,
            "revision": revision,
            "path": checkout.as_posix(),
        }
    if not revision:
        return {
            "status": "revision_missing",
            "repository": repository,
            "revision": revision,
            "actual_revision": None,
            "path": checkout.as_posix(),
        }
    if not (checkout / ".git").exists():
        return {
            "status": "git_metadata_missing",
            "repository": repository,
            "revision": revision,
            "actual_revision": None,
            "path": checkout.as_posix(),
        }
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else None
    status = (
        "verified"
        if actual == revision
        else ("head_unavailable" if actual is None else "revision_mismatch")
    )
    return {
        "status": status,
        "repository": repository,
        "revision": revision,
        "actual_revision": actual,
        "path": checkout.as_posix(),
    }


def disk_guard(expected_bytes: int = 0) -> None:
    registry = load_encoder_environment_registry(PROJECT_ROOT)
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free - max(0, expected_bytes) < registry.minimum_free_bytes:
        raise RuntimeError(
            f"projected free disk {free - expected_bytes} is below floor "
            f"{registry.minimum_free_bytes}"
        )


def acquire_huggingface(
    candidate: dict[str, Any],
    entry: dict[str, Any],
    spec: CheckpointSpec,
    cache_root: Path,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    checkpoint = candidate["checkpoint"]
    repo_id = checkpoint.get("repo_id")
    revision = checkpoint.get("revision")
    if not repo_id or not revision:
        raise RuntimeError("missing Hugging Face repo_id or revision")
    expected_size = int(checkpoint.get("expected_size_bytes") or 0)
    disk_guard(expected_size)
    final = local_asset_path(entry)
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing asset path: {final}")
    temporary = cache_root / "downloads" / candidate["id"]
    temporary.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="snapshot-", dir=temporary))
    try:
        snapshot_download(
            repo_id=str(repo_id),
            revision=str(revision),
            local_dir=str(staging),
            allow_patterns=list(checkpoint.get("allow_patterns") or []),
            cache_dir=str(cache_root / "huggingface"),
            max_workers=1,
        )
        verify_checkpoint(spec, staging)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return {"status": "downloaded_verified", "path": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--id", action="append")
    parser.add_argument(
        "--output-root",
        default="outputs/environment-migration-v2",
    )
    args = parser.parse_args(argv)

    environment = load_encoder_environment_registry(PROJECT_ROOT)
    candidates = load_encoder_candidates(PROJECT_ROOT)
    selected = set(args.id or [item["id"] for item in candidates])
    unknown = selected - {item["id"] for item in candidates}
    if unknown:
        raise SystemExit(f"unknown encoder ids: {sorted(unknown)}")
    checkpoint_data = load_yaml(PROJECT_ROOT / "registry/checkpoints.yaml")["checkpoints"]
    checkpoint_specs = load_checkpoint_registry(PROJECT_ROOT / "registry/checkpoints.yaml")
    catalog = load_default_integration_catalog(PROJECT_ROOT)
    records = {record.id: record for record in catalog.integrations}
    runtime_candidates = [
        item
        for item in candidates
        if item["registration_state"] != "candidate_only" and item["id"] in selected
    ]
    items = []
    manual = []
    for candidate in runtime_candidates:
        checkpoint_id = candidate["checkpoint"]["registry_id"]
        entry = checkpoint_data[checkpoint_id]
        spec = checkpoint_specs[checkpoint_id]
        checkout = verify_checkout(candidate, records[candidate["id"]].definition)
        verified = verify_entry(entry, spec)
        if verified["status"] == "verified" and checkout["status"] in {"verified", "not_required"}:
            items.append(
                {
                    "integration_id": candidate["id"],
                    "checkpoint_id": checkpoint_id,
                    "status": "verified_existing",
                    "code": checkout,
                    "files": verified["files"],
                }
            )
            continue
        reason = None
        acquired = None
        can_auto = (
            args.execute
            and candidate["license_state"] == "verified"
            and entry.get("source") == "huggingface"
            and checkout["status"] in {"verified", "not_required"}
        )
        if can_auto:
            try:
                acquired = acquire_huggingface(candidate, entry, spec, environment.cache_root)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        else:
            if checkout["status"] not in {"verified", "not_required"}:
                reason = f"code checkout is {checkout['status']}"
            elif not args.execute:
                reason = "automatic download disabled"
            else:
                reason = "manual or license-gated asset"
        if acquired is not None:
            verified_after_download = verify_entry(entry, spec)
            items.append(
                {
                    "integration_id": candidate["id"],
                    "checkpoint_id": checkpoint_id,
                    "status": (
                        "verified_existing"
                        if verified_after_download["status"] == "verified"
                        else "missing_or_mismatch"
                    ),
                    "code": checkout,
                    "files": verified_after_download["files"],
                    "acquisition": acquired,
                }
            )
        else:
            record = {
                "integration_id": candidate["id"],
                "checkpoint_id": checkpoint_id,
                "status": "manual_required",
                "reason": reason,
                "asset": verified,
                "official_repo": candidate["checkpoint"].get("repo_url"),
                "code": checkout,
                "code_incoming_path": (environment.new_external_root / candidate["id"]).as_posix(),
                "artifact_url": candidate["checkpoint"].get("artifact_url"),
                "revision": candidate["checkpoint"].get("revision"),
                "license": candidate["checkpoint"].get("license"),
                "allow_patterns": candidate["checkpoint"].get("allow_patterns") or [],
                "expected_size_bytes": candidate["checkpoint"].get("expected_size_bytes"),
                "incoming_path": (
                    PROJECT_ROOT / ".incoming/encoder-v2" / candidate["id"]
                ).as_posix(),
                "final_path": str(entry.get("local_path")),
            }
            manual.append(record)
            items.append(record)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    if PROJECT_ROOT not in output_root.parents:
        raise RuntimeError("output_root must remain under project root")
    output_root.mkdir(parents=True, exist_ok=True)
    asset_payload = {
        "schema_version": 2,
        "hostname": socket.gethostname(),
        "selected_count": len(runtime_candidates),
        "counts": {
            status: sum(item["status"] == status for item in items)
            for status in sorted({item["status"] for item in items})
        },
        "items": items,
    }
    (output_root / "asset-matrix.json").write_text(
        json.dumps(asset_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manual_payload = {
        "schema_version": 1,
        "manual_count": len(manual),
        "items": manual,
    }
    (output_root / "manual-download-manifest.json").write_text(
        json.dumps(manual_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asset_payload, ensure_ascii=False, indent=2))
    return 0 if items and all(item["status"] == "verified_existing" for item in items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
