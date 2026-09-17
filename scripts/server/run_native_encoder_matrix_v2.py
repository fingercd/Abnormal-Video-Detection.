#!/usr/bin/env python3
"""Run native encoder smoke tests only through the isolated v2 environments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import (  # noqa: E402
    build_encoder_runtime_environment,
    load_encoder_candidates,
    load_encoder_environment_registry,
    resolve_encoder_runtime,
)
from vadbench.integrations.catalog import load_default_integration_catalog  # noqa: E402
from vadbench.smoke import read_smoke_result_v2  # noqa: E402


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_is_runnable(
    candidate: dict[str, Any], include_license_blocked: bool
) -> tuple[bool, str | None]:
    if candidate["registration_state"] == "candidate_only":
        return False, "candidate_only"
    if candidate["registration_state"] == "awaiting_manual_asset":
        return False, "manual_asset_missing"
    if candidate["license_state"] != "verified" and not include_license_blocked:
        return False, "license_blocked"
    return True, None


def require_available_gpu(device: str) -> None:
    if not device.startswith("cuda"):
        return
    index = device.partition(":")[2] or "0"
    used_mib = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if used_mib > 1024:
        raise SystemExit(f"GPU {index} already uses {used_mib} MiB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-license-blocked", action="store_true")
    parser.add_argument(
        "--video",
        default="data/smoke/mlvu-surveil-8.mp4",
    )
    parser.add_argument(
        "--output-root",
        default=None,
    )
    args = parser.parse_args(argv)
    require_available_gpu(args.device)

    environment = load_encoder_environment_registry(PROJECT_ROOT)
    catalog = load_default_integration_catalog(PROJECT_ROOT)
    by_candidate = {item["id"]: item for item in load_encoder_candidates(PROJECT_ROOT)}
    selected = list(args.id or catalog.ids)
    unknown = sorted(set(selected) - set(catalog.ids))
    if unknown:
        raise SystemExit(f"not registered runtime encoders: {unknown}")
    run_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output_root = (
        Path(args.output_root) if args.output_root else PROJECT_ROOT / "outputs/encoder-v2" / run_id
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    if PROJECT_ROOT not in output_root.parents:
        raise RuntimeError("output_root must remain under project root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"matrix output already exists: {output_root}")
    output_root.mkdir(parents=True)

    items = []
    for encoder_id in selected:
        candidate = by_candidate[encoder_id]
        runnable, skip_reason = candidate_is_runnable(candidate, args.include_license_blocked)
        runtime = resolve_encoder_runtime(
            encoder_id,
            project_root=PROJECT_ROOT,
            registry=environment,
        )
        group = runtime.group
        python = runtime.python
        overlay = runtime.overlay
        item = {
            "integration_id": encoder_id,
            "group": group.id,
            "python_executable": str(python),
            "overlay": str(overlay) if overlay is not None else None,
            "base_marker_sha256": file_hash(group.prefix / ".vadbench-env-v2.json"),
            "overlay_marker_sha256": (
                file_hash(overlay / ".overlay-v2.json") if overlay is not None else None
            ),
        }
        if not runnable:
            item.update({"status": "skipped", "reason": skip_reason})
            items.append(item)
            continue
        if not python.is_file():
            item.update({"status": "blocked", "reason": "new_environment_missing"})
            items.append(item)
            continue
        item_root = output_root / encoder_id
        item_root.mkdir(parents=True, exist_ok=True)
        log_path = item_root / "launcher.log"
        command = [
            str(python),
            "-m",
            "vadbench",
            "integrations",
            "smoke",
            "--video",
            str(PROJECT_ROOT / args.video),
            "--id",
            encoder_id,
            "--device",
            args.device,
            "--output-root",
            str(item_root),
        ]
        env = build_encoder_runtime_environment(
            runtime,
            project_root=PROJECT_ROOT,
            registry=environment,
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        log_path.write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr,
            encoding="utf-8",
        )
        result_path = item_root / encoder_id / "result.json"
        status = "failed"
        reason = None
        if completed.returncode != 0:
            reason = "launcher_exit_nonzero"
        elif not result_path.is_file():
            reason = "result_missing"
        else:
            try:
                status = str(
                    read_smoke_result_v2(result_path, expected_record=catalog.get(encoder_id))[
                        "status"
                    ]
                )
            except (OSError, ValueError) as exc:
                reason = f"result_invalid: {exc}"
                status = "failed"
        technical_status = status
        if candidate["license_state"] != "verified" and status == "smoke_pass":
            status = "blocked_license"
        item.update(
            {
                "status": status,
                "technical_status": technical_status,
                "reason": "license_blocked_after_technical_pass"
                if status == "blocked_license"
                else reason,
                "exit_code": completed.returncode,
                "result_path": (
                    result_path.relative_to(PROJECT_ROOT).as_posix()
                    if result_path.is_file()
                    else None
                ),
                "log_path": log_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
        items.append(item)

    counts = {
        status: sum(item["status"] == status for item in items)
        for status in sorted({item["status"] for item in items})
    }
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "device": args.device,
        "video": args.video,
        "counts": counts,
        "items": items,
    }
    (output_root / "matrix-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if counts.get("smoke_pass") and set(counts) == {"smoke_pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
