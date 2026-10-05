#!/usr/bin/env python3
"""Report one explicit v2 smoke run without promoting historical attempts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import load_encoder_candidates  # noqa: E402


def _project_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if PROJECT_ROOT not in path.parents:
        raise ValueError(f"{label} must remain under project root: {path}")
    return path


def load_current_run(run_root: Path) -> dict[str, Any]:
    matrix_path = run_root / "matrix-v2.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("items"), list) or not payload.get("run_id"):
        raise ValueError(f"invalid matrix-v2 result: {matrix_path}")
    payload["matrix_path"] = matrix_path.relative_to(PROJECT_ROOT).as_posix()
    return payload


def list_history(history_root: Path | None) -> list[str]:
    if history_root is None:
        return []
    return sorted(
        path.relative_to(PROJECT_ROOT).as_posix() for path in history_root.glob("**/matrix-v2.json")
    )


def consolidate(current_run: dict[str, Any], history_root: Path | None = None) -> dict[str, Any]:
    attempts = {str(item["integration_id"]): dict(item) for item in current_run["items"]}
    items = []
    for candidate in load_encoder_candidates(PROJECT_ROOT):
        encoder_id = candidate["id"]
        attempt = attempts.get(encoder_id)
        if candidate["registration_state"] == "candidate_only":
            status = "unregistered"
        elif candidate["registration_state"] == "awaiting_manual_asset":
            status = "manual_required"
        elif candidate["license_state"] != "verified":
            status = "blocked_license"
        elif attempt is None:
            status = "not_run"
        else:
            status = str(attempt.get("status", "failed"))
        items.append(
            {
                "integration_id": encoder_id,
                "group": candidate["group"],
                "registration_state": candidate["registration_state"],
                "asset_state": candidate["asset_state"],
                "license_state": candidate["license_state"],
                "status": status,
                "attempt": attempt,
            }
        )
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in sorted({item["status"] for item in items})
    }
    current_items = current_run["items"]
    return {
        "schema_version": 3,
        "target_count": len(items),
        "current_run": {
            "run_id": current_run["run_id"],
            "matrix_path": current_run["matrix_path"],
            "success": bool(current_items)
            and all(item.get("status") == "smoke_pass" for item in current_items),
        },
        "history_matrix_paths": list_history(history_root),
        "counts": counts,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--history-root")
    parser.add_argument(
        "--output",
        default="outputs/environment-migration-v2/native-smoke-matrix.json",
    )
    args = parser.parse_args(argv)
    run_root = _project_path(args.run_root, "run_root")
    history_root = _project_path(args.history_root, "history_root") if args.history_root else None
    output = _project_path(args.output, "output")
    payload = consolidate(load_current_run(run_root), history_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"consolidated output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"counts": payload["counts"], "output": str(output)}, ensure_ascii=False))
    return 0 if payload["current_run"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
