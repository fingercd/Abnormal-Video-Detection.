"""Summarize the detached UR-DMU GPU monitor without claiming unobserved peaks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(path: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pid = int(row["pid"])
                memory = float(row["used_memory_mib"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid monitor row: {row}") from exc
            if not math.isfinite(memory) or memory < 0:
                raise ValueError(f"invalid memory value: {row}")
            key = f"{pid}:{row.get('gpu_uuid', '')}"
            groups[key].append({"timestamp_utc": row["timestamp_utc"], "pid": pid, "gpu_uuid": row.get("gpu_uuid", ""), "used_memory_mib": memory, "command": row.get("command", "")})
    if not groups:
        raise ValueError(f"monitor log has no samples: {path}")
    processes = []
    for rows in groups.values():
        rows.sort(key=lambda row: row["timestamp_utc"])
        values = [row["used_memory_mib"] for row in rows]
        processes.append({
            "pid": rows[0]["pid"],
            "gpu_uuid": rows[0]["gpu_uuid"],
            "first_sample_utc": rows[0]["timestamp_utc"],
            "last_sample_utc": rows[-1]["timestamp_utc"],
            "samples": len(rows),
            "peak_used_memory_mib": max(values),
            "mean_used_memory_mib": sum(values) / len(values),
            "command": rows[-1]["command"],
        })
    processes.sort(key=lambda row: (row["first_sample_utc"], row["pid"]))
    all_rows = [row for rows in groups.values() for row in rows]
    peak = max(all_rows, key=lambda row: row["used_memory_mib"])
    return {
        "schema": "icassp2027.urdmu-gpu-monitor-summary/v1",
        "status": "completed",
        "monitor_log": str(path),
        "observed_from_utc": min(row["timestamp_utc"] for row in all_rows),
        "observed_until_utc": max(row["timestamp_utc"] for row in all_rows),
        "sample_rows": len(all_rows),
        "processes": processes,
        "observed_peak": {
            "pid": peak["pid"],
            "gpu_uuid": peak["gpu_uuid"],
            "used_memory_mib": peak["used_memory_mib"],
            "timestamp_utc": peak["timestamp_utc"],
        },
        "coverage_note": "Peak values are maxima observed after monitor start; they are not claims about the pre-monitor portion of any process.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.log.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "processes": len(result["processes"]), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
