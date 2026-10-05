"""Inventory the existing ICASSP asset roots without reading model scores.

The output is a dated discovery catalog. A discovered feature is not marked
verified until its manifest, index, blobs and completion receipt pass QA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
from pathlib import Path


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}
KEEP_RATIOS = {"0p80": 0.8, "0p60": 0.6, "0p40": 0.4}


def _entry(path: Path) -> dict[str, object]:
    if path.is_symlink():
        return {"path": str(path), "kind": "symlink", "target": os.readlink(path)}
    if not path.exists():
        return {"path": str(path), "kind": "missing"}
    stat = path.stat()
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _children(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name) if path.is_dir() else []


def _small_json(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _marker(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "bytes": stat.st_size}
    if stat.st_size <= 1024 * 1024:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _feature(
    path: Path, *, family: str, dataset: str, split: str, encoder: str,
    method: str, keep_ratio: float | None, role: str,
) -> dict[str, object]:
    possible_indexes = [path / "index.jsonl", path / "features" / "train" / "index.jsonl"]
    index = next((p for p in possible_indexes if p.is_file()), None)
    markers = (
        "merge-contract.json", "extraction-contract.json", "status.json", "result.json",
        "summary.json", "resolved.json", "resolved_reducer.json", "sealed_features.json",
    )
    return {
        "family": family,
        "dataset": dataset,
        "split": split,
        "encoder": encoder,
        "method": method,
        "keep_ratio": keep_ratio,
        "role": role,
        "source": _entry(path),
        "index": _marker(index) if index else None,
        "markers": {name: value for name in markers if (value := _marker(path / name)) is not None},
        "summary_status": (_small_json(path / "status.json") or {}).get("status")
        or (_small_json(path / "result.json") or {}).get("status")
        or (_small_json(path / "summary.json") or {}).get("status"),
        "qa_status": "discovered_unverified",
    }


def _video_count(root: Path) -> int:
    if not root.is_dir():
        return 0
    count = 0
    for base, _, files in os.walk(root):
        count += sum(Path(name).suffix.lower() in VIDEO_SUFFIXES for name in files)
    return count


def collect(args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    kimi = Path(args.kimi_root)
    runs = Path(args.runs_root)
    dsanet = runs / "dsanet-extension-20260923-r01"
    dsanet_train = Path(args.dsanet_train_root)
    datasets = Path(args.datasets_root)
    framework = Path(args.framework_root)

    roots = [
        {"role": role, **_entry(path)} for role, path in (
            ("framework", framework), ("runs", runs), ("datasets", datasets),
            ("video_vit_assets", kimi), ("dsanet_train_features", dsanet_train),
        )
    ]
    dataset_rows = []
    for path in _children(datasets):
        if not (path.name.startswith("UCF-Crime") or path.name.startswith("XD-Violence")):
            continue
        dataset_rows.append({
            "source": _entry(path), "video_files": _video_count(path),
            "role": "source_candidate", "qa_status": "discovered_unverified",
        })

    features = []
    merged = kimi / "merged-full"
    for encoder_dir in _children(merged):
        for view in _children(encoder_dir):
            if view.name == "dense-full-view":
                features.append(_feature(view, family="video_vit", dataset="ucf_crime",
                    split="train", encoder=encoder_dir.name, method="dense", keep_ratio=None,
                    role="merged_view"))
            elif view.name == "xd-dense-full-view":
                for run in _children(view):
                    features.append(_feature(run, family="video_vit", dataset="xd_violence",
                        split="train", encoder=encoder_dir.name, method="dense", keep_ratio=None,
                        role="merged_view"))

    for encoder_dir in _children(kimi / "merged"):
        for view in _children(encoder_dir):
            if view.name.startswith("xd-dense-test-view"):
                dataset, split = "xd_violence", "test"
            elif view.name.startswith("dense-test-view"):
                dataset, split = "ucf_crime", "test"
            elif view.name.startswith("dense-train-view"):
                dataset, split = "ucf_crime", "train"
            else:
                continue
            features.append(_feature(view, family="video_vit", dataset=dataset,
                split=split, encoder=encoder_dir.name, method="dense", keep_ratio=None,
                role="merged_view"))

    for source_name, role in (("compressed-test", "formal_candidate"),
                              ("compressed-test-node2", "supplemental_candidate")):
        for encoder_dir in _children(kimi / source_name):
            for dataset_dir in _children(encoder_dir):
                for run in _children(dataset_dir):
                    method = next((m for m in ("pair_select", "group_uniform", "group_random")
                                   if run.name.startswith(m)), "unknown")
                    features.append(_feature(run, family="video_vit",
                        dataset={"ucf": "ucf_crime", "xd": "xd_violence"}.get(dataset_dir.name, dataset_dir.name),
                        split="test", encoder=encoder_dir.name, method=method,
                        keep_ratio=0.6, role=role))

    for encoder_dir in _children(kimi / "compressed-budgetsweep-r01"):
        for dataset_dir in _children(encoder_dir):
            for ratio_dir in _children(dataset_dir):
                for run in _children(ratio_dir / "merged"):
                    role = "historical_attempt" if "partial-stale" in run.name or "failed" in run.name else "budget_sweep_candidate"
                    features.append(_feature(run, family="video_vit",
                        dataset={"ucf": "ucf_crime", "xd": "xd_violence"}.get(dataset_dir.name, dataset_dir.name),
                        split="test", encoder=encoder_dir.name, method="pair_select",
                        keep_ratio=KEEP_RATIOS.get(ratio_dir.name), role=role))

    campaign = dsanet / "test-campaign-r02"
    for tier in _children(campaign):
        if tier.name in {"logs", "state"}:
            continue
        dataset = "ucf_crime" if tier.name.startswith("ucf") else "xd_violence"
        ratio = {"dense": None, "0.8": 0.8, "0.6": 0.6, "0.4": 0.4}.get(tier.name.split("-", 1)[-1])
        features.append(_feature(tier, family="clip_dsanet", dataset=dataset, split="test",
            encoder="clip_vit_b16", method="dense" if ratio is None else "pair_select",
            keep_ratio=ratio, role="published_head_input_candidate"))
    for shard in _children(dsanet_train / "train-features-r01"):
        if shard.name.startswith(("ucf-shard", "xd-shard")):
            features.append(_feature(shard, family="clip_dsanet",
                dataset="ucf_crime" if shard.name.startswith("ucf") else "xd_violence",
                split="train", encoder="clip_vit_b16", method="dense", keep_ratio=None,
                role="local_head_attempt_input"))

    run_rows = []
    for root, family in ((runs, "orchestration"), (kimi, "video_vit"), (dsanet, "clip_dsanet")):
        for path in _children(root):
            if path.name.startswith("code-") or path.name == "project":
                continue
            run_rows.append({"family": family, "source": _entry(path),
                "role": "campaign_root", "qa_status": "discovered_unverified"})
    code = [{"source": _entry(path), "qa_status": "identity_pending"}
            for path in _children(runs) if path.name.startswith("code-")]
    return {"roots": roots, "datasets": dataset_rows, "features": features,
            "runs": run_rows, "code_snapshots": code}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--kimi-root", required=True)
    parser.add_argument("--dsanet-train-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error(f"output already exists: {output}")
    records = collect(args)
    output.mkdir(parents=True)
    for name, rows in records.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output / "summary.json").write_text(json.dumps({
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "scope": "discovery_only_no_scores_read",
        "counts": {name: len(rows) for name, rows in records.items()},
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in records.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
