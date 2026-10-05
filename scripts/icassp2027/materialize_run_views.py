"""Create non-destructive navigation links for existing experiment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


KIMI_CATEGORIES = {
    "merged-full": "video_vit/dense-extraction/full-train-views",
    "merged": "video_vit/dense-extraction/other-views",
    "training-full": "video_vit/dense-head/full-train",
    "compressed-test": "video_vit/keep60-extraction/node3-runs",
    "compressed-test-node2": "video_vit/keep60-extraction/node2-supplements",
    "urdmu-official-predictions-r08": "video_vit/keep60-score/predictions-r08",
    "urdmu-quality-exports-r08": "video_vit/keep60-score/quality-r08",
    "compressed-budgetsweep-r01": "video_vit/budget-sweep/features",
    "urdmu-official-predictions-budgetsweep-r01": "video_vit/budget-sweep/predictions-r01",
    "urdmu-official-predictions-budgetsweep-r02-reuse": "video_vit/budget-sweep/predictions-r02-reuse",
    "urdmu-quality-exports-budgetsweep-r01": "video_vit/budget-sweep/quality-r01",
    "urdmu-quality-exports-budgetsweep-r02-reuse": "video_vit/budget-sweep/quality-r02-reuse",
    "efficiency-20260923-r01": "video_vit/benchmark/efficiency-r01",
    "batch-scaling-20260923-r01": "video_vit/benchmark/batch-scaling-r01",
    "screen-64x64-native": "development/idea-screen/64x64-native",
    "v0": "development/v0/runs",
    "v0-explore": "development/v0/explore",
    "probe-np-tensor": "development/probes/np-tensor",
}

DSANET_CATEGORIES = {
    "test-campaign-r02": "clip_dsanet/raw-features/test-campaign-r02",
    "head-training-r01": "clip_dsanet/local-head-attempt/head-training-r01",
    "train-csv-r01": "clip_dsanet/local-head-attempt/train-csv-r01",
    "score-published-r01": "clip_dsanet/published-head-score/predictions-r01",
    "analysis-published-r02": "clip_dsanet/published-head-score/quality-r02",
    "benchmark-published-r02": "clip_dsanet/published-head-score/benchmark-r02",
    "pipeline-exclusive-r02": "clip_dsanet/published-head-score/pipeline-exclusive-r02",
    "test-campaign-r01": "clip_dsanet/historical-attempt/test-campaign-r01",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def plan(asset_root: Path, inventory: Path) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}

    def add(destination: Path, source: Path, role: str) -> None:
        key = str(destination)
        record = {"destination": key, "source": str(source), "role": role}
        if key in rows and rows[key]["source"] != record["source"]:
            raise ValueError(f"collision: {key}")
        rows[key] = record

    for item in read_jsonl(inventory / "runs.jsonl"):
        source = Path(str(item["source"]["path"]))
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        add(asset_root / "experiments" / "_discovered" / str(item["family"])
            / f"{digest}-{source.name}", source, "discovered_unverified")
        if item["family"] == "video_vit" and source.name in KIMI_CATEGORIES:
            add(asset_root / "experiments" / KIMI_CATEGORIES[source.name] / "source",
                source, "curated_campaign_root")
        elif item["family"] == "clip_dsanet" and source.name in DSANET_CATEGORIES:
            add(asset_root / "experiments" / DSANET_CATEGORIES[source.name] / "source",
                source, "curated_campaign_root")

    for item in read_jsonl(inventory / "code_snapshots.jsonl"):
        source = Path(str(item["source"]["path"]))
        add(asset_root / "provenance" / "code" / "execution-snapshots" / source.name,
            source, "execution_snapshot_identity_pending")

    for item in read_jsonl(inventory / "datasets.jsonl"):
        source = Path(str(item["source"]["path"]))
        family = "ucf_crime" if source.name.startswith("UCF-Crime") else "xd_violence"
        add(asset_root / "datasets" / family / "_source_candidates" / source.name,
            source, "source_candidate_unverified")

    return [rows[key] for key in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asset_root = Path(args.asset_root).resolve()
    records = plan(asset_root, Path(args.inventory))
    for record in records:
        destination = Path(record["destination"])
        source = Path(record["source"])
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"existing link points elsewhere: {destination}")
        elif destination.exists():
            raise ValueError(f"destination already exists: {destination}")
    if args.apply:
        output = Path(args.output)
        if output.exists():
            parser.error(f"output already exists: {output}")
        for record in records:
            destination = Path(record["destination"])
            if not destination.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(record["source"], destination, target_is_directory=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                          for row in records), encoding="utf-8")
    print(json.dumps({"views": len(records), "applied": args.apply}, sort_keys=True))


if __name__ == "__main__":
    main()
