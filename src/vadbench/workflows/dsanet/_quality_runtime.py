"""Export paired DSANet frame quality from sealed scores and frame truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vadbench.workflows.quality_comparison import compare_paired_quality
from vadbench.workflows.urdmu_quality_export import _aligned_video, _prediction_rows, _truth_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth_id(dataset: str, source_id: str) -> str:
    if dataset == "ucf":
        return source_id
    return "xd-" + hashlib.sha256(source_id.casefold().encode()).hexdigest()


def convert(
    dataset: str, branch: str, scores_path: Path, truth: dict, output: Path
) -> dict:
    seen = set()
    rows = 0
    with scores_path.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for raw in source:
            record = json.loads(raw)
            video_id = truth_id(dataset, record["video_id"])
            if video_id not in truth or video_id in seen:
                raise ValueError(f"prediction video ID missing or repeated: {video_id}")
            seen.add(video_id)
            values = record[f"{branch}_scores"]
            frame_count = record["raw_frames"]
            if len(values) != record["feature_rows"] or len(values) != (frame_count + 15) // 16:
                raise ValueError(f"invalid feature and score coverage: {video_id}")
            if frame_count < truth[video_id]["num_frames"]:
                raise ValueError(f"prediction shorter than canonical frame truth: {video_id}")
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    raise ValueError(f"nonfinite score: {video_id}")
                interval = {
                    "video_id": video_id,
                    "frame_start": index * 16,
                    "frame_end": min(frame_count, (index + 1) * 16),
                    "anomaly_score": float(value),
                }
                target.write(json.dumps(interval, separators=(",", ":")) + "\n")
                rows += 1
    if seen != set(truth):
        raise ValueError(f"predictions do not cover all truth videos: {dataset}/{branch}")
    return {"videos": len(seen), "score_intervals": rows,
            "source_scores_sha256": sha256(scores_path), "sha256": sha256(output)}


def run(options) -> None:
    truth = _truth_rows(options.truth)
    expected = 290 if options.dataset == "ucf" else 800
    if len(truth) != expected:
        raise ValueError(f"{options.dataset} truth does not contain {expected} videos")
    options.output.mkdir(parents=True, exist_ok=False)
    conversion = {}
    score_runs = {}
    results = {}
    for branch in ("coarse", "refined"):
        interval_paths = {}
        for tier in ("dense", *options.tiers):
            source = options.scores_root / f"{options.dataset}-{tier}" / "predictions.jsonl"
            summary = json.loads(source.with_name("summary.json").read_text(encoding="utf-8"))
            if (summary["status"] != "completed" or summary["dataset"] != options.dataset
                    or summary["videos"] != expected):
                raise ValueError(f"scoring receipt incomplete: {options.dataset}/{tier}")
            score_runs[tier] = {
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "manifest_sha256": summary["manifest_sha256"],
                "score_file_sha256": sha256(source),
            }
            destination = options.output / f"{branch}-{tier}-intervals.jsonl"
            conversion[f"{branch}-{tier}"] = convert(
                options.dataset, branch, source, truth, destination
            )
            interval_paths[tier] = destination
        if len({item["checkpoint_sha256"] for item in score_runs.values()}) != 1:
            raise ValueError("dense and compressed scores use different head checkpoints")
        if len({item["manifest_sha256"] for item in score_runs.values()}) != 1:
            raise ValueError("dense and compressed scores use different test manifests")
        dense = _prediction_rows(interval_paths["dense"])
        for tier in options.tiers:
            method = _prediction_rows(interval_paths[tier])
            if set(dense) != set(method) or set(dense) != set(truth):
                raise ValueError(f"video identity mismatch: {branch}/{tier}")
            videos = [
                _aligned_video(video_id, truth[video_id], dense[video_id], method[video_id])
                for video_id in sorted(truth)
            ]
            sources = {
                "truth": sha256(options.truth),
                "dense_intervals": sha256(interval_paths["dense"]),
                "method_intervals": sha256(interval_paths[tier]),
            }
            results[f"{branch}-{tier}"] = {
                metric: compare_paired_quality(videos, metric=metric, source_sha256=sources)
                for metric in ("frame_roc_auc", "frame_pr_auc", "frame_ap")
            }
            print(f"exported {options.dataset}/{branch}/{tier}", flush=True)
    payload = {
        "status": "completed",
        "dataset": options.dataset,
        "seed": options.seed,
        "checkpoint_origin": options.checkpoint_origin,
        "primary_branch": "coarse",
        "primary_metric": "frame_roc_auc" if options.dataset == "ucf" else "frame_ap",
        "truth_sha256": sha256(options.truth),
        "score_runs": score_runs,
        "conversion": conversion,
        "results": results,
    }
    (options.output / "quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "dataset": options.dataset,
                      "output": str(options.output / "quality.json")}))

