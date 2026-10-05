"""Assemble the frozen six-cell UR-DMU evaluation matrix.

The command is intentionally fail-closed.  It consumes only completed
post-freeze quality exports and their provenance-bound prediction/feature
artifacts; it never selects a method or reads labels to make a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ENCODERS = ("videomaev2", "videomae", "timesformer")
DATASETS = ("ucf_crime", "xd_violence")
METHODS = ("group_uniform", "group_random", "pair_select")
SHORT_DATASET = {"ucf_crime": "ucf", "xd_violence": "xd"}
EXPECTED_VIDEOS = {"ucf_crime": 290, "xd_violence": 800}

METHOD_RUN_UCF = {
    "videomaev2": {"group_uniform": "group_uniform-0p60-r03", "group_random": "group_random-0p60-r03", "pair_select": "pair_select-0p60-r03"},
    "videomae": {"group_uniform": "group_uniform-0p60-r02", "group_random": "group_random-0p60-r02", "pair_select": "pair_select-0p60-r02"},
    "timesformer": {"group_uniform": "group_uniform-0p60-r02", "group_random": "group_random-0p60-r02", "pair_select": "pair_select-0p60-r02"},
}
METHOD_RUN_XD = {
    "videomaev2": {"group_uniform": "group_uniform-0p60-r01", "group_random": "group_random-0p60-r01", "pair_select": "pair_select-0p60-r03"},
    "videomae": {"group_uniform": "group_uniform-0p60-r01", "group_random": "group_random-0p60-r01", "pair_select": "pair_select-0p60-r01"},
    "timesformer": {"group_uniform": "group_uniform-0p60-r01", "group_random": "group_random-0p60-r01", "pair_select": "pair_select-0p60-r01"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"required artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    return path


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _run_id(dataset: str, encoder: str, method: str) -> str:
    table = METHOD_RUN_UCF if dataset == "ucf_crime" else METHOD_RUN_XD
    return table[encoder][method]


def _dense_view(base: Path, dataset: str, encoder: str) -> Path:
    if dataset == "ucf_crime":
        name = "dense-test-view-v2" if encoder == "videomae" else "dense-test-view"
    else:
        name = "xd-dense-test-view"
    return base / "merged" / encoder / name


def _head(base: Path, dataset: str, encoder: str) -> Path:
    if dataset == "ucf_crime":
        return base / "training-full" / encoder / "formal-dense-s0"
    if encoder == "videomaev2":
        return base / "training-full" / encoder / "xd-formal-dense-s0-r02"
    return base / "training-full" / encoder / "xd-formal-dense-s0"


def _feature_contract(base: Path, dataset: str, encoder: str, method: str) -> tuple[Path, Path]:
    if method == "dense":
        view = _dense_view(base, dataset, encoder)
        return view / "index.jsonl", view / "merge-contract.json"
    split = SHORT_DATASET[dataset]
    run = base / "compressed-test" / encoder / split / _run_id(dataset, encoder, method)
    return run / "features" / "train" / "index.jsonl", run / "extraction-contract.json"


def _prediction_paths(base: Path, dataset: str, encoder: str, method: str) -> tuple[Path, Path]:
    root = base / "urdmu-official-predictions-r08" / dataset / encoder / method
    return root / "predictions.jsonl", root / "resolved.json"


def _token_stats(index_path: Path, *, method: str) -> dict[str, Any]:
    """Stream a FeatureStore index and summarize actual token execution."""

    rows = 0
    videos: set[str] = set()
    native: list[float] = []
    retained: list[float] = []
    ratios: list[float] = []
    plugin_ms: list[float] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid FeatureStore index JSON: {index_path}:{line_number}") from exc
            rows += 1
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise RuntimeError(f"FeatureStore row lacks video_id: {index_path}:{line_number}")
            videos.add(video_id)
            encoder_output = row.get("metadata", {}).get("encoder_output", {})
            reduction = row.get("metadata", {}).get("reduction_execution") or {}
            native_tokens = reduction.get("native_input_tokens", encoder_output.get("token_count"))
            retained_tokens = reduction.get("gathered_tokens", encoder_output.get("token_count"))
            if not isinstance(native_tokens, (int, float)) or not isinstance(retained_tokens, (int, float)):
                raise RuntimeError(f"token counts are missing: {index_path}:{line_number}")
            if native_tokens <= 0 or retained_tokens <= 0 or retained_tokens > native_tokens:
                raise RuntimeError(f"invalid token counts: {index_path}:{line_number}")
            native.append(float(native_tokens))
            retained.append(float(retained_tokens))
            ratios.append(float(retained_tokens) / float(native_tokens))
            overhead = reduction.get("plugin_overhead_ms") or {}
            plugin_value = sum(float(overhead.get(key, 0.0)) for key in ("gather_ms", "transform_ms"))
            if not math.isfinite(plugin_value) or plugin_value < 0:
                raise RuntimeError(f"invalid plugin overhead: {index_path}:{line_number}")
            plugin_ms.append(plugin_value)
    if not rows:
        raise RuntimeError(f"FeatureStore index is empty: {index_path}")
    return {
        "index_sha256": _sha256(index_path),
        "clips": rows,
        "videos": len(videos),
        "native_tokens": {"min": min(native), "median": _percentile(native, 50), "max": max(native)},
        "retained_tokens": {"min": min(retained), "median": _percentile(retained, 50), "max": max(retained)},
        "retained_ratio": {"min": min(ratios), "mean": sum(ratios) / len(ratios), "median": _percentile(ratios, 50), "max": max(ratios)},
        "plugin_overhead_ms": {"mean": sum(plugin_ms) / len(plugin_ms), "median": _percentile(plugin_ms, 50), "p90": _percentile(plugin_ms, 90), "max": max(plugin_ms)},
        "method": method,
        "execution_source": "per-clip FeatureStore reduction_execution metadata",
    }


def _provenance(base: Path, dataset: str, encoder: str, method: str, method_contract: Path, freeze: Path) -> dict[str, Any]:
    prediction, resolved = _prediction_paths(base, dataset, encoder, method)
    feature_index, feature_contract = _feature_contract(base, dataset, encoder, method)
    head = _head(base, dataset, encoder)
    checkpoint = _json(head / "result.json").get("checkpoint_path")
    checkpoint_path = Path(checkpoint) if isinstance(checkpoint, str) else head / "checkpoints" / "final.pt"
    return {
        "prediction": {"path": str(_require_file(prediction, label="prediction JSONL")), "sha256": _sha256(prediction)},
        "prediction_resolved": {"path": str(_require_file(resolved, label="prediction resolved receipt")), "sha256": _sha256(resolved)},
        "feature_index": {"path": str(_require_file(feature_index, label="feature index")), "sha256": _sha256(feature_index)},
        "feature_contract": {"path": str(_require_file(feature_contract, label="feature contract")), "sha256": _sha256(feature_contract)},
        "head_checkpoint": {"path": str(_require_file(checkpoint_path, label="head checkpoint")), "sha256": _sha256(checkpoint_path)},
        "head_result": {"path": str(_require_file(head / "result.json", label="head result")), "sha256": _sha256(head / "result.json")},
        "training_qa": {"path": str(_require_file(head / "training_qa.json", label="training QA")), "sha256": _sha256(head / "training_qa.json")},
        "method_contract": {"path": str(_require_file(method_contract, label="method contract")), "sha256": _sha256(method_contract)},
        "freeze_receipt": {"path": str(_require_file(freeze, label="freeze receipt")), "sha256": _sha256(freeze)},
    }


def assemble(*, base: Path, quality_root: Path, method_contract: Path, freeze: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for encoder in ENCODERS:
            quality: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                path = quality_root / dataset / encoder / f"{method}.json"
                quality[method] = _json(_require_file(path, label="quality export"))
                metric = quality[method].get("metric")
                if not isinstance(metric, dict) or metric.get("status") != "available":
                    raise RuntimeError(f"quality export is not available: {path}")
                if metric.get("n_videos") != EXPECTED_VIDEOS[dataset]:
                    raise RuntimeError(f"quality export video coverage mismatch: {path}")
            dense_values = [float(quality[method]["metric"]["dense_value"]) for method in METHODS]
            if max(dense_values) - min(dense_values) > 1e-12:
                raise RuntimeError(f"dense metric differs across paired exports: {dataset}/{encoder}")
            methods: dict[str, Any] = {}
            for method in METHODS:
                quality_payload = quality[method]
                method_index, method_contract_path = _feature_contract(base, dataset, encoder, method)
                method_contract_doc = _json(_require_file(method_contract_path, label="method extraction contract"))
                if method_contract_doc.get("status") != "ready" or method_contract_doc.get("test_only") is not True:
                    raise RuntimeError(f"method extraction contract is not ready/test-only: {method_contract_path}")
                token_execution = _token_stats(method_index, method=method)
                if token_execution["clips"] != method_contract_doc.get("clips"):
                    raise RuntimeError(f"FeatureStore clip count differs from extraction contract: {method_contract_path}")
                if token_execution["videos"] != EXPECTED_VIDEOS[dataset]:
                    raise RuntimeError(f"FeatureStore video coverage differs from extraction contract: {method_contract_path}")
                methods[method] = {
                    "quality": quality_payload["metric"],
                    "xd_average_precision": quality_payload.get("average_precision"),
                    "video_level": quality_payload.get("video_level"),
                    "token_execution": token_execution,
                    "quality_export": {
                        "path": str(quality_root / dataset / encoder / f"{method}.json"),
                        "sha256": _sha256(quality_root / dataset / encoder / f"{method}.json"),
                    },
                    "provenance": _provenance(base, dataset, encoder, method, method_contract, freeze),
                }
            dense_index, dense_contract = _feature_contract(base, dataset, encoder, "dense")
            dense_contract_doc = _json(_require_file(dense_contract, label="dense merge contract"))
            if dense_contract_doc.get("status") != "ready" or dense_contract_doc.get("complete") is not True:
                raise RuntimeError(f"dense view is not ready: {dense_contract}")
            if dense_contract_doc.get("videos") != EXPECTED_VIDEOS[dataset]:
                raise RuntimeError(f"dense video coverage mismatch: {dense_contract}")
            cells.append({
                "dataset": dataset,
                "encoder": encoder,
                "videos": EXPECTED_VIDEOS[dataset],
                "dense": {
                    "metric": {"value": dense_values[0], "metric": quality["pair_select"]["metric"]["metric"], "definition": quality["pair_select"]["metric"].get("metric_definition")},
                    "feature_contract": {"path": str(dense_contract), "sha256": _sha256(dense_contract)},
                    "feature_index": {"path": str(dense_index), "sha256": _sha256(dense_index)},
                    "clips": dense_contract_doc.get("clips"),
                },
                "methods": methods,
                "freeze": {"path": str(freeze), "sha256": _sha256(freeze)},
            })
    return {
        "schema": "icassp2027.urdmu-six-cell-matrix/v1",
        "status": "completed",
        "scope": {"encoders": list(ENCODERS), "datasets": list(DATASETS), "methods": list(METHODS), "mode": "direct_insert", "test_scores_read_only_after_freeze": True},
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--quality-root", required=True, type=Path)
    parser.add_argument("--method-contract", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    matrix = assemble(base=args.base.resolve(), quality_root=args.quality_root.resolve(), method_contract=args.method_contract.resolve(), freeze=args.freeze.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "cells": len(matrix["cells"]), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
