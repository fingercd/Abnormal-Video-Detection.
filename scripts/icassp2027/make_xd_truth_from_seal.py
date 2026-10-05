"""Create an XD canonical-prefix frame-truth JSONL from a sealed audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(values: np.ndarray) -> list[dict[str, int]]:
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("GT slice must be finite binary one-dimensional values")
    result: list[dict[str, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            result.append({"start": start, "end": index, "label": int(values[start])})
            start = index
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--canonical-metadata", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt_path = Path(args.receipt).resolve()
    manifest_path = Path(args.test_manifest).resolve()
    metadata_path = Path(args.canonical_metadata).resolve()
    gt_path = Path(args.ground_truth).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "sealed" or len(receipt.get("records") or []) != 800:
        raise ValueError("receipt is not the sealed 800-video XD audit")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != 800:
        raise ValueError("canonical metadata does not contain 800 records")
    by_feature = {row["feature_name"]: row for row in records}
    gt = np.load(gt_path, allow_pickle=False)
    if tuple(gt.shape) != (int(metadata["gt_reference_length"]),):
        raise ValueError("GT shape differs from canonical metadata")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["video_id"]: row for row in manifest}
    if len(manifest) != 800 or len(by_id) != 800:
        raise ValueError("XD test manifest is not the complete 800-video cohort")
    output_rows = []
    for source in receipt["records"]:
        feature = by_feature.get(source["feature_name"])
        if feature is None or feature["video_id"] != source["feature_name"].rsplit("__", 1)[0]:
            # Canonical metadata uses original feature names; the receipt binds
            # the hashed video identity.  Match by the feature name only and
            # preserve the receipt's sealed identity.
            feature = by_feature.get(source["feature_name"])
        if feature is None:
            raise ValueError(f"receipt feature is absent from canonical metadata: {source['feature_name']}")
        video_id = source["video_id"]
        row = by_id.get(video_id)
        if row is None:
            raise ValueError(f"receipt video is absent from test manifest: {video_id}")
        start, end = int(feature["gt_start"]), int(feature["gt_end_exclusive"])
        prefix = gt[start:end]
        output_rows.append({
            "video_id": video_id,
            "weak_label": int(bool(row["is_anomaly"])),
            "num_frames": int(end - start),
            "source_num_frames": int(source["num_frames"]),
            "truncate_predictions": True,
            "intervals": _runs(prefix),
        })
    output_rows.sort(key=lambda row: row["video_id"])
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    receipt_out = {
        "schema": "icassp2027.xd-frame-truth-from-seal/v1",
        "status": "completed",
        "videos": len(output_rows),
        "output": str(output),
        "output_sha256": _sha(output),
        "raw_coordinate_receipt_sha256": _sha(receipt_path),
        "test_manifest_sha256": _sha(manifest_path),
        "canonical_metadata_sha256": _sha(metadata_path),
        "ground_truth_sha256": _sha(gt_path),
        "mapping": receipt.get("mapping"),
        "model_scores_read": False,
    }
    output.with_name("truth-receipt.json").write_text(json.dumps(receipt_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
