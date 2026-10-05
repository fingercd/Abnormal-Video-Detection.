"""创建不含大模型/真实数据的 train+evaluate 编排冒烟夹具。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from vadbench.artifacts import PredictionRecord
from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
    write_manifest_jsonl,
)
from vadbench.features import FeatureStore, compute_encoder_fingerprint


def create_fixture(root: Path) -> dict[str, str]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    feature_store = FeatureStore(root / "features")
    fingerprint = compute_encoder_fingerprint({"adapter": "pipeline-smoke", "version": 1})

    train_records: list[VideoManifestRecord] = []
    for index in range(4):
        anomaly = index >= 2
        video_id = f"train-{index}"
        train_records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=f"{'Anomaly' if anomaly else 'Normal'}/{video_id}.mp4",
                split="train",
                category="Abuse" if anomaly else "Normal",
                is_anomaly=anomaly,
            )
        )
        for clip_index in range(2):
            value = (1.0 if anomaly else -1.0) + clip_index * 0.1
            feature_store.write(
                video_id=video_id,
                clip_id=f"{video_id}:segment-{clip_index}",
                clip_index=clip_index,
                encoder_fingerprint=fingerprint,
                features=np.full((1, 4), value, dtype=np.float32),
                pooled=np.full((4,), value, dtype=np.float32),
                start_s=float(clip_index * 2),
                end_s=float((clip_index + 1) * 2),
                frame_start=clip_index * 2,
                frame_end=(clip_index + 1) * 2,
                metadata={"source": {"split": "train", "is_anomaly": anomaly}},
            )

    test_records = (
        VideoManifestRecord(
            video_id="test-normal",
            path="Normal/test-normal.mp4",
            split="test",
            category="Normal",
            is_anomaly=False,
            num_frames=4,
            fps=1.0,
        ),
        VideoManifestRecord(
            video_id="test-anomaly",
            path="Abuse/test-anomaly.mp4",
            split="test",
            category="Abuse",
            is_anomaly=True,
            num_frames=4,
            fps=1.0,
            annotations=(
                SupervisionAnnotation(
                    scope="frame",
                    label="Abuse",
                    is_anomaly=True,
                    span=TemporalSpan(start=2, end=4, unit="frame"),
                ),
            ),
        ),
    )
    train_manifest = write_manifest_jsonl(train_records, root / "train.jsonl")
    test_manifest = write_manifest_jsonl(test_records, root / "test.jsonl")

    predictions = (
        PredictionRecord("pipeline-smoke", "test-normal", "normal-0", 0, 0, 2, 0.1, 0, 2),
        PredictionRecord("pipeline-smoke", "test-normal", "normal-1", 1, 2, 4, 0.1, 2, 4),
        PredictionRecord("pipeline-smoke", "test-anomaly", "anomaly-0", 0, 0, 2, 0.1, 0, 2),
        PredictionRecord("pipeline-smoke", "test-anomaly", "anomaly-1", 1, 2, 4, 0.9, 2, 4),
    )
    prediction_path = root / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "dataset": {
            "name": "pipeline_smoke",
            "root": str(root / "videos"),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
        },
        "sampler": {"segments_per_video": 2},
        "encoder": {"adapter": "cached_features", "trainable": False},
        "streaming": {"enabled": False},
        "task": {"kind": "weak_mil", "supervision": "video", "pooling": "attention"},
        "training": {"batch_size": 2, "epochs": 1, "max_steps": 1, "seed": 7},
        "output": {"root": str(root / "outputs"), "run_name": "pipeline-smoke"},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "root": str(root),
        "config": str(config_path),
        "features": str(feature_store.root),
        "predictions": str(prediction_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(create_fixture(Path(args.output)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
