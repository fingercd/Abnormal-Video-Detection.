from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from vadbench.artifacts import (
    ArtifactStore,
    CacheTelemetryRecord,
    PredictionRecord,
    RunProvenance,
    record_stage,
)
from vadbench.features import compute_encoder_fingerprint


def test_stage_failure_keeps_input_hash_and_cache_parameters(tmp_path: Path) -> None:
    source = tmp_path / "manifest.jsonl"
    source.write_text("first", encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="model failure"),
        record_stage(
            tmp_path / "run",
            "extract",
            config={"kv_budget_tokens": 64, "api_token": "hidden"},
            inputs={"manifest": source},
        ),
    ):
        raise RuntimeError("model failure")
    artifact = next((tmp_path / "run/provenance/stages").glob("*.json"))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["inputs"]["manifest"]["sha256"]
    assert payload["config"]["kv_budget_tokens"] == 64
    assert payload["config"]["api_token"] == "<redacted>"
    from jsonschema import Draft202012Validator

    schema = json.loads((Path(__file__).parents[1] / "schemas/stage-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(payload)


def _fingerprint() -> str:
    return compute_encoder_fingerprint({"adapter": "synthetic"})


def test_artifact_store_builds_canonical_run_layout_and_redacts_secrets(tmp_path: Path) -> None:
    provenance = RunProvenance(
        run_id="smoke-001",
        command=("vadbench", "extract"),
        config={"seed": 7, "api_token": "must-not-leak"},
        dataset={"name": "UCF-Crime", "manifest_sha256": "abc"},
        encoder_fingerprint=_fingerprint(),
        git={"available": True, "commit": "deadbeef", "dirty": False},
        runtime={"python": "test"},
    )
    store = ArtifactStore.create(tmp_path, run_id="smoke-001", provenance=provenance)
    assert store.provenance_path == tmp_path / "runs" / "smoke-001" / "provenance" / "run.json"
    assert store.metrics_dir.is_dir()
    assert store.predictions_dir.is_dir()
    assert store.cache_telemetry_dir.is_dir()
    raw = store.provenance_path.read_text(encoding="utf-8")
    assert "must-not-leak" not in raw
    assert store.read_provenance().config["api_token"] == "<redacted>"


def test_metrics_are_atomic_strict_json_and_reject_dense_arrays(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", run_id="run")
    written = store.write_metrics({"frame_auc": 0.8123}, split="test", step=4)
    assert written["schema_version"] == "vadbench.metrics.v1"
    assert store.read_metrics()["metrics"]["frame_auc"] == 0.8123
    assert store.metrics_path.read_bytes().endswith(b"\n")
    store.append_metrics({"loss": 0.9}, split="train", step=0)
    store.append_metrics({"loss": 0.5}, split="train", step=1)
    history = list(store.iter_metric_history())
    assert [row["step"] for row in history] == [0, 1]
    assert [row["metrics"]["loss"] for row in history] == [0.9, 0.5]

    with pytest.raises(TypeError, match="NPZ/NPY"):
        store.write_metrics({"curve": np.zeros(4096, dtype=np.float32)})
    # The last known-good metrics file was not truncated by failed validation.
    assert store.read_metrics()["metrics"]["frame_auc"] == 0.8123


def test_predictions_and_cache_telemetry_round_trip_jsonl(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", run_id="run")
    prediction = PredictionRecord(
        run_id="run",
        video_id="Abuse001",
        clip_id="Abuse001:0003",
        clip_index=3,
        start_s=3.0,
        end_s=4.0,
        frame_start=90,
        frame_end=120,
        anomaly_score=0.92,
        predicted_label=1,
        ground_truth=1,
        encoder_fingerprint=_fingerprint(),
    )
    store.append_prediction(prediction)
    store.append_prediction(
        PredictionRecord(
            run_id="run",
            video_id="Normal001",
            clip_id="Normal001:0000",
            clip_index=0,
            start_s=0.0,
            end_s=1.0,
            anomaly_score=0.03,
        )
    )
    assert [record.anomaly_score for record in store.iter_predictions()] == [0.92, 0.03]
    assert store.predictions_path.read_bytes().endswith(b"\n")

    telemetry = CacheTelemetryRecord(
        run_id="run",
        encoder_fingerprint=_fingerprint(),
        video_id="Abuse001",
        clip_id="Abuse001:0003",
        mode="streaming",
        cache_type="token",
        cache_hit=True,
        input_tokens=128,
        reused_tokens=96,
        output_tokens=32,
        cache_bytes=65536,
        encode_ms=12.5,
    )
    store.append_cache_telemetry(telemetry)
    loaded = list(store.iter_cache_telemetry())
    assert loaded == [telemetry]
    assert json.loads(store.cache_telemetry_path.read_text(encoding="utf-8"))["reuse_ratio"] == 0.75

    visual_memory = CacheTelemetryRecord(
        run_id="run",
        encoder_fingerprint=_fingerprint(),
        video_id="Abuse001",
        clip_id="Abuse001:0004",
        mode="streaming",
        cache_type="visual_memory",
        cache_hit=True,
        input_tokens=64,
        reused_tokens=32,
        output_tokens=32,
        cache_bytes=32768,
        encode_ms=8.0,
    )
    store.append_cache_telemetry(visual_memory)
    assert list(store.iter_cache_telemetry())[-1] == visual_memory


def test_prediction_validation_preserves_temporal_invariants() -> None:
    with pytest.raises(ValueError, match="end_s"):
        PredictionRecord(
            run_id="run",
            video_id="video",
            clip_id="clip",
            clip_index=0,
            start_s=2.0,
            end_s=1.0,
            anomaly_score=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        PredictionRecord(
            run_id="run",
            video_id="video",
            clip_id="clip",
            clip_index=0,
            start_s=0.0,
            end_s=1.0,
            anomaly_score=float("nan"),
        )


def test_prediction_jsonl_append_is_concurrent_writer_safe(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run", run_id="run")

    def append(index: int) -> None:
        store.append_prediction(
            PredictionRecord(
                run_id="run",
                video_id=f"video-{index % 2}",
                clip_id=f"clip-{index}",
                clip_index=index,
                start_s=float(index),
                end_s=float(index + 1),
                anomaly_score=index / 10,
            )
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(20)))
    records = list(store.iter_predictions())
    assert len(records) == 20
    assert {record.clip_id for record in records} == {f"clip-{index}" for index in range(20)}


def test_prediction_rows_validate_against_published_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    store = ArtifactStore(tmp_path / "run", run_id="run")
    store.append_prediction(
        PredictionRecord(
            run_id="run",
            video_id="video",
            clip_id="clip",
            clip_index=0,
            start_s=0.0,
            end_s=1.0,
            anomaly_score=0.5,
        )
    )
    schema_path = Path(__file__).parents[1] / "schemas" / "prediction-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    row = json.loads(store.predictions_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(row)
