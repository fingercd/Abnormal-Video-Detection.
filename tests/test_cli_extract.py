from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

import vadbench.cli as cli
from vadbench.config import load_experiment, load_yaml
from vadbench.contracts import ClipBatch, EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl


class FakeAdapter:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_training=False,
        fixed_num_frames=2,
        min_frames=2,
        max_frames=2,
    )

    def encode(self, batch, train=False):
        assert not train
        timeline = TokenTimeline(
            start_s=np.asarray([[0.0]], dtype=np.float32),
            end_s=np.asarray([[1.0]], dtype=np.float32),
            valid_mask=np.asarray([[True]]),
            source_frame_start=np.asarray([[0]], dtype=np.int64),
            source_frame_end=np.asarray([[2]], dtype=np.int64),
        )
        return EncoderOutput(
            features=np.ones((batch.batch_size, 1, 2), dtype=np.float32),
            pooled=np.ones((batch.batch_size, 2), dtype=np.float32),
            timeline=timeline,
        )


def test_cli_extract_builds_feature_index(tmp_path: Path, monkeypatch, capsys) -> None:
    record = VideoManifestRecord(
        video_id="normal",
        path="Normal/normal.mp4",
        split="train",
        category="Normal",
        is_anomaly=False,
    )
    manifest = write_manifest_jsonl((record,), tmp_path / "train.jsonl")
    config = {
        "schema_version": 1,
        "dataset": {
            "root": str(tmp_path),
            "train_manifest": str(manifest),
            "test_manifest": str(manifest),
        },
        "sampler": {"segments_per_video": 1, "clip_frames": 2, "frame_stride": 1},
        "encoder": {"adapter": "fake", "checkpoint": "fake", "micro_batch_size": 1},
        "streaming": {"enabled": False},
        "task": {"supervision": "video"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "extract"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    batch = ClipBatch(
        frames=np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0.0, 1.0]], dtype=np.float32),
        video_ids=("normal",),
        frame_indices=np.asarray([[0, 1]], dtype=np.int64),
        metadata={"clip_ids": ["normal:0"], "clip_indices": [0]},
    )
    monkeypatch.setattr(
        cli,
        "create_encoder_from_experiment",
        lambda config, project_root: (
            FakeAdapter(),
            {"adapter": "fake", "constructor": {}, "checkpoint": {}},
        ),
    )
    monkeypatch.setattr(cli, "iter_fixed_segment_batches", lambda *args, **kwargs: iter((batch,)))

    assert cli.main(["extract", "-c", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["feature_records"] == 1
    assert Path(payload["feature_index"]).is_file()


def test_reference_checkpoint_path_and_id_reach_real_extraction_engine(
    tmp_path, monkeypatch, capsys
):
    root = Path(__file__).resolve().parents[1]
    config = load_experiment(root / "configs/experiments/ucf_videomaev2_weak.yaml")
    definition = load_yaml(root / "configs/encoders/videomaev2-base.yaml")
    weight = tmp_path / "weights"
    weight.mkdir()
    (weight / "model.bin").write_bytes(b"tiny-reference")
    digest = hashlib.sha256(b"tiny-reference").hexdigest()
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry/checkpoints.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoints": {
                    "tiny": {
                        "adapter": "videomaev2",
                        "source": "huggingface",
                        "repo_id": "test/tiny",
                        "revision": "0" * 40,
                        "license": "mit",
                        "sha256": {"model.bin": digest},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    definition["checkpoint"].update(id="tiny", local_path=str(weight))
    definition["constructor"]["model_name"] = str(weight)
    definition.pop("upstream_lock")
    config["encoder"]["checkpoint"] = "tiny"
    manifest = write_manifest_jsonl(
        (
            VideoManifestRecord(
                video_id="normal",
                path="normal.mp4",
                split="train",
                category="Normal",
                is_anomaly=False,
            ),
        ),
        tmp_path / "manifest.jsonl",
    )
    config["dataset"]["train_manifest"] = str(manifest)
    config["output"] = {"root": str(tmp_path / "runs"), "run_name": "reference"}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    batch = ClipBatch(
        frames=np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0.0, 1.0]]),
        video_ids=("normal",),
        frame_indices=np.asarray([[0, 1]]),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "create_encoder_from_experiment", lambda *a, **k: (FakeAdapter(), definition)
    )
    monkeypatch.setattr(cli, "iter_fixed_segment_batches", lambda *a, **k: iter((batch,)))
    assert cli.main(["extract", "-c", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["feature_records"] == 1
    attempt = next((tmp_path / "runs/reference/provenance/stages").glob("*.json"))
    assert json.loads(attempt.read_text())["status"] == "completed"
