from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vadbench.data.manifest import VideoManifestRecord
from vadbench.paper.profile import PaperProject, output_path
from vadbench.paper.stages import _probe_shard_path, _write_probe_shard, run_probe


class _Observation:
    def to_rows(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "status": "available",
                "probe_id": "P01",
                "layer_index": 0,
                "encoder_id": kwargs["encoder_id"],
                "clip_id": kwargs["clip_ids"][0],
                "video_id": kwargs["video_ids"][0],
                "partition": kwargs["partitions"][0],
            }
        ]


def _project_and_plan(tmp_path: Path) -> tuple[PaperProject, dict[str, Any]]:
    cv2 = pytest.importorskip("cv2")
    movie = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"MJPG"), 10, (8, 8))
    assert writer.isOpened()
    for frame in range(12):
        writer.write(np.full((8, 8, 3), frame * 10, dtype=np.uint8))
    writer.release()
    record = VideoManifestRecord(
        video_id="fixture-video",
        path="source.avi",
        split="train",
        category="fixture",
        is_anomaly=False,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text(
        "\n".join(
            json.dumps(
                {
                    "video_id": "fixture-video",
                    "clip_id": f"fixture-video:segment-{index:02d}",
                    "official_split": "train",
                    "partition": "fit",
                    "role": "debug",
                    "weak_label": 0,
                    "label_source": "video_weak",
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("fixture: true\n", encoding="utf-8")
    project = PaperProject(
        profile_path,
        tmp_path,
        {"output_root": "outputs"},
        {"annotation_policy": "weak_development"},
        {},
        {},
    )
    return project, {
        "blockers": [],
        "resolved": {
            "cohort": str(cohort),
            "manifest": str(manifest),
            "dataset_root": str(tmp_path),
            "encoders": {"videomaev2": {"constructor": {"num_frames": 4}}},
            "suite": {
                "partition": "fit",
                "role": "debug",
                "max_videos": 1,
                "sampling": {"windows_per_video": 2, "frame_stride": 1},
                "observation": {
                    "depths": [1.0],
                    "max_records": 8,
                    "max_tokens": 8,
                    "max_queries": 2,
                    "probes": ["P01"],
                },
            },
        },
    }


def _patch_probe_dependencies(monkeypatch: pytest.MonkeyPatch, *, fail_second: bool) -> None:
    import vadbench.orchestration as orchestration
    import vadbench.paper.stages as stages
    from vadbench.registry import ENCODER_REGISTRY

    monkeypatch.setattr(
        orchestration,
        "encoder_identity",
        lambda *args, **kwargs: {"checkpoint": {"sha256": {"fixture": "streaming"}}},
    )
    monkeypatch.setattr(ENCODER_REGISTRY, "create", lambda *args, **kwargs: object())
    calls = 0

    def observe(_adapter: Any, _name: str, _clip: Any, _observation: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if fail_second and calls == 2:
            raise RuntimeError("intentional second-clip crash")
        return {
            "architecture": {"block_count": 1, "attention_backend": "fixture-native"},
            "observations": [_Observation()],
        }

    monkeypatch.setattr(stages, "observe_clip", observe)


def _single_output(project: PaperProject) -> Path:
    runs = list(output_path(project.root, project.profile["output_root"]).iterdir())
    assert len(runs) == 1
    return runs[0]


def test_probe_merges_two_clip_shards_streamingly_into_legacy_jsonl(tmp_path: Path, monkeypatch) -> None:
    project, plan = _project_and_plan(tmp_path)
    _patch_probe_dependencies(monkeypatch, fail_second=False)

    summary = run_probe(project, plan)
    output = output_path(project.root, project.profile["output_root"]) / summary["run_id"]

    assert summary["status"] == "completed"
    assert summary["clips"] == 2
    assert summary["probe_status_counts"] == {"available": 2}
    assert len((output / "architecture_receipts.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "input_controls.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "probe_summary.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress == {
        "run_id": summary["run_id"],
        "status": "completed",
        "expected_clips": 2,
        "completed_clips": 2,
        "video_count": 1,
        "completed_videos": 1,
    }
    shards = sorted((output / "output_shards").glob("*.json"))
    assert len(shards) == 2
    assert all(":" not in path.name and "fixture-video" not in path.name for path in shards)


def test_crash_keeps_committed_first_clip_without_final_summary(tmp_path: Path, monkeypatch) -> None:
    project, plan = _project_and_plan(tmp_path)
    _patch_probe_dependencies(monkeypatch, fail_second=True)

    with pytest.raises(RuntimeError, match="second-clip crash"):
        run_probe(project, plan)

    output = _single_output(project)
    assert len(list((output / "output_shards").glob("*.json"))) == 1
    assert not (output / "summary.json").exists()
    assert not (output / "architecture_receipts.jsonl").exists()
    assert not (output / "input_controls.jsonl").exists()
    assert not (output / "probe_summary.jsonl").exists()
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["completed_clips"] == 1
    assert progress["completed_videos"] == 0


def test_shard_path_uses_digest_and_refuses_overwrite(tmp_path: Path) -> None:
    path = _probe_shard_path(tmp_path, 0, "encoder", "../../unsafe:clip")
    assert path.parent.name == "output_shards"
    assert ":" not in path.name and "unsafe" not in path.name
    payload = {"schema_version": "vadbench.probe-shard.v1", "receipts": [], "controls": [], "rows": []}
    _write_probe_shard(path, payload)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        _write_probe_shard(path, payload)
