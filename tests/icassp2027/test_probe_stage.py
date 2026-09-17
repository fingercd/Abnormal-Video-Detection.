from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.contracts import ClipBatch
from vadbench.paper.profile import PaperProject
from vadbench.paper.stages import observe_clip, run_probe
from vadbench.token_reduction.bridges.geometry import TubeletGeometry


class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 4, kernel_size=2, stride=2)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.qkv = nn.Linear(4, 12)
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        a = self.attn_drop(((q @ k.transpose(-1, -2)) / 2).unsqueeze(1).softmax(-1))
        return self.proj(a.squeeze(1) @ v)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FixtureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = Patch()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.fc_norm = nn.LayerNorm(4)
        self.pos_embed = torch.zeros(1, 8, 4)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return x


class FixtureAdapter:
    def __init__(self):
        self.encoder = FixtureEncoder()

    def encode(self, batch):
        assert batch.video_ids == ("sample-0",)
        assert not {"category", "is_anomaly", "video_path"} & batch.metadata.keys()
        x = torch.tensor(batch.frames, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        features = self.encoder(x)
        return SimpleNamespace(features=features, pooled=self.encoder.fc_norm(features.mean(1)))


def test_end_to_end_observer_and_identity_parity_and_coordinates():
    adapter = FixtureAdapter()
    adapter.encoder.blocks[0].eval()
    initial_modes = [module.training for module in adapter.encoder.modules()]
    batch = ClipBatch(
        frames=np.zeros((1, 4, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.array([[0.0, 0.2, 0.4, 0.6]]),
        frame_indices=np.array([[0, 2, 4, 6]]),
        valid_mask=np.ones((1, 4), dtype=bool),
        video_ids=("contains-label",),
        metadata={"category": "secret", "is_anomaly": True, "video_path": "label.mp4"},
    )
    result = observe_clip(
        adapter,
        "videomaev2",
        batch,
        {"depths": [0.5, 1.0], "max_records": 64, "max_tokens": 8, "max_queries": 4},
    )
    receipt = result["architecture"]
    assert receipt["grid"] == [2, 2, 2]
    assert receipt["source_frame_indices"] == [[[0, 2], [4, 6]]]
    assert receipt["has_cls"] is False and receipt["flatten_verified"]
    assert all(delta == 0 for delta in receipt["parity"].values())
    assert not receipt["reduction_ready"]
    assert adapter.encoder.training  # caller's mode restored
    assert [module.training for module in adapter.encoder.modules()] == initial_modes
    with pytest.raises(ValueError, match="source frame indices"):
        observe_clip(
            adapter,
            "videomaev2",
            replace(batch, frame_indices=None),
            {"depths": [1.0], "max_records": 64, "max_tokens": 8, "max_queries": 4},
        )
    assert [module.training for module in adapter.encoder.modules()] == initial_modes
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in adapter.encoder.modules())
    assert {row["probe_id"] for item in result["observations"] for row in item.rows} == {
        "P01",
        "P02",
        "P04",
        "P07",
        "P10",
        "P11",
        "P13",
        "P16",
    }


def test_geometry_excludes_partial_padding_and_rejects_changed_flatten_order():
    patch = Patch()
    geometry = TubeletGeometry(patch, [[0, 2, 4, 4]], [[True, True, True, False]])
    with geometry:
        patch(torch.zeros(1, 3, 4, 4, 4))
    assert geometry.layout.valid_token_counts.tolist() == [4]
    assert geometry.layout.original_token_ids.tolist() == [[0, 1, 2, 3, -1, -1, -1, -1]]

    class WrongPatch(Patch):
        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2).flip(1)

    wrong = WrongPatch()
    geometry = TubeletGeometry(wrong, [[0, 2, 4, 6]], [[True] * 4])
    with pytest.raises(ValueError, match="flattened"), geometry:
        wrong(torch.randn(1, 3, 4, 4, 4))
    assert not wrong._forward_hooks and not wrong.proj._forward_hooks


def test_probe_execution_reuses_manifest_video_path_and_joins_labels_only_after_forward(
    tmp_path, monkeypatch
):
    cv2 = pytest.importorskip("cv2")
    import vadbench.orchestration as orchestration
    from vadbench.data.manifest import VideoManifestRecord
    from vadbench.registry import ENCODER_REGISTRY

    movie = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    assert writer.isOpened()
    for frame in range(24):
        writer.write(np.full((16, 16, 3), frame * 4, dtype=np.uint8))
    writer.release()
    manifest = tmp_path / "train.jsonl"
    record = VideoManifestRecord(
        video_id="labelled",
        path="clip.avi",
        split="train",
        category="test-fixture",
        is_anomaly=True,
    )
    manifest.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text(
        json.dumps(
            {
                "video_id": "labelled",
                "clip_id": "labelled:segment-00",
                "official_split": "train",
                "partition": "fit",
                "role": "debug",
                "weak_label": 1,
                "label_source": "video_weak",
            }
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
    plan = {
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
                "sampling": {"windows_per_video": 1, "frame_stride": 2},
                "observation": {
                    "depths": [1.0],
                    "max_records": 64,
                    "max_tokens": 8,
                    "max_queries": 4,
                    "probes": ["P01", "P10", "P13"],
                },
            },
        },
    }
    # Synthetic adapter/checkpoint identity is a test fixture, never a run claimed
    # as real-weight evidence. Video decoding and manifest/cohort checks are real.
    monkeypatch.setattr(
        orchestration,
        "encoder_identity",
        lambda *a, **k: {"checkpoint": {"sha256": {"fixture": "test"}}},
    )
    monkeypatch.setattr(ENCODER_REGISTRY, "create", lambda *a, **k: FixtureAdapter())
    summary = run_probe(project, plan)
    assert summary["status"] == "completed" and summary["clips"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "outputs" / summary["run_id"] / "probe_summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows and all(row["weak_label"] == 1 and row["video_id"] == "labelled" for row in rows)
    assert {row["probe_id"] for row in rows} == {"P01", "P10", "P13"}
    assert summary["research_conclusions"] is None
