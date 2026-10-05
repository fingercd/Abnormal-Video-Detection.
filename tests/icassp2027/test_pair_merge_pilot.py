from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vadbench.contracts import ClipBatch, EncoderOutput, TokenTimeline
from vadbench.data.manifest import VideoManifestRecord

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/icassp2027/run_pair_merge_pilot.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("pair_merge_pilot_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


def test_pair_merge_pilot_dry_run_never_initializes_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", "videomaev2"])
    monkeypatch.setattr(
        pilot,
        "load_project",
        lambda _path: SimpleNamespace(encoder=lambda _name: {"definition": {"constructor": {"num_frames": 64}}}),
    )
    monkeypatch.setattr(pilot.ENCODER_REGISTRY, "create", lambda *_args, **_kwargs: pytest.fail("dry run initialized model"))

    assert pilot.main() == 0
    output = capsys.readouterr().out
    assert '"engineering_only_pair_merge": true' in output
    assert '"execute": false' in output


@pytest.mark.parametrize("encoder", ["videomaev2", "timesformer", "videomae", "vjepa2"])
def test_pair_merge_pilot_parser_keeps_only_active_encoders(monkeypatch, encoder) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", encoder])
    assert pilot.parse_args().encoder == encoder


def _production_scale(tmp_path: Path, monkeypatch, *, fail_ordinal: int | None = None):
    cases = [
        {"video_id": f"video-{index:02d}", "weak_label": index < 13, "category_for_analysis_only": "analysis"}
        for index in range(26)
    ]
    records = {
        case["video_id"]: VideoManifestRecord(
            video_id=case["video_id"], path=f"{case['video_id']}.mp4", split="train", category="Normal", is_anomaly=False, num_frames=200, fps=10.0
        )
        for case in cases
    }
    monkeypatch.setattr(pilot.neutral, "_validate_assets", lambda **_kwargs: ({}, cases, records))
    monkeypatch.setattr(pilot.neutral, "_clip_jobs", lambda _cases: tuple((index, cases[index // 2], (1, 6)[index % 2]) for index in range(52)))
    monkeypatch.setattr(pilot, "load_project", lambda _path: SimpleNamespace(root=tmp_path, encoder=lambda _name: {"definition": {"constructor": {"num_frames": 16}}}))
    monkeypatch.setattr(pilot, "encoder_identity", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(pilot.ENCODER_REGISTRY, "create", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(pilot.neutral, "_adapter_details", lambda *_args, **_kwargs: {"parameter": None})

    class Reader:
        info = SimpleNamespace(num_frames=200)
        def __init__(self, _path): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return None
    monkeypatch.setattr(pilot, "OpenCVVideoReader", Reader)
    monkeypatch.setattr(pilot, "_batch_from_reader", lambda _reader, video_id, samples: SimpleNamespace(video_ids=(video_id,), frame_indices=samples[0].frame_indices))
    calls = []
    def controls(_adapter, _encoder, batch, *, seed):
        ordinal = len(calls)
        assert batch.video_ids == (f"video-{ordinal // 2:02d}",)
        calls.append(seed)
        if fail_ordinal == ordinal:
            raise RuntimeError("fixture pair failure")
        return {name: {"forward_count": 4} for name in ("dense", "identity", "pairmean", "pairedrandom")}
    monkeypatch.setattr(pilot, "_pair_controls", controls)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", "videomaev2", "--dataset-root", str(tmp_path), "--output", str(output), "--execute"])
    return output, calls


def test_production_scale_runs_exactly_52_four_control_jobs(tmp_path: Path, monkeypatch) -> None:
    output, calls = _production_scale(tmp_path, monkeypatch)
    assert pilot.main() == 0
    assert calls == [20260918 + ordinal for ordinal in range(52)]
    shards = sorted((output / "shards").glob("*.json"))
    assert len(shards) == 52
    assert all(set(json.loads(path.read_text())["controls"]) == {"dense", "identity", "pairmean", "pairedrandom"} for path in shards)
    assert json.loads((output / "summary.json").read_text())["clips"] == 52


def test_partial_pair_failure_keeps_shard_and_omits_summary(tmp_path: Path, monkeypatch) -> None:
    output, calls = _production_scale(tmp_path, monkeypatch, fail_ordinal=3)
    with pytest.raises(RuntimeError, match="partial"):
        pilot.main()
    assert len(calls) == 52
    assert not (output / "summary.json").exists()
    progress = json.loads((output / "progress.json").read_text())
    assert progress["completed_clips"] == 51 and progress["failures"][0]["ordinal"] == 3
    assert json.loads((output / "shards/003.json").read_text())["status"] == "failed"


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 4, kernel_size=2, stride=2, bias=False)
    def forward(self, value):
        return self.proj(value).flatten(2).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.attn = nn.Linear(4, 4, bias=False)
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Linear(4, 4, bias=False)
    def forward(self, value):
        return value + self.attn(self.norm1(value)) + self.mlp(self.norm2(value))


class _Adapter:
    def __init__(self):
        self.encoder = nn.Module()
        self.encoder.patch_embed = _Patch()
        self.encoder.blocks = nn.ModuleList([_Block(), _Block()])
        self.calls = 0
    def encode(self, batch, train=False):
        assert not train and batch.video_ids == ("sample-0",) and not batch.metadata
        self.calls += 1
        tokens = self.encoder.patch_embed(torch.from_numpy(np.asarray(batch.frames)).permute(0, 4, 1, 2, 3).float())
        for block in self.encoder.blocks:
            tokens = block(tokens)
        return EncoderOutput(features=tokens, pooled=tokens.mean(1), timeline=TokenTimeline(start_s=torch.zeros((1, tokens.shape[1])), end_s=torch.ones((1, tokens.shape[1]))))


def test_pair_controls_use_clean_batch_four_native_forwards_and_cleanup_hooks():
    adapter = _Adapter()
    batch = ClipBatch(frames=np.zeros((1, 4, 4, 4, 3), dtype=np.uint8), timestamps_s=np.arange(4, dtype=float)[None], frame_indices=np.arange(4)[None], valid_mask=np.ones((1, 4), dtype=bool), video_ids=("labelled",), metadata={"is_anomaly": True, "path": "hidden"})
    result = pilot._pair_controls(adapter, "videomaev2", batch, seed=7)
    assert adapter.calls == 4
    assert set(result) >= {"dense", "identity", "pairmean", "pairedrandom"}
    assert result["identity"]["receipt"]["gathered_shape"][1] == 8
    assert result["pairmean"]["receipt"]["gathered_shape"][1] == 4
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in adapter.encoder.modules())
