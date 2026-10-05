from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.contracts import ClipBatch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/icassp2027/verify_pair_merge_active.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("pair_merge_active_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
active = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = active
SPEC.loader.exec_module(active)


def test_dry_run_never_initializes_model(monkeypatch, capsys, tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"partitions": {"Abuse005_x264": "fit"}}), encoding="utf-8")
    monkeypatch.setattr(active, "ROLE_LOCK", lock)
    monkeypatch.setattr(active, "ROLE_LOCK_SHA256", active.sha256_file(lock))
    monkeypatch.setattr(
        active,
        "load_project",
        lambda _path: SimpleNamespace(
            path=tmp_path / "profile.yaml",
            encoder=lambda _name: {"definition": {"constructor": {"num_frames": 16}, "checkpoint": {"local_path": str(tmp_path / "missing")}}},
        ),
    )
    monkeypatch.setattr(
        active.ENCODER_REGISTRY,
        "create",
        lambda *_args, **_kwargs: pytest.fail("dry run initialized model"),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", "videomaev2"])
    assert active.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned"
    assert plan["partition"] == "fit"
    assert plan["steps"] == 2


def test_nonfit_video_is_rejected_before_execution(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"partitions": {"Abuse001_x264": "confirm"}}), encoding="utf-8")
    monkeypatch.setattr(active, "ROLE_LOCK_SHA256", active.sha256_file(lock))
    with pytest.raises(ValueError, match="fit"):
        active._fit_video_id(tmp_path / "Abuse001_x264.mp4", lock)


def test_execute_asset_failure_keeps_failed_receipt(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"partitions": {"Abuse005_x264": "fit"}}), encoding="utf-8")
    video = tmp_path / "Abuse005_x264.mp4"
    video.write_bytes(b"fixture")
    output = tmp_path / "out"
    monkeypatch.setattr(active, "ROLE_LOCK", lock)
    monkeypatch.setattr(active, "ROLE_LOCK_SHA256", active.sha256_file(lock))
    monkeypatch.setattr(
        active,
        "load_project",
        lambda _path: SimpleNamespace(
            root=tmp_path,
            path=tmp_path / "profile.yaml",
            encoder=lambda _name: {"definition": {"constructor": {"num_frames": 16}, "checkpoint": {"local_path": str(tmp_path / "missing")}}},
        ),
    )
    monkeypatch.setattr(
        active.ENCODER_REGISTRY,
        "create",
        lambda *_args, **_kwargs: pytest.fail("missing checkpoint initialized model"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--encoder", "videomae", "--video", str(video), "--output", str(output), "--execute"],
    )
    with pytest.raises(FileNotFoundError):
        active.main()
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["failure"] == "FileNotFoundError"


def test_execute_accepts_checkpoint_directory_and_reloads_saved_gate(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"partitions": {"Abuse005_x264": "fit"}}), encoding="utf-8")
    video = tmp_path / "Abuse005_x264.mp4"
    video.write_bytes(b"fixture")
    checkpoint = tmp_path / "hf-checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "out"
    frames = np.arange(4 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 4, 3)
    batch = ClipBatch(
        frames=frames,
        timestamps_s=np.asarray([[0.0, 0.1, 0.2, 0.3]]),
        video_ids=("fixture",),
        valid_mask=np.asarray([[True, True, True, True]]),
        frame_indices=np.asarray([[0, 1, 2, 3]]),
    )
    sample = SimpleNamespace(frame_indices=(0, 1, 2, 3), valid_mask=(True, True, True, True))
    monkeypatch.setattr(active, "ROLE_LOCK", lock)
    monkeypatch.setattr(active, "ROLE_LOCK_SHA256", active.sha256_file(lock))
    monkeypatch.setattr(
        active,
        "load_project",
        lambda _path: SimpleNamespace(
            root=tmp_path,
            path=tmp_path / "profile.yaml",
            encoder=lambda _name: {"definition": {"constructor": {"num_frames": 4}, "checkpoint": {"local_path": str(checkpoint)}}},
        ),
    )
    monkeypatch.setattr(active, "probe_video", lambda _path: SimpleNamespace(num_frames=4, fps=10.0, height=4, width=4))
    monkeypatch.setattr(active, "sample_fixed_clip", lambda *_args, **_kwargs: sample)
    monkeypatch.setattr(active, "build_clip_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(active, "encoder_identity", lambda *_args, **_kwargs: {"fixture": True})
    monkeypatch.setattr(active.ENCODER_REGISTRY, "create", lambda *_args, **_kwargs: _Adapter())
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--encoder", "videomae", "--video", str(video), "--output", str(output), "--execute"],
    )
    assert active.main() == 0
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["checkpoint_exists"] is True
    assert (output / "pair_linear_gate.pt").is_file()
    assert receipt["gate_state_sha256"] == active.sha256_file(output / "pair_linear_gate.pt")


class _Patch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv3d(3, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels).flatten(2).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(self.norm1(hidden))


class _Native(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = _Patch()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Block(), _Block()])

    def forward(self, pixel_values: torch.Tensor):
        hidden = self.embeddings.patch_embeddings(pixel_values)
        for block in self.encoder.layer:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Adapter:
    def __init__(self) -> None:
        self.model = _Native()
        self.device = None

    def _prepare_inputs(self, batch: ClipBatch):
        pixels = torch.from_numpy(batch.frames).permute(0, 4, 1, 2, 3).float()
        return {"pixel_values": pixels}, np.asarray(batch.valid_lengths)

    def _forward(self, inputs, *, train: bool):
        return self.model(**inputs)

    def _output_from_raw(self, raw, batch: ClipBatch, *, lengths):
        return SimpleNamespace(pooled=raw.last_hidden_state.mean(dim=1))

    def encode(self, batch: ClipBatch, train: bool = False):
        inputs, lengths = self._prepare_inputs(batch)
        return self._output_from_raw(self._forward(inputs, train=train), batch, lengths=lengths)


def test_real_suffix_qa_trains_only_local_gate_and_reloads(tmp_path: Path) -> None:
    frames = np.arange(4 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 4, 3)
    batch = ClipBatch(
        frames=frames,
        timestamps_s=np.asarray([[0.0, 0.1, 0.2, 0.3]]),
        video_ids=("fixture",),
        valid_mask=np.asarray([[True, True, True, True]]),
        frame_indices=np.asarray([[0, 1, 2, 3]]),
    )
    gate_path = tmp_path / "gate.pt"
    result = active._run_pair_merge_qa(_Adapter(), "videomae", batch, gate_path=gate_path)
    assert result["identity"]["max_absolute_error"] == 0.0
    assert result["pair_mean"]["shape"][1] == 4
    assert result["zero_gate"]["max_absolute_error_to_mean"] == 0.0
    assert result["gate"]["steps"] == 2
    assert result["gate"]["gradients_finite_nonzero"]
    assert result["gate"]["weights_changed"]
    assert result["backbone"]["all_grad_none"]
    assert gate_path.is_file() and result["gate"]["state_dict_sha256"] == active.sha256_file(gate_path)


@pytest.mark.parametrize("encoder", ["videomaev2", "timesformer", "videomae", "vjepa2"])
def test_parser_accepts_only_active_encoders(monkeypatch, encoder: str) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--encoder", encoder])
    assert active.parse_args().encoder == encoder
