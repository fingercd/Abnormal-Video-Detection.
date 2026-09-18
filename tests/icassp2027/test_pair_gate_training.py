"""Small native attention models exercise training, real hooks, and disk reload."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.checkpoints import sha256_file
from vadbench.contracts import ClipBatch
from vadbench.integrations.foundation.vjepa2 import VJEPA2Adapter, _VJEPA2Encoder
from vadbench.integrations.transformers_video import TransformersVideoAdapter
from vadbench.token_reduction import training
from vadbench.token_reduction.pair_merge import PairLinearGate
from vadbench.token_reduction.training import (
    CalibrationSample,
    gate_training_step,
    relative_pooled_mse,
    train_pair_gate,
    validate_calibration_assets,
)


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class _Processor:
    def __call__(self, videos, **_kwargs):
        values = torch.from_numpy(np.asarray(videos)).permute(0, 1, 4, 2, 3).float() / 255
        return {"pixel_values": values}


class _VJProcessor:
    def __call__(self, videos, **_kwargs):
        return {"pixel_values_videos": torch.from_numpy(np.asarray(videos)).permute(0, 1, 4, 2, 3).float() / 255}


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 8, 2, 2)

    def forward(self, values):
        return self.proj(values).flatten(2).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attention = nn.MultiheadAttention(8, 2, dropout=0.4, batch_first=True)

    def forward(self, hidden):
        normalized = self.norm1(hidden)
        return hidden + self.attention(normalized, normalized, normalized, need_weights=False)[0]


class _MAEv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _Patch()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, pixels):
        hidden = self.patch_embed(pixels)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _MAEv2Adapter:
    def __init__(self):
        self.encoder = _MAEv2().eval()

    def encode(self, batch, train=False):
        assert batch.video_ids == ("sample-0",)
        assert "is_anomaly" not in batch.metadata
        pixels = torch.from_numpy(batch.frames).permute(0, 4, 1, 2, 3).float() / 255
        with torch.set_grad_enabled(train):
            tokens = self.encoder(pixels)
            # Deliberately not raw token mean: the trainer must use this
            # adapter readout for both the teacher and differentiable path.
            pooled = tokens.square().mean(dim=1)
        return SimpleNamespace(features=tokens, pooled=pooled)


def _adapter(encoder):
    torch.manual_seed(39)
    if encoder == "videomaev2":
        return _MAEv2Adapter(), 4
    import transformers
    if encoder == "vjepa2":
        model = transformers.VJEPA2Model(transformers.VJEPA2Config(
            crop_size=4, patch_size=2, frames_per_clip=4, tubelet_size=2,
            hidden_size=24, num_hidden_layers=2, num_attention_heads=2,
            pred_hidden_size=24, pred_num_hidden_layers=1, pred_num_attention_heads=2,
        ))
        return VJEPA2Adapter(encoder=_VJEPA2Encoder(model, _VJProcessor(), device="cpu"), num_frames=4), 4
    frames = 2 if encoder == "timesformer" else 4
    config_class = transformers.TimesformerConfig if encoder == "timesformer" else transformers.VideoMAEConfig
    model_class = transformers.TimesformerModel if encoder == "timesformer" else transformers.VideoMAEModel
    model = model_class(config_class(image_size=4, patch_size=2, num_frames=frames, tubelet_size=2,
                                    hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
                                    hidden_dropout_prob=0.4, attention_probs_dropout_prob=0.4))
    return TransformersVideoAdapter(variant=encoder, model=model, processor=_Processor(),
                                    clip_frames=frames, image_size=4, pooling="mean"), frames


def _loader(frames):
    def load(sample):
        indices = np.arange(frames)[None] + 100 * sample.window_index
        return ClipBatch(
            frames=np.random.default_rng(sample.window_index).integers(0, 256, (1, frames, 4, 4, 3), dtype=np.uint8),
            timestamps_s=indices.astype(float) / 30, video_ids=(sample.video_id,), frame_indices=indices,
            valid_mask=np.ones_like(indices, dtype=bool), metadata={"is_anomaly": True, "filename": "positive.mp4"},
        )
    return load


def test_seeded_epoch_orders_repeat_cover_all_samples_and_preserve_global_rng():
    state = torch.get_rng_state().clone()
    first = training._epoch_orders(256)
    assert torch.equal(torch.get_rng_state(), state)
    assert training._epoch_orders(256) == first
    assert torch.equal(torch.get_rng_state(), state)
    assert len(first) == 3 and len(set(first)) == 3
    for order in first:
        assert len(order) == len(set(order)) == 256
        assert set(order) == set(range(256))


@pytest.mark.parametrize("encoder", training.ENCODERS)
def test_real_native_training_reload_actual_readout_and_frozen_backbone(tmp_path, encoder):
    adapter, frames = _adapter(encoder)
    output = tmp_path / encoder
    samples = (CalibrationSample("fit-video", 1), CalibrationSample("fit-video", 6))
    receipt = train_pair_gate(adapter, encoder, samples, _loader(frames), output_dir=output,
                              identity={"fixture": "native_cpu_fit_only"})
    assert receipt["status"] == "completed"
    assert receipt["steps"] == 6
    assert receipt["sample_order"] == "seeded_epoch_permutation"
    expected_orders = [[samples[index].key for index in order] for order in training._epoch_orders(len(samples))]
    assert [epoch["ordered_sample_keys"] for epoch in receipt["epoch_results"]] == expected_orders
    assert receipt["nonzero_gradient_steps"] > 0
    assert receipt["weights_changed"]
    assert receipt["backbone_all_grad_none"] and receipt["backbone_all_requires_grad_false"]
    assert receipt["backbone_eval"] and receipt["optimizer_only_gate"] and receipt["teacher_detached"]
    assert receipt["reload_max_absolute_error"] == 0
    assert all(np.isfinite(receipt["losses"]))
    assert receipt["trained_execution"]["suffix_shapes"] == receipt["reload_execution"]["suffix_shapes"]
    assert receipt["gate_state_sha256"] == sha256_file(output / "pair_linear_gate.pt")
    cache = torch.load(output / "teacher_pooled_cache.pt", weights_only=True)
    assert set(cache) == {sample.key for sample in samples}
    assert all(not value.requires_grad and value.device.type == "cpu" and value.ndim == 2 for value in cache.values())
    inputs = json.loads((output / "input_receipts.json").read_text())
    assert inputs[samples[0].key]["frame_indices"] != inputs[samples[1].key]["frame_indices"]
    assert json.loads((output / "progress.json").read_text())["completed_steps"] == 6
    with pytest.raises(FileExistsError):
        train_pair_gate(adapter, encoder, samples, _loader(frames), output_dir=output, identity={})


def test_disconnected_gradient_publishes_failed_receipt_only(tmp_path, monkeypatch):
    adapter, frames = _adapter("videomaev2")
    original = training.native_pooled_forward

    def disconnected(*args, **kwargs):
        tokens, pooled = original(*args, **kwargs)
        return tokens, pooled.detach()

    monkeypatch.setattr(training, "native_pooled_forward", disconnected)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="disconnected"):
        train_pair_gate(adapter, "videomaev2", (CalibrationSample("fit", 1),), _loader(frames),
                        output_dir=output, identity={})
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["completed_steps"] == 0
    assert json.loads((output / "progress.json").read_text())["status"] == "failed"
    assert not (output / "pair_linear_gate.pt").exists()
    assert not (output / "summary.json").exists()
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in adapter.encoder.blocks)


def test_objective_rejects_teacher_graph_and_optimizer_backbone_parameters():
    prediction = torch.ones(1, 3, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        relative_pooled_mse(prediction, prediction * 2)
    gate = PairLinearGate(3)
    backbone = nn.Linear(3, 3).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([gate.weight, backbone.weight], lr=1e-3)
    with pytest.raises(RuntimeError, match="only"):
        gate_training_step(gate, backbone, optimizer, lambda: prediction, prediction.detach())


@pytest.fixture
def assets(tmp_path, monkeypatch):
    ids = [f"fit-{index:03d}" for index in range(128)]
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"basis": "complete_official_train_source_groups", "partitions": dict.fromkeys(ids, "fit")}), encoding="utf-8")
    monkeypatch.setattr(training, "ROLE_LOCK_SHA256", sha256_file(lock))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"schema_version": 1, "cases": [
        {"video_id": video_id, "window_index": index} for video_id in ids for index in (1, 6)]}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps({
        "schema_version": 1, "video_id": video_id, "path": f"{video_id}.mp4", "split": "train",
        "category": "audit", "is_anomaly": index >= 64, "annotations": [], "num_frames": 300, "fps": 30,
        "metadata": {"training_asset_sha256": "a" * 64, "size_bytes": 10},
    }) for index, video_id in enumerate(ids)), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "protocol": "W", "role": "fit", "video_count": 128,
                               "clip_count": 256, "epochs": 3, "seed": 0, "sampling": training.SAMPLING,
                               "manifest_sha256": sha256_file(manifest), "cases_sha256": sha256_file(cases),
                               "role_lock_sha256": sha256_file(lock)}), encoding="utf-8")
    return {"plan_path": plan, "cases_path": cases, "manifest_path": manifest,
            "role_lock_path": lock, "dataset_root": tmp_path}


def test_authorized_asset_contract(assets):
    plan, samples, records = validate_calibration_assets(**assets)
    assert len(samples) == 256 and len(records) == 128 and plan["role"] == "fit"


@pytest.mark.parametrize("bad_role", ["test", "confirm", "select"])
def test_assets_reject_forbidden_roles(assets, bad_role):
    plan = json.loads(assets["plan_path"].read_text())
    plan["role"] = bad_role
    assets["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        validate_calibration_assets(**assets)


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "fine", "fit_lock", "video_bytes"])
def test_assets_reject_identity_fine_labels_and_unlocked_fit(assets, mutation):
    plan = json.loads(assets["plan_path"].read_text())
    if mutation in {"hash", "duplicate"}:
        cases = json.loads(assets["cases_path"].read_text())
        cases["cases"][0] = cases["cases"][1]
        assets["cases_path"].write_text(json.dumps(cases), encoding="utf-8")
        if mutation == "duplicate":
            plan["cases_sha256"] = sha256_file(assets["cases_path"])
    elif mutation in {"fine", "fit_lock"}:
        records = [json.loads(line) for line in assets["manifest_path"].read_text().splitlines()]
        if mutation == "fine":
            records[0]["annotations"] = [{"scope": "caption", "text": "forbidden"}]
        else:
            records[0]["split"] = "test"
        assets["manifest_path"].write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        plan["manifest_sha256"] = sha256_file(assets["manifest_path"])
    assets["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_calibration_assets(**assets, verify_video_bytes=mutation == "video_bytes")
