"""Native CPU setup and disk calibration checks; formal metadata below is fixture-only."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.checkpoints import sha256_file
from vadbench.contracts import ClipBatch
from vadbench.integrations.videomaev2 import VideoMAEv2Adapter
from vadbench.integrations.videomaev2_encoder import VideoMAEv2EncoderConfig
from vadbench.paper.reduction_setup import prepare_reduction
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.training import CalibrationSample, train_pair_gate

ROLE_LOCK = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 8, 2, 2)

    def forward(self, pixels):
        return self.proj(pixels).flatten(2).transpose(1, 2)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.attention = nn.MultiheadAttention(8, 2, dropout=0.2, batch_first=True)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        return hidden + self.attention(normalized, normalized, normalized, need_weights=False)[0]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _Patch()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, pixels):
        hidden = self.patch_embed(pixels)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _NativeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = VideoMAEv2EncoderConfig(model_name="fixture-only", image_size=4, num_frames=4, use_half=False)
        self.backbone = _Backbone()

    def forward(self, clips):
        pixels = torch.from_numpy(np.asarray(clips)).permute(0, 4, 1, 2, 3).float() / 255
        # Keep an actual adapter-owned readout instead of replacing it during setup.
        return self.backbone(pixels).square().mean(dim=1)


def _adapter():
    torch.manual_seed(39)
    return VideoMAEv2Adapter(encoder=_NativeEncoder(), image_size=4, num_frames=4, use_half=False)


def _batch(offset=0):
    indices = np.arange(4)[None] + offset
    return ClipBatch(
        frames=np.random.default_rng(offset).integers(0, 256, (1, 4, 4, 4, 3), dtype=np.uint8),
        frame_indices=indices, valid_mask=np.ones((1, 4), dtype=bool),
        timestamps_s=indices.astype(float) / 30, video_ids=("fit-fixture",),
        metadata={"is_anomaly": True, "filename": "fit-fixture.mp4"},
    )


def _identity():
    return {"adapter": "videomaev2", "checkpoint": {"sha256": {"fixture.pt": "a" * 64}},
            "constructor": {"pooling": "auto", "num_frames": 4}}


def _prepare(adapter, **kwargs):
    return prepare_reduction(
        adapter, "videomaev2", _batch(123), output_dim=8,
        verified_encoder_identity=_identity(), **kwargs,
    )


@pytest.mark.parametrize("reducer", ["global_uniform", "paired_random"])
def test_no_training_setup_runs_once_cleans_input_and_restores_native_modes(reducer, monkeypatch):
    adapter = _adapter()
    model = adapter.encoder
    model.train()
    model.backbone.blocks[0].norm.eval()
    next(model.parameters()).requires_grad_(False)
    modes = [module.training for module in model.modules()]
    gradients = [parameter.requires_grad for parameter in model.parameters()]
    encode = adapter.encode
    calls = []

    def observed(batch, train=False):
        calls.append(batch)
        assert batch.video_ids == ("sample-0",) and not batch.metadata
        assert not any(module.training for module in model.modules())
        assert not any(parameter.requires_grad for parameter in model.parameters())
        return encode(batch, train=train)

    monkeypatch.setattr(adapter, "encode", observed)
    deployment, receipt = _prepare(adapter, reducer=reducer)
    from vadbench.token_reduction.deployment_contracts import (
        ReductionDeployment,
        ReductionExecutionContext,
    )
    assert isinstance(deployment, ReductionDeployment)
    assert isinstance(deployment(_batch()), ReductionExecutionContext)
    assert len(calls) == 1
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == gradients
    assert receipt["status"] == "prepared"
    assert receipt["depth"] == 0
    assert receipt["dense_geometry"]["flatten_verified"] is True
    assert receipt["dense_output"]["features_shape"] == [1, 8, 8]
    assert receipt["calibration"] is None
    assert receipt["cross_runtime_numerical_validation"] == "not_performed"
    assert receipt["current_runtime"]["device"] == "cpu"
    json.dumps(receipt, allow_nan=False)
    monkeypatch.setattr(adapter, "encode", encode)
    model.eval()
    with torch.no_grad(), deployment(_batch()) as context:
        output = adapter.encode(_batch())
        context.validate_execution()
    assert output.features.shape == (1, 4, 8)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.fixture
def trained_fixture(tmp_path):
    adapter = _adapter()
    run = tmp_path / "tiny-real-training"
    receipt = train_pair_gate(
        adapter, "videomaev2", (CalibrationSample("fit-fixture", 1), CalibrationSample("fit-fixture", 6)),
        lambda sample: _batch(sample.window_index), output_dir=run,
        identity={"fixture_only": True, "encoder_identity": _identity()},
    )
    assert receipt["status"] == "completed" and receipt["steps"] == 6
    return adapter, run, receipt


def _write_receipt(run, receipt):
    (run / "receipt.json").write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")


@pytest.fixture
def formal_metadata_fixture(trained_fixture):
    """Synthetic formal-shaped metadata only; this is not a 256-sample experiment."""
    adapter, run, tiny = trained_fixture
    receipt = copy.deepcopy(tiny)
    receipt.update(fixture_only=True, sample_count=256, steps=768)
    receipt["losses"] *= 128
    receipt["grad_l1"] *= 128
    receipt["nonzero_gradient_steps"] *= 128
    for epoch in receipt["epoch_results"]:
        epoch["steps"] = 256
        epoch["ordered_sample_keys"] = [f"synthetic-fixture-{i}" for i in range(256)]
    receipt["identity"]["plan"] = {
        "schema_version": 1, "protocol": "W", "role": "fit", "video_count": 128,
        "clip_count": 256, "epochs": 3, "seed": 0, "role_lock_sha256": ROLE_LOCK,
        "manifest_sha256": "b" * 64, "fine_annotations_or_captions_used": False,
        "test_annotations_read": False,
        "sampling": {"windows_per_video": 8, "selected_window_indices": [1, 6],
                     "stride": 2, "short_policy": "stride1_if_needed"},
    }
    _write_receipt(run, receipt)
    return adapter, run, receipt


def test_real_tiny_training_cannot_masquerade_as_formal_fit128(trained_fixture):
    adapter, run, _receipt = trained_fixture
    with pytest.raises(ValueError, match="sample_count"):
        _prepare(adapter, reducer="pair_linear", calibration_run=run)


def test_formal_shaped_fixture_loads_real_gate_preserves_full_source_and_separates_runtime(formal_metadata_fixture):
    adapter, run, source = formal_metadata_fixture
    source["torch"] = "fixture-only-different-source-runtime"
    _write_receipt(run, source)
    deployment, receipt = _prepare(adapter, reducer="pair_linear", calibration_run=run)
    assert deployment.reducer_identity["name"] == "pair_linear"
    assert deployment.reducer_identity["calibration_manifest_digest"] == "b" * 64
    assert receipt["calibration"]["receipt_sha256"] == sha256_file(run / "receipt.json")
    assert receipt["calibration"]["source_receipt"] == json.loads(json.dumps(source))
    assert receipt["calibration"]["source_runtime"]["torch"] != receipt["current_runtime"]["torch"]
    assert receipt["cross_runtime_numerical_validation"] == "not_performed"
    assert receipt["dense_geometry"]["source_frame_indices"] != source["geometry"]["source_frame_indices"]
    with torch.no_grad(), deployment(_batch()) as context:
        output = adapter.encode(_batch())
        context.validate_execution()
    assert output.features.shape == (1, 4, 8)
    assert not deployment._merger.gate.weight.requires_grad


@pytest.mark.parametrize("mutation,expected", [
    ("status", "status"), ("encoder", "encoder"), ("epochs", "epochs"),
    ("steps", "steps"), ("gradients", "grad_l1"), ("no_gradient", "nonzero"),
    ("weights_changed", "weights_changed"), ("frozen", "backbone_all_requires_grad_false"),
    ("reload", "reload"), ("role", "role"), ("role_lock", "role_lock_sha256"),
    ("protocol", "protocol"), ("checkpoint", "checkpoint SHA"), ("readout", "readout"),
    ("geometry", "geometry"), ("depth", "depth"), ("dimension", "output_dim"),
    ("file_sha", "checkpoint SHA"), ("path", "basename"),
])
def test_calibration_rejects_incomplete_mismatched_or_escaping_source(formal_metadata_fixture, mutation, expected):
    adapter, run, receipt = formal_metadata_fixture
    if mutation == "status":
        receipt["status"] = "failed"
    elif mutation == "encoder":
        receipt["encoder"] = "timesformer"
    elif mutation == "epochs":
        receipt["epochs"] = 2
    elif mutation == "steps":
        receipt["steps"] = 6
    elif mutation == "gradients":
        receipt["grad_l1"] = receipt["grad_l1"][:6]
    elif mutation == "no_gradient":
        receipt["grad_l1"] = [0] * 768
        receipt["nonzero_gradient_steps"] = 0
    elif mutation == "weights_changed":
        receipt["weights_changed"] = False
    elif mutation == "frozen":
        receipt["backbone_all_requires_grad_false"] = False
    elif mutation == "reload":
        receipt["reload_max_absolute_error"] = 0.01
    elif mutation == "role":
        receipt["identity"]["plan"]["role"] = "confirm"
    elif mutation == "role_lock":
        receipt["identity"]["plan"]["role_lock_sha256"] = "c" * 64
    elif mutation == "protocol":
        receipt["identity"]["plan"]["protocol"] = "D"
    elif mutation == "checkpoint":
        receipt["identity"]["encoder_identity"]["checkpoint"]["sha256"]["fixture.pt"] = "c" * 64
    elif mutation == "readout":
        receipt["identity"]["encoder_identity"]["constructor"]["pooling"] = "mean"
    elif mutation == "geometry":
        receipt["geometry"]["grid"] = [2, 1, 4]
    elif mutation == "depth":
        receipt["depth"] = 1
    elif mutation == "dimension":
        receipt["trained_execution"]["gathered_shape"][-1] = 9
    elif mutation == "file_sha":
        receipt["gate_state_sha256"] = "c" * 64
    else:
        receipt["gate_state_dict"] = "../escape.pt"
    _write_receipt(run, receipt)
    with pytest.raises(ValueError, match=expected):
        _prepare(adapter, reducer="pair_linear", calibration_run=run)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in adapter.encoder.modules())


@pytest.mark.parametrize("mutation,expected", [
    ("keys", "only weight"), ("shape", "output_dim"), ("finite", "finite"), ("dtype", "dtype"),
])
def test_checkpoint_schema_finiteness_and_dtype_are_checked(formal_metadata_fixture, mutation, expected):
    adapter, run, receipt = formal_metadata_fixture
    path = run / receipt["gate_state_dict"]
    state = torch.load(path, weights_only=True)
    if mutation == "keys":
        state["bias"] = torch.zeros(1)
    elif mutation == "shape":
        state["weight"] = state["weight"][:7]
    elif mutation == "finite":
        state["weight"][0] = float("nan")
    else:
        state["weight"] = state["weight"].half()
    torch.save(state, path)
    receipt["gate_state_sha256"] = sha256_file(path)
    _write_receipt(run, receipt)
    with pytest.raises(ValueError, match=expected):
        _prepare(adapter, reducer="pair_linear", calibration_run=run)


def test_setup_failure_restores_flags_and_cleans_geometry_hooks():
    adapter = _adapter()
    bridge = create_observation_bridge("videomaev2", adapter)
    adapter.encoder.train()
    original_modes = [module.training for module in adapter.encoder.modules()]
    original_gradients = [parameter.requires_grad for parameter in adapter.encoder.parameters()]

    def fail(_module, _inputs):
        raise RuntimeError("native fixture failure")

    handle = bridge._blocks[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="native fixture failure"):
            _prepare(adapter, reducer="global_uniform")
    finally:
        handle.remove()
    assert [module.training for module in adapter.encoder.modules()] == original_modes
    assert [parameter.requires_grad for parameter in adapter.encoder.parameters()] == original_gradients
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in adapter.encoder.modules())


def test_selection_reducers_require_keep_ratio_and_prepare_three_tiers():
    adapter = _adapter()
    for bad in (None, 1.0, 0.5):
        with pytest.raises(ValueError, match="keep_ratio"):
            _prepare(adapter, reducer="pair_select", keep_ratio=bad)
    for legacy in ("global_uniform", "paired_random"):
        with pytest.raises(ValueError, match="keep_ratio"):
            _prepare(adapter, reducer=legacy, keep_ratio=0.8)
    for tier in (0.8, 0.6, 0.4):
        for reducer in ("pair_select", "group_uniform", "group_random"):
            deployment, receipt = _prepare(
                adapter, reducer=reducer, keep_ratio=tier, seed=3,
            )
            identity = deployment.reducer_identity
            from vadbench.token_reduction.deployment_contracts import ReductionDeployment
            assert isinstance(deployment, ReductionDeployment)
            assert identity["keep_ratio"] == tier
            assert identity["budget_rule"] == "group_budget_v1"
            assert identity["same_budget_for_all_controls"] is True
            assert receipt["keep_ratio"] == tier
            spec = deployment.selection_spec
            assert receipt["reducer_identity"]["kept_tokens"] == spec.output_token_count
            assert 0 < spec.actual_keep_ratio <= 1
    # identical budgets across the three rules at the same tier
    budgets = {}
    for reducer in ("pair_select", "group_uniform", "group_random"):
        deployment, _receipt = _prepare(adapter, reducer=reducer, keep_ratio=0.6, seed=3)
        budgets[reducer] = deployment.selection_spec.output_token_count
    assert len(set(budgets.values())) == 1


def test_setup_requires_calibration_only_for_learned_reducer_and_checks_output_dim(tmp_path):
    adapter = _adapter()
    with pytest.raises(ValueError, match="calibration_run"):
        _prepare(adapter, reducer="pair_linear")
    with pytest.raises(ValueError, match="calibration_run"):
        _prepare(adapter, reducer="paired_random", calibration_run=tmp_path)
    with pytest.raises(ValueError, match="output_dim"):
        prepare_reduction(adapter, "videomaev2", _batch(), reducer="global_uniform", output_dim=7,
                          verified_encoder_identity=_identity())


class _VJProcessor:
    def __call__(self, videos, **_kwargs):
        pixels = torch.from_numpy(np.asarray(videos)).permute(0, 1, 4, 2, 3).float() / 255
        return {"pixel_values_videos": pixels}

    def to_dict(self):
        return {"fixture_only": True}


@pytest.mark.parametrize("encoder_id", ["videomae", "timesformer", "vjepa2"])
def test_other_real_native_adapters_verify_geometry_and_restore_model_modes(encoder_id):
    transformers = pytest.importorskip("transformers")
    from vadbench.integrations.foundation.vjepa2 import VJEPA2Adapter, _VJEPA2Encoder
    from vadbench.integrations.transformers_video import TransformersVideoAdapter

    frames = 2 if encoder_id == "timesformer" else 4
    dim = 24 if encoder_id == "vjepa2" else 8
    if encoder_id == "vjepa2":
        model = transformers.VJEPA2Model(transformers.VJEPA2Config(
            crop_size=4, patch_size=2, frames_per_clip=4, tubelet_size=2,
            hidden_size=24, num_hidden_layers=2, num_attention_heads=2,
            pred_hidden_size=24, pred_num_hidden_layers=1, pred_num_attention_heads=2,
        ))
        adapter = VJEPA2Adapter(encoder=_VJEPA2Encoder(model, _VJProcessor(), device="cpu"), num_frames=4)
        constructor = {"feature_stage": "backbone_tokens"}
    else:
        config_class = transformers.TimesformerConfig if encoder_id == "timesformer" else transformers.VideoMAEConfig
        model_class = transformers.TimesformerModel if encoder_id == "timesformer" else transformers.VideoMAEModel
        model = model_class(config_class(
            image_size=4, patch_size=2, num_frames=frames, tubelet_size=2,
            hidden_size=dim, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
        ))
        processor = transformers.VideoMAEImageProcessor(
            size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4},
        )
        adapter = TransformersVideoAdapter(
            variant=encoder_id, model=model, processor=processor, clip_frames=frames, pooling="mean",
        )
        constructor = {"pooling": "mean", "feature_stage": "last_hidden_state"}
    batch = _batch()
    batch = replace(batch, frames=batch.frames[:, :frames], frame_indices=batch.frame_indices[:, :frames],
                    valid_mask=batch.valid_mask[:, :frames], timestamps_s=batch.timestamps_s[:, :frames])
    model.train()
    modes = [module.training for module in model.modules()]
    gradients = [parameter.requires_grad for parameter in model.parameters()]
    identity = {"adapter": encoder_id, "checkpoint": {"sha256": {"fixture.pt": "a" * 64}},
                "constructor": constructor}
    deployment, receipt = prepare_reduction(
        adapter, encoder_id, batch, reducer="global_uniform", output_dim=dim,
        verified_encoder_identity=identity,
    )
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == gradients
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert receipt["dense_output"]["pooled_shape"] == [1, dim]
    model.eval()
    with torch.no_grad(), deployment(batch) as context:
        output = adapter.encode(batch, train=False)
        execution = context.validate_execution()
    expected = 5 if encoder_id == "timesformer" else 4
    assert output.features.shape == (1, expected, dim)
    assert execution["gathered_tokens"] == expected
