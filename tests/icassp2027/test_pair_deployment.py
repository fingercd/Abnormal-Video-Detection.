"""CPU native-forward checks of frozen pair deployment, including tail batches."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.contracts import ClipBatch
from vadbench.paper.extraction import representation_from_verified_encoder
from vadbench.token_reduction import deployment as deployment_module
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.deployment import PairMergeDeployment
from vadbench.token_reduction.pair_merge import PairLinearGate, PairMergeError


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch(size=1, frames=4, offset=0):
    indices = np.broadcast_to(np.arange(frames) + offset, (size, frames)).copy()
    return ClipBatch(
        frames=np.random.default_rng(7).integers(0, 255, (size, frames, 4, 4, 3), dtype=np.uint8),
        timestamps_s=indices.astype(float) / 30,
        video_ids=tuple(f"sample-{i}" for i in range(size)),
        frame_indices=indices,
        valid_mask=np.ones((size, frames), dtype=bool),
    )


class _NativePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 8, 2, 2)

    def forward(self, pixels):
        return self.proj(pixels).flatten(2).transpose(1, 2)


class _NativeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)

    def forward(self, hidden):
        normalized = self.norm1(hidden)
        return hidden + self.attention(normalized, normalized, normalized, need_weights=False)[0]


class _NativeMAEv2(nn.Module):
    """Native Conv3d/patch-only/block contract without upstream dependencies."""
    def __init__(self):
        super().__init__()
        self.patch_embed = _NativePatch()
        self.blocks = nn.ModuleList([_NativeBlock(), _NativeBlock()])

    def forward(self, pixels):
        hidden = self.patch_embed(pixels)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Adapter:
    def __init__(self, model, encoder_id, frames):
        self.model = model.eval()
        self.encoder_id = encoder_id
        self.capabilities = SimpleNamespace(
            fixed_num_frames=frames, min_frames=frames, max_frames=frames,
            supports_fixed_clip=True, supports_training=True,
        )

    def encode(self, batch):
        pixels = torch.from_numpy(batch.frames).permute(0, 1, 4, 2, 3).float() / 255
        if self.encoder_id == "videomaev2":
            return self.model(pixels.transpose(1, 2)).last_hidden_state
        if self.encoder_id == "vjepa2":
            return self.model(pixel_values_videos=pixels, skip_predictor=True).last_hidden_state
        return self.model(pixel_values=pixels).last_hidden_state


def _case(encoder_id="videomae"):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(41)
    frames, dim = 4, 8
    if encoder_id == "timesformer":
        frames = 2
        model = transformers.TimesformerModel(transformers.TimesformerConfig(
            image_size=4, patch_size=2, num_frames=frames, hidden_size=dim,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
        ))
    elif encoder_id == "videomae":
        model = transformers.VideoMAEModel(transformers.VideoMAEConfig(
            image_size=4, patch_size=2, num_frames=frames, tubelet_size=2,
            hidden_size=dim, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
        ))
    elif encoder_id == "vjepa2":
        dim = 24
        model = transformers.VJEPA2Model(transformers.VJEPA2Config(
            crop_size=4, patch_size=2, frames_per_clip=frames, tubelet_size=2,
            hidden_size=dim, num_hidden_layers=2, num_attention_heads=2,
            pred_hidden_size=24, pred_num_hidden_layers=1, pred_num_attention_heads=2,
        ))
    else:
        model = _NativeMAEv2()
    adapter = _Adapter(model, encoder_id, frames)
    bridge = create_observation_bridge(encoder_id, model)
    initial = _batch(frames=frames, offset=100)
    with torch.no_grad(), bridge.geometry(initial.frame_indices, initial.valid_mask) as geometry:
        adapter.encode(initial)
    assert geometry.layout is not None
    return adapter, bridge, geometry.layout, frames, dim


@pytest.mark.parametrize("encoder_id", ["videomaev2", "videomae", "timesformer", "vjepa2"])
@pytest.mark.parametrize("strategy", ["mean", "paired_random", "global_uniform"])
def test_consecutive_batches_reuse_real_half_suffix_without_setup_work(encoder_id, strategy, monkeypatch):
    adapter, bridge, layout, frames, dim = _case(encoder_id)
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim, batch_sizes=(1, 2), strategy=strategy)
    identity = dict(deployment.reducer_identity)
    first = _batch(2, frames, 200)
    again = _batch(2, frames, 7000)
    assert deployment(first) is deployment(again)
    expected_count = (layout.token_capacity + (encoder_id == "timesformer")) // 2
    dense_coordinates = deployment(first).layout.source_coordinates.clone()

    def forbidden(*_args, **_kwargs):
        pytest.fail("setup-only CPU/hash/geometry/plan work reached inference")

    monkeypatch.setattr(bridge, "geometry", forbidden)
    monkeypatch.setattr(deployment_module, "horizontal_pair_merge_spec", forbidden)
    monkeypatch.setattr(deployment_module, "_implementation_digest", forbidden)
    monkeypatch.setattr(torch, "randint", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    for batch in (first, _batch(1, frames, 999), again):
        with torch.no_grad(), deployment(batch) as context:
            assert context.gathered_shape is None
            output = adapter.encode(batch)
            receipt = context.validate_execution()
        assert output.shape == (batch.batch_size, expected_count, dim)
        assert receipt["suffix_shapes"] == {"1": [batch.batch_size, expected_count, dim]}
        assert context.suffix_position_masks == {}
        assert not any(block._forward_hooks or block._forward_pre_hooks for block in bridge._blocks)
    assert torch.equal(dense_coordinates, deployment(again).layout.source_coordinates)
    assert "source_frame_indices" not in deployment(again).layout.provenance
    assert dict(deployment.reducer_identity) == identity


def test_native_adapter_and_backbone_identity_do_not_change():
    adapter, bridge, layout, frames, dim = _case()
    arguments = dict(
        runtime_id="videomae", adapter=adapter,
        verified_encoder_identity={"adapter": "videomae", "checkpoint": {"sha256": "a" * 64}, "constructor": {}},
        preprocessing={"fixture": "native"}, readout={"pooling": "mean"},
        output_dim=dim, precision="float32", position_strategy={"name": "native"},
    )
    dense = representation_from_verified_encoder(**arguments, reducer={"name": "identity"})
    bound_encode = adapter.encode.__func__
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim)
    with torch.no_grad(), deployment(_batch(frames=frames)) as context:
        adapter.encode(_batch(frames=frames))
        context.validate_execution()
    reduced = representation_from_verified_encoder(**arguments, reducer=dict(deployment.reducer_identity))
    assert adapter.encode.__func__ is bound_encode
    assert reduced.backbone == dense.backbone
    assert reduced.position_strategy == dense.position_strategy
    assert reduced.fingerprint != dense.fingerprint


def test_vjepa_suffix_receives_exact_first_anchor_positions():
    adapter, bridge, layout, frames, dim = _case("vjepa2")
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim, batch_sizes=(2,))
    observed = []
    batch = _batch(2, frames)
    with torch.no_grad(), deployment(batch) as context:
        handle = bridge._blocks[1].register_forward_pre_hook(lambda _module, args: observed.append(args[1].clone()))
        try:
            adapter.encode(batch)
            receipt = context.validate_execution()
        finally:
            handle.remove()
    assert len(observed) == 1
    assert torch.equal(observed[0], torch.tensor([[0, 2, 4, 6], [0, 2, 4, 6]]))
    assert receipt["vjepa2_position_injections"] == 1
    assert receipt["recorded_position_masks"] is False


def test_timesformer_gate_uses_complete_trajectory_and_preserves_cls():
    adapter, bridge, layout, frames, dim = _case("timesformer")
    gate = PairLinearGate(dim)
    with torch.no_grad():
        gate.weight.copy_(torch.arange(dim) / 10)
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim, gate=gate, calibration_manifest_digest="a" * 64)
    captured = {}
    before = bridge._blocks[0].register_forward_hook(lambda _module, _args, output: captured.update(dense=output[0].detach().clone()))
    batch = _batch(frames=frames)
    try:
        with torch.no_grad(), deployment(batch) as context:
            after = bridge._blocks[1].register_forward_pre_hook(lambda _module, args: captured.update(reduced=args[0].detach().clone()))
            try:
                adapter.encode(batch)
                context.validate_execution()
            finally:
                after.remove()
    finally:
        before.remove()
    dense = captured["dense"]
    expected = dense[:, [0, 1, 2, 5, 6]].clone()
    for anchor, partner, slot in ((1, 3, 1), (5, 7, 3)):
        left, right = dense[:, anchor:anchor + frames], dense[:, partner:partner + frames]
        logits = torch.stack(((left * gate.weight).sum(-1).mean(1), (right * gate.weight).sum(-1).mean(1)), -1)
        weights = logits.softmax(-1)
        expected[:, slot:slot + frames] = left * weights[:, None, :1] + right * weights[:, None, 1:]
    torch.testing.assert_close(captured["reduced"], expected)
    assert torch.equal(captured["reduced"][:, 0], dense[:, 0])


def test_gate_is_cloned_frozen_and_identity_is_readonly():
    adapter, bridge, layout, frames, dim = _case()
    gate = PairLinearGate(dim)
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim, gate=gate, calibration_manifest_digest="c" * 64)
    identity = dict(deployment.reducer_identity)
    batch = _batch(frames=frames)
    with torch.no_grad(), deployment(batch):
        first = adapter.encode(batch)
    with torch.no_grad():
        gate.weight.fill_(100)
    with torch.no_grad(), deployment(batch):
        second = adapter.encode(batch)
    assert torch.equal(first, second)
    assert dict(deployment.reducer_identity) == identity
    assert not deployment._merger.training
    assert all(not parameter.requires_grad for parameter in deployment._merger.parameters())
    with pytest.raises(TypeError):
        deployment.reducer_identity["depth"] = 1
    with pytest.raises(AttributeError):
        deployment.reducer_identity = {}
    changed = PairMergeDeployment(bridge, layout, depth=0, dim=dim, gate=gate, calibration_manifest_digest="c" * 64)
    assert changed.reducer_identity["gate_tensor_sha256"] != identity["gate_tensor_sha256"]


def test_failure_removes_hooks_and_context_is_reusable():
    adapter, bridge, layout, frames, dim = _case()
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim)
    batch = _batch(frames=frames)
    def fail(_module, _args):
        raise RuntimeError("suffix failed")
    handle = bridge._blocks[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="suffix failed"), deployment(batch):
            adapter.encode(batch)
    finally:
        handle.remove()
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in bridge._blocks)
    with pytest.raises(ValueError, match="did not execute"):
        deployment(batch).validate_execution()
    with torch.no_grad(), deployment(batch) as context:
        adapter.encode(batch)
        context.validate_execution()


def test_unprepared_batch_padding_and_frame_mismatch_fail_closed():
    _adapter, bridge, layout, frames, dim = _case()
    deployment = PairMergeDeployment(bridge, layout, depth=0, dim=dim, batch_sizes=(1,))
    with pytest.raises(PairMergeError, match="dense"):
        deployment(_batch(2, frames))
    with pytest.raises(PairMergeError, match="帧数"):
        deployment(_batch(1, 2))
    padded = replace(_batch(frames=frames), valid_mask=np.asarray([[True, True, True, False]]))
    with pytest.raises(PairMergeError, match="padded"):
        deployment(padded)
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in bridge._blocks)


def test_setup_rejects_unverified_or_already_merged_layout_and_missing_calibration():
    _adapter, bridge, layout, _frames, dim = _case()
    for bad in (replace(layout, mass=layout.mass * 2), replace(layout, provenance={**layout.provenance, "pair_merge": {}})):
        with pytest.raises(PairMergeError, match="merge"):
            PairMergeDeployment(bridge, bad, depth=0, dim=dim)
    with pytest.raises(PairMergeError, match="coordinate_source"):
        PairMergeDeployment(bridge, replace(layout, provenance={**layout.provenance, "coordinate_source": "guessed"}), depth=0, dim=dim)
    with pytest.raises(PairMergeError, match="calibration"):
        PairMergeDeployment(bridge, layout, depth=0, dim=dim, gate=PairLinearGate(dim))


@pytest.mark.parametrize("encoder_id", ["videomaev2", "videomae", "timesformer", "vjepa2"])
@pytest.mark.parametrize("strategy", ["paired_random", "global_uniform"])
def test_static_subsampling_preserves_native_members_mass_and_global_rng(encoder_id, strategy):
    adapter, bridge, layout, frames, dim = _case(encoder_id)
    global_rng = torch.get_rng_state().clone()
    deployment = PairMergeDeployment(
        bridge, layout, depth=0, dim=dim, batch_sizes=(1, 2), strategy=strategy,
    )
    assert torch.equal(torch.get_rng_state(), global_rng)
    single = deployment(_batch(frames=frames, offset=9))
    double = deployment(_batch(2, frames, offset=1000))
    assert single.transform is double.transform is None
    assert torch.equal(double.indices, single.indices.expand(2, -1))
    assert torch.equal(single.indices, single.indices.sort(dim=1).values)
    masses = double.layout.mass.gather(1, double.indices)
    assert torch.equal(masses, torch.ones_like(masses))
    assert masses.sum() < double.layout.mass.sum()
    identity = deployment.reducer_identity
    assert identity["name"] == strategy
    assert identity["operator"] == f"{strategy}_subsample"
    assert identity["position_anchor"] == ("selected_member_native_index" if strategy == "paired_random" else "selected_native_index")
    assert identity["position_semantics"] == "retained_native_position"
    assert identity["mass_convention"] == "retained_unit_mass"
    assert identity["seed"] == (0 if strategy == "paired_random" else None)
    assert identity["mask_scope"] == "static_setup_mask_shared_across_all_clips_and_batch_sizes"

    selected = layout.source_coordinates[0, single.indices[0]]
    if encoder_id == "timesformer":
        assert single.indices[0, 0] == 0
        selected = selected[1:]
        trajectories = selected.reshape(-1, frames, 3)
        assert torch.equal(trajectories[:, :, 0], torch.arange(frames).expand(len(trajectories), -1))
        assert torch.equal(trajectories[:, :, 1:], trajectories[:, :1, 1:].expand(-1, frames, -1))
    if strategy == "paired_random":
        # Exactly one native coordinate from each horizontal pair survives.
        pair_locations = selected.clone()
        pair_locations[:, 2] //= 2
        assert torch.unique(pair_locations, dim=0).shape[0] == selected.shape[0]
    else:
        from vadbench.research.interventions import fixed_budget_indices

        expected_control = fixed_budget_indices(
            torch.zeros_like(layout.mass), layout, single.indices.shape[1], "uniform",
        )
        assert torch.equal(single.indices, expected_control.indices)
        assert identity["random_generator"] is None

    captured = {}
    def capture_dense(_module, _inputs, output):
        captured["dense"] = bridge.block_output_tensor(0, output).detach().clone()
    dense_handle = bridge._blocks[0].register_forward_hook(capture_dense)
    batch = _batch(2, frames)
    try:
        with torch.no_grad(), deployment(batch) as context:
            def capture_suffix(_module, inputs):
                captured["suffix"] = inputs[0].detach().clone()
                if encoder_id == "vjepa2":
                    captured["positions"] = inputs[1].detach().clone()
            suffix_handle = bridge._blocks[1].register_forward_pre_hook(capture_suffix)
            try:
                output = adapter.encode(batch)
                context.validate_execution()
            finally:
                suffix_handle.remove()
    finally:
        dense_handle.remove()
    expected = captured["dense"].gather(1, double.indices[:, :, None].expand(-1, -1, dim))
    assert torch.equal(captured["suffix"], expected)
    assert output.shape[1] == double.indices.shape[1]
    if encoder_id == "vjepa2":
        assert torch.equal(captured["positions"], double.indices)
    if encoder_id == "timesformer":
        assert torch.equal(captured["suffix"][:, 0], captured["dense"][:, 0])
    repeated = PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy=strategy)
    assert torch.equal(repeated(batch).indices, double.indices)
    assert dict(repeated.reducer_identity) == dict(identity)


@pytest.mark.parametrize("encoder_id", ["videomae", "timesformer"])
def test_random_seed_changes_static_indices_and_identity(encoder_id):
    _adapter, bridge, layout, frames, dim = _case(encoder_id)
    first = PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="paired_random", seed=0)
    second = PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="paired_random", seed=1)
    batch = _batch(frames=frames)
    assert not torch.equal(first(batch).indices, second(batch).indices)
    assert first.reducer_identity["selected_indices_sha256"] != second.reducer_identity["selected_indices_sha256"]
    assert first.reducer_identity["seed"] == 0
    assert second.reducer_identity["seed"] == 1


def test_random_strategy_rejects_gate_and_invalid_options():
    _adapter, bridge, layout, _frames, dim = _case()
    with pytest.raises(PairMergeError, match="strategy"):
        PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="unknown")
    with pytest.raises(PairMergeError, match="gate"):
        PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="paired_random", gate=PairLinearGate(dim))
    with pytest.raises(PairMergeError, match="seed"):
        PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="paired_random", seed=True)
    with pytest.raises(PairMergeError, match="gate"):
        PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="global_uniform", gate=PairLinearGate(dim))


def test_uniform_mask_and_identity_do_not_depend_on_seed():
    _adapter, bridge, layout, frames, dim = _case()
    first = PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="global_uniform", seed=0)
    second = PairMergeDeployment(bridge, layout, depth=0, dim=dim, strategy="global_uniform", seed=456)
    batch = _batch(frames=frames)
    assert torch.equal(first(batch).indices, second(batch).indices)
    assert dict(first.reducer_identity) == dict(second.reducer_identity)
    assert first.reducer_identity["seed"] is None
