from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vadbench.research import ProbeCollector
from vadbench.token_reduction.bridges import (
    IndexedInterventionError,
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)


class _TimePatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, pixels: torch.Tensor):
        batch, frames, channels, height, width = pixels.shape
        projected = self.projection(pixels.reshape(batch * frames, channels, height, width))
        return projected.flatten(2).transpose(1, 2), frames, projected.shape[-1]


class _TimeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = _TimePatch()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch = pixels.shape[0]
        patches, frames, _ = self.patch_embeddings(pixels)
        patches = patches.reshape(batch, frames, patches.shape[1], patches.shape[2])
        # This is the upstream divided-space-time public layout: P-major with
        # time changing fastest, plus a unique global CLS.
        patches = patches.permute(0, 2, 1, 3).reshape(batch, -1, patches.shape[-1])
        return torch.cat((self.cls_token.expand(batch, -1, -1), patches), dim=1)


class _TimeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal_layernorm = nn.Identity()
        self.temporal_attention = nn.Identity()
        self.layernorm_before = nn.Identity()
        self.attention = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = value.shape
        frames = 2
        temporal = (
            value[:, 1:].reshape(batch, 2, 2, frames, width).reshape(batch * 4, frames, width)
        )
        temporal = self.temporal_attention(self.temporal_layernorm(temporal))
        # The test only needs the temporal native execution seam.
        return (
            value
            + temporal.reshape(batch, 2, 2, frames, width)
            .reshape(batch, tokens - 1, width)
            .mean(1, keepdim=True)
            * 0
        )


class _TimeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(attention_type="divided_space_time")
        self.embeddings = _TimeEmbeddings()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_TimeLayer()])

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        value = self.embeddings(pixels)
        for layer in self.encoder.layer:
            value = layer(value)
        return value


class _TubeletPatch(nn.Module):
    def __init__(self, *, projection_name: str) -> None:
        super().__init__()
        projection = nn.Conv3d(3, 4, kernel_size=2, stride=2, bias=False)
        setattr(self, projection_name, projection)
        self._projection_name = projection_name

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        projection = getattr(self, self._projection_name)
        return projection(pixels).flatten(2).transpose(1, 2)


class _TubeletModel(nn.Module):
    def __init__(self, encoder_id: str, projection_name: str) -> None:
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = _TubeletPatch(projection_name=projection_name)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Identity()])
        self.encoder_id = encoder_id


def test_timesformer_geometry_proves_spatial_major_time_minor_and_collector_restores_batch():
    model = _TimeModel()
    bridge = create_observation_bridge("timesformer", model)
    geometry = bridge.geometry([[2, 4], [10, 12]], [[True, True], [True, True]])
    pixels = torch.randn(2, 2, 3, 4, 4)
    with geometry:
        model(pixels)
    assert geometry.receipt["patch_flatten_verified"]
    assert geometry.receipt["divided_layout_verified"]
    assert geometry.layout.source_coordinates[0, 1:].tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
    ]
    sites = bridge.observation_sites([0])
    metadata = bridge.probe_token_metadata(geometry, sites)
    collector = ProbeCollector(
        {
            "block.0.temporal.attn.pre_projection.output": sites[
                "block.0.temporal.attn.pre_projection.output"
            ]
        },
        metadata,
    )
    with collector:
        model(pixels)
    observation = collector.observations[0]
    assert observation.batch_size == 2
    assert {row["batch_index"] for row in observation.rows} == {0, 1}
    assert all(
        row["num_valid_tokens"] == 8 for row in observation.rows if row["status"] == "available"
    )


def test_timesformer_p16_uses_post_temporal_projection_not_raw_attention_output():
    transformers = pytest.importorskip("transformers")
    model = transformers.TimesformerModel(
        transformers.TimesformerConfig(
            image_size=4,
            patch_size=2,
            num_frames=2,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    bridge = create_observation_bridge("timesformer", model)
    pixels = torch.randn(1, 2, 3, 4, 4)
    geometry = bridge.geometry([[0, 1]], [[True, True]])
    with geometry:
        model(pixel_values=pixels)
    sites = bridge.observation_sites([0])
    selected = {
        key: sites[key]
        for key in (
            "block.0.temporal.attn.pre_norm.input",
            "block.0.temporal.attn.pre_projection.output",
            "block.0.temporal.attn.projection.output",
        )
    }
    collector = ProbeCollector(selected, bridge.probe_token_metadata(geometry, selected))
    with collector:
        model(pixel_values=pixels)
    raw = next(item for item in collector.observations if "pre_projection" in item.site)
    projected = next(item for item in collector.observations if ".projection." in item.site)
    assert not any("update_to_input_norm_ratio" in row["statistic_name"] for row in raw.rows)
    assert any(
        "attention_update_to_input_norm_ratio" in row["statistic_name"] for row in projected.rows
    )


def test_external_indices_shorten_real_timesformer_suffix_and_require_complete_trajectories():
    transformers = pytest.importorskip("transformers")
    model = transformers.TimesformerModel(
        transformers.TimesformerConfig(
            image_size=4,
            patch_size=2,
            num_frames=2,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).eval()
    bridge = create_observation_bridge("timesformer", model)
    pixels = torch.randn(1, 2, 3, 4, 4)
    geometry = bridge.geometry([[0, 1]], [[True, True]])
    with geometry:
        dense = model(pixel_values=pixels).last_hidden_state
    with indexed_gather(bridge, 0, identity_indices(geometry.layout), geometry.layout) as identity:
        parity = model(pixel_values=pixels).last_hidden_state
        identity_receipt = identity.validate_execution()
    torch.testing.assert_close(parity, dense)
    assert identity_receipt["gathered_tokens"] == 9
    indices = torch.tensor([[0, 1, 2, 3, 4]])  # CLS plus two full T=2 spatial tracks
    with indexed_gather(bridge, 0, indices, geometry.layout) as intervention:
        reduced = model(pixel_values=pixels).last_hidden_state
        receipt = intervention.validate_execution()
    assert reduced.shape[1] == 5
    assert receipt["suffix_shapes"] == {"1": [1, 5, 8]}
    with pytest.raises(IndexedInterventionError, match="complete time trajectories"):
        indexed_gather(bridge, 0, torch.tensor([[0, 1, 3]]), geometry.layout)


def test_external_indices_shorten_vjepa_suffix_with_original_rope_positions():
    transformers = pytest.importorskip("transformers")
    model = transformers.VJEPA2Model(
        transformers.VJEPA2Config(
            crop_size=4,
            patch_size=2,
            frames_per_clip=4,
            tubelet_size=2,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            pred_hidden_size=24,
            pred_num_hidden_layers=1,
            pred_num_attention_heads=2,
        )
    ).eval()
    bridge = create_observation_bridge("vjepa2", model)
    pixels = torch.randn(1, 4, 3, 4, 4)
    geometry = bridge.geometry([[0, 1, 2, 3]], [[True] * 4])
    with geometry:
        model(pixel_values_videos=pixels, skip_predictor=True)
    indices = torch.tensor([[0, 1, 4, 5]])
    with indexed_gather(bridge, 0, indices, geometry.layout) as intervention:
        output = model(pixel_values_videos=pixels, skip_predictor=True).last_hidden_state
        receipt = intervention.validate_execution()
    assert output.shape[1] == 4
    assert receipt["suffix_shapes"] == {"1": [1, 4, 24]}
    assert receipt["vjepa2_original_rope_positions"]


def test_external_indices_shorten_hf_videomae_suffix_at_native_patch_positions():
    transformers = pytest.importorskip("transformers")
    model = transformers.VideoMAEModel(
        transformers.VideoMAEConfig(
            image_size=4,
            patch_size=2,
            num_frames=4,
            tubelet_size=2,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).eval()
    bridge = create_observation_bridge("videomae", model)
    pixels = torch.randn(1, 4, 3, 4, 4)
    geometry = bridge.geometry([[0, 1, 2, 3]], [[True] * 4])
    with geometry:
        model(pixel_values=pixels)
    with indexed_gather(bridge, 0, torch.tensor([[0, 1, 4, 5]]), geometry.layout) as intervention:
        output = model(pixel_values=pixels).last_hidden_state
        receipt = intervention.validate_execution()
    assert output.shape == (1, 4, 8)
    assert receipt["suffix_shapes"] == {"1": [1, 4, 8]}
    assert not receipt["vjepa2_original_rope_positions"]


def test_hf_videomae_and_vjepa2_conv3d_geometry_accept_native_projection_names():
    for encoder_id, projection_name in (("videomae", "projection"), ("vjepa2", "proj")):
        model = _TubeletModel(encoder_id, projection_name)
        bridge = create_observation_bridge(encoder_id, model)
        geometry = bridge.geometry([[0, 2, 4, 6]], [[True, True, True, True]])
        with geometry:
            model.embeddings.patch_embeddings(torch.randn(1, 3, 4, 4, 4))
        assert geometry.receipt["flatten_verified"]
        assert geometry.layout.source_coordinates.tolist() == [
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        ]


def test_real_hf_small_configs_execute_all_four_native_geometry_paths():
    transformers = pytest.importorskip("transformers")
    cases = [
        (
            "timesformer",
            transformers.TimesformerModel(
                transformers.TimesformerConfig(
                    image_size=4,
                    patch_size=2,
                    num_frames=2,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                )
            ),
            torch.randn(1, 2, 3, 4, 4),
            lambda model, pixels: model(pixel_values=pixels),
        ),
        (
            "videomae",
            transformers.VideoMAEModel(
                transformers.VideoMAEConfig(
                    image_size=4,
                    patch_size=2,
                    num_frames=4,
                    tubelet_size=2,
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=16,
                )
            ),
            torch.randn(1, 4, 3, 4, 4),
            lambda model, pixels: model(pixel_values=pixels),
        ),
        (
            "vjepa2",
            transformers.VJEPA2Model(
                transformers.VJEPA2Config(
                    crop_size=4,
                    patch_size=2,
                    frames_per_clip=4,
                    tubelet_size=2,
                    hidden_size=24,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    pred_hidden_size=24,
                    pred_num_hidden_layers=1,
                    pred_num_attention_heads=2,
                )
            ),
            torch.randn(1, 4, 3, 4, 4),
            lambda model, pixels: model(pixel_values_videos=pixels, skip_predictor=True),
        ),
    ]
    for encoder_id, model, pixels, forward in cases:
        frames = 2 if encoder_id == "timesformer" else 4
        bridge = create_observation_bridge(encoder_id, model)
        geometry = bridge.geometry([list(range(frames))], [[True] * frames])
        with geometry:
            forward(model, pixels)
        assert geometry.layout.valid_token_counts.tolist() == [
            9 if encoder_id == "timesformer" else 8
        ]
        assert geometry.receipt.get(
            "flatten_verified", geometry.receipt.get("divided_layout_verified")
        )
        assert bridge.receipt().reduction_ready is False

    # Native VideoMAEv2 remains a real patch-only Conv3d path.  The project
    # fixture uses its same upstream module contract without loading weights.
    v2 = _TubeletModel("videomaev2", "proj")
    v2.embeddings = nn.Module()
    v2.patch_embed = _TubeletPatch(projection_name="proj")
    v2.blocks = nn.ModuleList([nn.Identity()])
    bridge = create_observation_bridge("videomaev2", v2)
    geometry = bridge.geometry([[0, 1, 2, 3]], [[True] * 4])
    with geometry:
        v2.patch_embed(torch.randn(1, 3, 4, 4, 4))
    assert geometry.receipt["flatten_verified"]

    # Eager is an explicit native backend selection for this small CPU fixture,
    # never a production fallback.  It proves that the V-JEPA2 bridge reads
    # the model's returned pre-dropout probability tensor rather than context.
    eager = transformers.VJEPA2Model(
        transformers.VJEPA2Config(
            crop_size=4,
            patch_size=2,
            frames_per_clip=4,
            tubelet_size=2,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            pred_hidden_size=24,
            pred_num_hidden_layers=1,
            pred_num_attention_heads=2,
        )
    )
    eager.set_attn_implementation("eager")
    pixels = torch.randn(1, 4, 3, 4, 4)
    bridge = create_observation_bridge("vjepa2", eager)
    geometry = bridge.geometry([[0, 1, 2, 3]], [[True] * 4])
    with geometry:
        eager(pixel_values_videos=pixels, skip_predictor=True)
    sites = bridge.observation_sites([0])
    probability_site = {key: value for key, value in sites.items() if key.endswith(".probs.output")}
    collector = ProbeCollector(probability_site, bridge.probe_token_metadata(geometry, sites))
    with collector:
        eager(pixel_values_videos=pixels, skip_predictor=True)
    rows = collector.observations[0].rows
    assert any(row["probe_id"] == "P10" and row["status"] == "available" for row in rows)
    assert any(row["probe_id"] == "P11" and row["status"] == "available" for row in rows)
    assert any(row["probe_id"] == "P13" and row["status"] == "not_applicable" for row in rows)
