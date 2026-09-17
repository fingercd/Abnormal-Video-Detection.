"""Independent real-HF checks for the neutral indexed bridge intervention."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from vadbench.token_reduction.bridges import (
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)


def _case(encoder_id: str):
    transformers = pytest.importorskip("transformers")
    if encoder_id == "timesformer":
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
        )
        pixels = torch.randn(1, 2, 3, 4, 4)
        forward = partial(model, pixel_values=pixels)
        frame_indices = [[0, 1]]
        kept = torch.tensor([[0, 1, 2, 3, 4]])
    elif encoder_id == "videomae":
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
        )
        pixels = torch.randn(1, 4, 3, 4, 4)
        forward = partial(model, pixel_values=pixels)
        frame_indices = [[0, 1, 2, 3]]
        kept = torch.tensor([[0, 1, 4, 5]])
    elif encoder_id == "vjepa2":
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
        )
        pixels = torch.randn(1, 4, 3, 4, 4)
        forward = partial(model, pixel_values_videos=pixels, skip_predictor=True)
        frame_indices = [[0, 1, 2, 3]]
        kept = torch.tensor([[0, 1, 4, 5]])
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(encoder_id)
    bridge = create_observation_bridge(encoder_id, model)
    geometry = bridge.geometry(frame_indices, [[True] * len(frame_indices[0])])
    with torch.no_grad(), geometry:
        forward()
    return model, bridge, geometry.layout, forward, kept


@pytest.mark.parametrize("encoder_id", ["timesformer", "videomae", "vjepa2"])
def test_identity_indexed_suffix_matches_dense_and_retains_backward(encoder_id: str):
    model, bridge, layout, forward, _ = _case(encoder_id)
    model.eval()
    with torch.no_grad():
        dense = forward().last_hidden_state
        with indexed_gather(bridge, 0, identity_indices(layout), layout) as intervention:
            identity = forward().last_hidden_state
            receipt = intervention.validate_execution()
    torch.testing.assert_close(identity, dense)
    assert receipt["gathered_tokens"] == layout.token_capacity

    model.train()
    model.zero_grad(set_to_none=True)
    with indexed_gather(bridge, 0, identity_indices(layout), layout) as intervention:
        loss = forward().last_hidden_state.square().mean()
        loss.backward()
        intervention.validate_execution()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("encoder_id", ["timesformer", "videomae", "vjepa2"])
def test_reduced_indexed_suffix_has_real_shape_and_context_cleanup(encoder_id: str):
    model, bridge, layout, forward, kept = _case(encoder_id)
    model.eval()
    with torch.no_grad(), indexed_gather(bridge, 0, kept, layout) as intervention:
        output = forward().last_hidden_state
        receipt = intervention.validate_execution()
    assert output.shape[1] == kept.shape[1]
    assert receipt["suffix_shapes"] == {"1": [1, kept.shape[1], output.shape[-1]]}
    if encoder_id == "vjepa2":
        assert intervention.suffix_position_masks
        assert all(torch.equal(mask, kept) for mask in intervention.suffix_position_masks.values())

    with pytest.raises(RuntimeError, match="fixture failure"), indexed_gather(bridge, 0, kept, layout):
        raise RuntimeError("fixture failure")
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in bridge._blocks)
    with torch.no_grad():
        recovered = forward().last_hidden_state
    assert recovered.shape[1] == layout.token_capacity
