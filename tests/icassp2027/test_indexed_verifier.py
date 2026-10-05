from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vadbench.contracts import ClipBatch


def _script_module():
    path = Path(__file__).parents[2] / "scripts/icassp2027/verify_indexed_active.py"
    spec = importlib.util.spec_from_file_location("indexed_active_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_case(encoder_id: str):
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
        return model, lambda: model(pixel_values=pixels), [[0, 1]], {"pixel_values": pixels}
    if encoder_id == "videomae":
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
        return (
            model,
            lambda: model(pixel_values=pixels),
            [[0, 1, 2, 3]],
            {"pixel_values": pixels},
        )
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
    return (
        model,
        lambda: model.get_vision_features(pixel_values_videos=pixels),
        [[0, 1, 2, 3]],
        {"pixel_values_videos": pixels},
    )


@pytest.mark.parametrize("encoder_id", ["timesformer", "videomae", "vjepa2"])
def test_verify_native_uses_four_real_forwards_identity_shape_and_leaf_gradient(encoder_id: str):
    verifier = _script_module()
    model, forward, frames, inputs = _native_case(encoder_id)
    receipt = verifier._verify_native(
        model=model,
        forward=forward,
        encoder=encoder_id,
        frame_indices=frames,
        valid_mask=[[True] * len(frames[0])],
        depth=0,
        inputs=inputs,
    )
    assert receipt["identity_max_abs"] <= 1e-6
    assert receipt["identity_mean_pooled_max_abs"] <= 1e-6
    assert receipt["reduced_shape"][1] < receipt["dense_shape"][1]
    assert receipt["leaf_bias_grad_l1"] > 0
    assert receipt["initial_hook_counts"] == receipt["final_hook_counts"]
    assert receipt["actual_parameter"]["device"] == "cpu"
    assert receipt["actual_dense"]["device"] == "cpu"
    assert receipt["leaf_bias"]["device"] == receipt["actual_dense"]["device"]
    if encoder_id == "timesformer":
        assert receipt["reduced_shape"][1] == 5


def test_vjepa_components_uses_worker_mapping_and_get_vision_features():
    verifier = _script_module()
    pixels = torch.randn(1, 4, 3, 4, 4)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def get_vision_features(self, **inputs):
            self.seen = inputs
            return inputs["pixel_values_videos"]

    class Worker:
        def __init__(self) -> None:
            self.model = Model()

        def _prepare_inputs(self, _batch):
            from transformers.image_processing_base import BatchFeature

            return BatchFeature({"pixel_values_videos": pixels})

    worker = Worker()
    adapter = SimpleNamespace(bridge=SimpleNamespace(encoder=worker))
    batch = ClipBatch(
        frames=np.zeros((1, 4, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.arange(4, dtype=np.float64)[None],
        valid_mask=np.ones((1, 4), dtype=bool),
        video_ids=("fixture",),
    )
    model, forward, inputs = verifier._components(adapter, "vjepa2", batch)
    assert model is worker.model
    torch.testing.assert_close(forward(), pixels)
    assert inputs["pixel_values_videos"] is pixels
    assert worker.model.seen == {"pixel_values_videos": pixels}
