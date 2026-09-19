"""CPU integration contracts; fixture math is not native-model validation."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.contracts import ClipBatch
from vadbench.hashing import sha256_file
from vadbench.paper.stages import observe_clip
from vadbench.token_reduction.bridges import BridgeUnsupportedError, create_observation_bridge


class FixtureAttention(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.registry = registry
        self.num_attention_heads = 1
        self.query = nn.Linear(4, 4)
        self.key = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        q, k, v = (projection(x).unsqueeze(1) for projection in (self.query, self.key, self.value))
        context, _ = self.registry["sdpa"](
            self, q, k, v, None, scaling=0.5, is_causal=False, dropout=0.0
        )
        return (self.proj(context.flatten(2)),)


class FixturePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 4, kernel_size=2, stride=2)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class FixtureBlock(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, x):
        return (x + self.attention(x)[0],)


class FixtureModel(nn.Module):
    def __init__(self, attention_type, registry):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = FixturePatch()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([FixtureBlock(attention_type(registry)) for _ in range(2)])

    def forward(self, x):
        x = self.embeddings.patch_embeddings(x)
        for block in self.encoder.layer:
            x = block(x)[0]
        return x


class FixtureAdapter:
    def __init__(self, model):
        self.encoder = SimpleNamespace(model=model)

    def encode(self, batch):
        x = torch.as_tensor(batch.frames, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        features = self.encoder.model(x)
        return SimpleNamespace(features=features, pooled=features.mean(1))


@pytest.fixture
def native_seam(monkeypatch, tmp_path):
    # Deliberately emulate the inspected dispatch seam, without installing HF.
    calls = []

    def sdpa_attention_forward(module, q, k, v, attention_mask, *, scaling, is_causal, dropout):
        calls.append(module)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, scale=scaling,
            is_causal=is_causal, dropout_p=dropout,
        ).transpose(1, 2).contiguous(), None

    sdpa_attention_forward.__module__ = "transformers.integrations.sdpa_attention"
    registry = {"sdpa": sdpa_attention_forward}
    modeling = ModuleType("transformers.models.vjepa2.modeling_vjepa2")
    source = tmp_path / "fixture_modeling.py"
    source.write_text("# Explicit test fixture, not native V-JEPA source\n", encoding="utf-8")
    modeling.__file__ = str(source)
    modeling.ALL_ATTENTION_FUNCTIONS = registry
    attention_type = type("VJEPA2RopeAttention", (FixtureAttention,), {"__module__": modeling.__name__})
    modeling.VJEPA2RopeAttention = attention_type
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    model = FixtureModel(attention_type, registry)
    return model, registry, sdpa_attention_forward, calls, source


def batch():
    return ClipBatch(
        frames=np.ones((1, 4, 4, 4, 3), dtype=np.uint8),
        timestamps_s=np.array([[0., .2, .4, .6]]),
        frame_indices=np.array([[0, 2, 4, 6]]), valid_mask=np.ones((1, 4), dtype=bool),
        video_ids=("source",),
    )


def test_recipe_uses_selected_live_module_and_actual_source_hash(native_seam):
    model, registry, backend, _, source = native_seam
    model.eval()
    bridge = create_observation_bridge("vjepa2", model)
    recipe = bridge.sdpa_observation_recipe([1])
    assert recipe["registry"] is registry
    assert recipe["targets"] == {"block.1.attn.probs_reconstructed": model.encoder.layer[1].attention}
    assert recipe["source_identity"]["modeling_sha256"] == sha256_file(source)
    assert recipe["source_identity"]["sdpa_sha256"] == sha256_file(__file__)
    assert registry["sdpa"] is backend
    model.train()
    with pytest.raises(BridgeUnsupportedError, match="eval mode"):
        bridge.sdpa_observation_recipe([1])


def test_stage_preserves_native_missing_sites_and_records_reconstruction(native_seam):
    model, registry, backend, calls, _ = native_seam
    result = observe_clip(FixtureAdapter(model), "vjepa2", batch(), {
        "depths": [1.0], "probes": ["P10", "P11", "P13"],
        "max_records": 8, "max_tokens": 8, "max_queries": 3,
    })
    architecture = result["architecture"]
    assert architecture["missing_observation_sites"] == ["block.1.attn.probs.output"]
    assert not architecture["probe_ready"]
    assert list(architecture["reconstructed_attention_sites"]) == ["block.1.attn.probs_reconstructed"]
    assert architecture["cls_probe_applicability"] == "not_applicable_no_cls"
    assert all(value == 0 for value in architecture["parity"].values())
    rows = [row for observation in result["observations"] for row in observation.rows]
    assert {row["status"] for row in rows if row["probe_id"] == "P13"} == {"not_applicable"}
    reconstructed = [row for row in rows if "attention_capture" in row]
    assert len(reconstructed) == 5
    assert reconstructed[0]["attention_capture"]["key_count"] == 8
    assert len(reconstructed[0]["attention_capture"]["query_token_ids"]) == 3
    assert len(calls) == 6  # Three full passes, two blocks; no replacement forward.
    assert registry["sdpa"] is backend and model.training


def test_token_only_stage_does_not_install_reconstruction(native_seam, monkeypatch):
    model, registry, backend, _, _ = native_seam
    from vadbench.token_reduction.bridges import EncoderBridge

    def forbidden(*args):
        raise AssertionError("token-only observation must not request SDPA recipe")

    monkeypatch.setattr(EncoderBridge, "sdpa_observation_recipe", forbidden)
    result = observe_clip(FixtureAdapter(model), "vjepa2", batch(), {
        "depths": [1.0], "probes": ["P01"],
        "max_records": 32, "max_tokens": 8, "max_queries": 3,
    })
    assert result["architecture"]["reconstructed_attention_sites"] == {}
    assert registry["sdpa"] is backend


def test_stage_capture_failure_restores_registry_and_hooks(native_seam, monkeypatch):
    from vadbench.research.collectors import ProbeCollector

    model, registry, backend, _, _ = native_seam

    def fail_capture(*args):
        raise RuntimeError("intentional collector failure")

    monkeypatch.setattr(ProbeCollector, "capture_reconstructed_attention", fail_capture)
    with pytest.raises(RuntimeError, match="intentional collector failure"):
        observe_clip(FixtureAdapter(model), "vjepa2", batch(), {
            "depths": [1.0], "probes": ["P10"],
            "max_records": 8, "max_tokens": 8, "max_queries": 3,
        })
    assert registry["sdpa"] is backend and model.training
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_installed_hf_tiny_vjepa_sdpa_stage_parity_and_registry_restoration():
    transformers = pytest.importorskip("transformers")
    from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import ALL_ATTENTION_FUNCTIONS, VJEPA2Model

    config = VJEPA2Config(
        crop_size=4, patch_size=2, frames_per_clip=4, tubelet_size=2,
        hidden_size=24, num_attention_heads=2, num_hidden_layers=2,
        pred_hidden_size=24, pred_num_attention_heads=2, pred_num_hidden_layers=1,
    )
    config._attn_implementation = "sdpa"
    model = VJEPA2Model(config).eval()

    class NativeAdapter:
        def __init__(self):
            self.encoder = SimpleNamespace(model=model)

        def encode(self, clip):
            inputs = torch.as_tensor(clip.frames, dtype=torch.float32).permute(0, 1, 4, 2, 3)
            features = model.get_vision_features(inputs)
            return SimpleNamespace(features=features, pooled=features.mean(1))

    registry_before = dict(ALL_ATTENTION_FUNCTIONS._local_mapping)
    backend_before = ALL_ATTENTION_FUNCTIONS["sdpa"]
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    result = observe_clip(NativeAdapter(), "vjepa2", batch(), {
        "depths": [0.5, 1.0], "probes": ["P10", "P11", "P13"],
        "max_records": 8, "max_tokens": 8, "max_queries": 3,
    })
    receipt = result["architecture"]
    assert all(delta == 0 for delta in receipt["parity"].values()), transformers.__version__
    assert receipt["grid"] == [2, 2, 2] and receipt["flatten_verified"]
    assert len(receipt["missing_observation_sites"]) == 2
    assert len(receipt["reconstructed_attention_sites"]) == 2
    assert receipt["cls_probe_applicability"] == "not_applicable_no_cls"
    for capture in receipt["reconstructed_attention_sites"].values():
        assert capture["query_shape"] == (1, 2, 8, 12)
        assert capture["sampled_score_shape"] == (1, 2, 3, 8)
        assert capture["source_kind"] == "reconstructed_from_native_post_rope_qk"
    assert dict(ALL_ATTENTION_FUNCTIONS._local_mapping) == registry_before
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is backend_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
