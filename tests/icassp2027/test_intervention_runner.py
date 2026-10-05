from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from vadbench.contracts import ClipBatch, EncoderOutput, TokenTimeline
from vadbench.research.intervention_runner import (
    InterventionRunnerUnsupportedError,
    run_intervention_diagnostic,
)


def _batch(frames: int) -> ClipBatch:
    return ClipBatch(
        frames=np.arange(frames * 4 * 4 * 3, dtype=np.uint8).reshape(1, frames, 4, 4, 3),
        timestamps_s=np.arange(frames, dtype=np.float64)[None],
        frame_indices=np.arange(frames, dtype=np.int64)[None],
        valid_mask=np.ones((1, frames), dtype=bool),
        video_ids=("labelled-file-name-must-not-reach-adapter",),
        metadata={"is_anomaly": True, "path": "must-not-reach-adapter.mp4"},
    )


def _output(tokens: torch.Tensor) -> EncoderOutput:
    count = tokens.shape[1]
    return EncoderOutput(
        features=tokens,
        pooled=tokens.mean(dim=1),
        timeline=TokenTimeline(
            start_s=torch.zeros((1, count)),
            end_s=torch.ones((1, count)),
            valid_mask=torch.ones((1, count), dtype=torch.bool),
        ),
    )


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = _Attention(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class _V2PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, 8, kernel_size=2, stride=2, bias=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.proj(pixels).flatten(2).transpose(1, 2)


class _V2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _V2PatchEmbed()
        self.blocks = nn.ModuleList([_Block(8), _Block(8)])

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(pixels)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class _V2Adapter:
    def __init__(self) -> None:
        self.model = _V2Model()
        self.encoder = self.model
        self.seen_batch = None

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        assert not train
        assert batch.video_ids == ("sample-0",)
        assert not batch.metadata
        self.seen_batch = batch
        pixels = torch.from_numpy(np.asarray(batch.frames)).permute(0, 4, 1, 2, 3).float()
        return _output(self.model(pixels))


class _HFAdapter:
    def __init__(self, model: nn.Module, encoder_id: str) -> None:
        self.model = model
        self.encoder_id = encoder_id

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        assert not train
        pixels = torch.from_numpy(np.asarray(batch.frames)).permute(0, 1, 4, 2, 3).float()
        if self.encoder_id == "vjepa2":
            raw = self.model(pixel_values_videos=pixels, skip_predictor=True)
        else:
            raw = self.model(pixel_values=pixels)
        return _output(raw.last_hidden_state)


def _assert_receipt(receipt) -> None:
    assert receipt.offline_prefix_score_diagnostic is True
    assert set(receipt.results) == {"dense", "identity", "uniform", "seeded_random", "score_high", "score_low"}
    assert receipt.results["identity"].pooled.relative_l2 < 1e-5
    assert receipt.results["identity"].gathered_shape is not None
    assert receipt.results["dense"].suffix_shapes
    budgets = {receipt.results[name].effective_budget for name in ("uniform", "seeded_random", "score_high", "score_low")}
    assert len(budgets) == 1
    assert all(receipt.results[name].indices_sha256 for name in receipt.results if name != "dense")
    assert receipt.results["dense"].indices_sha256 is None


def test_runner_uses_clean_batch_and_real_v2_style_suffix_for_both_candidates() -> None:
    torch.manual_seed(3)
    adapter = _V2Adapter()
    for candidate in ("relative_attention_update", "midlayer_temporal_change"):
        receipt = run_intervention_diagnostic(
            adapter=adapter,
            encoder_id="videomaev2",
            batch=_batch(4),
            candidate=candidate,
            relative_depth=0.5,
            budget_ratio=0.5,
            seed=7,
        )
        _assert_receipt(receipt)
    assert adapter.model.training is True
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in adapter.model.modules())


@pytest.mark.parametrize("encoder_id", ["timesformer", "videomae", "vjepa2"])
def test_runner_executes_real_small_hf_suffixes(encoder_id: str) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(4)
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
        batch = _batch(2)
        candidate = "relative_attention_update"
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
        batch = _batch(4)
        candidate = "midlayer_temporal_change"
    else:
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
                drop_path_rate=0.2,
            )
        )
        batch = _batch(4)
        candidate = "relative_attention_update"
    receipt = run_intervention_diagnostic(
        adapter=_HFAdapter(model, encoder_id),
        encoder_id=encoder_id,
        batch=batch,
        candidate=candidate,
        relative_depth=0.5,
        budget_ratio=0.5,
        seed=11,
    )
    _assert_receipt(receipt)
    assert model.training is True


def test_runner_rejects_adapter_that_hides_the_shortened_suffix_shape() -> None:
    class _BadAdapter(_V2Adapter):
        def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
            output = super().encode(batch, train=train)
            return _output(output.features.mean(dim=1, keepdim=True))

    with pytest.raises(InterventionRunnerUnsupportedError, match="timeline/reshape"):
        run_intervention_diagnostic(
            adapter=_BadAdapter(),
            encoder_id="videomaev2",
            batch=_batch(4),
            candidate="relative_attention_update",
            relative_depth=0.5,
            budget_ratio=0.5,
        )


def test_runtime_oom_is_not_mislabeled_as_unsupported_adapter() -> None:
    class _OOMAdapter(_V2Adapter):
        calls = 0

        def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
            self.calls += 1
            if self.calls == 3:
                raise torch.cuda.OutOfMemoryError("synthetic allocation failure")
            return super().encode(batch, train=train)

    adapter = _OOMAdapter()
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic allocation failure"):
        run_intervention_diagnostic(
            adapter=adapter,
            encoder_id="videomaev2",
            batch=_batch(4),
            candidate="relative_attention_update",
        )
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in adapter.model.modules())


def test_paired_controls_execute_the_same_shortened_budget_as_global_controls() -> None:
    receipt = run_intervention_diagnostic(
        adapter=_V2Adapter(), encoder_id="videomaev2", batch=_batch(4),
        candidate="relative_attention_update", include_paired=True,
    )
    budget = receipt.results["uniform"].effective_budget
    for name in ("paired_first", "paired_random", "paired_high", "paired_low"):
        assert receipt.results[name].effective_budget == budget
        assert all(shape[1] == budget for shape in receipt.results[name].suffix_shapes.values())
