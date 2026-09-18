"""Exact long-dense inference tests for the pinned author UR-DMU Test graph.

Synthetic vectors here test execution equivalence only.  They are not anomaly
scores and never stand in for a research result.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vadbench.paper.urdmu_backend import build_urdmu  # noqa: E402
from vadbench.paper.urdmu_inference import (  # noqa: E402
    estimate_query_chunk_attention_workspace,
    query_chunked_attention,
    run_urdmu_query_chunked,
    state_dict_sha256,
)


@pytest.fixture(scope="module")
def upstream() -> Path:
    root = Path(
        os.environ.get(
            "URDMU_UPSTREAM_DIR",
            str(Path(__file__).parents[2] / "outputs/icassp2027/research/author-recipes-20260918/UR-DMU"),
        )
    ).resolve()
    if not root.is_dir():
        pytest.skip("pinned external UR-DMU checkout is not available")
    pytest.importorskip("einops")
    return root


@pytest.mark.parametrize(
    ("input_dim", "batch_size", "sequence_length", "query_chunk_size"),
    [
        (768, 1, 5, 2),
        (768, 2, 17, 4),
        (1024, 1, 23, 7),
    ],
)
def test_query_chunked_test_output_matches_original_full_attention(
    upstream: Path,
    input_dim: int,
    batch_size: int,
    sequence_length: int,
    query_chunk_size: int,
) -> None:
    torch.manual_seed(20260918 + sequence_length)
    model, source_receipt = build_urdmu(input_dim, upstream, mode="Test")
    inputs = torch.randn(batch_size, sequence_length, input_dim)
    before = state_dict_sha256(model)
    with torch.inference_mode():
        original = model(inputs)
    chunked, receipt = run_urdmu_query_chunked(
        model,
        inputs,
        query_chunk_size=query_chunk_size,
        source_receipt=source_receipt,
        return_receipt=True,
    )
    assert tuple(original["frame"].shape) == tuple(chunked["frame"].shape) == (
        batch_size,
        sequence_length,
    )
    # GEMM/batched-GEMM implementation choices can reorder floating-point
    # reductions.  Equivalence is numerical, not a claim of bitwise identity.
    torch.testing.assert_close(original["frame"], chunked["frame"], atol=2e-6, rtol=2e-5)
    assert state_dict_sha256(model) == before
    assert receipt["state_dict"]["unchanged"] is True
    assert receipt["source_identity_complete"] is True
    assert receipt["query_chunk_count"] > 1
    assert receipt["complete_key_value_sequence"] is True
    assert receipt["temporal_prior"]["broadcast_direction"] == "last_dimension_key_index"


def test_context_restores_original_forwards_after_exception(upstream: Path) -> None:
    model, source_receipt = build_urdmu(768, upstream, mode="Test")
    attentions = [module for module in model.modules() if type(module).__name__ == "Attention"]
    assert len(attentions) == 2
    had_instance_forwards = ["forward" in module.__dict__ for module in attentions]
    originals = [module.forward for module in attentions]
    inputs = torch.randn(1, 9, 768)

    def fail_after_chunked_attention(*_args: object) -> None:
        raise RuntimeError("test exception")

    hook = attentions[0].register_forward_hook(fail_after_chunked_attention)
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="test exception"),
        query_chunked_attention(model, query_chunk_size=3, source_receipt=source_receipt),
    ):
        model(inputs)
    hook.remove()
    assert ["forward" in module.__dict__ for module in attentions] == had_instance_forwards
    for attention, original in zip(attentions, originals, strict=True):
        assert attention.forward.__func__ is original.__func__
        assert attention.forward.__self__ is original.__self__


def test_rejects_training_grad_and_active_dropout(upstream: Path) -> None:
    model, source_receipt = build_urdmu(768, upstream, mode="Test")
    inputs = torch.randn(1, 9, 768)
    with (
        pytest.raises(ValueError, match="torch.no_grad"),
        query_chunked_attention(model, query_chunk_size=3, source_receipt=source_receipt),
    ):
        pass
    model.train()
    with pytest.raises(ValueError, match="eval-only"):
        run_urdmu_query_chunked(model, inputs, query_chunk_size=3)
    model.eval()
    dropout = next(module for module in model.modules() if isinstance(module, torch.nn.Dropout))
    dropout.train()
    with pytest.raises(ValueError, match="active dropout"):
        run_urdmu_query_chunked(model, inputs, query_chunk_size=3)
    dropout.eval()
    with pytest.raises(ValueError, match="requiring gradients"):
        run_urdmu_query_chunked(model, inputs.requires_grad_(), query_chunk_size=3)


def test_memory_estimate_is_linear_in_fixed_chunk_not_quadratic_in_full_queries() -> None:
    short = estimate_query_chunk_attention_workspace(
        batch_size=1, sequence_length=65_536, query_chunk_size=256
    )
    long = estimate_query_chunk_attention_workspace(
        batch_size=1, sequence_length=65_536, query_chunk_size=512
    )
    assert short["effective_query_chunk_size"] == 256
    assert long["effective_query_chunk_size"] == 512
    assert short["workspace_lower_bound_bytes"] < 2 * 1024**3
    assert long["workspace_lower_bound_bytes"] > short["workspace_lower_bound_bytes"]
    assert short["workspace_lower_bound_bytes"] < 13 * 65_536 * 65_536 * 4
