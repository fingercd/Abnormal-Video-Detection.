from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from vadbench.research.sdpa_observer import (
    SDPAQueryCapture,
    SDPAQueryRowObserver,
    UnsupportedSDPAObservationError,
)


class _FakeRegistry(dict[str, Callable]):
    """Minimal AttentionInterface model with class-style global/local lookup."""

    def __init__(self, backend: Callable) -> None:
        super().__init__()
        self._global_mapping = {"sdpa": backend}
        self._local_mapping: dict[str, Callable] = {}

    def __getitem__(self, key: str) -> Callable:
        return self._local_mapping[key] if key in self._local_mapping else self._global_mapping[key]

    def __setitem__(self, key: str, value: Callable) -> None:
        self._local_mapping[key] = value

    def __delitem__(self, key: str) -> None:
        del self._local_mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._local_mapping or key in self._global_mapping


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("offset", torch.tensor(3.0))
        self.eval()


def _backend(calls: list[tuple], returned: list[object]):
    def run(module, query, key, value, attention_mask=None, **kwargs):
        calls.append((module, query, key, value, attention_mask, kwargs))
        result = (query.sum() * module.weight + module.offset, None)
        returned.append(result)
        return result

    return run


def _inputs(*, heads: int = 1, q_len: int = 5, key_len: int = 4):
    query = torch.tensor(
        [[[[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25], [2.0, 1.0], [3.0, -0.5]]]],
    )[:, :heads, :q_len].detach().requires_grad_()
    key = torch.tensor(
        [[[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.25], [2.0, 0.0]]]]
    )[:, :heads, :key_len]
    value = key.clone()
    return query, key, value


def _rope(x: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)
    angles = positions * 0.31
    cos, sin = angles.cos(), angles.sin()
    return torch.stack(
        (x[..., 0] * cos - x[..., 1] * sin, x[..., 0] * sin + x[..., 1] * cos), dim=-1
    )


def test_capture_uses_post_rope_qk_and_returns_original_backend_result():
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    q_pre, k_pre, value = _inputs()

    with SDPAQueryRowObserver(
        registry,
        {"encoder.layers.5.attn": module},
        captures.append,
        max_queries=2,
        source_identity={"modeling_vjepa2_sha256": "known-source"},
    ):
        # This producer is the test's non-zero RoPE boundary: SDPA receives
        # rotated Q/K, while the inputs above remain the pre-RoPE projections.
        result = registry["sdpa"](module, _rope(q_pre), _rope(k_pre), value, None, scaling=0.37)

    assert result is returned[0]
    assert len(calls) == len(captures) == 1
    capture: SDPAQueryCapture = captures[0]
    assert capture.site == "encoder.layers.5.attn"
    assert capture.query_ids.tolist() == [0, 4]
    assert capture.metadata["source_kind"] == "reconstructed_from_native_post_rope_qk"
    assert capture.metadata["effective_scale"] == pytest.approx(0.37)
    assert capture.metadata["source_fingerprints"]["modeling_vjepa2_sha256"] == "known-source"

    q_post, k_post = _rope(q_pre).float(), _rope(k_pre).float()
    expected = torch.softmax(q_post @ k_post.transpose(-2, -1) * 0.37, dim=-1)
    torch.testing.assert_close(capture.sampled_probabilities, expected[:, :, [0, 4], :])
    pre_rope = torch.softmax(q_pre.float() @ k_pre.float().transpose(-2, -1) * 0.37, dim=-1)
    assert not torch.allclose(capture.sampled_probabilities, pre_rope[:, :, [0, 4], :])


def test_bounded_selected_rows_match_full_reference_and_do_not_allocate_full_scores():
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    query, key, value = _inputs()
    with SDPAQueryRowObserver(registry, {"site": module}, captures.append, 2, {"source": "fake"}):
        registry["sdpa"](module, query, key, value, scaling=0.25)

    capture = captures[0]
    reference = torch.softmax(query.float() @ key.float().transpose(-2, -1) * 0.25, dim=-1)
    torch.testing.assert_close(capture.sampled_probabilities, reference[:, :, capture.query_ids, :])
    assert capture.metadata["sampled_score_shape"] == (1, 1, 2, 4)
    assert capture.metadata["sampled_score_shape"][-2] < capture.metadata["query_count"]
    assert capture.metadata["key_count"] == 4


def test_plain_dict_registry_calls_original_backend_and_restores_its_entry():
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    backend = _backend(calls, returned)
    registry = {"sdpa": backend}
    query, key, value = _inputs()

    with SDPAQueryRowObserver(registry, {"site": module}, captures.append, 2, {}):
        actual = registry["sdpa"](module, query, key, value)
        assert actual is returned[0]

    assert len(calls) == len(captures) == 1
    assert registry == {"sdpa": backend}


def test_reconstruction_stays_float32_when_cpu_autocast_is_enabled():
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    query, key, value = _inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), SDPAQueryRowObserver(
        registry, {"site": module}, captures.append, 2, {}
    ):
        registry["sdpa"](module, query, key, value, scaling=0.25)

    capture = captures[0]
    expected = torch.softmax(query.float() @ key.float().transpose(-2, -1) * 0.25, dim=-1)
    assert capture.sampled_probabilities.dtype == torch.float32
    assert capture.metadata["reconstruction_dtype"] == "float32"
    torch.testing.assert_close(capture.sampled_probabilities, expected[:, :, capture.query_ids, :])


@pytest.mark.parametrize(
    ("extra", "heads"),
    [
        ({"attention_mask": torch.zeros(1, 1, 5, 4)}, 1),
        ({"head_mask": torch.ones(1)}, 1),
        ({"is_causal": True}, 1),
        ({"dropout": 0.1}, 1),
        ({"enable_gqa": True}, 2),
    ],
)
def test_selected_unsupported_sdpa_variants_are_rejected(extra, heads):
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    query, key, value = _inputs(heads=heads)
    if heads == 2:
        query = query.repeat(1, 2, 1, 1)
    with SDPAQueryRowObserver(
        registry, {"site": module}, captures.append, 2, {}
    ), pytest.raises(UnsupportedSDPAObservationError):
        registry["sdpa"](module, query, key, value, **extra)
    assert not calls and not captures


def test_backend_once_output_identity_and_grad_parameters_buffers_and_rng_are_unchanged():
    module = _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    query, key, value = _inputs()
    before_state = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    grad_enabled = torch.is_grad_enabled()
    with SDPAQueryRowObserver(registry, {"site": module}, captures.append, 2, {}):
        actual = registry["sdpa"](module, query, key, value)
    assert actual is returned[0] and len(calls) == 1 and torch.is_grad_enabled() is grad_enabled
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    actual[0].backward()
    assert module.weight.grad is not None and query.grad is not None
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, before_state[name], rtol=0, atol=0)


def test_unselected_module_is_transparent_even_for_unobserved_sdpa_variants():
    selected, other = _ToyAttention(), _ToyAttention()
    calls, returned, captures = [], [], []
    registry = _FakeRegistry(_backend(calls, returned))
    query, key, value = _inputs()
    with SDPAQueryRowObserver(registry, {"site": selected}, captures.append, 2, {}):
        actual = registry["sdpa"](other, query, key, value, dropout=0.5, is_causal=True)
    assert actual is returned[0]
    assert len(calls) == 1 and not captures


def test_registry_local_and_global_state_restore_after_exception_and_repeated_entries():
    module = _ToyAttention()
    global_calls, global_returned = [], []
    global_backend = _backend(global_calls, global_returned)
    registry = _FakeRegistry(global_backend)
    observer = SDPAQueryRowObserver(registry, {"site": module}, lambda _: None, 1, {})

    with pytest.raises(RuntimeError, match="boom"), observer:
        assert "sdpa" in registry._local_mapping
        raise RuntimeError("boom")
    assert "sdpa" not in registry._local_mapping
    assert registry["sdpa"] is global_backend
    with observer:
        pass
    assert "sdpa" not in registry._local_mapping

    local_calls, local_returned = [], []
    local_backend = _backend(local_calls, local_returned)
    registry._local_mapping["sdpa"] = local_backend
    with SDPAQueryRowObserver(registry, {"site": module}, lambda _: None, 1, {}):
        assert registry["sdpa"] is not local_backend
    assert registry._local_mapping["sdpa"] is local_backend
