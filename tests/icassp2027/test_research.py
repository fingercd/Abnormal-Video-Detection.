from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.research import (
    CohortError,
    CohortIndex,
    CohortRecord,
    ProbeCollector,
    ProbeLimits,
    ProbeTokenMetadata,
    join_probe_rows,
)


def record(video="a", clip="a:0", partition="fit", role="explore", **kwargs):
    return CohortRecord(
        video_id=video,
        clip_id=clip,
        official_split="train",
        partition=partition,
        role=role,
        weak_label=0,
        label_source="video_weak",
        **kwargs,
    )


def metadata(n=4):
    return ProbeTokenMetadata(
        valid_mask=np.ones((1, n), dtype=bool),
        coordinates=np.array([[[t, 0, w] for t in range(2) for w in range(n // 2)]]),
        coordinate_source="fixture-conv-grid",
        special_token_indices=(),
        has_cls=False,
    )


def test_cohort_isolation_checks_both_video_and_source_and_rejects_test_truth():
    for records in (
        [record(source_id="shared"), record("b", "b:0", "confirm", "confirm", source_id="shared")],
        [record(source_id="one"), record("a", "a:1", "confirm", "confirm", source_id="two")],
        [replace(record(), official_split="test")],
        [record(temporal_label=[1, 5])],
        [record(partition="confirm")],
    ):
        with pytest.raises(CohortError):
            CohortIndex.from_records(records, policy="W")


def test_join_rejects_wrong_video_and_multi_clip_lookup_is_ambiguous():
    cohort = CohortIndex.from_records([record(), record(clip="a:1")], policy="W")
    with pytest.raises(CohortError, match="多个 clip"):
        cohort.record_for_video("a")
    with pytest.raises(CohortError, match="video_id"):
        join_probe_rows([{"clip_id": "a:0", "video_id": "b"}], cohort)
    result = join_probe_rows([{"clip_id": "a:0", "video_id": "a"}], cohort)
    assert result[0]["weak_label"] == 0


def test_collector_preserves_output_cleans_hooks_on_success_and_failure():
    model = nn.Linear(3, 3)
    x = torch.randn(1, 4, 3, requires_grad=True)
    expected = model(x)
    observer = ProbeCollector({"block.0.input": model, "block.0.output": model}, metadata())
    actual = observer.run(model, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert x.grad is not None
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert observer.observations and not observer._input_norms
    with pytest.raises(RuntimeError, match="fixture failure"), observer:
        model(x)
        raise RuntimeError("fixture failure")
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_padding_and_special_tokens_do_not_bias_patch_statistics():
    m = metadata()
    m = replace(
        m,
        valid_mask=np.array([[True, True, True, False]]),
        special_token_indices=(0,),
        has_cls=True,
    )
    values = torch.tensor([[[1000.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1000.0, 0.0]]])
    observer = ProbeCollector({"block.0.output": nn.Identity()}, m)
    observer.run(next(iter(observer.observation_sites.values())), values)
    row = next(
        row
        for row in observer.observations[0].rows
        if row["statistic_name"] == "activation_norm_median"
    )
    assert row["statistic_value"] == pytest.approx(1 / np.sqrt(2))
    assert row["num_valid_tokens"] == 2


def test_known_spectrum_and_temporal_direction():
    values = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    layer = nn.Identity()
    observer = ProbeCollector({"block.0.output": layer}, metadata())
    observer.run(layer, values)
    rows = {row["statistic_name"]: row for row in observer.observations[0].rows}
    assert rows["effective_rank"]["statistic_value"] == pytest.approx(1)
    assert rows["same_position_adjacent_time_change_median"]["statistic_value"] == pytest.approx(0)
    assert not any("high_change_fraction" in name for name in rows)


def test_attention_uses_all_keys_and_checks_native_normalisation():
    layer = nn.Identity()
    observer = ProbeCollector(
        {"block.0.attn.probs.input": layer}, metadata(), ProbeLimits(max_attention_queries=2)
    )
    attention = torch.full((1, 2, 4, 4), 0.25)
    observer.run(layer, attention)
    rows = observer.observations[0].rows
    entropy = next(
        row for row in rows if row["statistic_name"] == "outgoing_attention_entropy_mean"
    )
    assert entropy["statistic_value"] == pytest.approx(1)
    assert entropy["query_count"] == 2
    assert all(row["status"] == "not_applicable" for row in rows if row["probe_id"] == "P13")
    observer.run(layer, attention * 2)
    assert all(
        row["status"] == ("not_applicable" if row["probe_id"] == "P13" else "unavailable")
        for row in observer.observations[0].rows
    )


def test_missing_geometry_is_explicit_and_limits_are_reported():
    layer = nn.Identity()
    observer = ProbeCollector(
        {"block.0.output": layer}, ProbeTokenMetadata(), ProbeLimits(max_observations=1)
    )
    with observer:
        layer(torch.ones(1, 4, 2))
        layer(torch.ones(1, 4, 2))
    assert observer.dropped_observations == 1
    assert all(row["statistic_value"] is None for row in observer.observations[0].rows)


def test_qkv_projection_norm_is_not_reported_as_a_residual_branch_update():
    block_input, qkv, update = nn.Identity(), nn.Identity(), nn.Identity()
    observer = ProbeCollector(
        {"block.0.input": block_input, "block.0.attn.qkv": qkv, "block.0.attn.output": update},
        metadata(),
    )
    with observer:
        block_input(torch.ones(1, 4, 2))
        qkv(torch.ones(1, 4, 6))
        update(torch.ones(1, 4, 2))
    by_site = {item.site: item for item in observer.observations}
    assert not any("update" in row["statistic_name"] for row in by_site["block.0.attn.qkv"].rows)
    ratio = next(
        row for row in by_site["block.0.attn.output"].rows if "update" in row["statistic_name"]
    )
    assert ratio["statistic_value"] == pytest.approx(1)


def test_registered_but_nontriggering_or_tensorless_sites_emit_explicit_receipts():
    active = nn.Identity()
    dormant = nn.Identity()
    collector = ProbeCollector(
        {"block.0.output": active, "block.0.attn.probs.output": dormant}, metadata()
    )
    collector.run(active, torch.ones(1, 4, 2))
    assert collector.missing_sites == ("block.0.attn.probs.output",)
    missing = next(item for item in collector.observations if item.site.endswith("probs.output"))
    assert {row["probe_id"] for row in missing.rows} == {"P10", "P11", "P13"}
    for row in missing.rows:
        assert row["status"] == ("not_applicable" if row["probe_id"] == "P13" else "unavailable")
        assert row["statistic_name"] == ("cls_attention" if row["probe_id"] == "P13" else "hook_not_triggered")


class _NativeAttentionTuple(nn.Module):
    def __init__(self, second: torch.Tensor | None | object) -> None:
        super().__init__()
        self.second = second

    def forward(self, value: torch.Tensor):
        context = value * 2
        if self.second is _CONTEXT_ONLY:
            return (context,)
        return context, self.second


_CONTEXT_ONLY = object()


@pytest.mark.parametrize("second", [_CONTEXT_ONLY, None], ids=["context_only", "context_none"])
def test_probability_site_rejects_context_only_native_attention_tuples(second):
    layer = _NativeAttentionTuple(second)
    values = torch.ones(1, 4, 2, requires_grad=True)
    collector = ProbeCollector({"block.0.attn.probs.output": layer}, metadata())

    output = collector.run(layer, values)
    torch.testing.assert_close(output[0], values * 2)
    output[0].sum().backward()
    assert values.grad is not None
    observation = collector.observations[0]
    assert {row["probe_id"] for row in observation.rows} == {"P10", "P11", "P13"}
    for row in observation.rows:
        assert row["status"] == ("not_applicable" if row["probe_id"] == "P13" else "unavailable")
        assert row["statistic_name"] == (
            "cls_attention" if row["probe_id"] == "P13" else "hook_output_has_no_usable_tensor"
        )
    assert collector.missing_sites == ("block.0.attn.probs.output",)


def test_probability_site_accepts_only_native_second_probability_tensor():
    attention = torch.full((1, 2, 4, 4), 0.25)
    layer = _NativeAttentionTuple(attention)
    collector = ProbeCollector({"block.0.attn.probs.output": layer}, metadata())

    output = collector.run(layer, torch.ones(1, 4, 2))

    assert tuple(output[1].shape) == (1, 2, 4, 4)
    rows = collector.observations[0].rows
    assert any(row["probe_id"] == "P10" and row["status"] == "available" for row in rows)
    assert any(row["probe_id"] == "P11" and row["status"] == "available" for row in rows)
    assert all(row["probe_id"] != "P01" for row in rows)


def test_reconstructed_attention_keeps_native_missing_and_serializes_source():
    import json

    native_site = "block.0.attn.probs.output"
    auxiliary_site = "block.0.attn.probs.reconstructed"
    layer = _NativeAttentionTuple(None)
    collector = ProbeCollector({native_site: layer}, metadata())
    probabilities = torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]])
    evidence = {"source_kind": "reconstructed_from_native_post_rope_qk", "key_count": 4}
    with collector:
        layer(torch.ones(1, 4, 2))
        collector.capture_reconstructed_attention(auxiliary_site, probabilities, [0, 3], evidence)
    assert collector.missing_sites == (native_site,)
    assert collector.reconstructed_attention_sites == (auxiliary_site,)
    auxiliary = next(item for item in collector.observations if item.site == auxiliary_site)
    assert {row["probe_id"] for row in auxiliary.rows} == {"P10", "P11"}
    assert all(row["status"] == "available" and row["query_count"] == 2 for row in auxiliary.rows)
    serialized = json.loads(json.dumps(auxiliary.to_rows(
        run_id="fixture", encoder_id="vjepa2", checkpoint_digest=None,
        clip_ids=["a:0"], video_ids=["a"],
    )))
    for row in serialized:
        assert row["attention_capture"]["query_token_ids"] == [0, 3]
        assert row["attention_capture"]["query_coordinates"] == [[0, 0, 0], [1, 0, 1]]
        assert row["attention_capture"]["source_kind"] == evidence["source_kind"]
    native = next(item for item in collector.observations if item.site == native_site)
    assert all(row["status"] == "not_applicable" for row in native.rows if row["probe_id"] == "P13")
