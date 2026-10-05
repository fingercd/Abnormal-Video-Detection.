from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from vadbench.research import ProbeCollector, ProbeLimits, ProbeSiteMetadata, ProbeTokenMetadata


def _metadata(
    site: str,
    layout: str,
    *,
    frames: int = 2,
    spatial_tokens: int = 2,
    leading_cls: bool = True,
) -> ProbeTokenMetadata:
    coordinates = np.asarray(
        [
            [[0, 0, 0]]
            + [[time, patch, 0] for patch in range(spatial_tokens) for time in range(frames)]
            for _source in range(2)
        ],
        dtype=np.int64,
    )
    return ProbeTokenMetadata(
        valid_mask=np.ones((2, 1 + frames * spatial_tokens), dtype=bool),
        coordinates=coordinates,
        coordinate_source="verified-p-major-time-fast-test-layout",
        special_token_indices=(0,) if leading_cls else (),
        has_cls=leading_cls,
        site_metadata={
            site: ProbeSiteMetadata(layout, frames=frames, spatial_tokens=spatial_tokens)
        },
    )


def _by(rows: tuple[dict, ...], index: int, statistic: str) -> dict:
    return next(
        row for row in rows if row["batch_index"] == index and row["statistic_name"] == statistic
    )


def test_temporal_groups_are_per_clip_and_emit_audit_fields():
    site, layer = "block.0.attn.probs.input", nn.Identity()
    attention = torch.tensor(
        [
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]]],
        ]
    )
    collector = ProbeCollector({site: layer}, _metadata(site, "timesformer_temporal"))
    collector.run(layer, attention)
    rows = collector.observations[0].rows
    assert _by(rows, 0, "outgoing_attention_entropy_mean")["statistic_value"] == pytest.approx(0.5)
    assert _by(rows, 1, "outgoing_attention_entropy_mean")["statistic_value"] == pytest.approx(0.0)
    row = _by(rows, 0, "sampled_query_incoming_attention_gini")
    assert row["num_valid_tokens"] == row["local_key_count"] == 2
    assert row["query_count"] == row["native_query_rows"] == 4
    assert row["native_domain"] == "temporal"
    assert row["groups_per_clip"] == 2
    assert row["temporary_cls_policy"] == "none"
    rendered = collector.observations[0].to_rows(
        run_id="r",
        encoder_id="timesformer",
        checkpoint_digest="d",
        clip_ids=("a:0", "b:0"),
        video_ids=("a", "b"),
    )
    assert {row["video_id"] for row in rendered} == {"a", "b"}
    assert all("native_domain" in row for row in rendered)
    cls = next(row for row in rows if row["probe_id"] == "P13")
    assert cls["statistic_name"] == "global_cls_aggregation_not_implemented"
    assert cls["status"] == "unavailable"


def test_spatial_k5_nonuniform_statistics_and_temporary_cls_policy():
    site, layer = "block.1.attn.probs.input", nn.Identity()
    p = [0.5, 0.2, 0.15, 0.1, 0.05]
    uniform, one_hot = [0.2] * 5, [1.0, 0.0, 0.0, 0.0, 0.0]
    # B*T groups; each local query row has the same distribution.
    attention = torch.tensor([[([p] * 5)], [([uniform] * 5)], [([one_hot] * 5)], [([one_hot] * 5)]])
    collector = ProbeCollector(
        {site: layer}, _metadata(site, "timesformer_spatial", spatial_tokens=4)
    )
    collector.run(layer, attention)
    rows = collector.observations[0].rows
    expected_entropy = (-(sum(x * math.log(x) for x in p) / math.log(5)) + 1.0) / 2
    # P10/P11 sum the four largest masses: p gives .95 and uniform .8, so the
    # equal-group mean is .875.
    assert _by(rows, 0, "outgoing_attention_entropy_mean")["statistic_value"] == pytest.approx(
        expected_entropy
    )
    assert _by(rows, 0, "outgoing_attention_top4_mass_mean")["statistic_value"] == pytest.approx(
        0.875
    )
    assert _by(rows, 0, "sampled_query_incoming_attention_gini")[
        "statistic_value"
    ] == pytest.approx(0.2)
    assert _by(rows, 1, "sampled_query_incoming_attention_entropy")[
        "statistic_value"
    ] == pytest.approx(0.0)
    assert _by(rows, 1, "sampled_query_incoming_attention_gini")[
        "statistic_value"
    ] == pytest.approx(0.8)
    assert (
        _by(rows, 0, "outgoing_attention_entropy_mean")["temporary_cls_policy"]
        == "included_as_local_key_and_query"
    )


def test_query_sampling_happens_before_local_group_aggregation():
    site, layer = "block.0.attn.probs.input", nn.Identity()
    attention = torch.tensor(
        [
            [[[1.0, 0.0], [0.5, 0.5]]],
            [[[1.0, 0.0], [0.5, 0.5]]],
            [[[1.0, 0.0], [0.5, 0.5]]],
            [[[1.0, 0.0], [0.5, 0.5]]],
        ]
    )
    collector = ProbeCollector(
        {site: layer}, _metadata(site, "timesformer_temporal"), ProbeLimits(max_attention_queries=1)
    )
    collector.run(layer, attention)
    row = _by(collector.observations[0].rows, 0, "outgoing_attention_entropy_mean")
    assert row["statistic_value"] == pytest.approx(0.0)
    assert row["queries_per_group"] == 1
    assert row["native_query_rows"] == 2


def test_global_attention_rows_keep_the_existing_schema_without_local_audit_fields():
    site, layer = "block.0.attn.probs.input", nn.Identity()
    metadata = ProbeTokenMetadata(
        valid_mask=np.ones((1, 2), dtype=bool),
        coordinates=np.array([[[0, 0, 0], [0, 1, 0]]]),
        coordinate_source="global-test-layout",
        special_token_indices=(),
        has_cls=False,
    )
    collector = ProbeCollector({site: layer}, metadata)
    collector.run(layer, torch.full((1, 1, 2, 2), 0.5))
    rendered = collector.observations[0].to_rows(
        run_id="r", encoder_id="global", checkpoint_digest="d", clip_ids=("x:0",), video_ids=("x",)
    )
    assert all("native_domain" not in row for row in rendered)


@pytest.mark.parametrize(
    "kind", ["bad_shape", "bad_probabilities", "padding", "unknown_global_cls"]
)
def test_separated_attention_rejects_unverified_layouts_with_head_identity(kind: str):
    site, layer = "block.0.attn.probs.input", nn.Identity()
    metadata = _metadata(site, "timesformer_temporal")
    attention = torch.full((4, 2, 2, 2), 0.5)
    if kind == "bad_shape":
        attention = attention[:3]
    elif kind == "bad_probabilities":
        attention *= 2
    elif kind == "padding":
        metadata = _metadata(site, "timesformer_temporal")
        metadata = ProbeTokenMetadata(
            valid_mask=np.array([[True, True, True, True, False], [True] * 5]),
            coordinates=metadata.coordinates,
            coordinate_source=metadata.coordinate_source,
            special_token_indices=(0,),
            has_cls=True,
            site_metadata=metadata.site_metadata,
        )
    else:
        metadata = _metadata(site, "timesformer_temporal", leading_cls=False)
    collector = ProbeCollector({site: layer}, metadata)
    collector.run(layer, attention)
    rows = collector.observations[0].rows
    for row in rows:
        expected = "not_applicable" if kind == "unknown_global_cls" and row["probe_id"] == "P13" else "unavailable"
        assert row["status"] == expected
    assert {row["head_id"] for row in rows} == {0, 1}


def test_tolerance_normalises_rows_and_single_key_domain_is_explicit_na():
    site, layer = "block.0.attn.probs.input", nn.Identity()
    attention = torch.tensor([[[[1.001, 0.0], [1.001, 0.0]]]] * 4)
    collector = ProbeCollector({site: layer}, _metadata(site, "timesformer_temporal"))
    collector.run(layer, attention)
    assert _by(collector.observations[0].rows, 0, "outgoing_attention_entropy_mean")[
        "statistic_value"
    ] == pytest.approx(0.0)
    assert _by(collector.observations[0].rows, 0, "outgoing_attention_top4_mass_mean")[
        "statistic_value"
    ] == pytest.approx(1.0)
    single = torch.ones((2, 1, 1, 1))
    single_collector = ProbeCollector(
        {site: layer}, _metadata(site, "timesformer_temporal", frames=1, spatial_tokens=1)
    )
    single_collector.run(layer, single)
    p10 = next(row for row in single_collector.observations[0].rows if row["probe_id"] == "P10")
    assert p10["status"] == "not_applicable"
    assert p10["statistic_name"] == "insufficient_local_keys"


class _ParameterisedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("offset", torch.tensor(0.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight + self.offset


def test_hook_cleanup_preserves_parameter_buffer_output_and_gradient():
    site, layer = "block.0.attn.probs.input", _ParameterisedAttention()
    attention = torch.full((4, 1, 2, 2), 0.5, requires_grad=True)
    before = {key: value.detach().clone() for key, value in layer.state_dict().items()}
    collector = ProbeCollector({site: layer}, _metadata(site, "timesformer_temporal"))
    actual = collector.run(layer, attention)
    torch.testing.assert_close(actual, attention)
    actual.sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.item() > 0
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
