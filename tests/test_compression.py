from __future__ import annotations

import numpy as np
import pytest

from vadbench.compression import (
    IdentityCachePolicy,
    KeepRecentCachePolicy,
    append_cache_views,
    build_cache_policy,
    merge_cache_update,
)
from vadbench.contracts import (
    CacheKind,
    CacheUpdate,
    CacheUpdateMode,
    CacheView,
    ContractError,
    TokenTimeline,
)


def make_token_cache(start: int, stop: int, *, offset: int = 0) -> CacheView:
    indices = np.arange(start, stop, dtype=np.int64)
    timeline = TokenTimeline(
        start_s=indices[None, :].astype(np.float64),
        end_s=(indices + 1)[None, :].astype(np.float64),
        source_frame_start=indices[None, :],
        source_frame_end=(indices + 1)[None, :],
    )
    values = np.stack((indices + offset, indices + 100 + offset), axis=-1)[None, :, :]
    return CacheView(
        kind=CacheKind.TOKEN,
        tensors={"tokens": values},
        sequence_axis=1,
        timeline=timeline,
        metadata={"stop": stop},
    )


def make_kv_cache(start: int, stop: int) -> CacheView:
    length = stop - start
    starts = np.arange(start, stop, dtype=np.float64)[None, :]
    timeline = TokenTimeline(start_s=starts, end_s=starts + 1.0)
    return CacheView(
        kind=CacheKind.KV,
        tensors={
            "layer.0.key": np.full((1, 2, length, 3), start, dtype=np.float32),
            "layer.0.value": np.full((1, 2, length, 3), stop, dtype=np.float32),
        },
        sequence_axis=-2,
        timeline=timeline,
    )


def test_identity_policy_appends_losslessly_and_preserves_timeline() -> None:
    current = make_token_cache(0, 3)
    update = CacheUpdate.append(make_token_cache(3, 5))

    merged = IdentityCachePolicy().apply(current, update)

    assert merged.sequence_length == 5
    np.testing.assert_array_equal(merged.tensors["tokens"][0, :, 0], np.arange(5))
    np.testing.assert_array_equal(merged.timeline.start_s, [np.arange(5)])
    np.testing.assert_array_equal(merged.timeline.source_frame_start, [np.arange(5)])
    assert merged.metadata["stop"] == 5
    assert IdentityCachePolicy().compress(current) is current


def test_keep_recent_applies_after_append_and_slices_every_provenance_field() -> None:
    policy = KeepRecentCachePolicy(max_tokens=3)
    merged = policy.apply(
        make_token_cache(0, 3),
        CacheUpdate(view=make_token_cache(3, 6), mode=CacheUpdateMode.APPEND),
    )

    assert merged.sequence_length == 3
    np.testing.assert_array_equal(merged.tensors["tokens"][0, :, 0], [3, 4, 5])
    np.testing.assert_array_equal(merged.timeline.start_s, [[3.0, 4.0, 5.0]])
    np.testing.assert_array_equal(merged.timeline.source_frame_start, [[3, 4, 5]])

    already_short = make_token_cache(0, 2)
    assert policy.compress(already_short) is already_short


def test_keep_recent_respects_nontrivial_kv_sequence_axis() -> None:
    policy = KeepRecentCachePolicy(max_tokens=2)
    merged = policy.apply(make_kv_cache(0, 2), CacheUpdate.append(make_kv_cache(2, 4)))

    assert merged.sequence_axis == 2
    assert merged.sequence_length == 2
    assert merged.tensors["layer.0.key"].shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(merged.timeline.start_s, [[2.0, 3.0]])


def test_replace_update_does_not_mix_old_cache() -> None:
    current = make_token_cache(0, 5)
    replacement = make_token_cache(10, 12)

    result = merge_cache_update(current, CacheUpdate.replace(replacement, reason="reset"))

    assert result is replacement
    assert result.sequence_length == 2


def test_append_rejects_structural_or_temporal_mismatch() -> None:
    token_cache = make_token_cache(0, 2)
    kv_cache = make_kv_cache(2, 4)
    with pytest.raises(ContractError, match="kind"):
        append_cache_views(token_cache, kv_cache)

    wrong_keys = CacheView(
        kind=CacheKind.TOKEN,
        tensors={"other": np.zeros((1, 2, 2))},
        sequence_axis=1,
        timeline=TokenTimeline(
            start_s=np.array([[2.0, 3.0]]),
            end_s=np.array([[3.0, 4.0]]),
        ),
    )
    with pytest.raises(ContractError, match="tensor 键"):
        append_cache_views(token_cache, wrong_keys)

    backwards = make_token_cache(0, 2, offset=10)
    with pytest.raises(ContractError, match="单调不减"):
        append_cache_views(make_token_cache(5, 7), backwards)


def test_policy_builder_and_parameter_validation() -> None:
    assert isinstance(build_cache_policy("none"), IdentityCachePolicy)
    recent = build_cache_policy("keep-recent", max_tokens=8)
    assert isinstance(recent, KeepRecentCachePolicy)
    assert recent.max_tokens == 8

    with pytest.raises(ContractError, match="正整数"):
        KeepRecentCachePolicy(max_tokens=0)
    with pytest.raises(ContractError, match="必须提供"):
        build_cache_policy("keep_recent")
    with pytest.raises(ContractError, match="未知"):
        build_cache_policy("quantize")
