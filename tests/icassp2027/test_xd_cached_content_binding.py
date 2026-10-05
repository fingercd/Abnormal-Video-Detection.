"""The cached repeat reader must bind input bytes to the original raw seal."""

from __future__ import annotations

import pytest
import test_xd_quality_export as fixture

from vadbench.paper import xd_repeat_evaluation as repeats

pipeline = fixture.pipeline
synthetic_dataset = fixture.synthetic_dataset
_bounded_cpu_threads = fixture._bounded_cpu_threads
pair = fixture.pair


def test_cached_repeat_accepts_complete_computed_proof(pair):
    loaded = repeats._cache(pair.dense, expected_encoder="toy")
    assert loaded[0] == pair.dense
    assert loaded[4].manifest == pair.coordinates.manifest


def test_cached_repeat_rejects_coherent_but_unsealed_input_bytes(pair):
    # The fixture alters both caches and updates all cache/runtime/index/run
    # hashes consistently, while the independent original seal stays fixed.
    fixture.test_two_coherent_feature_stores_cannot_substitute_same_unsealed_raw_bytes(pair)
    with pytest.raises(ValueError, match="SHA/bytes differ from the raw-coordinate seal"):
        repeats._cache(pair.dense, expected_encoder="toy")
