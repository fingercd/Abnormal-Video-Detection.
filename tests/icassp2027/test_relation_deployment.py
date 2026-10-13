from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from vadbench.token_reduction.relation_deployment import RelationDeployment


def deployment():
    bridge = SimpleNamespace(receipt=lambda: SimpleNamespace(encoder_id="videomaev2", has_cls=False, block_count=12))
    layout = SimpleNamespace(batch_size=1, special_token_mask=torch.zeros(1, 20, dtype=torch.bool),
                             valid_mask=torch.ones(1, 20, dtype=torch.bool), mass=torch.ones(1, 20), token_capacity=20,
                             provenance={"coordinate_source": "verified-conv3d-flatten-t-h-w", "grid": [8, 1, 2], "stride": [2, 16, 16]})
    return RelationDeployment(bridge, layout, depth=11, method="visionzip_internal", keep_ratio=0.3)


def test_deployment_configuration_is_frozen():
    d = deployment()
    with pytest.raises(FrozenInstanceError):
        d.method = "uniform"
    with pytest.raises(FrozenInstanceError):
        d.depth = 5


def test_identity_nested_values_cannot_mutate_the_deployment():
    d = deployment()
    first = d.reducer_identity
    first["dominant_contextual_weights"][0] = 99
    first["implementation_sha256"].clear()
    assert d.reducer_identity["dominant_contextual_weights"] == [4, 1]
    assert len(d.reducer_identity["implementation_sha256"]) == 3


def test_exact_final_block_budget():
    d = deployment()
    assert d.output_tokens == 6
    assert d.reducer_identity["block_index"] == 11
