"""Exercise the actual deployment dispatch, not just the selector in isolation."""
import pytest
import torch
from test_pair_deployment import _batch, _case
from vadbench.token_reduction.deployment import GroupSelectDeployment


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("rule,expected", [
    ("pair_select", [0, 3, 4]),
    ("pair_fixed", [0, 2, 4]),
    ("pair_reverse", [1, 2, 5]),
])
def test_deployment_applies_the_requested_member_rule(rule, expected):
    adapter, bridge, layout, frames, dim = _case("videomaev2")
    hidden = torch.diag(torch.tensor([2., 1., 1., 2., .2, .1, .1, .05]))[None]
    # Inject known unequal members before the real deployment hook executes.
    handle = bridge._blocks[0].register_forward_hook(lambda _m, _a, _o: hidden.clone())
    deployment = GroupSelectDeployment(bridge, layout, depth=0, dim=dim,
        keep_ratio=.4, rule=rule, batch_sizes=(1,), device="cpu", seed=11)
    assert deployment.reducer_identity["mask_scope"] == "current_forward_mask_per_batch_item"
    try:
        with torch.no_grad(), deployment(_batch(frames=frames)) as ctx:
            output = adapter.encode(_batch(frames=frames))
            receipt = ctx.validate_execution()
        assert deployment._selectors[1].last_selection[0].tolist() == expected
        assert deployment.selection_digest(1) is not None
        assert output.shape == (1, 3, dim)
        assert receipt["gathered_tokens"] == 3
    finally:
        handle.remove()


def test_random_member_deployment_repeats_same_mask_and_does_not_fall_back():
    adapter, bridge, layout, frames, dim = _case("videomaev2")
    hidden = torch.diag(torch.tensor([2., 1., 1., 2., .2, .1, .1, .05]))[None]
    handle = bridge._blocks[0].register_forward_hook(lambda _m, _a, _o: hidden.clone())
    deployment = GroupSelectDeployment(bridge, layout, depth=0, dim=dim,
        keep_ratio=.4, rule="pair_random_member", batch_sizes=(1,), device="cpu", seed=11)
    try:
        masks = []
        for _ in range(2):
            with torch.no_grad(), deployment(_batch(frames=frames)):
                adapter.encode(_batch(frames=frames))
            masks.append(deployment._selectors[1].last_selection[0].tolist())
        assert masks[0] == masks[1]
        assert sorted(i // 2 for i in masks[0]) == [0, 1, 2]
        assert deployment.selection_digest(1) is not None
    finally:
        handle.remove()
