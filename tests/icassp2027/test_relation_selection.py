import pytest
import torch

from vadbench.token_reduction.relation_selection import METHODS, retained_count, select_relational_tokens


@pytest.mark.parametrize("method", METHODS)
def test_identity_is_exact_and_needs_no_signal(method):
    x = torch.randn(2, 20, 8)
    out = select_relational_tokens(x, method=method, keep_ratio=1)
    assert out.tokens is x
    assert torch.equal(out.mass.sum(1), torch.tensor([20, 20]))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("keep", [0.5, 0.3])
def test_exact_budgets_and_sources(method, keep):
    x = torch.randn(2, 30, 8, requires_grad=True)
    scores = torch.arange(30).float().expand(2, -1)
    keys = torch.randn(2, 2, 30, 4)
    out = select_relational_tokens(x, method=method, keep_ratio=keep, keys=keys, attention_scores=scores)
    k = retained_count(30, keep)
    assert out.tokens.shape == (2, k, 8)
    assert out.representatives.unique(dim=1).shape == (2, k)
    expected_mass = 30 if method == "visionzip_internal" else k
    assert out.mass.sum(1).tolist() == [expected_mass] * 2
    assert (out.source_to_output >= 0).sum(1).tolist() == [expected_mass] * 2
    out.tokens.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_attention_uses_scores_not_hidden_norm_and_preserves_order():
    x = torch.arange(8).float().view(1, 8, 1) * 100
    scores = torch.tensor([[0, 9, 0, 7, 0, 8, 0, 0.]])
    out = select_relational_tokens(x, method="attention_topk", keep_ratio=0.3, attention_scores=scores)
    assert out.representatives.tolist() == [[1, 5]]


def test_ties_choose_lowest_native_index():
    out = select_relational_tokens(torch.randn(1, 10, 3), method="attention_topk", keep_ratio=0.3,
                                   attention_scores=torch.ones(1, 10))
    assert out.representatives.tolist() == [[0, 1, 2]]


def test_visionzip_reference_formula_and_order():
    x = torch.arange(1, 11).float().reshape(1, 10, 1)
    out = select_relational_tokens(x, method="visionzip_internal", keep_ratio=0.5,
                                  attention_scores=torch.arange(10).float()[None], keys=torch.ones(1, 2, 10, 2))
    # Four dominant tokens 9,8,7,6. Context target 0 absorbs 1..5.
    assert out.representatives.tolist() == [[0, 6, 7, 8, 9]]
    assert out.mass.tolist() == [[6, 1, 1, 1, 1]]
    assert out.tokens.flatten().tolist() == [5, 7, 8, 9, 10]
    assert out.source_to_output.tolist() == [[0, 0, 0, 0, 0, 0, 1, 2, 3, 4]]


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan")])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        retained_count(20, ratio)


def test_visionzip_requires_real_keys():
    with pytest.raises(ValueError, match="actual keys"):
        select_relational_tokens(torch.randn(1, 20, 8), method="visionzip_internal", keep_ratio=0.5,
                                 attention_scores=torch.ones(1, 20))


def test_half_up_rounding():
    assert retained_count(1568, 0.3) == 470
    assert retained_count(5, 0.5) == 3
