from __future__ import annotations

import unittest

from vadbench.models import (
    TORCH_AVAILABLE,
    AttentionMILHead,
    TemporalSupervisedHead,
    TopKMILHead,
    mil_ranking_loss,
    temporal_supervised_loss,
)

if TORCH_AVAILABLE:
    import torch


class OptionalDependencyTests(unittest.TestCase):
    @unittest.skipIf(TORCH_AVAILABLE, "only relevant in the minimal environment")
    def test_constructing_head_without_torch_has_focused_error(self) -> None:
        with self.assertRaises(ImportError):
            AttentionMILHead(4)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional dependency")
class NeuralHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)

    def test_attention_mil_shapes_masking_and_backward(self) -> None:
        head = AttentionMILHead(4, hidden_dim=3)
        features = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        output = head(features, mask)
        self.assertEqual(tuple(output.video_logits.shape), (2,))
        self.assertEqual(tuple(output.snippet_logits.shape), (2, 5))
        torch.testing.assert_close(output.attention.sum(dim=1), torch.ones(2))
        torch.testing.assert_close(output.attention[0, 3:], torch.zeros(2))
        torch.testing.assert_close(output.valid_mask, mask)
        output.video_logits.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_attention_all_padding_row_is_finite_zero_weight(self) -> None:
        head = AttentionMILHead(2)
        output = head(torch.randn(1, 3, 2), torch.zeros(1, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output.video_logits).all())
        torch.testing.assert_close(output.attention, torch.zeros_like(output.attention))

    def test_topk_selects_only_valid_highest_logits(self) -> None:
        head = TopKMILHead(1, k=2)
        with torch.no_grad():
            head.classifier[-1].weight.fill_(1.0)
            head.classifier[-1].bias.zero_()
        features = torch.tensor([[[1.0], [4.0], [2.0], [10.0]]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        output = head(features, mask)
        self.assertAlmostEqual(float(output.video_logits.detach()), 3.0)
        self.assertEqual(output.selected_mask.tolist(), [[False, True, True, False]])

    def test_ranking_loss_known_value_and_regularizers(self) -> None:
        anomaly = torch.tensor([[0.9, 0.1]], requires_grad=True)
        normal = torch.tensor([[0.2, 0.3]])
        loss = mil_ranking_loss(anomaly, normal, margin=1.0)
        self.assertAlmostEqual(float(loss.detach()), 0.4, places=6)
        regularized = mil_ranking_loss(
            anomaly, normal, margin=1.0, smoothness_weight=1.0, sparsity_weight=1.0
        )
        self.assertGreater(float(regularized.detach()), float(loss.detach()))
        regularized.backward()
        self.assertIsNotNone(anomaly.grad)

    def test_ranking_loss_accepts_single_bag_masks(self) -> None:
        loss = mil_ranking_loss(
            torch.tensor([0.9, 10.0]),
            torch.tensor([0.2, 20.0]),
            anomaly_mask=torch.tensor([1, 0], dtype=torch.bool),
            normal_mask=torch.tensor([1, 0], dtype=torch.bool),
        )
        self.assertAlmostEqual(float(loss.detach()), 0.3, places=6)

    def test_temporal_supervised_loss_honors_mask(self) -> None:
        logits = torch.tensor([[0.0, 1.0, -5.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 1.0, 1.0]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        loss = temporal_supervised_loss(logits, targets, mask=mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :2], targets[:, :2]
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()

    def test_temporal_head_shape(self) -> None:
        head = TemporalSupervisedHead(6, hidden_dim=3)
        self.assertEqual(tuple(head(torch.randn(2, 7, 6)).shape), (2, 7))

    def test_heads_reject_empty_bags(self) -> None:
        with self.assertRaises(ValueError):
            AttentionMILHead(4)(torch.empty(0, 2, 4))
        with self.assertRaises(ValueError):
            TopKMILHead(4)(torch.empty(1, 0, 4))


if __name__ == "__main__":
    unittest.main()
