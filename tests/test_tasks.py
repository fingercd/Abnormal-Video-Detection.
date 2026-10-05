from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vadbench.engine.evaluate import evaluate_batches
from vadbench.engine.train import (
    load_checkpoint,
    move_to_device,
    save_checkpoint,
    train_one_step,
)
from vadbench.models import TORCH_AVAILABLE, AttentionMILHead, TemporalSupervisedHead
from vadbench.tasks import (
    TemporalSupervisedTask,
    WeaklySupervisedMILTask,
    build_task,
    build_temporal_targets,
    extract_encoder_features,
)

if TORCH_AVAILABLE:
    import torch


class EncoderFeatureExtractionTests(unittest.TestCase):
    def test_canonical_encoder_output_and_timeline(self) -> None:
        features = object()
        mask = object()
        output = SimpleNamespace(features=features, timeline=SimpleNamespace(valid_mask=mask))
        extracted = extract_encoder_features(output)
        self.assertIs(extracted.features, features)
        self.assertIs(extracted.valid_mask, mask)
        self.assertIs(extracted.raw_output, output)

    def test_mapping_and_direct_array_are_supported(self) -> None:
        import numpy as np

        array = np.zeros((1, 2, 3), dtype=np.float32)
        self.assertIs(extract_encoder_features(array).features, array)
        mapped = extract_encoder_features({"features": array, "valid_mask": [[True, False]]})
        self.assertIs(mapped.features, array)
        self.assertEqual(mapped.valid_mask, [[True, False]])

    def test_missing_features_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            extract_encoder_features({"pooled": [1.0]})

    def test_strong_annotations_map_to_half_open_token_ranges(self) -> None:
        import numpy as np

        from vadbench.contracts import TokenTimeline
        from vadbench.data import SupervisionAnnotation, TemporalSpan

        timeline = TokenTimeline(
            start_s=np.array([[0.0, 1.0, 2.0]]),
            end_s=np.array([[1.0, 2.0, 3.0]]),
            valid_mask=np.array([[True, True, False]]),
            source_frame_start=np.array([[0, 4, 8]]),
            source_frame_end=np.array([[4, 8, 12]]),
        )
        annotations = [
            SupervisionAnnotation(
                scope="frame",
                is_anomaly=True,
                span=TemporalSpan(3, 5, "frame"),
            ),
            SupervisionAnnotation(scope="caption", text="visual description"),
        ]
        targets = build_temporal_targets(timeline, annotations)
        np.testing.assert_array_equal(targets.labels, [[1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(targets.valid_mask, [[True, True, False]])

    def test_strong_target_overlap_threshold(self) -> None:
        import numpy as np

        from vadbench.contracts import TokenTimeline
        from vadbench.data import SupervisionAnnotation, TemporalSpan

        timeline = TokenTimeline(
            start_s=np.array([[0.0, 1.0]]),
            end_s=np.array([[1.0, 2.0]]),
        )
        annotation = SupervisionAnnotation(
            scope="segment",
            is_anomaly=True,
            span=TemporalSpan(0.75, 1.25, "second"),
        )
        targets = build_temporal_targets(timeline, [annotation], min_overlap_fraction=0.5)
        np.testing.assert_array_equal(targets.labels, [[0.0, 0.0]])

    def test_caption_only_annotations_have_no_strong_valid_targets(self) -> None:
        import numpy as np

        from vadbench.contracts import TokenTimeline
        from vadbench.data import SupervisionAnnotation

        timeline = TokenTimeline(start_s=np.array([[0.0, 1.0]]), end_s=np.array([[1.0, 2.0]]))
        targets = build_temporal_targets(
            timeline,
            [SupervisionAnnotation(scope="caption", text="not a binary label")],
        )
        np.testing.assert_array_equal(targets.labels, [[0.0, 0.0]])
        np.testing.assert_array_equal(targets.valid_mask, [[False, False]])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional dependency")
class TaskAndTrainingTests(unittest.TestCase):
    class FakeEncoder:
        def __init__(self, feature_dim: int) -> None:
            self.feature_dim = feature_dim
            self.train_flags: list[bool] = []

        def encode(self, batch, *, train: bool = False):
            self.train_flags.append(train)
            features = batch["input_features"]
            valid = batch.get("valid_mask")
            return SimpleNamespace(
                features=features,
                timeline=SimpleNamespace(valid_mask=valid),
                pooled=features.mean(dim=1),
            )

    class TrainableAdapter:
        def __init__(self) -> None:
            self.capabilities = SimpleNamespace(supports_training=True)
            self.encoder = torch.nn.Linear(3, 3, bias=False)

        def encode(self, batch, *, train: bool = False):
            features = self.encoder(batch["input_features"])
            return SimpleNamespace(features=features, timeline=None)

    def setUp(self) -> None:
        torch.manual_seed(9)

    def test_weak_task_and_one_optimizer_step(self) -> None:
        encoder = self.FakeEncoder(4)
        task = WeaklySupervisedMILTask(
            encoder,
            AttentionMILHead(4, hidden_dim=3),
            ranking_weight=0.1,
        )
        optimizer = torch.optim.SGD(task.parameters(), lr=0.05)
        batch = {
            "input_features": torch.randn(2, 5, 4),
            "valid_mask": torch.ones(2, 5, dtype=torch.bool),
            "video_labels": torch.tensor([0.0, 1.0]),
        }
        before = task.head.classifier.weight.detach().clone()
        result = train_one_step(task, batch, optimizer, step=7, max_grad_norm=2.0)
        self.assertEqual(result.step, 8)
        self.assertGreater(result.loss, 0.0)
        self.assertEqual(result.metrics["ranking_pairs"], 1)
        self.assertFalse(torch.equal(before, task.head.classifier.weight.detach()))
        self.assertTrue(encoder.train_flags[-1])

    def test_cached_numpy_features_are_normalized_for_head(self) -> None:
        import numpy as np

        task = WeaklySupervisedMILTask(None, AttentionMILHead(3))
        batch = {
            "features": np.ones((2, 4, 3), dtype=np.float64),
            "valid_mask": np.ones((2, 4), dtype=bool),
            "video_labels": [0.0, 1.0],
        }
        output = task.training_step(batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.predictions.snippet_logits.dtype, torch.float32)

    def test_trainable_adapter_module_is_registered(self) -> None:
        adapter = self.TrainableAdapter()
        task = WeaklySupervisedMILTask(adapter, AttentionMILHead(3))
        state_keys = set(task.state_dict())
        self.assertIn("registered_encoder_module.weight", state_keys)
        task.eval()
        self.assertFalse(adapter.encoder.training)
        task.to("cpu")
        self.assertEqual(adapter.encoder.weight.device.type, "cpu")

    def test_prediction_step_preserves_valid_mask(self) -> None:
        task = WeaklySupervisedMILTask(self.FakeEncoder(2), AttentionMILHead(2))
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        result = task.prediction_step({"input_features": torch.randn(1, 3, 2), "valid_mask": mask})
        torch.testing.assert_close(result.auxiliary["valid_mask"], mask)
        torch.testing.assert_close(result.predictions.valid_mask, mask)

    def test_evaluate_batches_uses_task_prediction_step(self) -> None:
        task = WeaklySupervisedMILTask(self.FakeEncoder(2), AttentionMILHead(2))
        mask = torch.tensor([[1, 0]], dtype=torch.bool)
        outputs = evaluate_batches(
            task,
            [{"input_features": torch.randn(1, 2, 2), "valid_mask": mask}],
            device="cpu",
        )
        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0].auxiliary["valid_mask"], mask)

    def test_temporal_task_training_step(self) -> None:
        encoder = self.FakeEncoder(3)
        task = TemporalSupervisedTask(encoder, TemporalSupervisedHead(3))
        batch = {
            "input_features": torch.randn(2, 4, 3),
            "valid_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
            "temporal_labels": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
        }
        output = task.training_step(batch)
        self.assertEqual(tuple(output.predictions.shape), (2, 4))
        self.assertTrue(torch.isfinite(output.loss))

    def test_temporal_task_consumes_structured_target_mask(self) -> None:
        import numpy as np

        encoder = self.FakeEncoder(3)
        task = TemporalSupervisedTask(encoder, TemporalSupervisedHead(3))
        target_bundle = SimpleNamespace(
            labels=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            valid_mask=np.array([[True, True, False]]),
        )
        output = task.training_step(
            {
                "input_features": torch.randn(1, 3, 3),
                "valid_mask": torch.ones(1, 3, dtype=torch.bool),
                "temporal_targets": target_bundle,
            }
        )
        torch.testing.assert_close(
            output.auxiliary["valid_mask"],
            torch.tensor([[True, True, False]]),
        )

    def test_build_task(self) -> None:
        task = build_task(
            "wsvad",
            self.FakeEncoder(2),
            feature_dim=2,
            head="topk",
            head_kwargs={"k": 1},
        )
        self.assertIsInstance(task, WeaklySupervisedMILTask)
        config_alias = build_task("weak_mil", self.FakeEncoder(2), feature_dim=2, head="attention")
        self.assertIsInstance(config_alias, WeaklySupervisedMILTask)
        strong_alias = build_task("temporal_supervised", self.FakeEncoder(2), feature_dim=2)
        self.assertIsInstance(strong_alias, TemporalSupervisedTask)

    def test_checkpoint_roundtrip_and_manifest(self) -> None:
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-0001.pt"
            artifact = save_checkpoint(
                path,
                model,
                optimizer=optimizer,
                step=1,
                epoch=2,
                metadata={"encoder": "smoke"},
            )
            self.assertTrue(path.is_file())
            self.assertTrue(Path(artifact.manifest_path).is_file())
            original = {key: value.detach().clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                model.weight.add_(10.0)
            restored = load_checkpoint(path, model, optimizer=optimizer)
            self.assertEqual(restored["step"], 1)
            self.assertEqual(restored["epoch"], 2)
            self.assertEqual(restored["metadata"]["encoder"], "smoke")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key])
            path.write_bytes(path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum verification failed"):
                load_checkpoint(path, model)

    def test_invalid_checkpoint_metadata_leaves_no_partial_file(self) -> None:
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            with self.assertRaises(TypeError):
                save_checkpoint(path, model, metadata={"not_json": object()})
            self.assertFalse(path.exists())

    def test_non_finite_gradient_never_updates_parameters(self) -> None:
        class NonFiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value * 0.0

            @staticmethod
            def backward(ctx, gradient):
                return torch.full_like(gradient, float("nan"))

        class BadGradientTask(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def training_step(self, batch):
                return {"loss": NonFiniteBackward.apply(self.weight)}

        task = BadGradientTask()
        optimizer = torch.optim.SGD(task.parameters(), lr=1.0)
        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient"):
            train_one_step(task, {}, optimizer)
        self.assertEqual(float(task.weight.detach()), 1.0)

    def test_move_canonical_batch_with_frozen_metadata(self) -> None:
        from vadbench.contracts import ClipBatch

        batch = ClipBatch(
            frames=torch.zeros(1, 2, 2, 2, 3, dtype=torch.uint8),
            timestamps_s=torch.tensor([[0.0, 0.1]]),
            video_ids=("v",),
            metadata={"nested": {"value": torch.tensor(1.0)}},
        )
        moved = move_to_device(batch, "cpu")
        self.assertIsInstance(moved, ClipBatch)
        self.assertEqual(moved.frames.device.type, "cpu")
        self.assertEqual(moved.metadata["nested"]["value"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
