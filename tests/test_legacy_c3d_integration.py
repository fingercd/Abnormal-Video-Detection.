from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG
from vadbench.integrations.legacy import (
    DEFAULT_CAPABILITIES,
    LegacyAssetError,
    LegacyDependencyError,
    LegacyVideoAdapter,
    LegacyWorkerError,
)


def _batch(*, frames: int = 16, batch_size: int = 1) -> ClipBatch:
    pixels = np.arange(batch_size * frames * 4 * 6 * 3, dtype=np.uint8).reshape(
        batch_size, frames, 4, 6, 3
    )
    timestamps = (np.arange(frames, dtype=np.float64) / 30.0)[None, :].repeat(batch_size, 0)
    indices = np.arange(frames, dtype=np.int64)[None, :].repeat(batch_size, 0)
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        frame_indices=indices,
        video_ids=tuple(f"video-{i}" for i in range(batch_size)),
    )


class _FakeEncoder:
    def __init__(self) -> None:
        self.last_input: Any = None
        self.last_train: bool | None = None

    def encode(self, frames: np.ndarray, *, train: bool) -> dict[str, np.ndarray]:
        self.last_input = frames
        self.last_train = train
        batch = int(frames.shape[0])
        return {"fc6": np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)}


class _FakeBlob:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def reshape(self, *shape: int) -> None:
        self.data = np.empty(shape, dtype=np.float32)


class _FakeCaffeNet:
    def __init__(self, batch_size: int) -> None:
        self.blobs = {
            "data": _FakeBlob(np.empty((0,), dtype=np.float32)),
            "fc6": _FakeBlob(
                np.arange(batch_size * 4, dtype=np.float32).reshape(batch_size, 4, 1, 1, 1)
            ),
            "features": _FakeBlob(np.full((batch_size, 2), -1.0, dtype=np.float32)),
        }

    def forward(self) -> dict[str, np.ndarray]:
        return {"logits": np.full((len(self.blobs["fc6"].data), 487), 99.0)}


def test_fake_encoder_receives_contiguous_c3d_bcthw_and_emits_contract() -> None:
    encoder = _FakeEncoder()
    adapter = LegacyVideoAdapter(encoder=encoder)

    output = adapter.encode(_batch())

    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert encoder.last_input.dtype == np.float32
    assert encoder.last_train is False
    assert output.features.shape == (1, 1, 4)
    assert output.pooled.shape == (1, 4)
    assert output.timeline.start_s.tolist() == [[0.0]]
    assert output.timeline.source_frame_start.tolist() == [[0]]
    assert output.timeline.source_frame_end.tolist() == [[16]]
    assert output.aux["feature_stage"] == "fc_features"
    assert output.aux["sequence_source"] == "fc6"
    assert output.aux["preprocess_profile"] == "c3d-16x112-v1"
    assert output.aux["input_layout"] == "BCTHW"
    assert output.aux["input_shape"] == [1, 3, 16, 112, 112]
    assert output.aux["model_output_type"] == "dict"


def test_caffe_fc_blob_is_selected_explicitly_and_flattened_to_singleton() -> None:
    adapter = LegacyVideoAdapter(encoder=_FakeCaffeNet(batch_size=2), feature_layer="fc6")

    output = adapter.encode(_batch(batch_size=2))

    np.testing.assert_array_equal(
        output.features[:, 0], np.arange(8, dtype=np.float32).reshape(2, 4)
    )
    assert output.features.shape == (2, 1, 4)
    assert output.timeline.source_frame_start.tolist() == [[0], [0]]
    assert output.timeline.source_frame_end.tolist() == [[16], [16]]
    np.testing.assert_allclose(output.timeline.end_s, 16.0 / 30.0)
    assert output.aux["sequence_source"] == "caffe:fc6"


@pytest.mark.parametrize(
    "raw_output",
    [
        {"features": np.ones((1, 4), dtype=np.float32)},
        SimpleNamespace(features=np.ones((1, 4), dtype=np.float32)),
        (
            np.ones((1, 4), dtype=np.float32),
            np.ones((1, 487), dtype=np.float32),
        ),
    ],
    ids=("mapping", "object", "tuple"),
)
def test_container_output_must_name_the_declared_feature_layer(raw_output: Any) -> None:
    class Encoder:
        def encode(self, frames: np.ndarray, *, train: bool) -> Any:
            return raw_output

    with pytest.raises(LegacyWorkerError) as captured:
        LegacyVideoAdapter(encoder=Encoder(), feature_layer="fc6").encode(_batch())

    assert captured.value.code == "invalid_output"


def test_direct_tensor_output_keeps_its_real_model_output_type() -> None:
    class Encoder:
        def encode(self, frames: np.ndarray, *, train: bool) -> np.ndarray:
            return np.ones((len(frames), 4), dtype=np.float32)

    output = LegacyVideoAdapter(encoder=Encoder()).encode(_batch())

    assert output.features.shape == (1, 1, 4)
    assert output.aux["model_output_type"] == "numpy.ndarray"


def test_encode_receives_train_once_and_callable_receives_only_bcthw() -> None:
    encoder = _FakeEncoder()
    LegacyVideoAdapter(encoder=encoder).encode(_batch(), train=True)
    assert encoder.last_train is True

    class CallableEncoder:
        def __init__(self) -> None:
            self.input_shape: tuple[int, ...] | None = None

        def __call__(self, frames: np.ndarray) -> dict[str, np.ndarray]:
            self.input_shape = frames.shape
            return {"fc6": np.ones((len(frames), 4), dtype=np.float32)}

    callable_encoder = CallableEncoder()
    output = LegacyVideoAdapter(encoder=callable_encoder).encode(_batch())
    assert callable_encoder.input_shape == (1, 3, 16, 112, 112)
    assert output.features.shape == (1, 1, 4)


def test_encoder_type_error_is_not_retried_with_callable_fallback() -> None:
    class Encoder:
        def __init__(self) -> None:
            self.encode_calls = 0
            self.callable_calls = 0

        def encode(self, frames: np.ndarray, *, train: bool) -> Any:
            self.encode_calls += 1
            raise TypeError("upstream encode failure")

        def __call__(self, frames: np.ndarray) -> dict[str, np.ndarray]:
            self.callable_calls += 1
            return {"fc6": np.ones((len(frames), 4), dtype=np.float32)}

    encoder = Encoder()
    with pytest.raises(LegacyWorkerError, match="upstream encode failure") as captured:
        LegacyVideoAdapter(encoder=encoder).encode(_batch())

    assert captured.value.code == "forward_failed"
    assert (encoder.encode_calls, encoder.callable_calls) == (1, 0)


def test_forward_only_non_caffe_encoder_is_rejected() -> None:
    class Encoder:
        def forward(self, frames: np.ndarray) -> dict[str, np.ndarray]:
            return {"fc6": np.ones((len(frames), 4), dtype=np.float32)}

    with pytest.raises(LegacyWorkerError) as captured:
        LegacyVideoAdapter(encoder=Encoder())

    assert captured.value.code == "worker_protocol_error"


def test_short_or_long_input_is_defensively_normalized_to_profile() -> None:
    encoder = _FakeEncoder()
    adapter = LegacyVideoAdapter(encoder=encoder, clip_frames=16, image_size=112)

    short_output = adapter.encode(_batch(frames=4))
    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert short_output.timeline.source_frame_end.tolist() == [[4]]
    assert short_output.aux["temporal_adjustment"] == ["repeat_last_frame"]

    long_output = adapter.encode(_batch(frames=20))
    assert encoder.last_input.shape == (1, 3, 16, 112, 112)
    assert long_output.timeline.source_frame_end.tolist() == [[20]]
    assert long_output.aux["temporal_adjustment"] == ["truncate_first_contiguous"]


def test_capabilities_match_catalog_generic_fixed_profile() -> None:
    adapter = LegacyVideoAdapter(encoder=_FakeEncoder())
    assert adapter.capabilities == DEFAULT_CAPABILITIES
    assert adapter.capabilities == DEFAULT_INTEGRATION_CATALOG.get("c3d").capabilities
    assert adapter.capabilities.supports_fixed_clip is True
    assert adapter.capabilities.supports_streaming is False
    assert adapter.capabilities.supports_kv_cache is False
    assert adapter.capabilities.supports_token_cache is False
    assert adapter.capabilities.fixed_num_frames is None


def test_explicit_loader_receives_only_local_asset_paths(tmp_path: Path) -> None:
    checkout = tmp_path / "external" / "C3D"
    checkpoint = tmp_path / "weights" / "c3d.caffemodel"
    prototxt = checkout / "deploy.prototxt"
    checkout.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"local-checkpoint")
    prototxt.write_text("name: 'c3d'\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def loader(
        *,
        checkout_path: str,
        checkpoint_path: str,
        prototxt_path: str,
        device: str,
        feature_layer: str,
    ) -> _FakeEncoder:
        seen.update(
            checkout=checkout_path,
            checkpoint=checkpoint_path,
            prototxt=prototxt_path,
            device=device,
            feature_layer=feature_layer,
        )
        return _FakeEncoder()

    adapter = LegacyVideoAdapter(
        loader=loader,
        project_root=tmp_path,
        checkout_path="external/C3D",
        checkpoint_path="weights/c3d.caffemodel",
        prototxt_path="external/C3D/deploy.prototxt",
        device="cpu",
    )
    adapter.encode(_batch())
    assert seen == {
        "checkout": str(checkout.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "prototxt": str(prototxt.resolve()),
        "device": "cpu",
        "feature_layer": "fc6",
    }


def test_loader_type_error_is_called_once_without_positional_retry() -> None:
    calls = 0

    def loader(*, device: str) -> Any:
        nonlocal calls
        calls += 1
        raise TypeError(f"loader failed on {device}")

    with pytest.raises(LegacyWorkerError, match="loader failed on cpu") as captured:
        LegacyVideoAdapter(loader=loader)

    assert captured.value.code == "model_load_failed"
    assert calls == 1


def test_caffe_net_uses_the_pinned_positional_signature(tmp_path: Path) -> None:
    checkout = tmp_path / "C3D"
    checkpoint = tmp_path / "model.caffemodel"
    prototxt = checkout / "deploy.prototxt"
    checkout.mkdir()
    checkpoint.write_bytes(b"weights")
    prototxt.write_text("name: 'c3d'\n", encoding="utf-8")
    calls: list[tuple[str, str, int]] = []

    def net_constructor(prototxt_arg: str, checkpoint_arg: str, mode: int, /) -> _FakeCaffeNet:
        calls.append((prototxt_arg, checkpoint_arg, mode))
        return _FakeCaffeNet(batch_size=1)

    caffe = SimpleNamespace(Net=net_constructor, TEST=17)
    adapter = LegacyVideoAdapter(
        project_root=tmp_path,
        checkout_path=checkout,
        checkpoint_path=checkpoint,
        prototxt_path=prototxt,
        caffe_module=caffe,
    )

    assert adapter.encode(_batch()).features.shape == (1, 1, 4)
    assert calls == [(str(prototxt.resolve()), str(checkpoint.resolve()), 17)]


def test_missing_assets_fail_closed_with_structured_error(tmp_path: Path) -> None:
    with pytest.raises(LegacyAssetError) as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path="external/C3D",
            checkpoint_path="weights/missing.caffemodel",
        )
    error = captured.value.to_dict()
    assert error["code"] == "missing_asset"
    assert error["integration_id"] == "c3d"
    assert {item["kind"] for item in error["details"]["missing"]} == {"checkout", "checkpoint"}


def test_missing_caffe_is_reported_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "C3D"
    checkpoint = tmp_path / "model.caffemodel"
    prototxt = checkout / "deploy.prototxt"
    checkout.mkdir()
    checkpoint.write_bytes(b"weights")
    prototxt.write_text("name: 'c3d'\n", encoding="utf-8")
    original_import = builtins.__import__

    def deny_caffe(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "caffe":
            raise ImportError("caffe unavailable in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_caffe)
    with pytest.raises(LegacyDependencyError) as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path=checkout,
            checkpoint_path=checkpoint,
            prototxt_path=prototxt,
        )
    assert captured.value.code == "missing_dependency"
    assert captured.value.to_dict()["details"]["dependency"] == "caffe"


@pytest.mark.parametrize(
    "entrypoint",
    ["../outside.py:load", "/tmp/loader.py:load", "loader.txt:load", "loader.py"],
)
def test_dynamic_entrypoint_validation_fails_closed(tmp_path: Path, entrypoint: str) -> None:
    checkout = tmp_path / "C3D"
    checkout.mkdir()
    with pytest.raises(LegacyAssetError, match="entrypoint") as captured:
        LegacyVideoAdapter(
            project_root=tmp_path,
            checkout_path=checkout,
            entrypoint=entrypoint,
            strict_assets=False,
        )
    assert captured.value.code == "invalid_entrypoint"
