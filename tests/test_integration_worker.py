from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    StreamState,
    StreamStep,
    TokenTimeline,
)
from vadbench.integrations.common import OutputHealthError
from vadbench.integrations.worker import run_worker_once
from vadbench.integrations.worker_protocol import (
    ArraySidecarRef,
    RemoteWorkerError,
    SidecarIntegrityError,
    SidecarStore,
    StreamWorkerResult,
    WorkerErrorInfo,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResponse,
    read_worker_request,
    read_worker_response,
    serialize_encoder_output,
    write_worker_request,
)
from vadbench.registry import EncoderRegistry

FIXED_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=True,
    fixed_num_frames=2,
    min_frames=2,
    max_frames=2,
)
STREAM_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=False,
    supports_streaming=True,
    min_frames=1,
)


def _batch(*, video_id: str = "video", start_frame: int = 0, start_s: float = 0.0) -> ClipBatch:
    return ClipBatch(
        frames=np.arange(1 * 2 * 3 * 4 * 3, dtype=np.uint8).reshape(1, 2, 3, 4, 3),
        timestamps_s=np.array([[start_s, start_s + 0.5]], dtype=np.float64),
        video_ids=(video_id,),
        valid_mask=np.array([[True, True]], dtype=bool),
        frame_indices=np.array([[start_frame, start_frame + 1]], dtype=np.int64),
        metadata={"clip_ids": [f"{video_id}:{start_frame}"], "nested": {"ok": True}},
    )


def _output(
    batch: ClipBatch, value: float, *, aux: dict[str, object] | None = None
) -> EncoderOutput:
    features = np.full((batch.batch_size, 1, 3), value, dtype=np.float32)
    starts = np.asarray(batch.timestamps_s)[:, :1].copy()
    ends = np.asarray(batch.timestamps_s)[:, -1:].copy()
    frame_start = np.asarray(batch.frame_indices)[:, :1].copy()
    frame_end = np.asarray(batch.frame_indices)[:, -1:].copy() + 1
    return EncoderOutput(
        features=features,
        pooled=features[:, 0],
        timeline=TokenTimeline(
            start_s=starts,
            end_s=ends,
            source_frame_start=frame_start,
            source_frame_end=frame_end,
        ),
        aux={
            "feature_stage": "backbone_tokens",
            "sequence_source": "synthetic",
            **(aux or {}),
        },
    )


class _FixedAdapter:
    capabilities = FIXED_CAPABILITIES

    def __init__(self, *, value: float = 2.0) -> None:
        self.value = value

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        return _output(batch, self.value, aux={"train": train})


class _FailingAdapter(_FixedAdapter):
    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        raise RuntimeError("synthetic forward failure")


class _BadAuxAdapter(_FixedAdapter):
    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        return _output(batch, 3.0, aux={"unsafe": Path("not-json")})


class _NonFiniteAdapter(_FixedAdapter):
    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        output = _output(batch, 3.0)
        np.asarray(output.features)[0, 0, 0] = np.nan
        return output


class _StreamingAdapter:
    capabilities = STREAM_CAPABILITIES

    def init_state(self, video_id: str) -> StreamState:
        return StreamState(video_id=video_id, opaque=object(), metadata={"phase": "init"})

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: object | None = None,
    ) -> StreamStep:
        next_state = state.replace(
            step_index=state.step_index + 1,
            opaque=object(),
            next_timestamp_s=float(np.asarray(chunk.timestamps_s)[0, -1] + 0.5),
            metadata={"phase": "running"},
        )
        return StreamStep(
            output=_output(chunk, float(next_state.step_index)),
            state=next_state,
            telemetry={"compression_is_none": compression is None, "train": train},
        )

    def finalize(self, state: StreamState) -> EncoderOutput | None:
        return None


def _registry(
    name: str, factory: type[object], capabilities: EncoderCapabilities
) -> EncoderRegistry:
    registry = EncoderRegistry()
    registry.register_factory(name, factory, capabilities=capabilities)
    return registry


def test_npy_and_npz_sidecars_round_trip_with_declared_identity(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = store.write_array("input/run/first.npy", first)

    assert ref.dtype == np.dtype(np.float32).str
    assert ref.nbytes == first.nbytes
    np.testing.assert_array_equal(store.load_array(ref, required_prefix="input/run"), first)

    path = tmp_path / ref.path
    with path.open("wb") as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {"descr": first.dtype.str, "fortran_order": False, "shape": first.shape},
        )
        handle.write(first.tobytes())
    payload = bytearray(path.read_bytes())
    payload[6:8] = b"\x03\x00"
    path.write_bytes(payload)
    v3_ref = replace(ref, sha256=hashlib.sha256(payload).hexdigest())
    np.testing.assert_array_equal(store.load_array(v3_ref, required_prefix="input/run"), first)

    refs = store.write_npz(
        "input/run/bundle.npz",
        {"frames": np.arange(4, dtype=np.uint8), "times": np.array([0.0, 0.5])},
    )
    np.testing.assert_array_equal(
        store.load_array(refs["frames"], required_prefix="input/run"),
        np.arange(4, dtype=np.uint8),
    )
    np.testing.assert_allclose(
        store.load_array(refs["times"], required_prefix="input/run"), [0.0, 0.5]
    )


def test_sidecar_shape_dtype_checksum_and_file_size_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    store = SidecarStore(tmp_path, create=True)
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = store.write_array("input/run/value.npy", array)

    with pytest.raises(SidecarIntegrityError, match="shape mismatch"):
        store.load_array(replace(ref, shape=(3, 2)), required_prefix="input/run")
    with pytest.raises(SidecarIntegrityError, match="dtype mismatch"):
        store.load_array(replace(ref, dtype=np.dtype(np.int32).str), required_prefix="input/run")
    with pytest.raises(SidecarIntegrityError, match="file_size mismatch"):
        store.load_array(replace(ref, file_size=ref.file_size + 1), required_prefix="input/run")

    path = tmp_path / ref.path
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(SidecarIntegrityError, match="SHA256 mismatch"):
        store.load_array(ref, required_prefix="input/run")


def test_sidecar_rejects_huge_header_shape_before_array_load(tmp_path: Path) -> None:
    path = tmp_path / "input" / "run" / "huge.npy"
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle,
            {
                "descr": np.dtype(np.float32).str,
                "fortran_order": False,
                "shape": (10**12,),
            },
        )
    ref = ArraySidecarRef(
        path="input/run/huge.npy",
        format="npy",
        key=None,
        shape=(1,),
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        file_size=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    with pytest.raises(SidecarIntegrityError, match="header array bytes"):
        SidecarStore(tmp_path).load_array(ref, required_prefix="input/run")


def test_sidecar_rejects_compressed_npz(tmp_path: Path) -> None:
    path = tmp_path / "input" / "run" / "compressed.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, value=np.array([1.0], dtype=np.float32))
    ref = ArraySidecarRef(
        path="input/run/compressed.npz",
        format="npz",
        key="value",
        shape=(1,),
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        file_size=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    with pytest.raises(SidecarIntegrityError, match="unsafe members"):
        SidecarStore(tmp_path).load_array(ref, required_prefix="input/run")


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/value.npy",
        "../value.npy",
        "input/../../value.npy",
        "C:/value.npy",
        "//server/share/value.npy",
        "input\\value.npy",
        "input//value.npy",
    ],
)
def test_array_reference_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(WorkerProtocolError):
        ArraySidecarRef(
            path=path,
            format="npy",
            key=None,
            shape=(1,),
            dtype="|u1",
            nbytes=1,
            file_size=129,
            sha256="0" * 64,
        )


def test_sidecar_rejects_symlink_even_when_checksum_matches(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    store = SidecarStore(root, create=True)
    outside = tmp_path / "outside.npy"
    with outside.open("wb") as handle:
        np.save(handle, np.array([1], dtype=np.uint8), allow_pickle=False)
    link = root / "input" / "run" / "link.npy"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"symlink unsupported: {exc}")
    ref = ArraySidecarRef(
        path="input/run/link.npy",
        format="npy",
        key=None,
        shape=(1,),
        dtype="|u1",
        nbytes=1,
        file_size=outside.stat().st_size,
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(WorkerProtocolError, match="symlink"):
        store.load_array(ref, required_prefix="input/run")


def test_object_array_and_non_json_metadata_are_rejected(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    with pytest.raises(WorkerProtocolError, match="unsupported portable dtype"):
        store.write_array("input/run/object.npy", np.array([object()], dtype=object))

    batch = _batch()
    unsafe = ClipBatch(
        frames=batch.frames,
        timestamps_s=batch.timestamps_s,
        video_ids=batch.video_ids,
        valid_mask=batch.valid_mask,
        frame_indices=batch.frame_indices,
        metadata={"unsafe": Path("value")},
    )
    with pytest.raises(WorkerProtocolError, match="non-portable type"):
        write_worker_request(
            tmp_path / "unsafe",
            "request.json",
            request_id="unsafe",
            encoder_id="fixed",
            operation="encode",
            clips=[unsafe],
        )


def test_request_round_trip_preserves_bthwc_time_and_frame_metadata(tmp_path: Path) -> None:
    batch = _batch()
    request = write_worker_request(
        tmp_path,
        "request.json",
        request_id="roundtrip",
        encoder_id="fixed",
        operation="encode",
        clips=[batch],
        adapter_kwargs={"variant": "tiny"},
    )
    loaded_request, clips = read_worker_request(tmp_path, "request.json")

    assert loaded_request == request
    assert loaded_request.adapter_kwargs["variant"] == "tiny"
    assert clips[0].video_ids == batch.video_ids
    assert clips[0].metadata["nested"] == {"ok": True}
    np.testing.assert_array_equal(clips[0].frames, batch.frames)
    np.testing.assert_array_equal(clips[0].timestamps_s, batch.timestamps_s)
    np.testing.assert_array_equal(clips[0].frame_indices, batch.frame_indices)


def test_fixed_worker_loopback_reconstructs_encoder_output(tmp_path: Path) -> None:
    registry = _registry("fixed", _FixedAdapter, FIXED_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="fixed-run",
        encoder_id="fixed",
        operation="encode",
        clips=[_batch()],
        adapter_kwargs={"value": 4.0},
    )

    assert (
        run_worker_once(
            tmp_path,
            "request.json",
            "response.json",
            registry=registry,
        )
        == 0
    )
    response, result = read_worker_response(
        tmp_path, "response.json", expected_request_id="fixed-run"
    )

    assert response.status == "ok"
    assert isinstance(result, EncoderOutput)
    assert result.features.shape == (1, 1, 3)
    np.testing.assert_allclose(result.features, 4.0)
    assert result.pooled.shape == (1, 3)
    assert result.aux == {
        "feature_stage": "backbone_tokens",
        "sequence_source": "synthetic",
        "train": False,
    }
    assert result.timeline.source_frame_end.tolist() == [[2]]


def test_worker_constructs_the_adapter_once_per_request(tmp_path: Path) -> None:
    calls: list[float] = []

    def factory(*, value: float = 2.0) -> _FixedAdapter:
        calls.append(value)
        return _FixedAdapter(value=value)

    registry = EncoderRegistry()
    registry.register_factory("fixed", factory, capabilities=FIXED_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="one-construction",
        encoder_id="fixed",
        operation="encode",
        clips=[_batch()],
        adapter_kwargs={"value": 7.0},
    )

    assert run_worker_once(tmp_path, "request.json", "response.json", registry=registry) == 0
    assert calls == [7.0]


def test_stream_worker_keeps_opaque_in_process_and_returns_step_records(tmp_path: Path) -> None:
    registry = _registry("stream", _StreamingAdapter, STREAM_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="stream-run",
        encoder_id="stream",
        operation="encode_stream",
        clips=[
            _batch(start_frame=0, start_s=0.0),
            _batch(start_frame=2, start_s=1.0),
        ],
    )

    assert (
        run_worker_once(
            tmp_path,
            "request.json",
            "response.json",
            registry=registry,
        )
        == 0
    )
    response, result = read_worker_response(tmp_path, "response.json")

    assert response.result_kind == "stream_result"
    assert isinstance(result, StreamWorkerResult)
    assert len(result.steps) == 2
    assert [step.state["step_index"] for step in result.steps] == [1, 2]
    assert all(step.state["opaque_present"] is True for step in result.steps)
    assert all(step.state["caches"] == {} for step in result.steps)
    assert result.steps[1].output is not None
    np.testing.assert_allclose(result.steps[1].output.features, 2.0)
    assert result.steps[1].telemetry["compression_is_none"] is True
    assert result.final_output is None
    assert "object at" not in (tmp_path / "response.json").read_text(encoding="utf-8")


def test_stream_worker_rejects_non_progressing_chunk_boundaries(tmp_path: Path) -> None:
    registry = _registry("stream", _StreamingAdapter, STREAM_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="bad-stream",
        encoder_id="stream",
        operation="encode_stream",
        clips=[_batch(start_frame=0), _batch(start_frame=1, start_s=1.0)],
    )

    assert run_worker_once(tmp_path, "request.json", "response.json", registry=registry) == 1
    response, result = read_worker_response(tmp_path, "response.json")
    assert result is None
    assert response.error is not None
    assert response.error.code == "execution_failed"
    assert "progress strictly" in response.error.message


def test_worker_returns_structured_forward_error(tmp_path: Path) -> None:
    registry = _registry("failing", _FailingAdapter, FIXED_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="failure-run",
        encoder_id="failing",
        operation="encode",
        clips=[_batch()],
    )

    assert run_worker_once(tmp_path, "request.json", "response.json", registry=registry) == 1
    response, result = read_worker_response(tmp_path, "response.json")

    assert result is None
    assert response.error is not None
    assert response.error.code == "execution_failed"
    assert response.error.stage == "execution"
    assert response.error.exception_type.endswith("RuntimeError")
    with pytest.raises(RemoteWorkerError, match="synthetic forward failure"):
        response.raise_for_error()


def test_worker_rejects_non_finite_encoder_output(tmp_path: Path) -> None:
    registry = _registry("nonfinite", _NonFiniteAdapter, FIXED_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="nonfinite-run",
        encoder_id="nonfinite",
        operation="encode",
        clips=[_batch()],
    )

    assert run_worker_once(tmp_path, "request.json", "response.json", registry=registry) == 1
    response, result = read_worker_response(tmp_path, "response.json")
    assert result is None
    assert response.error is not None
    assert response.error.code == "execution_failed"
    assert "健康检查" in response.error.message


def test_worker_returns_serialization_error_for_non_json_aux(tmp_path: Path) -> None:
    registry = _registry("bad_aux", _BadAuxAdapter, FIXED_CAPABILITIES)
    write_worker_request(
        tmp_path,
        "request.json",
        request_id="bad-aux-run",
        encoder_id="bad_aux",
        operation="encode",
        clips=[_batch()],
    )

    assert run_worker_once(tmp_path, "request.json", "response.json", registry=registry) == 2
    response, result = read_worker_response(tmp_path, "response.json")
    assert result is None
    assert response.error is not None
    assert response.error.code == "serialization_failed"
    assert "non-portable type" in response.error.message


def test_controller_rechecks_non_finite_response_output(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    output = _output(_batch(), 1.0)
    np.asarray(output.features)[0, 0, 0] = np.nan
    payload = serialize_encoder_output(output, store, prefix="output/untrusted/result")
    response = WorkerResponse.success(
        request_id="untrusted",
        output_dir="output/untrusted",
        result_kind="encoder_output",
        result=payload,
    )
    store.write_json("response.json", response.to_dict())

    with pytest.raises(OutputHealthError, match="健康检查"):
        read_worker_response(tmp_path, "response.json")


def test_request_rejects_unknown_import_target_field(tmp_path: Path) -> None:
    request = write_worker_request(
        tmp_path,
        "request.json",
        request_id="no-injection",
        encoder_id="fixed",
        operation="encode",
        clips=[_batch()],
    )
    payload = request.to_dict()
    payload["target"] = "os:system"
    SidecarStore(tmp_path).write_json("injected.json", payload)

    with pytest.raises(WorkerProtocolError, match="extra=.*target"):
        read_worker_request(tmp_path, "injected.json")


def test_wire_dataclasses_round_trip_only_exact_fields() -> None:
    reference = ArraySidecarRef(
        path="input/run/value.npy",
        format="npy",
        key=None,
        shape=(1,),
        dtype="|u1",
        nbytes=1,
        file_size=129,
        sha256="0" * 64,
    )
    error = WorkerErrorInfo(
        code="execution_failed",
        stage="execution",
        exception_type="RuntimeError",
        message="failed",
    )
    values = (
        reference,
        WorkerRequest("request", "fixed", "encode", ({},), output_dir="output/request"),
        error,
        WorkerResponse.failure(request_id="request", output_dir="output/request", error=error),
    )

    for value in values:
        payload = value.to_dict()
        assert type(value).from_dict(payload) == value
        with pytest.raises(WorkerProtocolError, match="extra=.*unexpected"):
            type(value).from_dict({**payload, "unexpected": None})


def test_json_parser_rejects_duplicate_keys_and_non_finite_constants(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    (tmp_path / "duplicate.json").write_text('{"a":1,"a":2}\n', encoding="utf-8")
    (tmp_path / "nan.json").write_text('{"value":NaN}\n', encoding="utf-8")

    with pytest.raises(WorkerProtocolError, match="duplicate JSON key"):
        store.read_json("duplicate.json")
    with pytest.raises(WorkerProtocolError, match="non-finite JSON number"):
        store.read_json("nan.json")


def test_worker_request_validates_operation_cardinality() -> None:
    with pytest.raises(WorkerProtocolError, match="exactly one"):
        WorkerRequest(
            request_id="bad",
            encoder_id="fixed",
            operation="encode",
            clips=({}, {}),
            output_dir="output/bad",
        )

    with pytest.raises(WorkerProtocolError, match="operation"):
        WorkerRequest(
            request_id="bad-type",
            encoder_id="fixed",
            operation=[],  # type: ignore[arg-type]
            clips=({},),
            output_dir="output/bad-type",
        )


def test_ref_rejects_declared_nbytes_inconsistent_with_shape_and_dtype() -> None:
    with pytest.raises(WorkerProtocolError, match="nbytes mismatch"):
        ArraySidecarRef(
            path="input/run/value.npy",
            format="npy",
            key=None,
            shape=(2, 3),
            dtype="<f4",
            nbytes=4,
            file_size=132,
            sha256="0" * 64,
        )


def test_response_ref_checksum_covers_entire_file(tmp_path: Path) -> None:
    store = SidecarStore(tmp_path, create=True)
    ref = store.write_array("output/run/value.npy", np.array([1.0], dtype=np.float32))
    assert ref.sha256 == hashlib.sha256((tmp_path / ref.path).read_bytes()).hexdigest()
    assert os.path.getsize(tmp_path / ref.path) == ref.file_size
