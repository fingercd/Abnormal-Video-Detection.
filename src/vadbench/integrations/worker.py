"""One-request external Python worker for dependency-isolated encoders."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    ContractError,
    EncoderOutput,
    StreamState,
    StreamStep,
    validate_clip_for_capabilities,
    validate_encoder_output,
    validate_stream_step,
)
from vadbench.integrations.common import validate_output_health
from vadbench.integrations.worker_protocol import (
    DEFAULT_LIMITS,
    ProtocolLimits,
    SidecarStore,
    WorkerErrorInfo,
    WorkerRequest,
    WorkerResponse,
    read_worker_request,
    serialize_encoder_output,
    serialize_stream_result,
)
from vadbench.registry import ENCODER_REGISTRY, EncoderRegistry


@dataclass(frozen=True, slots=True)
class StreamExecution:
    steps: tuple[StreamStep, ...]
    final_output: EncoderOutput | None


def _valid_row(value: Any, batch: ClipBatch, row: int = 0) -> np.ndarray:
    array = np.asarray(value)
    length = int(batch.valid_lengths[row])
    return array[row, :length]


def _validate_stream_chunks(chunks: Sequence[ClipBatch]) -> None:
    if not chunks:
        raise ContractError("encode_stream requires at least one chunk")
    video_id = chunks[0].video_ids[0] if chunks[0].batch_size == 1 else None
    previous_time: float | None = None
    previous_frame: int | None = None
    for index, chunk in enumerate(chunks):
        if chunk.batch_size != 1:
            raise ContractError(f"stream chunk {index} must have B=1")
        if chunk.video_ids != (video_id,):
            raise ContractError("all stream chunks must have the same video_id")
        times = _valid_row(chunk.timestamps_s, chunk).astype(np.float64, copy=False)
        if previous_time is not None and float(times[0]) < previous_time:
            raise ContractError("stream chunk timestamps must progress monotonically")
        previous_time = float(times[-1])
        if chunk.frame_indices is not None:
            frames = _valid_row(chunk.frame_indices, chunk).astype(np.int64, copy=False)
            if previous_frame is not None and int(frames[0]) <= previous_frame:
                raise ContractError("stream chunk source frame ranges must progress strictly")
            previous_frame = int(frames[-1])


def execute_request(
    request: WorkerRequest,
    clips: Sequence[ClipBatch],
    *,
    adapter: Any,
) -> EncoderOutput | StreamExecution:
    """Execute one fixed/stream request with its already constructed adapter."""

    capabilities = adapter.capabilities
    if request.operation == "encode":
        if len(clips) != 1:
            raise ContractError("encode requires exactly one ClipBatch")
        batch = clips[0]
        validate_clip_for_capabilities(batch, capabilities, train=request.train)
        output = adapter.encode(batch, train=request.train)
        validate_encoder_output(output, batch)
        validate_output_health(output)
        return output

    capabilities.require("supports_streaming")
    _validate_stream_chunks(clips)
    video_id = clips[0].video_ids[0]
    state = adapter.init_state(video_id)
    if not isinstance(state, StreamState):
        raise ContractError("init_state must return StreamState")
    steps: list[StreamStep] = []
    for chunk in clips:
        validate_clip_for_capabilities(
            chunk,
            capabilities,
            streaming=True,
            train=request.train,
        )
        step = adapter.encode_step(
            chunk,
            state,
            train=request.train,
            compression=None,
        )
        validate_stream_step(
            step,
            previous_state=state,
            chunk=chunk,
            capabilities=capabilities,
        )
        if step.output is not None:
            validate_output_health(step.output)
        steps.append(step)
        state = step.state
    final_output = adapter.finalize(state)
    if final_output is not None:
        validate_encoder_output(final_output, clips[-1])
        validate_output_health(final_output)
    return StreamExecution(steps=tuple(steps), final_output=final_output)


def _error(stage: str, exc: BaseException) -> WorkerErrorInfo:
    code = {
        "request_decode": "invalid_request",
        "adapter_load": "adapter_load_failed",
        "execution": "execution_failed",
        "response_encode": "serialization_failed",
    }.get(stage, "internal_error")
    message = str(exc).strip() or type(exc).__name__
    return WorkerErrorInfo(
        code=code,
        stage=stage,
        exception_type=f"{type(exc).__module__}.{type(exc).__name__}",
        message=message[:2048],
    )


def _write_error_response(
    store: SidecarStore,
    response_path: str,
    *,
    request_id: str,
    output_dir: str,
    stage: str,
    exc: BaseException,
) -> None:
    response = WorkerResponse.failure(
        request_id=request_id,
        output_dir=output_dir,
        error=_error(stage, exc),
    )
    store.write_json(response_path, response.to_dict())


def run_worker_once(
    exchange_root: str | Path,
    request_path: str,
    response_path: str,
    *,
    registry: EncoderRegistry = ENCODER_REGISTRY,
    limits: ProtocolLimits = DEFAULT_LIMITS,
) -> int:
    """Process exactly one request, atomically writing success or error JSON."""

    store = SidecarStore(exchange_root, limits=limits)
    try:
        request, clips = read_worker_request(exchange_root, request_path, limits=limits)
    except Exception as exc:
        try:
            _write_error_response(
                store,
                response_path,
                request_id="unknown",
                output_dir="output/unknown",
                stage="request_decode",
                exc=exc,
            )
        except Exception as response_exc:
            print(f"failed to write worker error response: {response_exc}", file=sys.stderr)
        return 2

    try:
        # Keep adapter construction a distinct error identity for asset/env diagnosis.
        adapter = registry.create(request.encoder_id, **dict(request.adapter_kwargs))
    except Exception as exc:
        _write_error_response(
            store,
            response_path,
            request_id=request.request_id,
            output_dir=request.output_dir,
            stage="adapter_load",
            exc=exc,
        )
        return 1

    try:
        result = execute_request(request, clips, adapter=adapter)
    except Exception as exc:
        _write_error_response(
            store,
            response_path,
            request_id=request.request_id,
            output_dir=request.output_dir,
            stage="execution",
            exc=exc,
        )
        return 1

    try:
        if isinstance(result, EncoderOutput):
            result_kind = "encoder_output"
            result_payload = serialize_encoder_output(
                result,
                store,
                prefix=f"{request.output_dir}/result",
            )
        else:
            result_kind = "stream_result"
            result_payload = serialize_stream_result(
                result.steps,
                result.final_output,
                store,
                prefix=request.output_dir,
            )
        response = WorkerResponse.success(
            request_id=request.request_id,
            output_dir=request.output_dir,
            result_kind=result_kind,
            result=result_payload,
        )
        store.write_json(response_path, response.to_dict())
    except Exception as exc:
        try:
            _write_error_response(
                store,
                response_path,
                request_id=request.request_id,
                output_dir=request.output_dir,
                stage="response_encode",
                exc=exc,
            )
        except Exception as response_exc:
            print(f"failed to write worker error response: {response_exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one VADBench external encoder request")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument(
        "--request", required=True, help="request JSON path relative to bundle root"
    )
    parser.add_argument(
        "--response", required=True, help="response JSON path relative to bundle root"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_worker_once(args.bundle_root, args.request, args.response)


if __name__ == "__main__":  # pragma: no cover - exercised through module invocation
    raise SystemExit(main())


__all__ = ["StreamExecution", "build_parser", "execute_request", "main", "run_worker_once"]
