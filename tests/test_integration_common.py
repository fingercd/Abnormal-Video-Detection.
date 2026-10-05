from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from vadbench.contracts import ContractError, EncoderOutput, TokenTimeline
from vadbench.integrations.common import (
    OutputHealthError,
    inspect_tensor_health,
    normalize_encoder_output,
    normalize_feature_stage,
    normalize_feature_tensor,
    pool_feature_sequence,
    select_feature_tensor,
    validate_output_health,
)


def _timeline(*, batch: int = 1, tokens: int = 2) -> TokenTimeline:
    starts = np.broadcast_to(np.linspace(0.0, 0.5, tokens), (batch, tokens)).copy()
    ends = starts + 0.5
    frame_starts = np.broadcast_to(np.arange(tokens), (batch, tokens)).copy()
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=frame_starts,
        source_frame_end=frame_starts + 1,
    )


def test_normalize_feature_tensor_supports_bd_and_bsd_without_copy() -> None:
    pooled = np.arange(8, dtype=np.float32).reshape(2, 4)
    singleton = normalize_feature_tensor(pooled, batch_size=2)
    assert singleton.shape == (2, 1, 4)
    np.testing.assert_array_equal(singleton[:, 0], pooled)

    sequence = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert normalize_feature_tensor(sequence, batch_size=2) is sequence

    with pytest.raises(ContractError, match=r"\[B,D\].*\[B,S,D\]"):
        normalize_feature_tensor(np.zeros((2, 3, 4, 5), dtype=np.float32))
    with pytest.raises(ContractError, match="期望 batch"):
        normalize_feature_tensor(sequence, batch_size=1)
    with pytest.raises(ContractError, match="非空"):
        normalize_feature_tensor(np.zeros((1, 0, 4), dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_normalize_bsd_preserves_standard_float_identity_and_dtype(dtype: type) -> None:
    sequence = np.arange(12, dtype=dtype).reshape(1, 3, 4)
    normalized = normalize_feature_tensor(sequence)
    assert normalized is sequence
    assert normalized.dtype == dtype


def test_bfloat16_health_uses_float32_only_for_finite_inspection() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[[1.0, float("nan")]]], dtype=torch.bfloat16)
    health = inspect_tensor_health(tensor, name="features")
    assert health.dtype == "torch.bfloat16"
    assert health.shape == (1, 1, 2)
    assert health.finite is False
    assert health.non_finite_count == 1
    assert tensor.dtype == torch.bfloat16


def test_pool_feature_sequence_respects_valid_prefix_mask() -> None:
    features = np.array(
        [
            [[1.0], [3.0], [100.0]],
            [[2.0], [4.0], [6.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, True, False], [True, True, True]])
    pooled = pool_feature_sequence(features, mask)
    np.testing.assert_allclose(pooled, [[2.0], [4.0]])

    with pytest.raises(ContractError, match="valid_mask"):
        pool_feature_sequence(features, np.ones((2, 2), dtype=bool))
    with pytest.raises(ContractError, match="bool"):
        pool_feature_sequence(features, np.ones((2, 3), dtype=np.int64))


def test_select_and_normalize_model_output_records_identity() -> None:
    class FakeModelOutput:
        def __init__(self) -> None:
            self.logits = np.full((1, 7), 99.0, dtype=np.float32)
            self.last_hidden_state = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
            self.pooler_output = np.full((1, 4), 5.0, dtype=np.float32)

    raw = FakeModelOutput()
    selected, source = select_feature_tensor(raw, batch_size=1)
    assert selected is raw.last_hidden_state
    assert source == "last_hidden_state"

    output = normalize_encoder_output(
        raw,
        timeline=_timeline(tokens=3),
        feature_stage="last-hidden-state",
        preprocess_profile="transformers-video-v1",
        aux={"family": "unit"},
    )
    assert output.features is raw.last_hidden_state
    assert output.pooled is raw.pooler_output
    assert output.aux["feature_stage"] == "last_hidden_state"
    assert output.aux["sequence_source"] == "last_hidden_state"
    assert output.aux["preprocess_profile"] == "transformers-video-v1"
    assert output.aux["family"] == "unit"
    assert output.aux["model_output_type"].endswith("FakeModelOutput")


def test_hidden_state_stack_prefers_last_layer_and_pooled_tensor_becomes_singleton() -> None:
    raw = {
        "hidden_states": (
            np.zeros((1, 2, 4), dtype=np.float32),
            np.ones((1, 2, 4), dtype=np.float32),
        )
    }
    selected, source = select_feature_tensor(raw, batch_size=1)
    np.testing.assert_array_equal(selected, 1.0)
    assert source == "hidden_states[1]"

    pooled = np.arange(4, dtype=np.float32).reshape(1, 4)
    output = normalize_encoder_output(
        pooled,
        timeline=_timeline(tokens=1),
        feature_stage="pooled",
    )
    assert output.features.shape == (1, 1, 4)
    assert output.pooled is pooled
    assert output.aux["sequence_source"] == "pooled_singleton"


def test_feature_stage_aliases_are_normalized_and_unknown_values_fail_closed() -> None:
    assert normalize_feature_stage("projected") == "projected_visual"
    assert normalize_feature_stage("decoder-contextual") == "decoder_contextual"
    with pytest.raises(ContractError, match="未知 feature_stage"):
        normalize_feature_stage("classification_logits")


def test_output_health_is_json_ready_and_checks_video_bounds() -> None:
    output = EncoderOutput(
        features=np.ones((1, 2, 4), dtype=np.float32),
        pooled=np.ones((1, 4), dtype=np.float32),
        timeline=_timeline(tokens=2),
        aux={
            "feature_stage": "backbone_tokens",
            "sequence_source": "last_hidden_state",
        },
    )
    health = validate_output_health(
        output,
        video_duration_seconds=1.0,
        video_num_frames=4,
        require_video_bounds=True,
    )
    assert health.passed is True
    assert health.timeline.token_count == 2
    assert health.timeline.monotonic is True
    assert health.timeline.in_video_range is True
    assert health.timeline.video_bounds_checked is True
    assert health.features.shape == (1, 2, 4)
    assert health.pooled is not None and health.pooled.shape == (1, 4)
    assert health.identity_inferred is False
    json.dumps(health.to_dict(), ensure_ascii=False)


def test_finite_checks_preserve_machine_readable_failure_evidence() -> None:
    output = EncoderOutput(
        features=np.array([[[1.0, np.nan]]], dtype=np.float32),
        pooled=np.array([[1.0, np.inf]], dtype=np.float32),
        timeline=_timeline(tokens=1),
        aux={"feature_stage": "pooled", "sequence_source": "pooled_singleton"},
    )
    with pytest.raises(OutputHealthError) as captured:
        validate_output_health(output, video_duration_seconds=1.0)
    assert captured.value.health["passed"] is False
    assert captured.value.health["features"]["finite"] is False


def test_output_health_can_infer_identity_for_legacy_output() -> None:
    output = EncoderOutput(
        features=np.ones((1, 1, 4), dtype=np.float32),
        pooled=np.ones((1, 4), dtype=np.float32),
        timeline=_timeline(tokens=1),
    )
    health = validate_output_health(output, video_duration_seconds=1.0)
    assert health.feature_stage == "pooled"
    assert health.sequence_source == "pooled_singleton"
    assert health.identity_inferred is True


def _tensor_health(shape: list[int]) -> dict[str, object]:
    return {
        "shape": shape,
        "dtype": "float32",
        "finite": True,
        "non_finite_count": 0,
    }


def _timeline_health(tokens: int) -> dict[str, object]:
    return {
        "token_count": tokens,
        "valid_tokens_per_batch": [tokens],
        "token_count_matches": True,
        "monotonic": True,
        "in_video_range": True,
        "video_bounds_checked": True,
        "has_source_frames": True,
        "min_start_seconds": 0.0,
        "max_end_seconds": 1.0,
        "min_source_frame": 0,
        "max_source_frame_end": 4,
    }


def _output_record(step_index: int = 0) -> dict[str, object]:
    return {
        "step_index": step_index,
        "passed": True,
        "feature_stage": "backbone_tokens",
        "sequence_source": "last_hidden_state",
        "identity_inferred": False,
        "features": _tensor_health([1, 2, 4]),
        "pooled": _tensor_health([1, 4]),
        "timeline": _timeline_health(2),
        "aux": {"preprocess_profile": "unit"},
    }


def _smoke_document(*, mode: str = "fixed") -> dict[str, object]:
    return {
        "schema_version": "vadbench.encoder-smoke.v2",
        "generated_at_utc": "2026-08-31T02:03:04Z",
        "run_id": "unit-fixed",
        "status": "smoke_pass",
        "encoder": {
            "id": "unit",
            "display_name": "Unit Encoder",
            "adapter": "unit.adapter",
            "backend": "unit",
            "run_mode": mode,
            "feature_stage": "backbone_tokens",
        },
        "input": {
            "video": {
                "path": "data/smoke/video.mp4",
                "sha256": "a" * 64,
                "num_frames": 64,
                "fps": 30.0,
                "duration_seconds": 64 / 30,
                "width": 320,
                "height": 240,
            },
            "batches": [
                {
                    "batch_index": 0,
                    "shape": [1, 4, 240, 320, 3],
                    "dtype": "uint8",
                    "layout": "BTHWC",
                    "video_ids": ["video"],
                    "frame_indices": [0, 1, 2, 3],
                    "timestamps_seconds": [0.0, 1 / 30, 2 / 30, 3 / 30],
                }
            ],
        },
        "outputs": [_output_record()],
        "streaming": None,
        "environment": {
            "profile": "unit-cpu",
            "hostname": "ibnode3",
            "python_version": "3.11.9",
            "python_executable": "/users/fotile/VAD/.venv/bin/python",
            "device": "cpu",
            "torch_version": None,
            "cuda_version": None,
            "gpu": None,
            "packages": {"numpy": "2.1.0", "torch": None},
        },
        "assets": {
            "upstream": {
                "repo": "https://example.com/upstream.git",
                "revision": "0123456789abcdef",
                "license": "MIT",
                "checkout_path": "external/unit",
            },
            "checkpoint": {
                "repo": "example/unit",
                "revision": "v1",
                "license": "MIT",
                "path": "weights/unit/model.bin",
                "sha256": "b" * 64,
                "size_bytes": 1024,
            },
        },
        "execution": {
            "command": ["python", "-m", "vadbench", "integrations", "smoke", "unit"],
            "started_at_utc": "2026-08-31T02:03:00Z",
            "finished_at_utc": "2026-08-31T02:03:04Z",
            "elapsed_seconds": 4.0,
            "exit_code": 0,
            "log_path": "outputs/encoder-integration/unit/run.log",
            "peak_gpu_memory_bytes": None,
        },
        "provenance": {
            "git_commit": "c" * 40,
            "git_dirty": True,
            "config_sha256": "d" * 64,
            "catalog_version": "vadbench.encoder-integrations.v1",
        },
        "error": None,
    }


def _validator() -> Draft202012Validator:
    schema_path = Path(__file__).parents[1] / "schemas" / "encoder-smoke-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_smoke_v2_schema_accepts_fixed_and_streaming_real_smokes() -> None:
    validator = _validator()
    fixed = _smoke_document()
    validator.validate(fixed)

    streaming = _smoke_document(mode="streaming")
    streaming["run_id"] = "unit-streaming"
    streaming["input"]["batches"].append(  # type: ignore[index,union-attr]
        {
            "batch_index": 1,
            "shape": [1, 4, 240, 320, 3],
            "dtype": "uint8",
            "layout": "BTHWC",
            "video_ids": ["video"],
            "frame_indices": [4, 5, 6, 7],
            "timestamps_seconds": [4 / 30, 5 / 30, 6 / 30, 7 / 30],
        }
    )
    streaming["outputs"] = [_output_record(1), _output_record(2)]
    streaming["streaming"] = {
        "chunks_requested": 2,
        "chunks_completed": 2,
        "state_steps": [1, 2],
        "state_present": True,
        "cache_kinds": ["decoder_kv"],
    }
    validator.validate(streaming)


@pytest.mark.parametrize("status,exit_code", [("failed", 1), ("blocked", None)])
def test_smoke_v2_schema_preserves_failed_and_blocked_evidence(
    status: str, exit_code: int | None
) -> None:
    document = _smoke_document()
    document["status"] = status
    document["outputs"] = []
    document["execution"]["exit_code"] = exit_code  # type: ignore[index]
    document["error"] = {
        "category": "missing_asset" if status == "blocked" else "runtime_error",
        "stage": "checkpoint" if status == "blocked" else "forward",
        "message": "checkpoint unavailable" if status == "blocked" else "forward failed",
        "recoverable": status != "blocked",
        "traceback": None,
        "evidence": {"log_path": "outputs/encoder-integration/unit/run.log"},
    }
    if status == "blocked":
        checkpoint = document["assets"]["checkpoint"]  # type: ignore[index]
        for key in ("repo", "revision", "license", "path", "sha256", "size_bytes"):
            checkpoint[key] = None
    _validator().validate(document)


def test_smoke_v2_schema_rejects_false_pass_and_mode_mismatch() -> None:
    validator = _validator()
    false_pass = _smoke_document()
    false_pass["outputs"][0]["passed"] = False  # type: ignore[index]
    false_pass["outputs"][0]["features"]["finite"] = False  # type: ignore[index]
    false_pass["outputs"][0]["features"]["non_finite_count"] = 1  # type: ignore[index]
    assert list(validator.iter_errors(false_pass))

    fixed_with_state = _smoke_document()
    fixed_with_state["streaming"] = {
        "chunks_requested": 1,
        "chunks_completed": 1,
        "state_steps": [1],
        "state_present": True,
        "cache_kinds": [],
    }
    assert list(validator.iter_errors(fixed_with_state))

    failed_without_error = _smoke_document()
    failed_without_error["status"] = "failed"
    failed_without_error["execution"]["exit_code"] = 1  # type: ignore[index]
    assert list(validator.iter_errors(failed_without_error))
