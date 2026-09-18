"""Synthetic engineering checks; never accesses real annotations or GT values."""

from __future__ import annotations

import hashlib
import json
import shutil
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.research import xd_feature_grid as grid
from vadbench.research import xd_raw_alignment as audit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _timing(n=32, *, pts=None, time_base="1/12288"):
    pts = [512 * i for i in range(n)] if pts is None else pts
    return {
        "streams": [
            {
                "nb_frames": str(n),
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "time_base": time_base,
            }
        ],
        "frames": [{"pts": p, "best_effort_timestamp": p} for p in pts],
    }


def _opencv(n=32):
    return {"container_frames": n, "decoded_frames": n, "fps": 24.0}


@pytest.mark.parametrize(
    "base,inclusive,expected",
    [(0, False, [1, 2]), (0, True, [1, 2, 3]), (1, False, [0, 1]), (1, True, [0, 1, 2])],
)
def test_suffix_four_endpoint_formulas(base, inclusive, expected):
    assert (
        np.flatnonzero(audit.project_spans([(5, 7)], 8, 12 - 8 + base, inclusive)).tolist()
        == expected
    )


def test_nonzero_pts_origin_and_exact_cfr():
    timing = audit.validate_timing(_timing(4, pts=[512, 1024, 1536, 2048]), _opencv(4))
    assert timing["pts_verified"] and timing["first_pts"] == 512
    assert timing["pts_tolerance_seconds"] == 0


def test_ffprobe44_packet_pts_is_an_explicit_presentation_timestamp_field():
    document = _timing(4)
    for frame in document["frames"]:
        frame["pkt_pts"] = frame.pop("pts")
    assert audit.validate_timing(document, _opencv(4))["pts_field"] == "pkt_pts"
    document["frames"][1]["pts"] = document["frames"][1].pop("pkt_pts")
    with pytest.raises(audit.XDAlignmentError, match="MISSING_CONTAINER_OR_FULL_PTS"):
        audit.validate_timing(document, _opencv(4))


def test_quantized_cfr_uses_only_preregistered_one_tick_tolerance():
    pts = [round(i * 12800 / 24) for i in range(12)]
    timing = audit.validate_timing(_timing(12, pts=pts, time_base="1/12800"), _opencv(12))
    assert timing["pts_tolerance_seconds"] == 1 / 12800
    pts[-1] += 2
    with pytest.raises(audit.XDAlignmentError, match="CADENCE_OR_DRIFT"):
        audit.validate_timing(_timing(12, pts=pts, time_base="1/12800"), _opencv(12))


@pytest.mark.parametrize(
    "failure",
    ["missing_pts", "count", "duplicate", "reverse", "vfr", "fps", "decode", "opencv_fps"],
)
def test_timing_cannot_be_replaced_by_average_fps(failure):
    document, cv = _timing(4), _opencv(4)
    if failure == "missing_pts":
        del document["frames"][1]["pts"]
    elif failure == "count":
        document["frames"].pop()
    elif failure == "duplicate":
        document["frames"][1] = dict(document["frames"][0])
    elif failure == "reverse":
        document["frames"].reverse()
    elif failure == "vfr":
        document["frames"][2]["pts"] += 1
        document["frames"][2]["best_effort_timestamp"] += 1
    elif failure == "fps":
        document["streams"][0]["avg_frame_rate"] = "25/1"
    elif failure == "decode":
        cv["decoded_frames"] = 3
    else:
        cv["fps"] = 30.0
    with pytest.raises(audit.XDAlignmentError):
        audit.validate_timing(document, cv)


def _comparison_inputs(n=32):
    rows = [{"video_id": "positive_label_B1", "t": 2, "gt_start": 0, "gt_end_exclusive": 32}]
    annotations = {rows[0]["video_id"]: [(3, 10)]}
    truth = audit.project_spans([(3, 10)], 32, 0, False)
    return rows, [n], annotations, truth


def test_same_coordinate_suffix_alias_is_not_false_ambiguity():
    result = audit.compare_candidates(*_comparison_inputs())
    assert result["prefix_exact_all"]
    assert result["suffix_resolved_without_coordinate_ambiguity"]
    assert set(result["exact_candidates"]) == {
        "b0-half_open-origin0",
        "b0-half_open-suffix-N-minus-L",
    }


def test_observational_alias_with_different_coordinates_is_reported_for_review():
    rows, counts, annotations, truth = _comparison_inputs(40)
    annotations[rows[0]["video_id"]] = [(0, 40)]
    result = audit.compare_candidates(rows, counts, annotations, np.ones(32, dtype=bool))
    assert result["prefix_exact_all"]
    assert not result["suffix_resolved_without_coordinate_ambiguity"]
    assert "b0-half_open-suffix-N-minus-L" in result["exact_but_distinct_coordinate_candidates"]


def test_short_raw_count_cannot_be_clamped_to_zero_origin():
    with pytest.raises(audit.XDAlignmentError, match="SHORTER_THAN_CANONICAL"):
        audit.compare_candidates(*_comparison_inputs(31))


def test_parser_rejects_unknown_and_missing_positive_without_disclosing_identity():
    rows = [{"video_id": "known_label_B1"}]
    for text in ("foreign_label_B1 2 5", "", "known_label_B1 1"):
        with pytest.raises(audit.XDAlignmentError) as error:
            audit.parse_annotations(text, rows)
        assert "known_label" not in str(error.value) and "foreign_label" not in str(error.value)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    prior.mkdir()
    original = {
        "deferred_nonprefix_family": {
            "deterministic_formula": "raw origin = verified_decoded_N_i - L_i"
        },
        "candidates": audit.fixed_candidates()[:12],
    }
    prior_files = {}
    for name, value in (
        ("preregistered-candidates.json", original),
        ("aggregate-results.json", {}),
        ("independent-validation.json", {}),
    ):
        prior_files[name] = _sha(_write(prior / name, value))
    monkeypatch.setattr(audit, "PRIOR_FILES", prior_files)
    freeze = _write(tmp_path / "method.json", {"synthetic_method_freeze": True})
    monkeypatch.setattr(audit, "METHOD_SHA", audit._canonical(json.loads(freeze.read_text())))
    root = tmp_path / "raw"
    rows, identities = [], []
    for i, suffix in enumerate(("B1", "A")):
        name = f"synthetic{i}_label_{suffix}"
        member = f"videos/{name}.mp4"
        path = root / "test_videos" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic raw bytes {i}".encode())
        identities.append(
            {
                "video_id": f"opaque{i}",
                "path": f"test_videos/{member}",
                "archive_member": member,
                "archive_volume": "test_videos",
                "official_split": "test",
                "central_directory_sha256": "d" * 64,
                "provider_crc32": f"{zlib.crc32(path.read_bytes()):08x}",
                "uncompressed_size": path.stat().st_size,
            }
        )
        rows.append(
            {
                "video_index": i,
                "video_id": name,
                "feature_name": name + "__0.npy",
                "t": 2,
                "gt_start": i * 32,
                "gt_end_exclusive": (i + 1) * 32,
                "header_verified": True,
                "status": "header_verified",
                "verification_method": "direct_npy_header",
                "npy_version": [1, 0],
                "npy_dtype": "<f4",
                "feature_dim": 1024,
                "npy_header_bytes": 128,
            }
        )
    index = tmp_path / "test.identity.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in identities), encoding="utf8")
    monkeypatch.setattr(audit, "TEST_INDEX_SHA", _sha(index))
    metadata = {
        "status": "ok",
        "video_count": 2,
        "gt_reference_length": 64,
        "records": rows,
        "provenance": {
            "official_test_list_sha256": grid.OFFICIAL_TEST_LIST_SHA256,
            "gt_npy_header_sha256": grid.OFFICIAL_GT_HEADER_SHA256,
            "ti_table_sha256": grid.feature_grid_table_sha256(rows),
        },
    }
    metadata_path = _write(tmp_path / "grid.json", metadata)
    monkeypatch.setattr(audit, "CANONICAL_SHA", grid.feature_grid_metadata_sha256(metadata))
    monkeypatch.setattr(audit, "VIDEO_COUNT", 2)
    monkeypatch.setattr(grid, "OFFICIAL_VIDEO_COUNT", 2)
    monkeypatch.setattr(grid, "OFFICIAL_GT_LENGTH", 64)
    volume = _write(
        tmp_path / "volume-receipt.json",
        {
            "volume": "test_videos",
            "state": "ready",
            "data_ready": True,
            "member_count": 2,
            "uncompressed_bytes": sum(r["uncompressed_size"] for r in identities),
            "central_directory_sha256": "d" * 64,
            "zip_sha256": "a" * 64,
            "crc_validation": "streamed_zipfile_crc_plus_local_crc32",
        },
    )
    sealed = prior / "sealed-inputs"
    sealed.mkdir()
    annotation = sealed / "annotations.txt"
    annotation.write_text(rows[0]["video_id"] + " 3 10\n", encoding="utf8")
    gt = sealed / "gt.npy"
    np.save(gt, np.r_[audit.project_spans([(3, 10)], 32, 0, False), np.zeros(32)].astype("<f4"))
    monkeypatch.setattr(audit, "ANNOTATION_SHA", _sha(annotation))
    monkeypatch.setattr(audit, "GT_SHA", _sha(gt))
    request = audit.RawAlignmentRequest(
        raw_root=str(root),
        test_index=str(index),
        canonical_metadata=str(metadata_path),
        prior_alignment_root=str(prior),
        method_freeze=str(freeze),
        volume_receipt=str(volume),
        output_root=str(tmp_path / "runs"),
        run_id="first",
    )
    calls = []

    def probe(path, directory, runtime):
        run = directory.parents[2]
        assert (run / "preregistration.json").is_file()
        directory.mkdir(parents=True)
        _write(directory / "ffprobe.json", _timing())
        _write(directory / "opencv.json", _opencv())
        (directory / "ffprobe.stderr.txt").write_bytes(b"")
        calls.append(path)

    return request, probe, calls


def test_complete_computed_audit_publishes_evaluator_compatible_seal(pipeline, monkeypatch):
    request, probe, calls = pipeline
    result = audit.run_raw_alignment_audit(
        request, probe=probe, runtime_factory=lambda _: {"cpu_synthetic": True}
    )
    assert result["status"] == "sealed" and len(calls) == 2
    assert "opaque0" not in json.dumps(result) and "synthetic0_label" not in json.dumps(result)
    from vadbench.paper import xd_evaluation as evaluator

    for name, value in (
        ("CANONICAL_METADATA_SHA256", audit.CANONICAL_SHA),
        ("GT_FILE_SHA256", audit.GT_SHA),
        ("OFFICIAL_VIDEO_COUNT", 2),
    ):
        monkeypatch.setattr(evaluator, name, value)
    root = Path(result["run_dir"])
    evaluation_request = evaluator.FrozenXDEvaluationRequest(
        encoder="videomaev2",
        device="cpu",
        dataset_root=request.raw_root,
        test_manifest=str(root / "private/test.jsonl"),
        audit_report=result["seal"]["path"],
        method_source_run="unused",
        output_root="unused",
        canonical_metadata=request.canonical_metadata,
        ground_truth=str(Path(request.prior_alignment_root) / "sealed-inputs/gt.npy"),
        raw_coordinate_receipt_sha256=result["seal"]["sha256"],
        head_data_contract_sha256="a" * 64,
        original_role_lock_path="unused",
    )
    assert len(evaluator.verify_raw_coordinates(evaluation_request).manifest) == 2


def test_missing_volume_proof_never_opens_raw_or_gt(pipeline, monkeypatch):
    request, probe, calls = pipeline
    Path(request.volume_receipt).unlink()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("GT opened before ready proof"))
    result = audit.run_raw_alignment_audit(request, probe=probe)
    assert result["status"] == "blocked_raw_volume_not_ready" and not calls
    assert not (Path(result["run_dir"]) / "raw-coordinate-receipt.json").exists()


def test_raw_crc_failure_does_not_read_gt_values(pipeline, monkeypatch):
    request, probe, calls = pipeline
    target = next(Path(request.raw_root).rglob("*.mp4"))
    target.write_bytes(b"changed bytes")
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("GT opened before complete raw evidence")
    )
    result = audit.run_raw_alignment_audit(request, probe=probe, runtime_factory=lambda _: {})
    assert result["status"] == "blocked_raw_coordinate_failures"
    assert result["failure_counts"] == {"RAW_VIDEO_CRC_OR_SIZE": 1}


def test_probe_cache_is_revalidated_and_never_claimed_as_new_decode(pipeline):
    request, probe, calls = pipeline
    first = audit.run_raw_alignment_audit(request, probe=probe, runtime_factory=lambda _: {})
    second = audit.run_raw_alignment_audit(
        replace(request, run_id="second", resume_source_run=first["run_dir"]),
        probe=lambda *a: pytest.fail("verified identical probe was decoded again"),
        runtime_factory=lambda _: {},
    )
    assert (
        second["status"] == "sealed"
        and second["reused_probe_videos"] == 2
        and second["newly_probed_videos"] == 0
    )


def test_tampered_probe_cache_blocks_seal(pipeline):
    request, probe, calls = pipeline
    first = audit.run_raw_alignment_audit(request, probe=probe, runtime_factory=lambda _: {})
    cache = Path(first["run_dir"]) / "private/probes/0000/ffprobe.json"
    cache.write_text("{}", encoding="utf8")
    second = audit.run_raw_alignment_audit(
        replace(request, run_id="second", resume_source_run=first["run_dir"]),
        probe=probe,
        runtime_factory=lambda _: {},
    )
    assert second["status"] == "blocked_raw_coordinate_failures"
    assert second["failure_counts"] == {"CACHED_PROBE_ARTIFACT_SHA": 1}


@pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe is unavailable in this runtime"
)
def test_real_ffprobe_and_opencv_decode_a_synthetic_video(tmp_path):
    import cv2

    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48))
    assert writer.isOpened()
    for i in range(32):
        writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    writer.release()
    result = audit.probe_raw_video(path, tmp_path / "probe", audit._runtime("ffprobe"))
    assert result["num_frames"] == result["pts_count"] == result["opencv_decoded_frames"] == 32
    assert result["fps"] == 24.0
