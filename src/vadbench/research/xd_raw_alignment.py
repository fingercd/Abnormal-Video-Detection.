"""Dataset-only, full raw-video audit for the frozen XD canonical prefix.

The audit registers all candidates before opening raw videos or label values.
No model or model score is read. Public results contain aggregate counts only;
identity, timestamps and source records are kept in the private run directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.data.xd_violence import parse_xd_labels
from vadbench.features import atomic_write_json
from vadbench.research.xd_feature_grid import _verified_rows, feature_grid_metadata_sha256

CANONICAL_SHA = "a03cdc6860326d52d0a767d4784e73e7f7c72a9b38cd2017a9c23ed3263537ce"
TEST_INDEX_SHA = "a984872a18adf766204fd0d4c6734e41316ca43b0399bdcaedb3c60eba6ab841"
METHOD_SHA = "54b81894e42e244c320b5d40a06c21199e3e77ebf1e53264b0f6069bda005271"
GT_SHA = "e2f9260882450e9ccabe8deaaac63a54b94558db4197f3e329125bf36e3b3863"
ANNOTATION_SHA = "27b583ba06fe2e096d7e3d281845365c3b24de7e30797ae5c1ec771667e04c2f"
PRIOR_FILES = {
    "preregistered-candidates.json": "2ad98cc546f4af2b1c90d177fb70136557e7fade745c534ea5ad09f00281c7ba",
    "aggregate-results.json": "f65c72d95e506e2e070dbd62eeb8e1206ae9be2daed4b760496fb566a38157a0",
    "independent-validation.json": "fc7edd10f48da3066793c9f9a10ce212841aafd0e3972c05b04862d5e1c7d9e9",
}
VIDEO_COUNT = 800
SEAL_PROTOCOL = "xd-violence/frozen-rgb-raw-prefix-pr-auc-v1"
SEAL_MAPPING = {
    "index_base": 0,
    "end_policy": "half_open",
    "prefix_origin": 0,
    "length": "16*Ti",
    "prediction_overlap_reduction": "mean",
}


class XDAlignmentError(ValueError):
    """Safe aggregate error code; never include IDs, intervals, GT or paths."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise XDAlignmentError(code)


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf8"))
    _require(isinstance(value, dict), "JSON_OBJECT_REQUIRED")
    return value


def _bound(path: Path, expected: str, code: str) -> Path:
    _require(path.is_file(), code + "_MISSING")
    _require(sha256_file(path) == expected, code + "_SHA")
    return path


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def fixed_candidates() -> list[dict[str, Any]]:
    """The original twelve prefix candidates, then the four deferred suffixes."""
    result = []
    for end_policy in ("inclusive", "half_open"):
        for base in (0, 1):
            for origin in (0, 1, 16):
                result.append(
                    {
                        "candidate_id": f"b{base}-{end_policy}-origin{origin}",
                        "index_base": base,
                        "end_policy": end_policy,
                        "family": "prefix",
                        "origin": origin,
                    }
                )
    for end_policy in ("inclusive", "half_open"):
        for base in (0, 1):
            result.append(
                {
                    "candidate_id": f"b{base}-{end_policy}-suffix-N-minus-L",
                    "index_base": base,
                    "end_policy": end_policy,
                    "family": "suffix",
                    "origin": "N-L",
                }
            )
    return result


@dataclass(frozen=True)
class RawAlignmentRequest:
    raw_root: str
    test_index: str
    canonical_metadata: str
    prior_alignment_root: str
    method_freeze: str
    volume_receipt: str
    output_root: str
    ffprobe: str = "ffprobe"
    resume_source_run: str | None = None
    wait_for_ready: bool = False
    poll_seconds: float = 60.0
    run_id: str | None = None
    cpu_lock_path: str | None = None


def _preregister(request: RawAlignmentRequest) -> tuple[Path, dict[str, Any]]:
    root = Path(request.output_root).resolve() / (request.run_id or new_run_id("xd-raw-alignment"))
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    prior = Path(request.prior_alignment_root).resolve()
    hashes = {}
    for name, expected in PRIOR_FILES.items():
        path = _bound(prior / name, expected, "PRIOR_ALIGNMENT")
        hashes[str(path)] = expected
    original = _json(prior / "preregistered-candidates.json")
    _require(
        original["deferred_nonprefix_family"]["deterministic_formula"].startswith(
            "raw origin = verified_decoded_N_i - L_i"
        ),
        "SUFFIX_WAS_NOT_PREREGISTERED",
    )
    original_candidates = original["candidates"]
    candidates = fixed_candidates()
    _require(
        [r["candidate_id"] for r in original_candidates]
        == [r["candidate_id"] for r in candidates[:12]],
        "ORIGINAL_CANDIDATE_ORDER",
    )
    freeze_path = Path(request.method_freeze).resolve()
    _require(_canonical(_json(freeze_path)) == METHOD_SHA, "METHOD_FREEZE_CHANGED")
    hashes[str(freeze_path)] = sha256_file(freeze_path)
    index_path = _bound(Path(request.test_index).resolve(), TEST_INDEX_SHA, "TEST_IDENTITY_INDEX")
    hashes[str(index_path)] = TEST_INDEX_SHA
    metadata_path = Path(request.canonical_metadata).resolve()
    metadata = _json(metadata_path)
    _require(feature_grid_metadata_sha256(metadata) == CANONICAL_SHA, "CANONICAL_METADATA_CHANGED")
    hashes[str(metadata_path)] = sha256_file(metadata_path)
    document = {
        "schema_version": 1,
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "dataset_only_full_raw_coordinate_audit",
        "registration_stage": "extension_after_prior_label_audit_before_this_attempt_raw_and_label_value_reads",
        "model_scores_read": False,
        "raw_videos_accessed_before_registration": False,
        "label_values_accessed_before_registration": False,
        "prior_input_sha256": hashes,
        "expected_gt_sha256": GT_SHA,
        "expected_annotations_sha256": ANNOTATION_SHA,
        "canonical_metadata_sha256": CANONICAL_SHA,
        "audit_implementation_sha256": sha256_file(Path(__file__)),
        "candidates": candidates,
        "coordinate_rule": "project [s-base-origin,e-base-origin+int(inclusive)) into [0,16Ti); never fit offsets",
        "suffix_rule": "origin=N-L using verified full decoded N and canonical L=16Ti",
        "raw_timing_rule": {
            "fps": 24,
            "ffprobe_counts_equal_container_and_opencv": True,
            "full_frame_pts_required": True,
            "pts_field_rule": "frame.pts or historical frame.pkt_pts, one complete field for all frames; never DTS; match best_effort_timestamp",
            "pts_strictly_increasing": True,
            "step_and_cumulative_error_max": "zero when time_base represents 1/24 exactly; otherwise one stream time_base tick",
            "time_base_must_be_finer_than_frame_period": True,
            "opencv_fps_absolute_tolerance": 1e-6,
            "no_padding_or_resampling": True,
        },
        "seal_rule": "prefix b0-half_open-origin0 exact for all; N>=L; all asset/timing audits pass; every other exact candidate must map to the same raw coordinates",
        "observational_equivalence_rule": "report exact-mask aliases separately from coordinate aliases; different coordinates remain blocked for review",
        "cache_rule": "new attempt; full raw CRC/SHA rechecked; previous probe reused only with identical code/runtime/rule and verified source artifact hashes; reuse is explicitly counted",
        "runtime_request": {
            "ffprobe": request.ffprobe,
            "decode_backend": "OpenCV VideoCapture sequential read",
        },
        "output_privacy": "aggregate stdout/results; private machine manifests and probe artifacts only",
    }
    _write_once(root / "preregistration.json", document)
    _write_once(root / "request.json", request.__dict__)
    return root, document


def parse_annotations(
    text: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[tuple[int, int]]]:
    identities = {row["video_id"] for row in rows}
    _require(len(identities) == len(rows), "CANONICAL_DUPLICATE_IDENTITY")
    parsed = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        name = PurePosixPath(tokens[0].replace("\\", "/")).name
        name = name[:-4] if name.endswith(".mp4") else name
        _require(name in identities and name not in parsed, "ANNOTATION_IDENTITY")
        _require(len(tokens[1:]) % 2 == 0, "ANNOTATION_ENDPOINT_COUNT")
        spans = []
        for start, end in zip(tokens[1::2], tokens[2::2], strict=True):
            _require(
                start.lstrip("-").isdigit() and end.lstrip("-").isdigit(),
                "ANNOTATION_ENDPOINT_TYPE",
            )
            left, right = int(start), int(end)
            if (left, right) == (-1, -1):
                continue
            _require(0 <= left <= right, "ANNOTATION_ENDPOINT_ORDER")
            spans.append((left, right))
        normal = "_label_A" in name
        _require((not spans) if normal else bool(spans), "ANNOTATION_WEAK_LABEL")
        parsed[name] = spans
    for name in identities - parsed.keys():
        _require("_label_A" in name, "MISSING_POSITIVE_ANNOTATION")
        parsed[name] = []
    return parsed


def project_spans(
    spans: Sequence[tuple[int, int]], length: int, shift: int, inclusive: bool
) -> np.ndarray:
    result = np.zeros(length, dtype=bool)
    for start, end in spans:
        left, right = max(0, start - shift), min(length, end - shift + int(inclusive))
        if left < right:
            result[left:right] = True
    return result


def compare_candidates(
    rows: Sequence[Mapping[str, Any]],
    raw_counts: Sequence[int],
    annotations: Mapping[str, Sequence[tuple[int, int]]],
    truth: np.ndarray,
) -> dict[str, Any]:
    _require(
        len(rows) == len(raw_counts)
        and all(n >= r["t"] * 16 for r, n in zip(rows, raw_counts, strict=True)),
        "RAW_LENGTH_SHORTER_THAN_CANONICAL",
    )
    _require(
        truth.shape == (sum(r["t"] * 16 for r in rows),) and np.all((truth == 0) | (truth == 1)),
        "GT_BINARY_SHAPE",
    )
    results, vectors, coordinate_signatures = [], [], []
    for candidate in fixed_candidates():
        vector = np.zeros(truth.size, dtype=bool)
        exact = coordinate_equal = 0
        shifts = []
        for row, count in zip(rows, raw_counts, strict=True):
            length = row["t"] * 16
            origin = count - length if candidate["family"] == "suffix" else candidate["origin"]
            shift = origin + candidate["index_base"]
            shifts.append(shift)
            projected = project_spans(
                annotations[row["video_id"]], length, shift, candidate["end_policy"] == "inclusive"
            )
            start, stop = row["gt_start"], row["gt_end_exclusive"]
            vector[start:stop] = projected
            exact += int(np.array_equal(projected, truth[start:stop]))
            # Same labels alone do not make distinct raw-frame supports identical.
            coordinate_equal += int(shift == 0 and candidate["end_policy"] == "half_open")
        results.append(
            {
                **candidate,
                "exact_video_count": exact,
                "mismatching_video_count": len(rows) - exact,
                "xor_position_count": int(np.count_nonzero(vector != truth)),
                "coordinate_equivalent_to_prefix_videos": coordinate_equal,
            }
        )
        vectors.append(vector)
        coordinate_signatures.append((candidate["end_policy"], tuple(shifts)))
    prefix_index = next(
        i for i, r in enumerate(results) if r["candidate_id"] == "b0-half_open-origin0"
    )
    prefix = vectors[prefix_index]
    equivalent_groups = []
    ungrouped = set(range(len(results)))
    for i in range(len(results)):
        if i not in ungrouped:
            continue
        group = [j for j in sorted(ungrouped) if np.array_equal(vectors[i], vectors[j])]
        equivalent_groups.append([results[j]["candidate_id"] for j in group])
        ungrouped.difference_update(group)
    for i, result in enumerate(results):
        result["projection_equivalent_to_prefix"] = bool(np.array_equal(vectors[i], prefix))
    coordinate_groups = {}
    for signature, candidate in zip(coordinate_signatures, results, strict=True):
        coordinate_groups.setdefault(signature, []).append(candidate["candidate_id"])
    exact = [r for r in results if r["exact_video_count"] == len(rows)]
    ambiguous = [
        r["candidate_id"] for r in exact if r["coordinate_equivalent_to_prefix_videos"] != len(rows)
    ]
    gaps = [n - r["t"] * 16 for r, n in zip(rows, raw_counts, strict=True)]
    return {
        "candidate_results": results,
        "observed_projection_equivalence_classes": equivalent_groups,
        "coordinate_transform_equivalence_classes": list(coordinate_groups.values()),
        "exact_candidates": [r["candidate_id"] for r in exact],
        "exact_but_distinct_coordinate_candidates": ambiguous,
        "prefix_exact_all": results[prefix_index]["exact_video_count"] == len(rows),
        "suffix_resolved_without_coordinate_ambiguity": not ambiguous,
        "raw_minus_canonical_length": {
            "min": min(gaps),
            "max": max(gaps),
            "zero_videos": gaps.count(0),
            "positive_videos": sum(x > 0 for x in gaps),
            "histogram": dict(
                sorted(Counter(str(x) for x in gaps).items(), key=lambda x: int(x[0]))
            ),
        },
    }


def validate_timing(
    ffprobe_document: Mapping[str, Any], opencv: Mapping[str, Any]
) -> dict[str, Any]:
    streams = ffprobe_document.get("streams", [])
    _require(len(streams) == 1, "VIDEO_STREAM_COUNT")
    stream = streams[0]
    try:
        frames = int(stream["nb_frames"])
        fps, average = Fraction(stream["r_frame_rate"]), Fraction(stream["avg_frame_rate"])
        tick = Fraction(stream["time_base"])
        observations = ffprobe_document["frames"]
        pts_field = "pts" if all("pts" in row for row in observations) else "pkt_pts"
        pts = [int(row[pts_field]) for row in observations]
        best_effort = [int(row["best_effort_timestamp"]) for row in ffprobe_document["frames"]]
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        raise XDAlignmentError("MISSING_CONTAINER_OR_FULL_PTS") from None
    _require(
        frames > 0 and len(pts) == frames and len(best_effort) == frames,
        "CONTAINER_PTS_COUNT_MISMATCH",
    )
    _require(fps == average == 24 and 0 < tick < Fraction(1, 24), "FPS_OR_TIME_BASE")
    _require(pts == best_effort, "PTS_BEST_EFFORT_DISAGREE")
    _require(
        all(
            int(row.get("pkt_pts", value)) == value
            for row, value in zip(observations, pts, strict=True)
        ),
        "PTS_FIELDS_DISAGREE",
    )
    _require(
        all(right > left for left, right in zip(pts, pts[1:], strict=False)),
        "PTS_NOT_STRICTLY_INCREASING",
    )
    step_error = max(
        (abs((b - a) * tick - Fraction(1, 24)) for a, b in zip(pts, pts[1:], strict=False)),
        default=Fraction(0),
    )
    drift = max(abs((p - pts[0]) * tick - Fraction(i, 24)) for i, p in enumerate(pts))
    period_ticks = Fraction(1, 24) / tick
    tolerance = Fraction(0) if period_ticks.denominator == 1 else tick
    _require(step_error <= tolerance and drift <= tolerance, "PTS_CADENCE_OR_DRIFT")
    _require(
        opencv["decoded_frames"] == frames and opencv["container_frames"] == frames,
        "OPENCV_CONTAINER_DECODE_COUNT",
    )
    _require(math.isfinite(opencv["fps"]) and abs(opencv["fps"] - 24.0) <= 1e-6, "OPENCV_FPS")
    return {
        "num_frames": frames,
        "fps": 24.0,
        "pts_verified": True,
        "time_base": str(tick),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "pts_count": len(pts),
        "pts_field": pts_field,
        "max_step_error_seconds": float(step_error),
        "max_cumulative_error_seconds": float(drift),
        "pts_tolerance_seconds": float(tolerance),
        "opencv_decoded_frames": opencv["decoded_frames"],
        "opencv_container_frames": opencv["container_frames"],
    }


def _runtime(ffprobe: str) -> dict[str, Any]:
    import cv2

    import vadbench

    _require(hasattr(cv2, "CAP_PROP_N_THREADS"), "OPENCV_THREAD_LIMIT_UNSUPPORTED")

    executable = shutil.which(ffprobe)
    _require(executable is not None, "FFPROBE_UNAVAILABLE")
    version = subprocess.run(
        [executable, "-version"], capture_output=True, check=True, timeout=30
    ).stdout
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "vadbench_file": vadbench.__file__,
        "opencv_version": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        "ffprobe": str(Path(executable).resolve()),
        "ffprobe_version_sha256": hashlib.sha256(version).hexdigest(),
        "ffprobe_binary_sha256": sha256_file(Path(executable)),
        "numpy_version": np.__version__,
        "device": "cpu",
        "gpu_used": False,
        "ffprobe_decoder_threads": 1,
        "opencv_decoder_threads": 2,
    }


def probe_raw_video(path: Path, directory: Path, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Decode every frame twice, once for ffprobe PTS and once with evaluator OpenCV."""
    import cv2

    directory.mkdir(parents=True, exist_ok=False)
    command = [
        runtime["ffprobe"],
        "-v",
        "error",
        "-threads",
        "1",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate,avg_frame_rate,time_base,start_time,duration:frame=pts,pkt_pts,best_effort_timestamp",
        "-show_frames",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, timeout=3600)
    (directory / "ffprobe.json").write_bytes(process.stdout)
    (directory / "ffprobe.stderr.txt").write_bytes(process.stderr)
    _require(process.returncode == 0 and not process.stderr.strip(), "FFPROBE_DECODE_ERROR")
    try:
        document = json.loads(process.stdout)
    except json.JSONDecodeError:
        raise XDAlignmentError("FFPROBE_INVALID_JSON") from None
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(
        str(path),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_N_THREADS, 2, cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE],
    )
    try:
        _require(capture.isOpened(), "OPENCV_OPEN_FAILED")
        header_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        _require(
            math.isfinite(header_count) and header_count.is_integer(), "OPENCV_FRAME_COUNT_INVALID"
        )
        opencv = {
            "container_frames": int(header_count),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "decoded_frames": 0,
        }
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _require(frame is not None and frame.size > 0, "OPENCV_EMPTY_FRAME")
            opencv["decoded_frames"] += 1
    finally:
        capture.release()
    _write_once(directory / "opencv.json", opencv)
    return validate_timing(document, opencv)


def _raw_digest(path: Path) -> tuple[str, str, int]:
    digest, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return digest.hexdigest(), f"{crc & 0xFFFFFFFF:08x}", size


def _read_volume_receipt(
    request: RawAlignmentRequest, root: Path, index: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    path = Path(request.volume_receipt)
    while True:
        receipt = _json(path) if path.is_file() else None
        if (
            receipt is not None
            and receipt.get("state") == "ready"
            and receipt.get("data_ready") is True
        ):
            _require(
                receipt.get("volume") == "test_videos"
                and receipt.get("member_count") == VIDEO_COUNT,
                "RAW_VOLUME_IDENTITY_OR_COUNT",
            )
            _require(
                receipt.get("crc_validation") == "streamed_zipfile_crc_plus_local_crc32",
                "RAW_VOLUME_CRC_PROOF",
            )
            _require(
                {r["central_directory_sha256"] for r in index}
                == {receipt.get("central_directory_sha256")},
                "RAW_VOLUME_CENTRAL_DIRECTORY",
            )
            _require(
                receipt.get("uncompressed_bytes") == sum(r["uncompressed_size"] for r in index),
                "RAW_VOLUME_BYTE_COUNT",
            )
            _require(
                isinstance(receipt.get("zip_sha256"), str)
                and len(receipt["zip_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in receipt["zip_sha256"]),
                "RAW_VOLUME_ZIP_SHA_MISSING",
            )
            return receipt
        atomic_write_json(
            root / "status.json",
            {"status": "waiting_for_complete_raw_volume", "model_scores_read": False},
        )
        if not request.wait_for_ready:
            return None
        time.sleep(request.poll_seconds)


def _input_index(
    request: RawAlignmentRequest, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(request.test_index).read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    _require(
        len(records) == VIDEO_COUNT and len({r["video_id"] for r in records}) == VIDEO_COUNT,
        "TEST_INDEX_CARDINALITY",
    )
    by_name = {PurePosixPath(r["archive_member"]).stem: r for r in records}
    _require(len(by_name) == VIDEO_COUNT, "TEST_INDEX_AMBIGUOUS_BASENAME")
    _require(set(by_name) == {r["video_id"] for r in rows}, "RAW_FEATURE_IDENTITY_JOIN")
    root = Path(request.raw_root).resolve()
    ordered = []
    for row in rows:
        raw = by_name[row["video_id"]]
        _require(
            raw["official_split"] == "test" and raw["archive_volume"] == "test_videos",
            "TEST_INDEX_SPLIT",
        )
        _require(raw["path"] == "test_videos/" + raw["archive_member"], "TEST_INDEX_ARCHIVE_PATH")
        _require((root / raw["path"]).resolve().is_relative_to(root), "RAW_PATH_ESCAPE")
        ordered.append(raw)
    return ordered


def run_raw_alignment_audit(
    request: RawAlignmentRequest,
    *,
    probe: Callable[..., Mapping[str, Any]] = probe_raw_video,
    runtime_factory: Callable[..., Mapping[str, Any]] = _runtime,
) -> dict[str, Any]:
    """Create an isolated attempt; only complete computed evidence can publish a seal."""
    _require(request.poll_seconds > 0, "POLL_INTERVAL")
    root, registration = _preregister(request)
    lease = None
    try:
        metadata = _json(Path(request.canonical_metadata))
        rows, _ = _verified_rows(metadata, CANONICAL_SHA)
        index = _input_index(request, rows)
        volume = _read_volume_receipt(request, root, index)
        if volume is None:
            result = {
                "status": "blocked_raw_volume_not_ready",
                "run_dir": str(root),
                "model_scores_read": False,
                "raw_videos_read": False,
                "ground_truth_values_read": False,
            }
            _write_once(root / "result.json", result)
            atomic_write_json(root / "status.json", result)
            return result
        if request.cpu_lock_path is not None:
            import fcntl

            lock_path = Path(request.cpu_lock_path).resolve()
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lock_path.open("a+")
            atomic_write_json(
                root / "status.json",
                {"status": "waiting_for_cpu_lease", "model_scores_read": False},
            )
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            _write_once(
                root / "cpu-lease.json",
                {
                    "path": str(lock_path),
                    "pid": os.getpid(),
                    "acquired_after_raw_volume_ready": True,
                    "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
        runtime = dict(runtime_factory(request.ffprobe))
        _write_once(root / "runtime.json", runtime)
        inputs = {
            **registration["prior_input_sha256"],
            str(Path(request.volume_receipt).resolve()): sha256_file(Path(request.volume_receipt)),
        }
        prior = Path(request.prior_alignment_root).resolve()
        gt_path = _bound(prior / "sealed-inputs" / "gt.npy", GT_SHA, "CANONICAL_GT")
        annotation_path = _bound(
            prior / "sealed-inputs" / "annotations.txt", ANNOTATION_SHA, "ANNOTATIONS"
        )
        inputs.update({str(gt_path): GT_SHA, str(annotation_path): ANNOTATION_SHA})
        previous_index = {}
        if request.resume_source_run:
            previous_index_path = Path(request.resume_source_run).resolve() / "probe-index.json"
            previous_index = _json(previous_index_path)
            inputs[str(previous_index_path)] = sha256_file(previous_index_path)
        _write_once(
            root / "input-lock.json",
            {
                "sha256": inputs,
                "volume_receipt": volume,
                "preregistration_sha256": sha256_file(root / "preregistration.json"),
            },
        )
        cache_identity = _canonical(
            {
                "implementation": registration["audit_implementation_sha256"],
                "runtime": runtime,
                "rule": registration["raw_timing_rule"],
            }
        )
        private_rows, failures, reused = [], [], 0
        probe_index = {}
        atomic_write_json(root / "probe-index.json", probe_index)
        for i, (raw, feature) in enumerate(zip(index, rows, strict=True)):
            directory = root / "private" / "probes" / f"{i:04d}"
            path = Path(request.raw_root).resolve() / raw["path"]
            try:
                _require(path.is_file(), "RAW_VIDEO_MISSING")
                before = path.stat()
                digest, crc, size = _raw_digest(path)
                _require(
                    crc == raw["provider_crc32"] and size == raw["uncompressed_size"],
                    "RAW_VIDEO_CRC_OR_SIZE",
                )
                timing = None
                origin = {"kind": "computed_in_this_attempt"}
                if request.resume_source_run:
                    previous = (
                        Path(request.resume_source_run).resolve()
                        / "private"
                        / "probes"
                        / f"{i:04d}"
                    )
                    receipt_path = previous / "receipt.json"
                    if f"{i:04d}" in previous_index:
                        _bound(receipt_path, previous_index[f"{i:04d}"], "CACHED_PROBE_RECEIPT")
                        cached = _json(receipt_path)
                        if (
                            cached.get("cache_identity") == cache_identity
                            and cached.get("raw_sha256") == digest
                            and cached.get("video_id") == raw["video_id"]
                            and cached.get("status") == "passed"
                        ):
                            _require(
                                set(cached["artifact_sha256"])
                                == {"ffprobe.json", "ffprobe.stderr.txt", "opencv.json"},
                                "CACHED_PROBE_ARTIFACT_SET",
                            )
                            for name, expected in cached["artifact_sha256"].items():
                                _bound(previous / name, expected, "CACHED_PROBE_ARTIFACT")
                            # Recompute timing verdict from old observations; do not trust a passed flag.
                            timing = validate_timing(
                                _json(previous / "ffprobe.json"), _json(previous / "opencv.json")
                            )
                            directory.mkdir(parents=True, exist_ok=False)
                            for name in cached["artifact_sha256"]:
                                shutil.copyfile(previous / name, directory / name)
                            origin = {
                                "kind": "reused_verified_probe",
                                "source_receipt": str(receipt_path),
                                "source_receipt_sha256": sha256_file(receipt_path),
                                "raw_crc_sha_rechecked": True,
                            }
                            reused += 1
                if timing is None:
                    probe(path, directory, runtime)
                    timing = validate_timing(
                        _json(directory / "ffprobe.json"), _json(directory / "opencv.json")
                    )
                _require(
                    not (directory / "ffprobe.stderr.txt").read_bytes().strip(),
                    "FFPROBE_DECODE_ERROR",
                )
                after = path.stat()
                _require(
                    (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    "RAW_CHANGED_DURING_PROBE",
                )
                _require(
                    timing["num_frames"] >= feature["t"] * 16, "RAW_LENGTH_SHORTER_THAN_CANONICAL"
                )
                artifact_hashes = {
                    name: sha256_file(directory / name)
                    for name in ("ffprobe.json", "ffprobe.stderr.txt", "opencv.json")
                }
                record = {
                    "video_index": i,
                    "video_id": raw["video_id"],
                    "feature_name": feature["feature_name"],
                    "path": raw["path"],
                    "raw_sha256": digest,
                    "raw_bytes": size,
                    "verified_crc32": crc,
                    **timing,
                    "cache_identity": cache_identity,
                    "probe_origin": origin,
                    "artifact_sha256": artifact_hashes,
                    "status": "passed",
                }
                _write_once(directory / "receipt.json", record)
                probe_index[f"{i:04d}"] = sha256_file(directory / "receipt.json")
                atomic_write_json(root / "probe-index.json", probe_index)
                private_rows.append(record)
            except (XDAlignmentError, OSError, subprocess.SubprocessError) as exc:
                code = str(exc) if isinstance(exc, XDAlignmentError) else type(exc).__name__
                failures.append({"video_index": i, "code": code})
                _write_once(root / "private" / "failures" / f"{i:04d}.json", {"code": code})
            atomic_write_json(
                root / "status.json",
                {
                    "status": "probing_raw_videos",
                    "checked": i + 1,
                    "passed": len(private_rows),
                    "failed": len(failures),
                    "reused_probe_videos": reused,
                },
            )
        if failures:
            result = {
                "status": "blocked_raw_coordinate_failures",
                "run_dir": str(root),
                "video_count": VIDEO_COUNT,
                "passed": len(private_rows),
                "failed": len(failures),
                "failure_counts": dict(Counter(r["code"] for r in failures)),
                "reused_probe_videos": reused,
                "model_scores_read": False,
                "ground_truth_values_read": False,
            }
            _write_once(root / "result.json", result)
            atomic_write_json(root / "status.json", result)
            return result
        # Both the new candidate document and completed raw timing evidence now exist.
        truth = np.load(gt_path, allow_pickle=False)
        annotations = parse_annotations(annotation_path.read_text(encoding="utf-8-sig"), rows)
        comparison = compare_candidates(
            rows, [r["num_frames"] for r in private_rows], annotations, truth
        )
        for path, expected in {
            **inputs,
            str(gt_path): GT_SHA,
            str(annotation_path): ANNOTATION_SHA,
        }.items():
            _bound(Path(path), expected, "AUDIT_INPUT_CHANGED")
        result = {
            "schema_version": 1,
            "run_dir": str(root),
            "video_count": VIDEO_COUNT,
            "model_scores_read": False,
            "ground_truth_values_exported": False,
            "test_ids_or_events_exported_in_summary": False,
            "reused_probe_videos": reused,
            "newly_probed_videos": VIDEO_COUNT - reused,
            "raw_timing_failures": 0,
            "fps_histogram": {"24": VIDEO_COUNT},
            "pts_quantized_time_base_videos": sum(
                r["pts_tolerance_seconds"] > 0 for r in private_rows
            ),
            "pts_max_cumulative_error_seconds": max(
                r["max_cumulative_error_seconds"] for r in private_rows
            ),
            **comparison,
        }
        accepted = (
            comparison["prefix_exact_all"]
            and comparison["suffix_resolved_without_coordinate_ambiguity"]
        )
        result["status"] = "sealed" if accepted else "blocked_alignment_requires_review"
        _write_once(root / "candidate-results.json", result)
        if accepted:
            seal = _publish_seal(root, request, private_rows, index, result, inputs)
            result["seal"] = {"path": str(seal), "sha256": sha256_file(seal)}
        _write_once(root / "result.json", result)
        atomic_write_json(
            root / "status.json", {"status": result["status"], "model_scores_read": False}
        )
        return result
    except Exception as exc:
        atomic_write_json(
            root / "status.json",
            {
                "status": "failed",
                "error_code": str(exc) if isinstance(exc, XDAlignmentError) else type(exc).__name__,
                "model_scores_read": False,
            },
        )
        raise
    finally:
        if lease is not None:
            import fcntl

            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()


def _publish_seal(
    root: Path,
    request: RawAlignmentRequest,
    records: list[dict[str, Any]],
    index: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> Path:
    manifest = []
    for record, raw in zip(records, index, strict=True):
        normal = parse_xd_labels(raw["archive_member"]) == ("A",)
        manifest.append(
            VideoManifestRecord(
                video_id=record["video_id"],
                path=record["path"],
                split="test",
                category="Normal" if normal else "Anomaly",
                is_anomaly=not normal,
                num_frames=record["num_frames"],
                fps=record["fps"],
                duration_seconds=record["num_frames"] / record["fps"],
                metadata={
                    "archive_member": raw["archive_member"],
                    "archive_volume": "test_videos",
                    "rawmember_sha256": record["raw_sha256"],
                    "verified_crc32": record["verified_crc32"],
                },
            )
        )
        record["suffix_audit_passed"] = True
    manifest_path = write_manifest_jsonl(manifest, root / "private" / "test.jsonl")
    common = {
        "schema_version": 1,
        "status": "passed",
        "model_scores_read": False,
        "video_count": VIDEO_COUNT,
        "test_manifest_sha256": sha256_file(manifest_path),
        "canonical_metadata_sha256": CANONICAL_SHA,
        "source_sha256": {**input_hashes, "gt.npy": GT_SHA, "annotations.txt": ANNOTATION_SHA},
        "candidate_results_sha256": sha256_file(root / "candidate-results.json"),
    }
    audits = {}
    timing_receipts = {
        f"private/probes/{i:04d}/receipt.json": sha256_file(
            root / "private" / "probes" / f"{i:04d}" / "receipt.json"
        )
        for i in range(VIDEO_COUNT)
    }
    payloads = {
        "annotation_alignment": {
            **common,
            "computed_prefix_exact_videos": VIDEO_COUNT,
            "computed_prefix_xor_positions": 0,
            "mapping": SEAL_MAPPING,
        },
        "raw_timing": {
            **common,
            "computed_full_pts_and_decode_passed": VIDEO_COUNT,
            "probe_receipt_sha256": timing_receipts,
            "runtime_sha256": sha256_file(root / "runtime.json"),
        },
        "prespecified_suffix": {
            **common,
            "exact_candidates": comparison["exact_candidates"],
            "exact_but_distinct_coordinate_candidates": comparison[
                "exact_but_distinct_coordinate_candidates"
            ],
            "observed_projection_equivalence_classes": comparison[
                "observed_projection_equivalence_classes"
            ],
            "preregistration": {
                "path": "../preregistration.json",
                "sha256": sha256_file(root / "preregistration.json"),
            },
        },
    }
    for name, document in payloads.items():
        path = root / "audits" / f"{name}.json"
        _write_once(path, document)
        audits[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
    seal = root / "raw-coordinate-receipt.json"
    _write_once(
        seal,
        {
            "schema_version": 1,
            "status": "sealed",
            "protocol": SEAL_PROTOCOL,
            "mapping": SEAL_MAPPING,
            "canonical_metadata_sha256": CANONICAL_SHA,
            "gt_file_sha256": GT_SHA,
            "test_manifest_sha256": sha256_file(manifest_path),
            "records": records,
            "audits": audits,
        },
    )
    return seal
