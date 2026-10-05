from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.data.xd_materialization import (
    ROLE_COUNTS,
    ROLE_LABELS,
    XDMaterializationError,
    audit_xd_ready_manifest,
    freeze_xd_canary_rule,
    materialize_xd_heads,
    validate_xd_head_data_contract,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    return path


def _record(video_id: str, role: str, is_anomaly: bool, *, ready: bool) -> VideoManifestRecord:
    payload = b"raw"
    crc = f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
    metadata: dict[str, Any] = {
        "source_group": "group-" + video_id,
        "provider_crc32": crc,
        "uncompressed_size": 3,
        "rawmember_sha256": hashlib.sha256(payload).hexdigest(),
        "verified_crc32": crc,
    }
    if ready:
        metadata.update({"actual_num_frames": 96, "actual_fps": 24.0})
    return VideoManifestRecord(
        video_id=video_id,
        path=f"videos/{video_id}.mp4",
        split="train",
        category="Anomaly" if is_anomaly else "Normal",
        is_anomaly=is_anomaly,
        annotations=(SupervisionAnnotation(scope="video", is_anomaly=is_anomaly),),
        num_frames=96 if ready else None,
        fps=24.0 if ready else None,
        duration_seconds=4.0 if ready else None,
        metadata=metadata,
    )


def _issue_ready_audit(
    inputs: dict[str, Any],
    records: list[VideoManifestRecord],
    name: str,
    *,
    frame_counts: dict[str, int] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    ready_path = write_manifest_jsonl(tuple(records), inputs["root"] / f"ready-{name}.jsonl")
    receipt_path = _write_json(
        inputs["root"] / f"ready-{name}-receipt.json",
        {
            "role_lock_sha256": _file_sha256(inputs["lock_path"]),
            "roles_never_reassigned": True,
            "ready_crc_verified": len(records),
        },
    )
    counts = frame_counts or {}
    import vadbench.data.xd_materialization as xd_materialization

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            xd_materialization,
            "probe_video",
            lambda path: SimpleNamespace(num_frames=counts.get(Path(path).stem, 96), fps=24.0),
        )
        audit = audit_xd_ready_manifest(
            plan_path=inputs["plan_path"],
            scope_path=inputs["scope_path"],
            original_lock_path=inputs["lock_path"],
            full_manifest_path=inputs["full_path"],
            ready_manifest_path=ready_path,
            upstream_receipt_path=receipt_path,
            dataset_root=inputs["dataset_root"],
            output_dir=inputs["root"] / f"audit-{name}",
        )
    return ready_path, receipt_path, audit


@pytest.fixture(scope="module")
def frozen_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("xd-materialization")
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    roles: dict[str, list[str]] = {role: [] for role in ROLE_COUNTS}
    full: list[VideoManifestRecord] = []
    lock_partitions: dict[str, str] = {}
    lock_groups: dict[str, str] = {}
    ready: list[VideoManifestRecord] = []
    for role in ("fit", "select", "confirm"):
        for label, count in ROLE_LABELS[role].items():
            for index in range(count):
                video_id = f"{role}-{label}-{index:04d}"
                record = _record(video_id, role, label == "1", ready=False)
                roles[role].append(video_id)
                full.append(record)
                lock_partitions[video_id] = role
                lock_groups[video_id] = record.metadata["source_group"]
                ready_record = _record(video_id, role, label == "1", ready=True)
                ready.append(ready_record)
                target = dataset_root / ready_record.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"raw")

    quarantine_ids = []
    for index in range(4):
        video_id = f"excluded-corrupt-{index}"
        quarantine_ids.append(video_id)
        record = _record(video_id, "fit", False, ready=False)
        full.append(record)
        lock_partitions[video_id] = "fit"
        lock_groups[video_id] = record.metadata["source_group"]
    for index in range(2352):
        video_id = f"excluded-overlap-{index:04d}"
        record = _record(video_id, "fit", False, ready=False)
        full.append(record)
        lock_partitions[video_id] = "fit"
        lock_groups[video_id] = record.metadata["source_group"]
    assert len(full) == 3954
    assert sum(map(len, roles.values())) == 1598

    full_path = write_manifest_jsonl(tuple(full), root / "full.jsonl")
    lock_path = _write_json(
        root / "role-lock.json",
        {"seed": 202709, "partitions": lock_partitions, "source_groups": lock_groups},
    )
    quarantine_path = _write_json(
        root / "quarantine.json", {"records": [{"video_id": value} for value in quarantine_ids]}
    )
    plan = {
        "derived_train_total": 1598,
        "role_reassignment": False,
        "membership_depends_on_availability": False,
        "role_video_ids": roles,
        "inputs": {
            "full_train_manifest_file_sha256": _file_sha256(full_path),
            "original_role_lock_file_sha256": _file_sha256(lock_path),
            "original_role_lock_canonical_sha256": _canonical_sha256(
                json.loads(lock_path.read_text(encoding="utf-8"))
            ),
            "source_corrupt_quarantine_file_sha256": _file_sha256(quarantine_path),
            "source_corrupt_membership_sha256": _canonical_sha256(sorted(quarantine_ids)),
        },
    }
    plan_path = _write_json(root / "plan.json", plan)
    scope = {
        "secondary_encoders": ["videomaev2", "timesformer"],
        "additional_reducer_calibration_on_xd": False,
        "derived_role_plan_file_sha256": _file_sha256(plan_path),
        "derived_role_plan_canonical_sha256": _canonical_sha256(plan),
        "original_role_lock_file_sha256": _file_sha256(lock_path),
        "original_role_lock_seed": 202709,
        "roles": {
            role: {
                "videos": ROLE_COUNTS[role],
                "normal": ROLE_LABELS[role]["0"],
                "positive": ROLE_LABELS[role]["1"],
            }
            for role in ROLE_COUNTS
        },
    }
    scope_path = _write_json(root / "scope.json", scope)
    freeze_path = _write_json(
        root / "freeze.json",
        {
            "xd_evaluation": {"additional_calibration_on_xd": False},
            "sampling": {
                "short_policy": "stride1_if_needed",
                "native_clip_frames": {"videomaev2": 16, "timesformer": 8},
            },
        },
    )
    ready_path = write_manifest_jsonl(tuple(ready), root / "ready.jsonl")
    ready_receipt_path = _write_json(
        root / "ready-receipt.json",
        {
            "role_lock_sha256": _file_sha256(lock_path),
            "roles_never_reassigned": True,
            "ready_crc_verified": len(ready),
        },
    )
    canary_path = root / "canary-rule.json"
    freeze_xd_canary_rule(plan_path, scope_path, canary_path)
    inputs = {
        "root": root,
        "dataset_root": dataset_root,
        "roles": roles,
        "full_path": full_path,
        "lock_path": lock_path,
        "quarantine_path": quarantine_path,
        "plan_path": plan_path,
        "scope_path": scope_path,
        "freeze_path": freeze_path,
        "ready_path": ready_path,
        "ready_receipt_path": ready_receipt_path,
        "canary_path": canary_path,
        "ready": ready,
    }
    _, _, audit = _issue_ready_audit(inputs, ready, "full")
    inputs["ready_audit_path"] = Path(audit["ready_audit_path"])
    inputs["ready_audit_sha256"] = audit["ready_audit_sha256"]
    return inputs


def _kwargs(
    inputs: dict[str, Any],
    output_dir: Path,
    *,
    ready_path: Path | None = None,
    ready_receipt_path: Path | None = None,
    canary_path: Path | None = None,
    ready_audit_path: Path | None = None,
    ready_audit_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "plan_path": inputs["plan_path"],
        "original_lock_path": inputs["lock_path"],
        "full_manifest_path": inputs["full_path"],
        "ready_manifest_path": ready_path or inputs["ready_path"],
        "ready_receipt_path": ready_receipt_path or inputs["ready_receipt_path"],
        "quarantine_path": inputs["quarantine_path"],
        "scope_path": inputs["scope_path"],
        "method_freeze_path": inputs["freeze_path"],
        "canary_rule_path": canary_path or inputs["canary_path"],
        "dataset_root": inputs["dataset_root"],
        "output_dir": output_dir,
        "ready_audit_path": ready_audit_path or inputs["ready_audit_path"],
        "ready_audit_sha256": ready_audit_sha256 or inputs["ready_audit_sha256"],
    }


def test_missing_metadata_and_raw_size_change_block_frozen_targets(
    frozen_inputs: dict[str, Any],
) -> None:
    missing = [
        row for row in frozen_inputs["ready"] if row.video_id != frozen_inputs["roles"]["fit"][0]
    ]
    missing_path, missing_receipt, missing_audit = _issue_ready_audit(
        frozen_inputs, missing, "missing"
    )
    readiness = materialize_xd_heads(
        **_kwargs(
            frozen_inputs,
            frozen_inputs["root"] / "blocked-metadata",
            ready_path=missing_path,
            ready_receipt_path=missing_receipt,
            ready_audit_path=Path(missing_audit["ready_audit_path"]),
            ready_audit_sha256=missing_audit["ready_audit_sha256"],
        )
    )
    assert readiness["head_training_ready"] is False
    assert readiness["members_reassigned_or_replaced"] is False

    target = frozen_inputs["dataset_root"] / frozen_inputs["ready"][1].path
    original_stat = target.stat()
    target.write_bytes(b"wrong-size")
    size_readiness = materialize_xd_heads(
        **_kwargs(frozen_inputs, frozen_inputs["root"] / "blocked-size")
    )
    assert size_readiness["head_training_ready"] is False
    target.write_bytes(b"raw")
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))


def test_complete_targets_materialize_source_controller_and_contract(
    frozen_inputs: dict[str, Any],
) -> None:
    output = frozen_inputs["root"] / "ready"
    readiness = materialize_xd_heads(**_kwargs(frozen_inputs, output))
    assert readiness["status"] == "ready"
    assert readiness["head_training_ready"] is True
    contract_payload = json.loads((output / "head-data-contract.json").read_text(encoding="utf-8"))
    assert contract_payload["schema_version"] == 2
    assert contract_payload["ready_evidence"]["kind"] == "independent_member_content_audit"
    for key in ("audit", "ready_manifest", "upstream_receipt", "member_proofs"):
        evidence = contract_payload["ready_evidence"][key]
        assert (output / evidence["path"]).is_file()
    validated = validate_xd_head_data_contract(
        output / "head-data-contract.json",
        scope_path=frozen_inputs["scope_path"],
        derived_plan_path=frozen_inputs["plan_path"],
        original_lock_path=frozen_inputs["lock_path"],
        method_freeze_path=frozen_inputs["freeze_path"],
    )
    for role, count in ROLE_COUNTS.items():
        source = validated["roles"][role]["source_records"]
        controller = validated["roles"][role]["controller_records"]
        assert len(source) == len(controller) == count
        assert {row.video_id for row in source} == set(frozen_inputs["roles"][role])
        assert {row.split.value for row in source} == {"train"}
        assert {row.split.value for row in controller} == ({"train"} if role == "fit" else {"val"})
    with pytest.raises(FileExistsError):
        materialize_xd_heads(**_kwargs(frozen_inputs, output))


@pytest.mark.parametrize("mutation", ["path", "label", "temporal", "raw_sha", "raw_sha_valid_zero"])
def test_ready_identity_and_weak_supervision_tampering_is_rejected(
    frozen_inputs: dict[str, Any], mutation: str
) -> None:
    records = list(frozen_inputs["ready"])
    original = records[0]
    if mutation == "path":
        records[0] = replace(original, path="other.mp4")
    elif mutation == "label":
        changed = not original.is_anomaly
        records[0] = replace(
            original,
            is_anomaly=changed,
            annotations=(SupervisionAnnotation(scope="video", is_anomaly=changed),),
        )
    elif mutation == "temporal":
        records[0] = replace(
            original,
            annotations=(
                SupervisionAnnotation(
                    scope="frame", is_anomaly=True, span=TemporalSpan(0, 1, "frame")
                ),
            ),
        )
    elif mutation == "raw_sha":
        records[0] = replace(
            original, metadata={**original.metadata, "rawmember_sha256": "not-a-sha"}
        )
    else:
        records[0] = replace(original, metadata={**original.metadata, "rawmember_sha256": "0" * 64})
    ready_path = write_manifest_jsonl(
        tuple(records), frozen_inputs["root"] / f"ready-{mutation}.jsonl"
    )
    with pytest.raises(XDMaterializationError):
        materialize_xd_heads(
            **_kwargs(
                frozen_inputs, frozen_inputs["root"] / f"tampered-{mutation}", ready_path=ready_path
            )
        )


def test_actual_ready_auditor_rejects_self_declared_raw_sha_and_legacy_receipts(
    frozen_inputs: dict[str, Any],
) -> None:
    bad = list(frozen_inputs["ready"])
    bad[0] = replace(bad[0], metadata={**bad[0].metadata, "rawmember_sha256": "0" * 64})
    bad_path = write_manifest_jsonl(tuple(bad), frozen_inputs["root"] / "ready-audit-bad-sha.jsonl")
    bad_receipt = _write_json(
        frozen_inputs["root"] / "ready-audit-bad-sha-receipt.json",
        {
            "role_lock_sha256": _file_sha256(frozen_inputs["lock_path"]),
            "roles_never_reassigned": True,
            "ready_crc_verified": len(bad),
        },
    )
    import vadbench.data.xd_materialization as xd_materialization

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            xd_materialization,
            "probe_video",
            lambda _path: SimpleNamespace(num_frames=96, fps=24.0),
        )
        with pytest.raises(XDMaterializationError):
            audit_xd_ready_manifest(
                plan_path=frozen_inputs["plan_path"],
                scope_path=frozen_inputs["scope_path"],
                original_lock_path=frozen_inputs["lock_path"],
                full_manifest_path=frozen_inputs["full_path"],
                ready_manifest_path=bad_path,
                upstream_receipt_path=bad_receipt,
                dataset_root=frozen_inputs["dataset_root"],
                output_dir=frozen_inputs["root"] / "audit-bad-sha",
            )
    legacy = _kwargs(frozen_inputs, frozen_inputs["root"] / "legacy-receipt-rejected")
    legacy.update({"ready_audit_path": None, "ready_audit_sha256": None})
    with pytest.raises(XDMaterializationError):
        materialize_xd_heads(**legacy)


def test_canary_requires_all_confirm_then_uses_duration_and_sha_tie_break(
    frozen_inputs: dict[str, Any],
) -> None:
    confirm_ids = set(frozen_inputs["roles"]["confirm"])
    incomplete = [
        row
        for row in frozen_inputs["ready"]
        if row.video_id != frozen_inputs["roles"]["confirm"][0]
    ]
    incomplete_path, incomplete_receipt, incomplete_audit = _issue_ready_audit(
        frozen_inputs, incomplete, "confirm-missing"
    )
    materialize_xd_heads(
        **_kwargs(
            frozen_inputs,
            frozen_inputs["root"] / "canary-blocked",
            ready_path=incomplete_path,
            ready_receipt_path=incomplete_receipt,
            ready_audit_path=Path(incomplete_audit["ready_audit_path"]),
            ready_audit_sha256=incomplete_audit["ready_audit_sha256"],
        )
    )
    status = json.loads(
        (frozen_inputs["root"] / "canary-blocked" / "canary-status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "blocked_incomplete_confirm"
    assert status["selected_video_id"] is None

    records = list(frozen_inputs["ready"])
    tied = [index for index, row in enumerate(records) if row.video_id in confirm_ids][:2]
    for index in tied:
        row = records[index]
        records[index] = replace(
            row,
            num_frames=48,
            fps=24.0,
            duration_seconds=2.0,
            metadata={**row.metadata, "actual_num_frames": 48, "actual_fps": 24.0},
        )
    tied_ids = [records[index].video_id for index in tied]
    expected = min(tied_ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    tied_path, tied_receipt, tied_audit = _issue_ready_audit(
        frozen_inputs, records, "canary-tie", frame_counts={value: 48 for value in tied_ids}
    )
    readiness = materialize_xd_heads(
        **_kwargs(
            frozen_inputs,
            frozen_inputs["root"] / "canary-ready",
            ready_path=tied_path,
            ready_receipt_path=tied_receipt,
            ready_audit_path=Path(tied_audit["ready_audit_path"]),
            ready_audit_sha256=tied_audit["ready_audit_sha256"],
        )
    )
    assert readiness["canary"]["status"] == "ready"
    assert readiness["canary"]["selected_video_id"] == expected


def test_complete_targets_with_short_native_input_stay_partial_and_block_canary(
    frozen_inputs: dict[str, Any],
) -> None:
    short_id = frozen_inputs["roles"]["confirm"][0]
    records = list(frozen_inputs["ready"])
    index = next(index for index, row in enumerate(records) if row.video_id == short_id)
    row = records[index]
    records[index] = replace(
        row,
        num_frames=7,
        fps=24.0,
        duration_seconds=7 / 24,
        metadata={**row.metadata, "actual_num_frames": 7, "actual_fps": 24.0},
    )
    short_path, short_receipt, short_audit = _issue_ready_audit(
        frozen_inputs, records, "short-native", frame_counts={short_id: 7}
    )
    output = frozen_inputs["root"] / "blocked-native"
    readiness = materialize_xd_heads(
        **_kwargs(
            frozen_inputs,
            output,
            ready_path=short_path,
            ready_receipt_path=short_receipt,
            ready_audit_path=Path(short_audit["ready_audit_path"]),
            ready_audit_sha256=short_audit["ready_audit_sha256"],
        )
    )

    assert readiness["status"] == "blocked_native_input"
    assert readiness["head_training_ready"] is False
    assert {role: readiness["targets"][role]["available"] for role in ROLE_COUNTS} == ROLE_COUNTS
    assert (output / "partial" / "fit.available.jsonl").is_file()
    assert not (output / "manifests").exists()
    assert not (output / "controller").exists()
    assert not (output / "head-data-contract.json").exists()
    assert readiness["canary"]["status"] == "blocked_native_input"
    assert readiness["canary"]["selected_video_id"] == short_id
    assert not (output / "engineering-canary.jsonl").exists()


def test_restoring_confirm_metadata_uses_same_frozen_plan_and_never_substitutes(
    frozen_inputs: dict[str, Any],
) -> None:
    missing_id = frozen_inputs["roles"]["confirm"][0]
    missing = [row for row in frozen_inputs["ready"] if row.video_id != missing_id]
    missing_path, missing_receipt, missing_audit = _issue_ready_audit(
        frozen_inputs, missing, "restore-missing"
    )
    plan_bytes = frozen_inputs["plan_path"].read_bytes()
    rule_bytes = frozen_inputs["canary_path"].read_bytes()
    blocked = materialize_xd_heads(
        **_kwargs(
            frozen_inputs,
            frozen_inputs["root"] / "restore-blocked",
            ready_path=missing_path,
            ready_receipt_path=missing_receipt,
            ready_audit_path=Path(missing_audit["ready_audit_path"]),
            ready_audit_sha256=missing_audit["ready_audit_sha256"],
        )
    )
    assert blocked["status"] == "blocked_incomplete"
    assert blocked["targets"]["confirm"] == {
        "required": 167,
        "available": 166,
        "pending": 1,
        "weak_labels_available": blocked["targets"]["confirm"]["weak_labels_available"],
    }
    assert blocked["members_reassigned_or_replaced"] is False

    restored_output = frozen_inputs["root"] / "restore-ready"
    restored = materialize_xd_heads(**_kwargs(frozen_inputs, restored_output))
    assert restored["status"] == "ready"
    assert frozen_inputs["plan_path"].read_bytes() == plan_bytes
    assert frozen_inputs["canary_path"].read_bytes() == rule_bytes
    for role, ids in frozen_inputs["roles"].items():
        records = load_manifest_jsonl(restored_output / "manifests" / f"{role}.jsonl")
        assert {row.video_id for row in records} == set(ids)


def test_frozen_canary_plan_manifest_and_contract_tampering_are_rejected(
    frozen_inputs: dict[str, Any],
) -> None:
    changed_rule = frozen_inputs["root"] / "changed-canary.json"
    changed_rule.write_text("{}", encoding="utf-8")
    with pytest.raises(XDMaterializationError):
        materialize_xd_heads(
            **_kwargs(
                frozen_inputs, frozen_inputs["root"] / "canary-changed", canary_path=changed_rule
            )
        )

    output = frozen_inputs["root"] / "contract-tamper"
    materialize_xd_heads(**_kwargs(frozen_inputs, output))
    contract_path = output / "head-data-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["roles"]["fit"]["source_identity_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(XDMaterializationError):
        validate_xd_head_data_contract(
            contract_path,
            scope_path=frozen_inputs["scope_path"],
            derived_plan_path=frozen_inputs["plan_path"],
            original_lock_path=frozen_inputs["lock_path"],
            method_freeze_path=frozen_inputs["freeze_path"],
        )

    materialize_xd_heads(**_kwargs(frozen_inputs, frozen_inputs["root"] / "source-row-tamper"))
    source_output = frozen_inputs["root"] / "source-row-tamper"
    source_path = source_output / "manifests" / "fit.jsonl"
    source_rows = list(load_manifest_jsonl(source_path))
    source_rows[0] = replace(
        source_rows[0], metadata={**source_rows[0].metadata, "rawmember_sha256": "0" * 64}
    )
    write_manifest_jsonl(tuple(source_rows), source_path)
    source_contract_path = source_output / "head-data-contract.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_contract["roles"]["fit"]["source_manifest_sha256"] = _file_sha256(source_path)
    source_contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
    with pytest.raises(XDMaterializationError):
        validate_xd_head_data_contract(
            source_contract_path,
            scope_path=frozen_inputs["scope_path"],
            derived_plan_path=frozen_inputs["plan_path"],
            original_lock_path=frozen_inputs["lock_path"],
            method_freeze_path=frozen_inputs["freeze_path"],
        )
