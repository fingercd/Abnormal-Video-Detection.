"""Materialize the frozen XD derived training view without changing membership.

Existing CRC/SHA/container audit metadata is the asset evidence. This module
checks file presence/size, freezes that evidence, and never decodes videos,
reads test truth, trains a model, or substitutes available videos for targets.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from vadbench.data.manifest import (
    DatasetSplit,
    SupervisionScope,
    VideoManifestRecord,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.data.video import probe_video
from vadbench.features import atomic_write_json

ROLE_COUNTS = {"fit": 1267, "select": 164, "confirm": 167}
ROLE_LABELS = {
    "fit": {"0": 637, "1": 630},
    "select": {"0": 84, "1": 80},
    "confirm": {"0": 86, "1": 81},
}

READY_AUDIT_KIND = "xd-ready-manifest-content-audit-v1"
READY_AUDIT_METHOD = "independent_streamed_crc32_sha256_and_container_probe"
# This pre-existing read-only audit was frozen before materialization and its
# P1 repro. It binds the real 2800-row producer output and original role lock.
# Newly assembled metadata-only receipts are not an equivalent authority.
FROZEN_METADATA_AUDIT_ROOTS = {
    "d698ed6bb3a4a066271b6388fa4456c97fd7476e48cc447e05fe55816f1d2ba8": "/users/fotile/datasets/XD-Violence-raw-verified",
}


class XDMaterializationError(ValueError):
    """A frozen identity or data-evidence contract is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XDMaterializationError(message)


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _weak_only(record: VideoManifestRecord) -> None:
    _require(record.split == DatasetSplit.TRAIN, "source records must retain official train split")
    _require(
        all(a.scope == SupervisionScope.VIDEO and a.span is None for a in record.annotations),
        "XD head preparation accepts video weak supervision only",
    )


def _labels(records: tuple[VideoManifestRecord, ...]) -> dict[str, int]:
    counter = Counter(str(int(record.is_anomaly)) for record in records)
    return {"0": counter["0"], "1": counter["1"]}


def _load_plan_scope(
    plan_path: str | Path, scope_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, scope = _json(plan_path), _json(scope_path)
    _require(
        scope.get("secondary_encoders") == ["videomaev2", "timesformer"],
        "XD scope must retain its two frozen secondary encoders",
    )
    _require(
        scope.get("additional_reducer_calibration_on_xd") is False,
        "XD preparation must not introduce reducer calibration",
    )
    _require(
        scope.get("derived_role_plan_file_sha256") == _file_sha(plan_path),
        "derived plan file SHA differs from frozen XD scope",
    )
    _require(
        scope.get("derived_role_plan_canonical_sha256") == _digest(plan),
        "derived plan canonical SHA differs from frozen XD scope",
    )
    _require(
        plan.get("derived_train_total") == 1598
        and plan.get("role_reassignment") is False
        and plan.get("membership_depends_on_availability") is False,
        "derived membership policy changed",
    )
    ids = plan.get("role_video_ids", {})
    _require(set(ids) == set(ROLE_COUNTS), "derived plan role set changed")
    flattened = []
    for role, count in ROLE_COUNTS.items():
        _require(
            len(ids[role]) == len(set(ids[role])) == count,
            f"{role} must retain its complete frozen ID set",
        )
        _require(
            scope["roles"][role]
            == {
                "videos": count,
                "normal": ROLE_LABELS[role]["0"],
                "positive": ROLE_LABELS[role]["1"],
            },
            "XD scope role counts or weak labels changed",
        )
        flattened.extend(ids[role])
    _require(len(set(flattened)) == 1598, "derived roles overlap")
    return plan, scope


def _canary_rule(plan: Mapping[str, Any], scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "engineering_only; no method, checkpoint or quality selection",
        "derived_role_plan_canonical_sha256": _digest(plan),
        "xd_scope_canonical_sha256": _digest(scope),
        "eligible_role": "confirm",
        "eligible_video_ids_sha256": _digest(plan["role_video_ids"]["confirm"]),
        "required_complete_candidates": 167,
        "duration_measure": "actual_container_num_frames / actual_container_fps",
        "duration_comparison": "exact Fraction(num_frames) / Fraction(str(fps))",
        "tie_breaker": "sha256(video_id UTF-8), then video_id",
        "selection_count": 1,
        "availability_substitution": False,
        "requires_all_confirm_metadata_and_files": True,
        "requires_name_groups_disjoint_from_fit_and_select": True,
    }


def freeze_xd_canary_rule(
    plan_path: str | Path, scope_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Freeze the engineering rule before consulting ready video metadata."""
    plan, scope = _load_plan_scope(plan_path, scope_path)
    rule = _canary_rule(plan, scope)
    destination = Path(output_path)
    if destination.exists():
        _require(_json(destination) == rule, "existing canary rule differs; it cannot be replaced")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(rule, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
    return rule


def _identity(records: tuple[VideoManifestRecord, ...]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": row.video_id,
            "path": row.path,
            "is_anomaly": row.is_anomaly,
            "num_frames": row.num_frames,
            "fps": row.fps,
            "rawmember_sha256": row.metadata["rawmember_sha256"],
            "verified_crc32": row.metadata["verified_crc32"],
            "source_group": row.metadata["source_group"],
        }
        for row in sorted(records, key=lambda item: item.video_id)
    ]


def _controller_records(
    records: tuple[VideoManifestRecord, ...], role: str
) -> tuple[VideoManifestRecord, ...]:
    split = "train" if role == "fit" else "val"
    return tuple(
        replace(
            row,
            split=split,
            metadata={
                **row.metadata,
                "xd_materialization": {
                    **row.metadata["xd_materialization"],
                    "controller_split": split,
                },
            },
        )
        for row in records
    )


def _policy_inputs(
    plan_path: str | Path,
    original_lock_path: str | Path,
    scope_path: str | Path,
    method_freeze_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, scope = _load_plan_scope(plan_path, scope_path)
    lock, freeze = _json(original_lock_path), _json(method_freeze_path)
    lock_sha = _file_sha(original_lock_path)
    _require(
        lock_sha
        == scope["original_role_lock_file_sha256"]
        == plan["inputs"]["original_role_lock_file_sha256"],
        "original role-lock file identity changed",
    )
    _require(
        _digest(lock) == plan["inputs"]["original_role_lock_canonical_sha256"],
        "original role-lock canonical identity changed",
    )
    _require(
        lock["seed"] == scope["original_role_lock_seed"] == 202709, "original role seed changed"
    )
    _require(
        freeze["xd_evaluation"]["additional_calibration_on_xd"] is False,
        "method freeze does not authorize XD calibration",
    )
    _require(
        freeze["sampling"]["short_policy"] == "stride1_if_needed",
        "frozen short-video sampling policy changed",
    )
    groups = {}
    for role, ids in plan["role_video_ids"].items():
        _require(
            all(lock["partitions"].get(key) == role for key in ids),
            "derived plan reassigns original roles",
        )
        groups[role] = {lock["source_groups"][key] for key in ids}
    _require(
        not (
            groups["fit"] & groups["confirm"]
            or groups["fit"] & groups["select"]
            or groups["confirm"] & groups["select"]
        ),
        "derived name groups cross training roles",
    )
    return plan, scope, lock, freeze


def _ready_record(
    row: VideoManifestRecord, original: VideoManifestRecord, role: str, plan_digest: str
) -> VideoManifestRecord:
    _weak_only(row)
    _require(
        (row.path, row.category, row.is_anomaly, row.annotations)
        == (original.path, original.category, original.is_anomaly, original.annotations),
        "ready video path/weak-label identity differs from original manifest",
    )
    meta = row.metadata
    for key in ("source_group", "provider_crc32", "uncompressed_size"):
        _require(
            meta.get(key) == original.metadata.get(key),
            f"ready {key} differs from original member identity",
        )
    _require(_sha(meta.get("rawmember_sha256")), "ready record needs its audited raw-member SHA256")
    _require(
        meta.get("verified_crc32") == original.metadata["provider_crc32"],
        "ready record lacks matching provider CRC evidence",
    )
    frames = meta.get("actual_num_frames", row.num_frames)
    fps = meta.get("actual_fps", row.fps)
    _require(type(frames) is int and frames > 0, "ready record lacks actual positive frame count")
    _require(
        not isinstance(fps, bool)
        and isinstance(fps, (int, float))
        and math.isfinite(fps)
        and fps > 0,
        "ready record lacks actual positive FPS",
    )
    _require(row.num_frames is None or row.num_frames == frames, "ready frame fields disagree")
    _require(row.fps is None or abs(row.fps - fps) <= 1e-3, "ready FPS fields disagree")
    duration = frames / fps
    _require(
        row.duration_seconds is None or abs(row.duration_seconds - duration) <= max(0.1, 1 / fps),
        "ready duration conflicts with audited N/FPS",
    )
    return replace(
        row,
        num_frames=frames,
        fps=float(fps),
        duration_seconds=duration,
        metadata={
            **meta,
            "xd_materialization": {
                "original_split": "train",
                "original_role": role,
                "derived_role_plan_canonical_sha256": plan_digest,
            },
        },
    )


def audit_xd_ready_manifest(
    *,
    plan_path: str | Path,
    scope_path: str | Path,
    original_lock_path: str | Path,
    full_manifest_path: str | Path,
    ready_manifest_path: str | Path,
    upstream_receipt_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Issue a NEW content-bound receipt after independent per-member checks.

    This is deliberately separate from materialization. It reads every ready
    video's bytes for CRC/SHA and independently probes its container; a caller
    cannot convert a count-only receipt into evidence by hashing JSON alone.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan, _ = _load_plan_scope(plan_path, scope_path)
        lock = _json(original_lock_path)
        _require(
            _file_sha(original_lock_path) == plan["inputs"]["original_role_lock_file_sha256"],
            "ready audit original role lock changed",
        )
        _require(
            _file_sha(full_manifest_path) == plan["inputs"]["full_train_manifest_file_sha256"],
            "ready audit original manifest changed",
        )
        source_paths = {
            "role-lock.json": original_lock_path,
            "train.full.jsonl": full_manifest_path,
            "ready_train.jsonl": ready_manifest_path,
            "ready_receipt.json": upstream_receipt_path,
        }
        source_files = {
            name: {"path": str(Path(path).resolve()), "sha256": _file_sha(path)}
            for name, path in source_paths.items()
        }
        full = {row.video_id: row for row in load_manifest_jsonl(full_manifest_path)}
        ready = load_manifest_jsonl(ready_manifest_path)
        upstream = _json(upstream_receipt_path)
        _require(
            upstream.get("role_lock_sha256") == _file_sha(original_lock_path)
            and upstream.get("roles_never_reassigned") is True
            and upstream.get("ready_crc_verified") == len(ready),
            "upstream ready count/role receipt is inconsistent",
        )
        _require(
            {row.video_id for row in ready} <= set(full),
            "ready audit contains unknown original training identities",
        )
        root = Path(dataset_root).expanduser().resolve()
        proofs = []
        for row in ready:
            original = full[row.video_id]
            normalized = _ready_record(
                row, original, lock["partitions"][row.video_id], _digest(plan)
            )
            path = original.resolve_path(root)
            before = path.stat()
            hasher, crc, count = hashlib.sha256(), 0, 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(block)
                    crc = zlib.crc32(block, crc)
                    count += len(block)
            observed_sha, observed_crc = hasher.hexdigest(), f"{crc & 0xFFFFFFFF:08x}"
            _require(
                count == before.st_size == original.metadata["uncompressed_size"],
                "ready audit actual member size mismatch",
            )
            _require(
                observed_crc
                == original.metadata["provider_crc32"]
                == row.metadata["verified_crc32"],
                "ready audit actual CRC mismatch",
            )
            _require(
                observed_sha == row.metadata["rawmember_sha256"],
                "ready audit actual SHA256 differs from ready metadata",
            )
            info = probe_video(path)
            _require(
                info.num_frames == normalized.num_frames and info.fps == normalized.fps,
                "ready audit actual N/FPS differs from ready metadata",
            )
            after = path.stat()
            _require(
                (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "raw member changed during ready audit",
            )
            proofs.append(
                {
                    "video_id": row.video_id,
                    "ready_record_sha256": _digest(row.to_dict()),
                    "identity": _identity((normalized,))[0],
                    "observed_sha256": observed_sha,
                    "observed_crc32": observed_crc,
                    "observed_size_bytes": count,
                    "observed_num_frames": info.num_frames,
                    "observed_fps": info.fps,
                    "observed_mtime_ns": after.st_mtime_ns,
                }
            )
            if len(proofs) % 25 == 0:
                atomic_write_json(
                    out / "progress.json",
                    {
                        "status": "running",
                        "verified_members": len(proofs),
                        "required_members": len(ready),
                    },
                )
        for name, path in source_paths.items():
            _require(
                _file_sha(path) == source_files[name]["sha256"],
                "upstream metadata changed during ready audit",
            )
        proof_bytes = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            + b"\n"
            for row in proofs
        )
        (out / "member-proofs.jsonl").write_bytes(proof_bytes)
        result = {
            "schema_version": 1,
            "kind": READY_AUDIT_KIND,
            "status": "verified",
            "verification_method": READY_AUDIT_METHOD,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_files": source_files,
            "role_lock_file_sha256": _file_sha(original_lock_path),
            "role_lock_canonical_sha256": _digest(lock),
            "ready_receipt_facts": upstream,
            "verified_members": len(ready),
            "member_proofs_file": "member-proofs.jsonl",
            "member_proofs_sha256": _file_sha(out / "member-proofs.jsonl"),
            "dataset_root": str(root),
            "new_independent_audit_not_relabelled_legacy_receipt": True,
            "python": sys.executable,
            "audit_implementation_sha256": _file_sha(__file__),
            "container_probe": f"{probe_video.__module__}.{probe_video.__qualname__}",
            "model_or_gt_access": False,
        }
        atomic_write_json(out / "ready-audit.json", result)
        return {
            "status": "verified",
            "verified_members": len(ready),
            "ready_audit_path": str(out / "ready-audit.json"),
            "ready_audit_sha256": _file_sha(out / "ready-audit.json"),
        }
    except Exception as exc:
        atomic_write_json(
            out / "audit-failure.json",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def _bound_ready_evidence(
    audit_path: str | Path | None,
    audit_sha256: str | None,
    *,
    ready_path: str | Path,
    upstream_path: str | Path,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        audit_path is not None and _sha(audit_sha256),
        "a content-bound upstream ready audit and its externally frozen SHA256 are required",
    )
    audit_path = Path(audit_path).expanduser().resolve()
    _require(_file_sha(audit_path) == audit_sha256, "upstream ready audit SHA256 changed")
    audit = _json(audit_path)
    source = audit.get("source_files", {})
    expected = {
        "role-lock.json": plan["inputs"]["original_role_lock_file_sha256"],
        "train.full.jsonl": plan["inputs"]["full_train_manifest_file_sha256"],
        "ready_train.jsonl": _file_sha(ready_path),
        "ready_receipt.json": _file_sha(upstream_path),
    }
    for name, value in expected.items():
        _require(
            isinstance(source.get(name), Mapping) and source[name].get("sha256") == value,
            "ready manifest bytes or upstream source identity differ from the frozen audit",
        )
    _require(
        audit.get("role_lock_file_sha256") == expected["role-lock.json"]
        and audit.get("role_lock_canonical_sha256") == _digest(lock),
        "ready authority is not bound to the original roles",
    )
    upstream = _json(upstream_path)
    _require(
        audit.get("ready_receipt_facts") == upstream,
        "ready authority upstream receipt payload changed",
    )
    rows = load_manifest_jsonl(ready_path)
    _require(
        upstream.get("ready_crc_verified") == len(rows)
        and upstream.get("roles_never_reassigned") is True,
        "ready authority record coverage is inconsistent",
    )
    proofs = None
    proof_path = None
    if audit.get("kind") == READY_AUDIT_KIND:
        _require(
            audit.get("status") == "verified"
            and audit.get("verification_method") == READY_AUDIT_METHOD
            and audit.get("verified_members") == len(rows),
            "new ready audit lacks completed independent member verification",
        )
        relative = Path(audit.get("member_proofs_file", ""))
        _require(
            str(relative) not in {"", "."}
            and not relative.is_absolute()
            and ".." not in relative.parts,
            "ready proof path is invalid",
        )
        proof_path = (audit_path.parent / relative).resolve()
        _require(
            proof_path.is_relative_to(audit_path.parent)
            and _file_sha(proof_path) == audit.get("member_proofs_sha256"),
            "ready member proof ledger changed",
        )
        ledger = [
            json.loads(line) for line in proof_path.read_text(encoding="utf-8").splitlines() if line
        ]
        proofs = {row["video_id"]: row for row in ledger}
        _require(
            len(proofs) == len(ledger) == len(rows)
            and set(proofs) == {row.video_id for row in rows},
            "ready member proof coverage is incomplete",
        )
        for row in rows:
            _require(
                row.video_id in lock["partitions"], "ready proof identity is outside original roles"
            )
            normalized = _ready_record(row, row, lock["partitions"][row.video_id], _digest(plan))
            proof = proofs[row.video_id]
            _require(
                proof.get("ready_record_sha256") == _digest(row.to_dict())
                and proof.get("identity") == _identity((normalized,))[0],
                "ready row differs from independently audited member proof",
            )
            _require(
                proof.get("observed_sha256") == row.metadata["rawmember_sha256"]
                and proof.get("observed_crc32") == row.metadata["provider_crc32"]
                and proof.get("observed_size_bytes") == row.metadata["uncompressed_size"]
                and proof.get("observed_num_frames") == normalized.num_frames
                and proof.get("observed_fps") == normalized.fps,
                "ready member observations differ from declared metadata",
            )
        kind = "independent_member_content_audit"
    else:
        # Existing, independently frozen metadata audit is an allowed authority.
        # Do not create this authority by hashing the candidate manifest here.
        _require(
            audit_sha256 in FROZEN_METADATA_AUDIT_ROOTS,
            "new metadata-only self-signatures are not an upstream authority; run the independent member audit",
        )
        _require(
            audit.get("remote_train_full_matches_local_file_sha256") is True
            and audit.get("roles_changed_or_videos_selected_for_deletion") is False
            and isinstance(audit.get("ready_train_role_statistics"), Mapping),
            "count-only or self-declared ready receipt is not a content-binding authority",
        )
        for role in ROLE_COUNTS:
            _require(
                audit["ready_train_role_statistics"][role]["videos"]
                == sum(lock["partitions"].get(row.video_id) == role for row in rows),
                "prior ready audit role coverage differs",
            )
        kind = "previous_externally_frozen_role_bound_manifest_audit"
    return {
        "authority": audit,
        "authority_path": audit_path,
        "authority_sha256": audit_sha256,
        "kind": kind,
        "ready_records": rows,
        "proofs": proofs,
        "proof_path": proof_path,
    }


def _copy_ready_evidence(
    destination: Path,
    evidence: Mapping[str, Any],
    ready_path: str | Path,
    upstream_path: str | Path,
) -> dict[str, Any]:
    folder = destination / "ready-evidence"
    folder.mkdir()
    files = {
        "audit": (evidence["authority_path"], "ready-audit.json"),
        "ready_manifest": (ready_path, "ready-manifest.jsonl"),
        "upstream_receipt": (upstream_path, "upstream-receipt.json"),
    }
    if evidence["proof_path"] is not None:
        files["member_proofs"] = (
            evidence["proof_path"],
            evidence["authority"]["member_proofs_file"],
        )
    result = {
        "kind": evidence["kind"],
        "externally_frozen_audit_sha256": evidence["authority_sha256"],
    }
    for key, (source, name) in files.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
        result[key] = {
            "path": target.relative_to(destination).as_posix(),
            "file_sha256": _file_sha(target),
        }
    return result


def materialize_xd_heads(
    *,
    plan_path: str | Path,
    original_lock_path: str | Path,
    full_manifest_path: str | Path,
    ready_manifest_path: str | Path,
    ready_receipt_path: str | Path,
    quarantine_path: str | Path,
    scope_path: str | Path,
    method_freeze_path: str | Path,
    canary_rule_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    ready_audit_path: str | Path | None = None,
    ready_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Create one immutable preparation attempt; missing targets stay blocked."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        return _materialize(
            plan_path=plan_path,
            original_lock_path=original_lock_path,
            full_manifest_path=full_manifest_path,
            ready_manifest_path=ready_manifest_path,
            ready_receipt_path=ready_receipt_path,
            quarantine_path=quarantine_path,
            scope_path=scope_path,
            method_freeze_path=method_freeze_path,
            canary_rule_path=canary_rule_path,
            dataset_root=dataset_root,
            destination=destination,
            ready_audit_path=ready_audit_path,
            ready_audit_sha256=ready_audit_sha256,
        )
    except Exception as exc:
        atomic_write_json(
            destination / "readiness.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "head_training_ready": False,
            },
        )
        raise


def _materialize(**paths: Any) -> dict[str, Any]:
    destination = paths["destination"]
    plan, scope, lock, freeze = _policy_inputs(
        paths["plan_path"],
        paths["original_lock_path"],
        paths["scope_path"],
        paths["method_freeze_path"],
    )
    plan_digest = _digest(plan)
    evidence = _bound_ready_evidence(
        paths["ready_audit_path"],
        paths["ready_audit_sha256"],
        ready_path=paths["ready_manifest_path"],
        upstream_path=paths["ready_receipt_path"],
        plan=plan,
        lock=lock,
    )
    rule = _json(paths["canary_rule_path"])
    _require(
        rule == _canary_rule(plan, scope),
        "canary rule was not frozen for this exact plan and scope",
    )
    _require(
        _file_sha(paths["full_manifest_path"]) == plan["inputs"]["full_train_manifest_file_sha256"],
        "original full train manifest changed",
    )
    _require(
        _file_sha(paths["quarantine_path"])
        == plan["inputs"]["source_corrupt_quarantine_file_sha256"],
        "frozen four-corrupt receipt changed",
    )
    quarantine = _json(paths["quarantine_path"])["records"]
    quarantined = {row["video_id"] for row in quarantine}
    _require(
        len(quarantine) == len(quarantined) == 4
        and _digest(sorted(quarantined)) == plan["inputs"]["source_corrupt_membership_sha256"],
        "frozen four-corrupt membership changed",
    )
    full = load_manifest_jsonl(paths["full_manifest_path"])
    _require(len(full) == 3954, "original train identity inventory is incomplete")
    full_by_id = {row.video_id: row for row in full}
    _require(quarantined <= set(full_by_id), "quarantine references an unknown training identity")
    _require(
        all(lock["partitions"][key] == "fit" for key in quarantined),
        "the frozen four source-corrupt roles changed",
    )
    _require(
        set(full_by_id) == set(lock["partitions"]),
        "original train inventory and role lock disagree",
    )
    ready = load_manifest_jsonl(paths["ready_manifest_path"])
    receipt = _json(paths["ready_receipt_path"])
    _require(
        receipt.get("role_lock_sha256") == _file_sha(paths["original_lock_path"])
        and receipt.get("roles_never_reassigned") is True,
        "ready metadata receipt is not bound to the frozen roles",
    )
    _require(
        receipt.get("ready_crc_verified") == len(ready),
        "ready metadata count differs from its receipt",
    )
    _require(
        all(row.video_id in full_by_id for row in ready),
        "ready metadata contains an unknown original train identity",
    )
    ready_by_id = {row.video_id: row for row in ready}
    targets = set().union(*(set(values) for values in plan["role_video_ids"].values()))
    _require(not (targets & quarantined), "source-corrupt identity re-entered derived plan")
    _require(len(set(full_by_id) - targets) == 2356, "frozen exclusion union changed")
    for row in full:
        _weak_only(row)
        _require(
            row.metadata["source_group"] == lock["source_groups"][row.video_id],
            "original source group differs from frozen lock",
        )
    for row in ready:
        _weak_only(row)
    data_root = Path(paths["dataset_root"]).expanduser().resolve()
    if evidence["kind"] == "independent_member_content_audit":
        _require(
            data_root == Path(evidence["authority"]["dataset_root"]).resolve(),
            "materialization root differs from independently audited raw root",
        )
    else:
        _require(
            data_root == Path(FROZEN_METADATA_AUDIT_ROOTS[evidence["authority_sha256"]]).resolve(),
            "raw root differs from the pre-existing frozen metadata audit",
        )
    available = {}
    blocked = {}
    unsupported = {encoder: [] for encoder in scope["secondary_encoders"]}
    for role, ids in plan["role_video_ids"].items():
        original_rows = tuple(full_by_id[key] for key in ids)
        _require(
            _labels(original_rows) == ROLE_LABELS[role],
            f"{role} weak labels differ from frozen XD scope",
        )
        complete = []
        missing = []
        for key in ids:
            if key not in ready_by_id:
                missing.append({"video_id": key, "reason": "verified_metadata_missing"})
                continue
            row = _ready_record(ready_by_id[key], full_by_id[key], role, plan_digest)
            path = row.resolve_path(data_root)
            if not path.is_file():
                missing.append({"video_id": key, "reason": "verified_raw_file_missing"})
                continue
            if path.stat().st_size != row.metadata["uncompressed_size"]:
                missing.append({"video_id": key, "reason": "verified_raw_size_changed"})
                continue
            if (
                evidence["proofs"] is not None
                and path.stat().st_mtime_ns != evidence["proofs"][key]["observed_mtime_ns"]
            ):
                missing.append(
                    {"video_id": key, "reason": "raw_member_changed_since_content_audit"}
                )
                continue
            complete.append(row)
            for encoder in unsupported:
                if row.num_frames < freeze["sampling"]["native_clip_frames"][encoder]:
                    unsupported[encoder].append(key)
        available[role] = tuple(complete)
        blocked[role] = missing
    partial_dir = destination / "partial"
    partial_dir.mkdir()
    for role, rows in available.items():
        write_manifest_jsonl(rows, partial_dir / f"{role}.available.jsonl")
    canary = {
        "status": "blocked_incomplete_confirm",
        "required_candidates": 167,
        "complete_candidates": len(available["confirm"]),
        "rule_sha256": _file_sha(paths["canary_rule_path"]),
        "selected_video_id": None,
        "engineering_only": True,
    }
    if not blocked["confirm"]:
        chosen = min(
            available["confirm"],
            key=lambda row: (
                Fraction(row.num_frames, 1) / Fraction(str(row.fps)),
                hashlib.sha256(row.video_id.encode()).hexdigest(),
                row.video_id,
            ),
        )
        canary_supported = all(
            chosen.num_frames >= freeze["sampling"]["native_clip_frames"][encoder]
            for encoder in scope["secondary_encoders"]
        )
        canary.update(
            status="ready" if canary_supported else "blocked_native_input",
            selected_video_id=chosen.video_id,
            selected_identity_sha256=_digest(_identity((chosen,))),
            duration_seconds=chosen.num_frames / chosen.fps,
            eligible_identity_sha256=_digest(_identity(available["confirm"])),
        )
        atomic_write_json(destination / "canary-lock.json", canary)
        if canary_supported:
            write_manifest_jsonl(
                _controller_records((chosen,), "confirm"), destination / "engineering-canary.jsonl"
            )
    atomic_write_json(destination / "canary-status.json", canary)
    complete = all(not values for values in blocked.values())
    native_supported = all(not values for values in unsupported.values())
    status = (
        "ready"
        if complete and native_supported
        else "blocked_incomplete"
        if not complete
        else "blocked_native_input"
    )
    identities = {
        name: {"path": str(Path(value).expanduser().resolve()), "file_sha256": _file_sha(value)}
        for name, value in paths.items()
        if name not in {"dataset_root", "destination", "ready_audit_sha256"}
    }
    readiness = {
        "schema_version": 1,
        "dataset": "xd_violence",
        "status": status,
        "head_training_ready": status == "ready",
        "head_encoders": scope["secondary_encoders"],
        "dataset_root": str(data_root),
        "input_files": identities,
        "derived_role_plan_canonical_sha256": plan_digest,
        "targets": {
            role: {
                "required": ROLE_COUNTS[role],
                "available": len(available[role]),
                "pending": len(blocked[role]),
                "weak_labels_available": _labels(available[role]),
            }
            for role in ROLE_COUNTS
        },
        "blocked_target_ids": blocked,
        "unsupported_native_input_ids": unsupported,
        "canary": canary,
        "members_reassigned_or_replaced": False,
        "ready_evidence_kind": evidence["kind"],
        "ready_audit_sha256": evidence["authority_sha256"],
        "bound_ready_manifest_sha256": _file_sha(paths["ready_manifest_path"]),
        "raw_evidence": "ready bytes authenticated by a separately frozen upstream audit; source rows remain bound to that snapshot; presence/size checked here",
        "model_or_gt_access": False,
        "head_data_contract": None,
    }
    if status == "ready":
        capsule = _copy_ready_evidence(
            destination, evidence, paths["ready_manifest_path"], paths["ready_receipt_path"]
        )
        for folder in ("manifests", "controller"):
            (destination / folder).mkdir()
        roles = {}
        for role, rows in available.items():
            source = write_manifest_jsonl(rows, destination / "manifests" / f"{role}.jsonl")
            controller = write_manifest_jsonl(
                _controller_records(rows, role), destination / "controller" / f"{role}.jsonl"
            )
            roles[role] = {
                "videos": len(rows),
                "weak_video_labels": _labels(rows),
                "source_manifest": source.relative_to(destination).as_posix(),
                "source_manifest_sha256": _file_sha(source),
                "controller_manifest": controller.relative_to(destination).as_posix(),
                "controller_manifest_sha256": _file_sha(controller),
                "source_split": "train",
                "controller_split": "train" if role == "fit" else "val",
                "source_identity_sha256": _digest(_identity(rows)),
            }
        contract = {
            "schema_version": 2,
            "dataset": "xd_violence",
            "status": "ready",
            "method_freeze_canonical_sha256": _digest(freeze),
            "xd_scope_file_sha256": _file_sha(paths["scope_path"]),
            "xd_scope_canonical_sha256": _digest(scope),
            "original_role_lock_file_sha256": _file_sha(paths["original_lock_path"]),
            "original_role_lock_canonical_sha256": _digest(lock),
            "derived_role_plan_file_sha256": _file_sha(paths["plan_path"]),
            "derived_role_plan_canonical_sha256": plan_digest,
            "source_quarantine_file_sha256": _file_sha(paths["quarantine_path"]),
            "source_corrupt_membership_sha256": _digest(sorted(quarantined)),
            "source_corrupt_excluded": 4,
            "derived_training_identity_count": 1598,
            "roles": roles,
            "input_files": identities,
            "dataset_root": str(data_root),
            "raw_evidence": readiness["raw_evidence"],
            "ready_evidence": capsule,
            "canary": canary,
            "source_content_near_duplicate_verification": "unknown",
        }
        atomic_write_json(destination / "head-data-contract.json", contract)
        readiness["head_data_contract"] = {
            "path": str(destination / "head-data-contract.json"),
            "file_sha256": _file_sha(destination / "head-data-contract.json"),
        }
    atomic_write_json(destination / "readiness.json", readiness)
    return readiness


def validate_xd_head_data_contract(
    contract_path: str | Path,
    *,
    scope_path: str | Path,
    derived_plan_path: str | Path,
    original_lock_path: str | Path,
    method_freeze_path: str | Path,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a complete data contract without reading video payloads or GT."""
    path = Path(contract_path).expanduser().resolve()
    contract = _json(path)
    _require(
        contract.get("schema_version") == 2
        and contract.get("dataset") == "xd_violence"
        and contract.get("status") == "ready",
        "XD head data contract is not complete and ready",
    )
    plan, scope, lock, freeze = _policy_inputs(
        derived_plan_path, original_lock_path, scope_path, method_freeze_path
    )
    plan_digest = _digest(plan)
    expected = {
        "method_freeze_canonical_sha256": _digest(freeze),
        "xd_scope_file_sha256": _file_sha(scope_path),
        "xd_scope_canonical_sha256": _digest(scope),
        "original_role_lock_file_sha256": _file_sha(original_lock_path),
        "original_role_lock_canonical_sha256": _digest(lock),
        "derived_role_plan_file_sha256": _file_sha(derived_plan_path),
        "derived_role_plan_canonical_sha256": plan_digest,
        "source_quarantine_file_sha256": plan["inputs"]["source_corrupt_quarantine_file_sha256"],
        "source_corrupt_membership_sha256": plan["inputs"]["source_corrupt_membership_sha256"],
        "source_corrupt_excluded": 4,
        "derived_training_identity_count": 1598,
    }
    _require(
        all(contract.get(name) == value for name, value in expected.items()),
        "XD head data contract provenance differs from frozen sources",
    )
    root = path.parent if source_root is None else Path(source_root).expanduser().resolve()
    source_hashes = {
        str(Path(item).expanduser().resolve()): _file_sha(item)
        for item in (path, scope_path, derived_plan_path, original_lock_path, method_freeze_path)
    }
    capsule = contract.get("ready_evidence")
    _require(
        isinstance(capsule, Mapping), "head contract lacks upstream content-bound ready evidence"
    )
    evidence_paths = {}
    for name in ("audit", "ready_manifest", "upstream_receipt"):
        item = capsule.get(name)
        _require(isinstance(item, Mapping), "head contract ready evidence entry is missing")
        relative = Path(item.get("path", ""))
        _require(
            str(relative) not in {"", "."}
            and not relative.is_absolute()
            and ".." not in relative.parts,
            "head ready evidence path is invalid",
        )
        evidence_path = (root / relative).resolve()
        _require(
            evidence_path.is_relative_to(root)
            and _file_sha(evidence_path) == item.get("file_sha256"),
            "head ready evidence snapshot changed",
        )
        evidence_paths[name] = evidence_path
        source_hashes[str(evidence_path)] = item["file_sha256"]
    evidence = _bound_ready_evidence(
        evidence_paths["audit"],
        capsule.get("externally_frozen_audit_sha256"),
        ready_path=evidence_paths["ready_manifest"],
        upstream_path=evidence_paths["upstream_receipt"],
        plan=plan,
        lock=lock,
    )
    _require(evidence["kind"] == capsule.get("kind"), "head ready evidence kind changed")
    if evidence["proof_path"] is not None:
        source_hashes[str(evidence["proof_path"])] = _file_sha(evidence["proof_path"])
    upstream_by_id = {row.video_id: row for row in evidence["ready_records"]}
    result = {}
    _require(
        set(contract.get("roles", {})) == set(ROLE_COUNTS), "head contract role set is incomplete"
    )
    for role, count in ROLE_COUNTS.items():
        entry = contract["roles"][role]
        _require(
            entry.get("videos") == count and entry.get("weak_video_labels") == ROLE_LABELS[role],
            "head contract role cardinality or labels changed",
        )
        _require(
            entry.get("source_split") == "train"
            and entry.get("controller_split") == ("train" if role == "fit" else "val"),
            "head contract split derivation changed",
        )
        loaded = {}
        for kind in ("source", "controller"):
            relative = Path(entry[f"{kind}_manifest"])
            _require(
                not relative.is_absolute() and ".." not in relative.parts,
                "head manifest path must remain relative to its source root",
            )
            manifest = (root / relative).resolve()
            _require(manifest.is_relative_to(root), "head manifest escapes its source root")
            _require(
                _file_sha(manifest) == entry[f"{kind}_manifest_sha256"],
                "head source/controller manifest changed",
            )
            records = load_manifest_jsonl(manifest)
            _require(
                len(records) == count
                and {row.video_id for row in records} == set(plan["role_video_ids"][role]),
                "head manifest does not cover the complete frozen role",
            )
            _require(_labels(records) == ROLE_LABELS[role], "head manifest weak labels changed")
            _require(
                _digest(_identity(records)) == entry["source_identity_sha256"],
                "head manifest video fields or raw hashes changed",
            )
            for row in records:
                _require(
                    row.video_id in upstream_by_id, "head role identity lacks upstream ready proof"
                )
                expected_source = _ready_record(
                    upstream_by_id[row.video_id], upstream_by_id[row.video_id], role, plan_digest
                )
                expected_row = (
                    expected_source
                    if kind == "source"
                    else _controller_records((expected_source,), role)[0]
                )
                _require(
                    row.to_dict() == expected_row.to_dict(),
                    "head source/controller row differs from immutable upstream ready evidence",
                )
                _require(
                    row.num_frames is not None
                    and row.fps is not None
                    and _sha(row.metadata.get("rawmember_sha256"))
                    and row.metadata.get("verified_crc32") == row.metadata.get("provider_crc32")
                    and isinstance(row.metadata.get("provider_crc32"), str),
                    "head source record lacks complete N/FPS/raw hash/CRC evidence",
                )
                _require(
                    row.split.value == entry[f"{kind}_split"],
                    "head manifest split differs from contract",
                )
                _require(
                    all(
                        a.scope == SupervisionScope.VIDEO and a.span is None
                        for a in row.annotations
                    ),
                    "temporal supervision entered XD head manifest",
                )
                _require(
                    row.metadata["source_group"] == lock["source_groups"][row.video_id],
                    "head manifest source group changed",
                )
                origin = row.metadata.get("xd_materialization", {})
                _require(
                    origin.get("original_split") == "train"
                    and origin.get("original_role") == role
                    and origin.get("derived_role_plan_canonical_sha256") == plan_digest,
                    "head manifest lacks explicit original-role provenance",
                )
            loaded[f"{kind}_path"] = manifest
            loaded[f"{kind}_records"] = records
            source_hashes[str(manifest)] = entry[f"{kind}_manifest_sha256"]
        _require(
            tuple(row.annotations for row in loaded["source_records"])
            == tuple(row.annotations for row in loaded["controller_records"]),
            "controller changed weak annotations",
        )
        result[role] = loaded
    return {"contract": contract, "roles": result, "source_hashes": source_hashes}
