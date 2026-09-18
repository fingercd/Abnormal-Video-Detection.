"""Create and consume official full-training views without changing old roles."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from vadbench.data.manifest import (
    DatasetSplit,
    SupervisionScope,
    VideoManifestRecord,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.data.ucf_crime import parse_ucf_split_file
from vadbench.data.video import probe_video
from vadbench.data.xd_materialization import (
    FROZEN_METADATA_AUDIT_ROOTS,
    _bound_ready_evidence,
    _digest,
    _ready_record,
    _sha,
)
from vadbench.features import atomic_write_json
from vadbench.hashing import sha256_file

VIEW = "official-fulltrain-final"
ROLES = ("fit", "confirm", "select")
EXPECTED = {"ucf_crime": (1610, 800, 810), "xd_violence": (3954, 2049, 1905)}
UCF_AUDIT_KIND = "ucf-official-training-content-audit-v1"


class OfficialTrainingError(ValueError):
    """An official membership, role, or independent evidence binding failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfficialTrainingError(message)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(value: str) -> str:
    result = {"ucf": "ucf_crime", "xd": "xd_violence"}.get(value, value)
    _require(result in EXPECTED, "unknown dataset")
    return result


def _weak(row: VideoManifestRecord) -> None:
    _require(row.split == DatasetSplit.TRAIN, "upstream record is not official TRAIN")
    _require(all(a.scope == SupervisionScope.VIDEO and a.span is None
                 and a.is_anomaly in (None, row.is_anomaly) for a in row.annotations),
             "temporal or inconsistent supervision entered official training")


def _official(dataset: str, source: Path) -> dict[str, dict[str, Any]]:
    if dataset == "ucf_crime":
        entries = parse_ucf_split_file(source, DatasetSplit.TRAIN)
        lines = [line for line in source.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        _require(len(lines) == len(entries), "official source contains duplicate IDs")
        result = {r.video_id: {"video_id": r.video_id, "path": r.path,
                              "category": r.category, "is_anomaly": r.category != "Normal"}
                  for r in entries}
    else:
        records = load_manifest_jsonl(source)
        for row in records:
            _weak(row)
        result = {r.video_id: {"video_id": r.video_id, "path": r.path,
                              "category": r.category, "is_anomaly": r.is_anomaly}
                  for r in records}
    total, normal, anomaly = EXPECTED[dataset]
    _require(len(result) == total, "official identity inventory is incomplete")
    _require(sum(not r["is_anomaly"] for r in result.values()) == normal
             and sum(r["is_anomaly"] for r in result.values()) == anomaly,
             "official identity weak-label totals differ")
    return result


def _role_lock(path: Path, official: Mapping[str, Any]) -> dict[str, Any]:
    lock = _json(path)
    _require(set(lock.get("partitions", {})) == set(official),
             "original role lock does not cover the exact official IDs")
    _require(set(lock["partitions"].values()) == set(ROLES), "original role set changed")
    return lock


def _row_identity(row: VideoManifestRecord, expected: Mapping[str, Any]) -> None:
    _weak(row)
    _require(all(getattr(row, key) == expected[key]
                 for key in ("video_id", "path", "category", "is_anomaly")),
             "official per-video path/category/weak-label identity differs")
    _require(type(row.num_frames) is int and row.num_frames > 0
             and row.fps is not None and math.isfinite(row.fps) and row.fps > 0,
             "member lacks authoritative positive N/FPS")


def _role_rows(paths, official, lock):
    rows = {}
    for role in ROLES:
        for row in load_manifest_jsonl(paths[role]):
            _require(row.video_id not in rows, "duplicate video across original roles")
            _require(row.video_id in official, "unknown official video ID")
            _require(lock["partitions"][row.video_id] == role, "original role lock mismatch")
            _row_identity(row, official[row.video_id])
            rows[row.video_id] = row
    return rows


def _file_bindings(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {key: {"path": str(value.resolve()), "sha256": sha256_file(value)}
            for key, value in paths.items()}


def _provider_members(path: Path) -> dict[str, Any]:
    members = _json(path)["video_members"]
    result = {}
    prefix = "UCF_Crimes/Videos/"
    for member in members:
        name = member["relative_path"]
        _require(name.startswith(prefix), "provider member is outside the official video root")
        relative = name[len(prefix):]
        _require(relative not in result and ".." not in Path(relative).parts,
                 "duplicate or unsafe provider member path")
        result[relative] = member
    return result


def audit_ucf_official_training(
    *, full_manifest: Path, full_manifest_sha256: str, role_lock: Path,
    role_lock_sha256: str, fit: Path, confirm: Path, select: Path,
    provider_metadata: Path, provider_metadata_sha256: str, dataset_root: Path, output: Path,
) -> dict[str, Any]:
    """Issue a NEW CPU byte/probe authority; no promotion of count-only receipts."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {"full_manifest": Path(full_manifest), "role_lock": Path(role_lock),
             "fit": Path(fit), "confirm": Path(confirm), "select": Path(select),
             "provider_metadata": Path(provider_metadata)}
    try:
        _require(_sha(full_manifest_sha256) and sha256_file(paths["full_manifest"]) == full_manifest_sha256,
                 "externally pinned official split SHA256 differs")
        _require(_sha(role_lock_sha256) and sha256_file(paths["role_lock"]) == role_lock_sha256,
                 "externally pinned original role lock SHA256 differs")
        _require(_sha(provider_metadata_sha256)
                 and sha256_file(paths["provider_metadata"]) == provider_metadata_sha256,
                 "externally pinned provider metadata SHA256 differs")
        provider = _provider_members(paths["provider_metadata"])
        official = _official("ucf_crime", paths["full_manifest"])
        lock = _role_lock(paths["role_lock"], official)
        rows = _role_rows(paths, official, lock)
        bindings = _file_bindings(paths)
        root = Path(dataset_root).resolve()
        proofs, blocked = [], []
        for key in sorted(official):
            if key not in rows:
                blocked.append({"video_id": key, "reason": "metadata_missing"})
                continue
            row = rows[key]
            _require(row.path in provider, "official member missing from provider archive metadata")
            member = provider[row.path]
            path = row.resolve_path(root)
            try:
                before = path.stat()
                hasher, crc, size = hashlib.sha256(), 0, 0
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        hasher.update(block)
                        crc = zlib.crc32(block, crc)
                        size += len(block)
                info = probe_video(path)
                after = path.stat()
                _require(size == before.st_size == after.st_size == member["uncompressed_size"]
                         and before.st_mtime_ns == after.st_mtime_ns,
                         "raw member changed during independent content audit")
                _require(info.num_frames == row.num_frames and info.fps == row.fps,
                         "actual container N/FPS differs from original metadata")
                observed_crc = f"{crc & 0xffffffff:08x}"
                _require(observed_crc == member["crc32"], "provider CRC mismatch")
                proofs.append({"video_id": key, "path": row.path,
                               "source_record_sha256": _digest(row.to_dict()),
                               "original_role": lock["partitions"][key],
                               "num_frames": info.num_frames, "fps": info.fps,
                               "content_sha256": hasher.hexdigest(), "size_bytes": size,
                               "observed_crc32": observed_crc,
                               "provider_crc_checked": True,
                               "observed_mtime_ns": after.st_mtime_ns})
            except (OSError, ValueError) as error:
                blocked.append({"video_id": key, "reason": type(error).__name__, "detail": str(error)})
            if (len(proofs) + len(blocked)) % 25 == 0:
                atomic_write_json(output / "progress.json", {
                    "status": "running", "verified_members": len(proofs),
                    "blocked_members": len(blocked), "expected_members": len(official)})
        _require(bindings == _file_bindings(paths), "source manifests changed during audit")
        ledger = output / "member-proofs.jsonl"
        with ledger.open("x", encoding="utf-8") as stream:
            for proof in proofs:
                stream.write(json.dumps(proof, sort_keys=True) + "\n")
        authority = {"schema_version": 1, "kind": UCF_AUDIT_KIND,
                     "status": "verified" if not blocked else "blocked",
                     "verification_method": "independent_streamed_crc32_sha256_size_and_container_probe",
                     "source_files": bindings, "dataset_root": str(root),
                     "official_identity_sha256": _digest(official), "verified_members": len(proofs),
                     "member_proofs_file": ledger.name, "member_proofs_sha256": sha256_file(ledger),
                     "blocked_members": blocked, "container_probe_is_full_decode": False,
                     "implementation_sha256": sha256_file(Path(__file__))}
        atomic_write_json(output / "authority.json", authority)
        receipt = {"state": "ready" if not blocked else "blocked", "verified_members": len(proofs),
                   "evidence": str(output / "authority.json"),
                   "evidence_sha256": sha256_file(output / "authority.json"), "blocked_members": blocked}
        atomic_write_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        atomic_write_json(output / "receipt.json", {"state": "blocked", "error_type": type(error).__name__,
                                                   "reason": str(error)})
        raise


def _ucf_evidence(paths, evidence, official, lock):
    authority = _json(evidence)
    _require(authority.get("kind") == UCF_AUDIT_KIND and authority.get("status") == "verified",
             "UCF requires a completed independent raw-content authority")
    _require(authority.get("verification_method") == "independent_streamed_crc32_sha256_size_and_container_probe",
             "UCF authority verification method differs")
    for key in ("full_manifest", "role_lock", "provider_metadata", *ROLES):
        _require(authority["source_files"][key]["sha256"] == sha256_file(paths[key]),
                 "UCF authority source manifest or role lock changed")
    _require(authority["official_identity_sha256"] == _digest(official), "official authority membership changed")
    relative = Path(authority["member_proofs_file"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "member proof path escapes authority")
    proof_path = (evidence.parent / relative).resolve()
    _require(proof_path.is_relative_to(evidence.parent.resolve()), "member proof symlink escapes authority")
    _require(sha256_file(proof_path) == authority["member_proofs_sha256"], "member proof ledger changed")
    ledger = [json.loads(line) for line in proof_path.read_text(encoding="utf-8").splitlines() if line]
    proofs = {row["video_id"]: row for row in ledger}
    rows = _role_rows(paths, official, lock)
    provider = _provider_members(paths["provider_metadata"])
    _require(len(proofs) == len(ledger) == authority["verified_members"]
             and set(proofs) == set(rows), "member proof coverage differs from source rows")
    for key, row in rows.items():
        proof = proofs[key]
        _require(proof["source_record_sha256"] == _digest(row.to_dict())
                 and proof["path"] == row.path and proof["original_role"] == lock["partitions"][key]
                 and proof["num_frames"] == row.num_frames and proof["fps"] == row.fps
                 and _sha(proof["content_sha256"]) and type(proof["size_bytes"]) is int
                 and proof["size_bytes"] > 0 and proof["size_bytes"] == provider[row.path]["uncompressed_size"]
                 and proof["observed_crc32"] == provider[row.path]["crc32"]
                 and proof["provider_crc_checked"] is True,
                 "member evidence identity/N/FPS/SHA/size/CRC mismatch")
    return rows, proofs, authority, proof_path


def _validated(dataset, paths, evidence_sha256, dataset_root):
    """One validation path for creation and every formal consumer."""
    _require(_sha(evidence_sha256) and sha256_file(paths["evidence"]) == evidence_sha256,
             "independent authority SHA256 missing or changed")
    official = _official(dataset, paths["full_manifest"])
    lock = _role_lock(paths["role_lock"], official)
    role_sha, full_sha = sha256_file(paths["role_lock"]), sha256_file(paths["full_manifest"])
    if dataset == "ucf_crime":
        rows, proofs, authority, proof_path = _ucf_evidence(paths, paths["evidence"], official, lock)
        audited_root = Path(authority["dataset_root"]).resolve()
    else:
        binding_plan = {"inputs": {"original_role_lock_file_sha256": role_sha,
                                   "full_train_manifest_file_sha256": full_sha}}
        bound = _bound_ready_evidence(paths["evidence"], evidence_sha256,
                                      ready_path=paths["ready_manifest"], upstream_path=paths["ready_receipt"],
                                      plan=binding_plan, lock=lock)
        authority, proof_path = bound["authority"], bound["proof_path"]
        audited_root = Path(authority["dataset_root"] if bound["proofs"] is not None
                            else FROZEN_METADATA_AUDIT_ROOTS[evidence_sha256]).resolve()
        full = {row.video_id: row for row in load_manifest_jsonl(paths["full_manifest"])}
        rows, proofs = {}, {}
        for row in bound["ready_records"]:
            _require(row.video_id in official, "ready evidence contains unknown official video")
            normalized = _ready_record(row, full[row.video_id], lock["partitions"][row.video_id], _digest(binding_plan))
            rows[row.video_id] = replace(normalized, metadata=row.metadata)
            proof = {"content_sha256": row.metadata["rawmember_sha256"],
                     "size_bytes": row.metadata["uncompressed_size"],
                     "verified_crc32": row.metadata["verified_crc32"]}
            if bound["proofs"] is not None:
                proof["observed_mtime_ns"] = bound["proofs"][row.video_id]["observed_mtime_ns"]
            proofs[row.video_id] = proof
    root = Path(dataset_root).resolve()
    _require(root == audited_root, "dataset root differs from independent member authority")
    missing, records = [], []
    for key in sorted(official):
        if key not in rows:
            missing.append({"video_id": key, "original_role": lock["partitions"][key],
                            "reason": "authoritative_member_evidence_missing_or_crc_failed"})
            continue
        row, proof = rows[key], proofs[key]
        _row_identity(row, official[key])
        raw = row.resolve_path(root)
        if not raw.is_file():
            missing.append({"video_id": key, "reason": "raw_file_missing"})
            continue
        stat = raw.stat()
        if stat.st_size != proof["size_bytes"]:
            missing.append({"video_id": key, "reason": "raw_size_changed"})
            continue
        if "observed_mtime_ns" in proof and stat.st_mtime_ns != proof["observed_mtime_ns"]:
            missing.append({"video_id": key, "reason": "raw_changed_since_independent_audit"})
            continue
        metadata = {**row.metadata, "original_role": lock["partitions"][key],
                    "original_split": row.split.value, "training_role": VIEW,
                    "content_sha256": proof["content_sha256"], "content_size_bytes": proof["size_bytes"],
                    "official_training": {"view": VIEW, "authority_sha256": evidence_sha256,
                                          "original_role_lock_sha256": role_sha,
                                          "official_source_sha256": full_sha,
                                          "source_record_sha256": _digest(row.to_dict())}}
        records.append(replace(row, split=DatasetSplit.TRAIN, metadata=metadata))
    return tuple(records), missing, official, lock, proof_path


def materialize(
    *, dataset: str, full_manifest: Path, role_lock: Path, evidence: Path,
    evidence_sha256: str, output: Path, dataset_root: Path,
    fit: Path | None = None, confirm: Path | None = None, select: Path | None = None,
    ready_manifest: Path | None = None, ready_receipt: Path | None = None,
    provider_metadata: Path | None = None,
) -> dict[str, Any]:
    """Create one unique complete view, or a blocked inventory without a contract."""
    dataset = _dataset(dataset)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {"full_manifest": Path(full_manifest), "role_lock": Path(role_lock), "evidence": Path(evidence)}
    extras = {"fit": fit, "confirm": confirm, "select": select,
              "provider_metadata": provider_metadata} if dataset == "ucf_crime" else {
        "ready_manifest": ready_manifest, "ready_receipt": ready_receipt}
    inventory = {"dataset": dataset, "training_view": VIEW, "expected_videos": EXPECTED[dataset][0],
                 "dataset_root": str(Path(dataset_root).resolve()), "role_reassignment": False,
                 "name_overlap_exclusions": 0, "authority_sha256": evidence_sha256}
    try:
        _require(all(value is not None for value in extras.values()), "required source manifests are missing")
        paths.update({key: Path(value) for key, value in extras.items()})
        bindings = _file_bindings(paths)
        records, missing, official, lock, proof_path = _validated(dataset, paths, evidence_sha256, dataset_root)
        _require(bindings == _file_bindings(paths), "upstream inputs changed during view validation")
        inventory.update(available_videos=len(records), missing_members=missing,
                         official_identity_sha256=_digest(official),
                         original_role_counts=dict(Counter(lock["partitions"].values())),
                         source_files=bindings)
        if missing:
            inventory.update(state="blocked", reason="official_full_training_members_incomplete")
            atomic_write_json(output / "blocked-inventory.json", inventory)
            atomic_write_json(output / "receipt.json", inventory)
            return inventory
        sources = output / "sources"
        sources.mkdir()
        capsule = {}
        for name, source in paths.items():
            suffix = ".txt" if name == "full_manifest" and dataset == "ucf_crime" else source.suffix
            target = sources / (name + suffix)
            if name == "evidence":
                target = sources / "evidence" / "authority.json"
                target.parent.mkdir()
            shutil.copyfile(source, target)
            _require(sha256_file(target) == bindings[name]["sha256"], "upstream input changed during snapshot")
            capsule[name] = {"path": target.relative_to(output).as_posix(), "sha256": sha256_file(target)}
        if proof_path is not None:
            target = sources / "evidence" / _json(paths["evidence"])["member_proofs_file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(proof_path, target)
            _require(sha256_file(target) == _json(paths["evidence"])["member_proofs_sha256"],
                     "member proof ledger changed during snapshot")
            capsule["member_proofs"] = {"path": target.relative_to(output).as_posix(), "sha256": sha256_file(target)}
        manifest = write_manifest_jsonl(records, output / "official-fulltrain.jsonl")
        contract = {**inventory, "schema_version": 1, "state": "ready", "status": "ready",
                    "official_counts": {"videos": EXPECTED[dataset][0], "normal": EXPECTED[dataset][1], "anomaly": EXPECTED[dataset][2]},
                    "inputs": capsule, "training_manifest": {"path": manifest.name, "file_sha256": sha256_file(manifest)},
                    "member_identity_sha256": _digest([row.to_dict() for row in records]),
                    "raw_validation": "independent content audit; current size and audited mtime checked; frozen legacy XD evidence supplies size"}
        atomic_write_json(output / "contract.json", contract)
        receipt = {**inventory, "state": "ready", "status": "ready", "video_count": len(records),
                   "contract_path": str(output / "contract.json"), "contract_sha256": sha256_file(output / "contract.json"),
                   "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest)}
        atomic_write_json(output / "receipt.json", receipt)
        return receipt
    except (OSError, ValueError, TypeError, KeyError) as error:
        inventory.update(state="blocked", reason=str(error), error_type=type(error).__name__)
        atomic_write_json(output / "blocked-inventory.json", inventory)
        atomic_write_json(output / "receipt.json", inventory)
        return inventory


def load_official_training_view(
    contract_path: str | Path, expected_sha256: str, *, dataset_root: str | Path | None = None,
) -> tuple[tuple[VideoManifestRecord, ...], Mapping[str, Any]]:
    """The sole formal consumer, reusing the creator's independent evidence checks."""
    contract_path = Path(contract_path).resolve()
    _require(_sha(expected_sha256) and sha256_file(contract_path) == expected_sha256,
             "official training contract SHA256 differs from external pin")
    contract = _json(contract_path)
    _require(contract.get("schema_version") == 1 and contract.get("state") == "ready"
             and contract.get("training_view") == VIEW, "official training contract is not ready")
    root = contract_path.parent
    paths, source_hashes = {}, {str(contract_path): expected_sha256}
    for key, item in contract["inputs"].items():
        relative = Path(item["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "source path escapes contract")
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and sha256_file(source) == item["sha256"], "official source snapshot changed")
        paths[key] = source
        source_hashes[str(source)] = item["sha256"]
    data_root = Path(dataset_root or contract["dataset_root"]).resolve()
    records, missing, official, _lock, _proof = _validated(
        contract["dataset"], paths, contract["authority_sha256"], data_root)
    _require(not missing and len(records) == EXPECTED[contract["dataset"]][0], "official training raw inventory is incomplete")
    _require(_digest(official) == contract["official_identity_sha256"], "official membership changed")
    relative = Path(contract["training_manifest"]["path"])
    manifest = (root / relative).resolve()
    _require(not relative.is_absolute() and ".." not in relative.parts and manifest.is_relative_to(root), "training manifest escapes contract")
    _require(sha256_file(manifest) == contract["training_manifest"]["file_sha256"], "official training manifest changed")
    actual = load_manifest_jsonl(manifest)
    expected = [row.to_dict() for row in records]
    _require([row.to_dict() for row in actual] == expected
             and _digest(expected) == contract["member_identity_sha256"], "official training rows differ from independently bound source evidence")
    source_hashes[str(manifest)] = contract["training_manifest"]["file_sha256"]
    return actual, {**contract, "manifest_path": str(manifest),
                    "manifest": {"path": str(manifest), "sha256": contract["training_manifest"]["file_sha256"]},
                    "manifest_sha256": contract["training_manifest"]["file_sha256"],
                    "contract_path": str(contract_path), "contract_sha256": expected_sha256,
                    "source_hashes": source_hashes}
