from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vadbench.data.official_training as official
import vadbench.data.xd_materialization as xd
from vadbench.data.manifest import (
    SupervisionAnnotation,
    VideoManifestRecord,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.hashing import sha256_file


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def record(key, path, anomaly, *, ready=True, frames=96, metadata=None):
    return VideoManifestRecord(
        video_id=key, path=path, split="train", category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly, annotations=(SupervisionAnnotation(scope="video", is_anomaly=anomaly),),
        num_frames=frames if ready else None, fps=24.0 if ready else None,
        duration_seconds=frames / 24 if ready else None, metadata=metadata or {},
    )


@pytest.fixture(scope="module")
def ucf_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("official-ucf")
    data = root / "raw"
    data.mkdir()
    groups = {role: [] for role in official.ROLES}
    members, partitions = [], {}
    rows = []
    for index in range(1610):
        anomaly = index >= 800
        key = ("Abuse" if anomaly else "Normal_Videos") + f"{index:04d}_x264"
        path = ("Abuse/" if anomaly else "Training_Normal_Videos_Anomaly/") + key + ".mp4"
        role = official.ROLES[index % 3]
        row = record(key, path, anomaly, frames=34 if index == 0 else 96)
        groups[role].append(row)
        rows.append(row)
        partitions[key] = role
        raw = data / path
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw")
        members.append({"relative_path": "UCF_Crimes/Videos/" + path,
                        "uncompressed_size": 3, "crc32": f"{zlib.crc32(b'raw'):08x}"})
    full = root / "Anomaly_Train.txt"
    full.write_text("\n".join(r.path for r in rows) + "\n", encoding="utf-8")
    roles = {role: write_manifest_jsonl(group, root / f"{role}.jsonl") for role, group in groups.items()}
    lock = dump(root / "role-lock.json", {"seed": 20260918, "partitions": partitions})
    provider = dump(root / "provider.json", {"video_members": members})
    with pytest.MonkeyPatch.context() as patch:
        frame_counts = {r.video_id: r.num_frames for r in rows}
        patch.setattr(official, "probe_video", lambda p: SimpleNamespace(num_frames=frame_counts[p.stem], fps=24.0))
        audit = official.audit_ucf_official_training(
            full_manifest=full, full_manifest_sha256=sha256_file(full), role_lock=lock,
            role_lock_sha256=sha256_file(lock), **roles, provider_metadata=provider,
            provider_metadata_sha256=sha256_file(provider), dataset_root=data, output=root / "audit",
        )
    assert audit["state"] == "ready"
    return {"root": root, "rows": rows, "roles": roles,
            "kwargs": {"dataset": "ucf", "full_manifest": full, "role_lock": lock,
                       "provider_metadata": provider, **roles, "dataset_root": data,
                       "evidence": Path(audit["evidence"]), "evidence_sha256": audit["evidence_sha256"]}}


@pytest.fixture(scope="module")
def xd_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("official-xd")
    data = root / "raw"
    data.mkdir()
    roles = {role: [] for role in xd.ROLE_COUNTS}
    rows, full, partitions, source_groups = [], [], {}, {}
    label_counts = {role: xd.ROLE_LABELS[role] for role in roles}
    for role, labels in label_counts.items():
        for label, count in labels.items():
            for index in range(count):
                roles[role].append((f"{role}-{label}-{index:04d}", label == "1"))
    official_members = [(role, key, label) for role, group in roles.items() for key, label in group]
    for label, count in ((False, 1242), (True, 1114)):
        official_members.extend(("fit", f"name-overlap-{int(label)}-{index:04d}", label) for index in range(count))
    assert len(official_members) == 3954
    for index, (role, key, label) in enumerate(official_members):
        path = f"videos/{key}.mp4"
        frames = 34 if index == 0 else 96
        metadata = {"source_group": "group-" + key, "provider_crc32": f"{zlib.crc32(b'raw'):08x}",
                    "uncompressed_size": 3}
        original = record(key, path, label, ready=False, metadata=metadata)
        actual = record(key, path, label, frames=frames, metadata={**metadata,
                        "rawmember_sha256": hashlib.sha256(b"raw").hexdigest(),
                        "verified_crc32": metadata["provider_crc32"], "actual_num_frames": frames, "actual_fps": 24.0})
        full.append(original)
        rows.append(actual)
        partitions[key] = role
        source_groups[key] = metadata["source_group"]
        raw = data / path
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw")
    full_path = write_manifest_jsonl(full, root / "full.jsonl")
    ready_path = write_manifest_jsonl(rows, root / "ready.jsonl")
    lock = dump(root / "role-lock.json", {"seed": 202709, "partitions": partitions, "source_groups": source_groups})
    plan = {"derived_train_total": 1598, "role_reassignment": False, "membership_depends_on_availability": False,
            "role_video_ids": {role: [key for key, _ in group] for role, group in roles.items()},
            "inputs": {"original_role_lock_file_sha256": sha256_file(lock),
                       "full_train_manifest_file_sha256": sha256_file(full_path)}}
    plan_path = dump(root / "plan.json", plan)
    scope = {"secondary_encoders": ["videomaev2", "timesformer"], "additional_reducer_calibration_on_xd": False,
             "derived_role_plan_file_sha256": sha256_file(plan_path), "derived_role_plan_canonical_sha256": xd._digest(plan),
             "roles": {role: {"videos": xd.ROLE_COUNTS[role], "normal": xd.ROLE_LABELS[role]["0"],
                              "positive": xd.ROLE_LABELS[role]["1"]} for role in roles}}
    scope_path = dump(root / "scope.json", scope)
    upstream = dump(root / "upstream.json", {"role_lock_sha256": sha256_file(lock), "roles_never_reassigned": True,
                                            "ready_crc_verified": 3954})
    with pytest.MonkeyPatch.context() as patch:
        frame_counts = {r.video_id: r.num_frames for r in rows}
        patch.setattr(xd, "probe_video", lambda p: SimpleNamespace(num_frames=frame_counts[p.stem], fps=24.0))
        audit = xd.audit_xd_ready_manifest(
            plan_path=plan_path, scope_path=scope_path, original_lock_path=lock, full_manifest_path=full_path,
            ready_manifest_path=ready_path, upstream_receipt_path=upstream, dataset_root=data, output_dir=root / "audit",
        )
    return {"root": root, "rows": rows,
            "kwargs": {"dataset": "xd", "full_manifest": full_path, "role_lock": lock,
                       "ready_manifest": ready_path, "ready_receipt": upstream, "dataset_root": data,
                       "evidence": Path(audit["ready_audit_path"]), "evidence_sha256": audit["ready_audit_sha256"]}}


@pytest.mark.parametrize("fixture_name,total", [("ucf_inputs", 1610), ("xd_inputs", 3954)])
def test_complete_official_union_is_ready_and_consumer_revalidates(request, tmp_path, fixture_name, total):
    inputs = request.getfixturevalue(fixture_name)
    before = {str(p): p.read_bytes() for p in inputs["kwargs"].values() if isinstance(p, Path) and p.is_file()}
    result = official.materialize(**inputs["kwargs"], output=tmp_path / "view")
    assert result["state"] == "ready", result
    rows, receipt = official.load_official_training_view(result["contract_path"], result["contract_sha256"])
    assert len(rows) == total
    assert {r.split.value for r in rows} == {"train"}
    assert {r.metadata["original_role"] for r in rows} == set(official.ROLES)
    assert all(r.metadata["original_split"] == "train" for r in rows)
    assert any(r.num_frames == 34 for r in rows)
    assert all(r.metadata["content_size_bytes"] == 3 and len(r.metadata["content_sha256"]) == 64 for r in rows)
    assert receipt["manifest"]["sha256"] == result["manifest_sha256"]
    assert receipt["name_overlap_exclusions"] == 0
    assert {str(p): p.read_bytes() for p in inputs["kwargs"].values() if isinstance(p, Path) and p.is_file()} == before
    if total == 3954:
        assert sum(r.video_id.startswith("name-overlap-") for r in rows) == 2356


@pytest.mark.parametrize("fixture_name", ["ucf_inputs", "xd_inputs"])
def test_missing_raw_member_blocks_whole_view_without_ready_manifest(request, tmp_path, fixture_name):
    inputs = request.getfixturevalue(fixture_name)
    raw = inputs["kwargs"]["dataset_root"] / inputs["rows"][0].path
    backup = raw.with_suffix(".held")
    raw.rename(backup)
    try:
        result = official.materialize(**inputs["kwargs"], output=tmp_path / "blocked")
    finally:
        backup.rename(raw)
    assert result["state"] == "blocked"
    assert result["missing_members"][0]["video_id"] == inputs["rows"][0].video_id
    assert not (tmp_path / "blocked/official-fulltrain.jsonl").exists()
    assert not (tmp_path / "blocked/contract.json").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "role", "metadata", "official_label", "evidence_missing"])
def test_ucf_invalid_inputs_never_publish_training_view(ucf_inputs, tmp_path, mutation):
    kwargs = dict(ucf_inputs["kwargs"])
    if mutation == "evidence_missing":
        kwargs["evidence"] = tmp_path / "missing-authority.json"
    elif mutation == "role":
        lock = json.loads(kwargs["role_lock"].read_text())
        key = ucf_inputs["rows"][0].video_id
        lock["partitions"][key] = "select"
        kwargs["role_lock"] = dump(tmp_path / "changed-lock.json", lock)
    elif mutation == "official_label":
        lines = kwargs["full_manifest"].read_text().splitlines()
        lines[0] = lines[0].replace("Training_Normal_Videos_Anomaly", "Abuse").replace("Normal_Videos", "Abuse")
        kwargs["full_manifest"] = tmp_path / "changed-official.txt"
        kwargs["full_manifest"].write_text("\n".join(lines) + "\n")
    else:
        rows = list(load_manifest_jsonl(kwargs["fit"]))
        if mutation == "duplicate":
            rows.append(load_manifest_jsonl(kwargs["confirm"])[0])
        else:
            rows[0] = replace(rows[0], num_frames=35, duration_seconds=35 / 24)
        kwargs["fit"] = write_manifest_jsonl(rows, tmp_path / "changed-fit.jsonl")
    result = official.materialize(**kwargs, output=tmp_path / "blocked")
    assert result["state"] == "blocked"
    assert not (tmp_path / "blocked/contract.json").exists()
    assert not (tmp_path / "blocked/official-fulltrain.jsonl").exists()


@pytest.mark.parametrize("mutation", ["sha", "fps", "crc", "duplicate"])
def test_xd_ready_metadata_remains_bound_to_independent_authority(xd_inputs, tmp_path, mutation):
    rows = list(xd_inputs["rows"])
    row = rows[0]
    if mutation == "sha":
        rows[0] = replace(row, metadata={**row.metadata, "rawmember_sha256": "0" * 64})
    elif mutation == "fps":
        rows[0] = replace(row, fps=25.0, duration_seconds=row.num_frames / 25,
                          metadata={**row.metadata, "actual_fps": 25.0})
    elif mutation == "crc":
        rows[0] = replace(row, metadata={**row.metadata, "verified_crc32": "00000000"})
    else:
        rows.append(rows[0])
    changed = tmp_path / "changed-ready.jsonl"
    changed.write_text("\n".join(json.dumps(r.to_dict()) for r in rows) + "\n", encoding="utf-8")
    kwargs = {**xd_inputs["kwargs"], "ready_manifest": changed}
    result = official.materialize(**kwargs, output=tmp_path / "blocked")
    assert result["state"] == "blocked"
    assert not (tmp_path / "blocked/contract.json").exists()


QUARANTINE_SCHEMA = "icassp2027.xd-official-quarantine/v1"


def _quarantine_file(root: Path, video_ids) -> Path:
    evidence_dir = root / "quarantine-evidence"
    evidence_dir.mkdir(exist_ok=True)
    items = []
    for video_id in video_ids:
        evidence = evidence_dir / f"{video_id}.json"
        dump(evidence, {"audit": f"two independent audits: official source corrupted: {video_id}"})
        items.append({
            "video_id": video_id,
            "reason": "official_source_corrupted",
            "evidence": {"path": f"quarantine-evidence/{video_id}.json", "sha256": sha256_file(evidence)},
        })
    return dump(root / "quarantine.json", {
        "schema": QUARANTINE_SCHEMA,
        "dataset": "xd_violence",
        "declared_members": 3954,
        "quarantined": items,
        "disclosure_note": "paper must report XD training as 3950/3954 accepted/quarantined",
    })


def _remove_raw(inputs, video_ids):
    held = []
    for video_id in video_ids:
        raw = inputs["kwargs"]["dataset_root"] / next(r.path for r in inputs["rows"] if r.video_id == video_id)
        backup = raw.with_suffix(".held")
        raw.rename(backup)
        held.append((raw, backup))
    return held


def _restore_raw(held):
    for raw, backup in held:
        backup.rename(raw)


def test_xd_quarantine_exemption_materializes_and_consumes(xd_inputs, tmp_path):
    quarantined_ids = [xd_inputs["rows"][index].video_id for index in range(4)]
    quarantine = _quarantine_file(xd_inputs["root"], quarantined_ids)
    held = _remove_raw(xd_inputs, quarantined_ids)
    try:
        result = official.materialize(
            **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "view"
        )
        assert result["state"] == "ready", result
        contract = json.loads(Path(result["contract_path"]).read_text())
        exemption = contract["quarantine_exemption"]
        assert exemption["declared_members"] == 3954
        assert exemption["accepted_members"] == 3950
        assert exemption["quarantined"] == sorted(quarantined_ids)
        assert "3950/3954" in exemption["disclosure_note"]
        assert contract["accepted_counts"]["videos"] == 3950
        assert contract["official_counts"]["videos"] == 3954  # declared, unchanged
        rows, receipt = official.load_official_training_view(result["contract_path"], result["contract_sha256"])
        assert len(rows) == 3950
        assert {row.video_id for row in rows}.isdisjoint(quarantined_ids)
        assert receipt["quarantine_exemption"]["accepted_members"] == 3950
    finally:
        _restore_raw(held)


def test_xd_quarantine_sha_mismatch_and_schema_drift_stay_blocked(xd_inputs, tmp_path):
    quarantined_ids = [xd_inputs["rows"][index].video_id for index in range(4)]
    quarantine = _quarantine_file(xd_inputs["root"], quarantined_ids)
    held = _remove_raw(xd_inputs, quarantined_ids)
    try:
        document = json.loads(quarantine.read_text())
        document["quarantined"][0]["evidence"]["sha256"] = "0" * 64
        dump(quarantine, document)
        blocked = official.materialize(
            **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "blocked-sha"
        )
        assert blocked["state"] == "blocked"
        assert not (tmp_path / "blocked-sha/contract.json").exists()
        # schema drift is equally fail-closed
        _quarantine_file(xd_inputs["root"], quarantined_ids)
        document = json.loads(quarantine.read_text())
        document["schema"] = "icassp2027.xd-official-quarantine/v2"
        dump(quarantine, document)
        blocked = official.materialize(
            **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "blocked-schema"
        )
        assert blocked["state"] == "blocked"
    finally:
        _restore_raw(held)


def test_xd_quarantine_cannot_cover_healthy_or_unknown_members(xd_inputs, tmp_path):
    healthy_id = xd_inputs["rows"][10].video_id  # raw file exists: not missing
    quarantine = _quarantine_file(xd_inputs["root"], [healthy_id])
    blocked = official.materialize(
        **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "blocked-healthy"
    )
    assert blocked["state"] == "blocked"
    quarantined_ids = [xd_inputs["rows"][index].video_id for index in range(4)]
    quarantine = _quarantine_file(xd_inputs["root"], [*quarantined_ids, "unknown-member"])
    held = _remove_raw(xd_inputs, quarantined_ids)
    try:
        blocked = official.materialize(
            **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "blocked-unknown"
        )
        assert blocked["state"] == "blocked"
    finally:
        _restore_raw(held)


def test_xd_quarantine_consumer_rejects_tampered_snapshot(xd_inputs, tmp_path):
    quarantined_ids = [xd_inputs["rows"][index].video_id for index in range(4)]
    quarantine = _quarantine_file(xd_inputs["root"], quarantined_ids)
    held = _remove_raw(xd_inputs, quarantined_ids)
    try:
        result = official.materialize(
            **xd_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "view"
        )
        snapshot = Path(result["contract_path"]).parent / "sources" / "quarantine" / "quarantine.json"
        document = json.loads(snapshot.read_text())
        document["quarantined"] = document["quarantined"][:-1]  # silently drop one member
        dump(snapshot, document)
        with pytest.raises(official.OfficialTrainingError, match="snapshot changed"):
            official.load_official_training_view(result["contract_path"], result["contract_sha256"])
    finally:
        _restore_raw(held)


def test_xd_quarantine_exemption_is_xd_only(ucf_inputs, tmp_path):
    quarantine = _quarantine_file(tmp_path, ["any-member"])
    with pytest.raises(official.OfficialTrainingError, match="XD-only"):
        official.materialize(
            **ucf_inputs["kwargs"], quarantine=quarantine, output=tmp_path / "blocked-ucf"
        )


def test_consumer_rejects_downstream_resigning_and_wrong_contract_pin(ucf_inputs, tmp_path):
    result = official.materialize(**ucf_inputs["kwargs"], output=tmp_path / "view")
    contract = Path(result["contract_path"])
    with pytest.raises(official.OfficialTrainingError, match="external pin"):
        official.load_official_training_view(contract, "0" * 64)
    manifest = Path(result["manifest_path"])
    rows = list(load_manifest_jsonl(manifest))
    rows[0] = replace(rows[0], metadata={**rows[0].metadata, "content_sha256": "0" * 64})
    write_manifest_jsonl(rows, manifest)
    document = json.loads(contract.read_text())
    document["training_manifest"]["file_sha256"] = sha256_file(manifest)
    document["member_identity_sha256"] = xd._digest([r.to_dict() for r in rows])
    dump(contract, document)
    with pytest.raises(official.OfficialTrainingError, match="independently bound"):
        official.load_official_training_view(contract, sha256_file(contract))


@pytest.mark.parametrize("mutation", ["duplicate", "role", "bad_crc"])
def test_ucf_audit_does_not_issue_ready_authority_for_invalid_sources(ucf_inputs, tmp_path, monkeypatch, mutation):
    source = ucf_inputs["kwargs"]
    kwargs = {key: source[key] for key in ("full_manifest", "role_lock", "fit", "confirm", "select", "provider_metadata", "dataset_root")}
    if mutation == "duplicate":
        rows = list(load_manifest_jsonl(kwargs["fit"])) + [load_manifest_jsonl(kwargs["confirm"])[0]]
        kwargs["fit"] = write_manifest_jsonl(rows, tmp_path / "bad-fit.jsonl")
    elif mutation == "role":
        lock = json.loads(kwargs["role_lock"].read_text())
        lock["partitions"][ucf_inputs["rows"][0].video_id] = "confirm"
        kwargs["role_lock"] = dump(tmp_path / "bad-lock.json", lock)
    else:
        provider = json.loads(kwargs["provider_metadata"].read_text())
        provider["video_members"][0]["crc32"] = "00000000"
        kwargs["provider_metadata"] = dump(tmp_path / "bad-provider.json", provider)
    kwargs.update(full_manifest_sha256=sha256_file(kwargs["full_manifest"]),
                  role_lock_sha256=sha256_file(kwargs["role_lock"]),
                  provider_metadata_sha256=sha256_file(kwargs["provider_metadata"]), output=tmp_path / "audit")
    frames = {r.video_id: r.num_frames for r in ucf_inputs["rows"]}
    monkeypatch.setattr(official, "probe_video", lambda p: SimpleNamespace(num_frames=frames[p.stem], fps=24.0))
    if mutation in {"duplicate", "role"}:
        with pytest.raises(official.OfficialTrainingError):
            official.audit_ucf_official_training(**kwargs)
    else:
        result = official.audit_ucf_official_training(**kwargs)
        assert result["state"] == "blocked" and len(result["blocked_members"]) == 1
