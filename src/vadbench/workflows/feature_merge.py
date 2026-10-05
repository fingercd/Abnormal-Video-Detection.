"""Merge several completed extraction runs into one auditable, complete view.

Multi-GPU extraction shards each publish an independent run root (per-video
``shards/<video_id>`` indexes plus a final root index).  This module is the
single safe publisher of a merged view over a target video list:

* every run must be ``completed=true`` — a partial run is rejected outright;
* every target video must appear in exactly one run (gaps and cross-run
  conflicts are both fatal and fully listed);
* representation/sampling/weight identity must agree across runs modulo the
  per-subset source digest and the scheduling-only ``micro_batch_size``;
* every video shard must match its run's published root index and every
  referenced bundle is reopened and checksum-verified before publication.

The merged index is never a raw JSONL concatenation: rows are reparsed,
revalidated and re-serialized, and the root index exists only after every
rewritten reference has been rechecked against the transported bytes.  Any
validation failure aborts the whole merge with a complete problem listing and
removes the unfinished destination, so no half-complete view is observable.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.features import (
    ArrayReference,
    FeatureRecord,
    FeatureStore,
    atomic_write_json,
    atomic_write_jsonl,
    compute_encoder_fingerprint,
    utc_now_iso,
)

MERGE_SCHEMA = "icassp2027.feature-run-merge/v1"
MERGED_VIEW_SCHEMA = "icassp2027.merged-feature-view/v1"
SUBSET_CONTRACT_SCHEMA = "icassp2027.feature-subset-contract/v1"
SUBSET_VIEW_SCHEMA = "icassp2027.feature-subset-view/v1"
VIEW_CONTRACT_SCHEMAS = (MERGE_SCHEMA, SUBSET_CONTRACT_SCHEMA)
VIEW_RESOLVED_SCHEMAS = (MERGED_VIEW_SCHEMA, SUBSET_VIEW_SCHEMA)
ROLE_EQUIVALENCE_FIT = "engineering-shards-of-official-fit"
ROLE_EQUIVALENCE_FULLTRAIN = "engineering-shards-of-official-fulltrain"
ROLE_EQUIVALENCE_VALUES = (ROLE_EQUIVALENCE_FIT, ROLE_EQUIVALENCE_FULLTRAIN)


class FeatureMergeError(ValueError):
    """A merge precondition failed; the message lists every problem found."""


@dataclass(frozen=True)
class FeatureRunMergeRequest:
    """One merge attempt over completed extraction runs.

    ``duplicate_policy="fail"`` (default) rejects any video claimed by more
    than one run. ``duplicate_policy="prefer_run"`` keeps the shard of the
    named run for duplicated videos, but only when both sides' shard bytes are
    identical (``duplicate_identical`` audit verdict); divergent duplicate
    bytes are always a hard error — choosing between different bytes is never
    silent.
    """

    target_video_ids: tuple[str, ...]
    run_roots: tuple[str, ...]
    output_root: str
    run_id: str | None = None
    transport: Literal["copy", "hardlink_npz"] = "hardlink_npz"
    duplicate_policy: Literal["fail", "prefer_run"] = "fail"
    prefer_run: str | None = None
    allow_partial: tuple[str, ...] = ()
    role_equivalence: str | None = None
    authority_contract_path: str | None = None
    authority_contract_sha256: str | None = None
    original_role_lock_path: str | None = None
    original_role_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.target_video_ids
            or any(not isinstance(video, str) or not video for video in self.target_video_ids)
            or len(set(self.target_video_ids)) != len(self.target_video_ids)
        ):
            raise ValueError("target_video_ids must be a nonempty list of unique video IDs")
        if not self.run_roots or len(set(self.run_roots)) != len(self.run_roots):
            raise ValueError("run_roots must contain at least one unique run root")
        if self.transport not in {"copy", "hardlink_npz"}:
            raise ValueError("transport must be copy or hardlink_npz")
        if self.duplicate_policy not in {"fail", "prefer_run"}:
            raise ValueError("duplicate_policy must be fail or prefer_run")
        if self.duplicate_policy == "prefer_run":
            if not self.prefer_run:
                raise ValueError("prefer_run duplicate policy requires an explicit prefer_run root")
        elif self.prefer_run is not None:
            raise ValueError("prefer_run is only meaningful with duplicate_policy=prefer_run")
        if len(set(self.allow_partial)) != len(self.allow_partial):
            raise ValueError("allow_partial must contain unique run roots")
        if self.role_equivalence is not None:
            if self.role_equivalence not in ROLE_EQUIVALENCE_VALUES:
                raise ValueError("unsupported role_equivalence declaration")
            if not (self.authority_contract_path and self.authority_contract_sha256):
                raise ValueError("role_equivalence requires the authority training view contract binding")
        if (self.authority_contract_path is None) != (self.authority_contract_sha256 is None):
            raise ValueError("authority contract path and SHA-256 must be provided together")
        if self.authority_contract_path is not None and self.role_equivalence is None:
            raise ValueError("authority contract binding requires an explicit role_equivalence declaration")
        if self.original_role_lock_path is not None and self.original_role_lock_sha256 is None:
            raise ValueError(
                "original role lock path requires its externally bound SHA-256 "
                "(SHA alone resolves the default sources/role_lock.json next to the authority contract)"
            )
        if self.original_role_lock_sha256 is not None and self.role_equivalence is None:
            raise ValueError("original role lock is only meaningful with a role_equivalence declaration")
        run_id = self.run_id
        if run_id is not None and (
            not run_id or run_id in {".", ".."} or any(char in run_id for char in "/\\")
        ):
            raise ValueError("run_id must be a basename")


@dataclass(frozen=True)
class FeatureSubsetRequest:
    """Build an explicit subset view from one completed extraction run.

    The subset view is not a native extraction run; it exists so v0/debug
    consumers can read a verified, integrity-checked slice without re-running
    an encoder.
    """

    source_run_root: str
    target_video_ids: tuple[str, ...]
    output_root: str
    run_id: str | None = None
    transport: Literal["copy", "hardlink_npz"] = "hardlink_npz"

    def __post_init__(self) -> None:
        if (
            not self.target_video_ids
            or any(not isinstance(video, str) or not video for video in self.target_video_ids)
            or len(set(self.target_video_ids)) != len(self.target_video_ids)
        ):
            raise ValueError("target_video_ids must be a nonempty list of unique video IDs")
        if not self.source_run_root:
            raise ValueError("source_run_root is required")
        if self.transport not in {"copy", "hardlink_npz"}:
            raise ValueError("transport must be copy or hardlink_npz")
        run_id = self.run_id
        if run_id is not None and (
            not run_id or run_id in {".", ".."} or any(char in run_id for char in "/\\")
        ):
            raise ValueError("run_id must be a basename")


@dataclass(frozen=True)
class _RunView:
    root: Path
    resolved: Mapping[str, Any]
    resolved_sha256: str
    status_sha256: str
    index_sha256: str | None
    rows: tuple[FeatureRecord, ...]
    fingerprint: str
    paper_identity: Mapping[str, Any]
    spec_identity: Mapping[str, Any]
    videos: frozenset[str]
    rescued_partial: bool = False
    partial_failures: tuple[Any, ...] = ()


def _json_object(path: Path, problems: list[str], label: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        problems.append(f"{label}: missing file {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{label}: unreadable JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        problems.append(f"{label}: {path} must contain a JSON object")
        return None
    return value


def _spec_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical run identity modulo scheduling batch size and subset digest."""

    identity = {key: value for key, value in spec.items() if key != "micro_batch_size"}
    sampling = dict(identity.get("sampling") or {})
    sampling.pop("source_digest", None)
    identity["sampling"] = sampling
    return json.loads(
        json.dumps(identity, ensure_ascii=False, allow_nan=False, sort_keys=True)
    )


def _video_shard_rows(run: Path, video_id: str) -> list[FeatureRecord] | None:
    shard = run / "shards" / video_id
    index = shard / "index.jsonl"
    if not index.is_file():
        return None
    try:
        store = FeatureStore(shard)
        return list(store.iter_records())
    except ValueError:
        return None


def _validate_video_rows(
    store: FeatureStore,
    fingerprint: str,
    paper_identity: Mapping[str, Any],
    rows: list[FeatureRecord],
) -> str | None:
    """Return an error message unless these rows form one fully valid video."""

    ordered = sorted(rows, key=lambda item: item.clip_index)
    if [item.clip_index for item in ordered] != list(range(len(ordered))):
        return "gapped clip indices"
    if len({item.clip_id for item in ordered}) != len(ordered):
        return "duplicate clip IDs"
    for row in ordered:
        if row.encoder_fingerprint != fingerprint:
            return f"row {row.clip_id} has a foreign encoder fingerprint"
        if row.metadata.get("paper_identity") != paper_identity:
            return f"row {row.clip_id} paper identity differs from resolved.json"
        try:
            bundle = store.load_bundle(row)
        except (OSError, ValueError) as exc:
            return f"{row.clip_id} bundle integrity check failed: {exc}"
        for name, array in bundle.items():
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                return f"{row.clip_id} array {name} is not finite float"
    return None


def _load_run(
    root_value: str, problems: list[str], *, allow_partial: bool = False
) -> _RunView | None:
    root = Path(root_value).expanduser().resolve()
    label = f"run {root}"
    resolved = _json_object(root / "resolved.json", problems, label)
    status = _json_object(root / "status.json", problems, label)
    if resolved is None or status is None:
        return None
    completed = status.get("status") == "completed" and status.get("completed") is True
    if not completed and not allow_partial:
        problems.append(
            f"{label}: run is not complete (status={status.get('status')!r}); "
            "partial runs are never merged (pass the run via allow_partial to "
            "rescue its complete per-video shards explicitly)"
        )
        return None
    if completed:
        if status.get("failures") not in (None, [], ()):
            problems.append(f"{label}: completed status carries failures: {status.get('failures')}")
            return None
        if not status.get("feature_root") or Path(status["feature_root"]).expanduser().resolve() != root:
            problems.append(f"{label}: status feature_root does not name this run root")
            return None
    fingerprint = resolved.get("encoder_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        problems.append(f"{label}: resolved.json lacks an encoder fingerprint")
        return None
    if completed and status.get("encoder_fingerprint") != fingerprint:
        problems.append(f"{label}: resolved/status encoder fingerprints differ")
        return None
    paper_identity = resolved.get("paper_identity")
    if not isinstance(paper_identity, dict):
        problems.append(f"{label}: resolved.json lacks paper_identity")
        return None
    spec = resolved.get("spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("sampling"), dict):
        problems.append(f"{label}: resolved.json lacks a complete extraction spec")
        return None

    store = FeatureStore(root)
    if not completed:
        # Partial rescue: only per-video shards that validate completely
        # (contiguous clips, run identity, every bundle checksum + finite)
        # contribute; everything else is ignored here and must be covered by
        # another run or the merge fails on the gap.
        shards_root = root / "shards"
        rescued: dict[str, list[FeatureRecord]] = {}
        rows: list[FeatureRecord] = []
        if shards_root.is_dir():
            for shard_dir in sorted(p for p in shards_root.iterdir() if p.is_dir()):
                video_id = shard_dir.name
                shard_rows = _video_shard_rows(root, video_id)
                if not shard_rows:
                    continue
                # Shard rows are relative to the shard store; validate there.
                if _validate_video_rows(
                    FeatureStore(shard_dir), fingerprint, paper_identity, shard_rows
                ) is not None:
                    continue
                prefix = shard_dir.relative_to(root).as_posix()
                rewritten = [
                    FeatureRecord(
                        video_id=row.video_id,
                        clip_id=row.clip_id,
                        clip_index=row.clip_index,
                        encoder_fingerprint=row.encoder_fingerprint,
                        storage_format=row.storage_format,
                        arrays={
                            name: ArrayReference(
                                path=f"{prefix}/{reference.path}",
                                key=reference.key,
                                shape=reference.shape,
                                dtype=reference.dtype,
                                sha256=reference.sha256,
                                nbytes=reference.nbytes,
                            )
                            for name, reference in row.arrays.items()
                        },
                        start_s=row.start_s,
                        end_s=row.end_s,
                        frame_start=row.frame_start,
                        frame_end=row.frame_end,
                        metadata=row.metadata,
                        created_at=row.created_at,
                    )
                    for row in shard_rows
                ]
                rescued[video_id] = rewritten
                rows.extend(rewritten)
        if not rows:
            problems.append(f"{label}: partial rescue found no complete valid video shards")
            return None
        return _RunView(
            root=root,
            resolved=resolved,
            resolved_sha256=sha256_file(root / "resolved.json"),
            status_sha256=sha256_file(root / "status.json"),
            index_sha256=None,
            rows=tuple(rows),
            fingerprint=fingerprint,
            paper_identity=paper_identity,
            spec_identity=_spec_identity(spec),
            videos=frozenset(rescued),
            rescued_partial=True,
            partial_failures=tuple(status.get("failures") or ()),
        )

    index_path = root / "index.jsonl"
    if not index_path.is_file():
        problems.append(f"{label}: missing final index.jsonl (run never published)")
        return None
    try:
        rows = list(FeatureStore(root).iter_records())
    except ValueError as exc:
        problems.append(f"{label}: invalid final index: {exc}")
        return None
    if not rows:
        problems.append(f"{label}: final index is empty")
        return None
    if status.get("records_written_to_shards") != len(rows):
        problems.append(f"{label}: record count differs from the completed status receipt")
        return None

    rows_by_video: dict[str, list[FeatureRecord]] = {}
    for row in rows:
        if row.encoder_fingerprint != fingerprint:
            problems.append(f"{label}: row {row.clip_id} has a foreign encoder fingerprint")
            return None
        if row.metadata.get("paper_identity") != paper_identity:
            problems.append(f"{label}: row {row.clip_id} paper identity differs from resolved.json")
            return None
        rows_by_video.setdefault(row.video_id, []).append(row)

    for video_id, video_rows in sorted(rows_by_video.items()):
        ordered = sorted(video_rows, key=lambda item: item.clip_index)
        shard_rows = _video_shard_rows(root, video_id)
        if shard_rows is None:
            problems.append(f"{label}: {video_id} has no per-video shard index")
            return None
        shard_keys = sorted((item.clip_id, item.clip_index) for item in shard_rows)
        if shard_keys != sorted((item.clip_id, item.clip_index) for item in ordered):
            problems.append(
                f"{label}: {video_id} shard index does not match the published root index"
            )
            return None
        error = _validate_video_rows(store, fingerprint, paper_identity, ordered)
        if error is not None:
            problems.append(f"{label}: {video_id} {error}")
            return None

    return _RunView(
        root=root,
        resolved=resolved,
        resolved_sha256=sha256_file(root / "resolved.json"),
        status_sha256=sha256_file(root / "status.json"),
        index_sha256=sha256_file(index_path),
        rows=tuple(rows),
        fingerprint=fingerprint,
        paper_identity=paper_identity,
        spec_identity=_spec_identity(spec),
        videos=frozenset(rows_by_video),
    )


def _merged_view_digest(
    spec_identity: Mapping[str, Any], runs: list[_RunView]
) -> str:
    """Content digest binding the merged view to its source run receipts."""

    sources = [
        {"resolved_sha256": run.resolved_sha256, "encoder_fingerprint": run.fingerprint}
        for run in sorted(runs, key=lambda item: item.resolved_sha256)
    ]
    return compute_encoder_fingerprint(
        {"icassp2027.merged-feature-view": {"canonical_spec": dict(spec_identity), "source_runs": sources}}
    )


def _copy_shard(shard: Path, destination: Path) -> None:
    shutil.copytree(shard, destination, symlinks=False)
    for source in sorted(path for path in shard.rglob("*") if path.is_file()):
        relative = source.relative_to(shard)
        if sha256_file(source) != sha256_file(destination / relative):
            raise FeatureMergeError(f"copied shard bytes differ from the source: {relative}")


def _hardlink_shard(run: Path, shard: Path, destination: Path, destination_root: Path) -> None:
    from .feature_resume import _blob_state, _same_blob, _unaliased_destination

    for source in sorted(path for path in shard.rglob("*") if path.is_file()):
        relative = source.relative_to(shard)
        target = _unaliased_destination(destination / relative)
        if not target.is_relative_to(destination_root):
            raise FeatureMergeError(f"hardlink target escapes the merged view: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        before = _blob_state(source, run)
        if before["device"] != destination_root.stat().st_dev:
            raise FeatureMergeError(
                f"hardlink_npz source and destination must share a filesystem: {source}"
            )
        # os.link never overwrites an existing target; copying is not a fallback.
        os.link(source, target, follow_symlinks=False)
        after = _blob_state(source, run)
        linked = _blob_state(target, destination_root)
        if (
            not _same_blob(before, after)
            or not _same_blob(before, linked)
            or after["nlink"] != before["nlink"] + 1
            or linked["nlink"] != after["nlink"]
        ):
            raise FeatureMergeError(f"hardlink_npz post-link inode verification failed: {source}")


def _transport_shard(
    transport: str, run: _RunView, video_id: str, destination: Path, destination_root: Path
) -> None:
    shard = run.root / "shards" / video_id
    if transport == "copy":
        _copy_shard(shard, destination)
    else:
        _hardlink_shard(run.root, shard, destination, destination_root)


def view_identity_digest(identity: Mapping[str, Any]) -> str:
    """Return the content digest recorded by a merged or subset view identity."""

    for key in ("merged_view_digest", "subset_view_digest"):
        value = identity.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("feature view identity lacks its view digest")


def _rows_bytes_identical(kept: _RunView, dropped: _RunView, video_id: str) -> bool:
    """True only when both runs' shard bytes for one video fully agree."""

    def signature(run: _RunView) -> list[tuple[Any, ...]]:
        rows = sorted(
            (row for row in run.rows if row.video_id == video_id),
            key=lambda item: item.clip_index,
        )
        return [
            (
                row.clip_id,
                row.clip_index,
                row.frame_start,
                row.frame_end,
                row.start_s,
                row.end_s,
                sorted(
                    (name, reference.sha256, reference.shape, reference.dtype)
                    for name, reference in row.arrays.items()
                ),
            )
            for row in rows
        ]

    if signature(kept) != signature(dropped):
        return False
    kept_content = {
        item.get("video_id"): item
        for item in kept.resolved.get("data_content_evidence", {}).get("videos", [])
        if isinstance(item, dict)
    }.get(video_id)
    dropped_content = {
        item.get("video_id"): item
        for item in dropped.resolved.get("data_content_evidence", {}).get("videos", [])
        if isinstance(item, dict)
    }.get(video_id)
    if kept_content is None or dropped_content is None:
        return True
    return kept_content.get("sha256") == dropped_content.get("sha256") and kept_content.get(
        "size_bytes"
    ) == dropped_content.get("size_bytes")


def _video_index_sha256(run: _RunView, video_id: str) -> str:
    return sha256_file(run.root / "shards" / video_id / "index.jsonl")


def _preflight_role_equivalence(request: FeatureRunMergeRequest) -> tuple[Path, Path | None] | None:
    """Validate authority and role-lock bindings before opening source runs."""

    if request.role_equivalence is None:
        return None
    authority = Path(request.authority_contract_path).expanduser().resolve()
    if not authority.is_file() or sha256_file(authority) != request.authority_contract_sha256:
        raise FeatureMergeError(
            f"role equivalence authority contract SHA differs: {authority}"
        )
    lock_path = (
        None
        if request.original_role_lock_path is None
        else Path(request.original_role_lock_path).expanduser().resolve()
    )
    if request.original_role_lock_path is None:
        default_lock = authority.parent / "sources" / "role_lock.json"
        if default_lock.is_file():
            lock_path = default_lock.resolve()
    if lock_path is None:
        return authority, None
    if not lock_path.is_file():
        raise FeatureMergeError(f"original role lock is missing: {lock_path}")
    if request.original_role_lock_sha256 is None:
        raise FeatureMergeError(
            "original role lock SHA-256 must be supplied before merging source runs"
        )
    if sha256_file(lock_path) != request.original_role_lock_sha256:
        raise FeatureMergeError(
            f"original role lock SHA-256 differs from its external binding: {lock_path}"
        )
    return authority, lock_path


def run_feature_merge(request: FeatureRunMergeRequest) -> dict[str, Any]:
    """Validate every run, then publish one complete merged FeatureStore view."""

    if not isinstance(request, FeatureRunMergeRequest):
        raise TypeError("request must be a FeatureRunMergeRequest")
    role_binding = _preflight_role_equivalence(request)
    target = set(request.target_video_ids)
    problems: list[str] = []
    partial_roots = {Path(value).expanduser().resolve() for value in request.allow_partial}
    run_roots = {Path(value).expanduser().resolve() for value in request.run_roots}
    unknown_partial = sorted(str(path) for path in partial_roots - run_roots)
    if unknown_partial:
        raise FeatureMergeError(
            "allow_partial entries must also be listed in run_roots: " + "; ".join(unknown_partial)
        )
    runs = [
        view
        for value in request.run_roots
        if (
            view := _load_run(
                value,
                problems,
                allow_partial=Path(value).expanduser().resolve() in partial_roots,
            )
        )
        is not None
    ]

    equivalence_fields: dict[str, Any] = {"role_equivalence": None, "original_role_lock": None}
    if role_binding is not None:
        authority, lock_path = role_binding
        equivalence_fields = {
            "role_equivalence": {
                "declared": request.role_equivalence,
                "authority_contract": {
                    "path": str(authority),
                    "sha256": request.authority_contract_sha256,
                },
                "target_video_ids": sorted(target),
                "note": (
                    "Declared equivalence is only a label: formal/development consumers "
                    "re-verify coverage against the authoritative training view, identity "
                    "against every constituent run receipt, and all SHA bindings."
                ),
            },
            "original_role_lock": None,
        }
        if lock_path is not None:
            equivalence_fields["original_role_lock"] = {
                "path": str(lock_path),
                "sha256": request.original_role_lock_sha256,
            }

    identities: dict[str, list[str]] = {}
    for run in runs:
        identities.setdefault(
            json.dumps(run.spec_identity, ensure_ascii=False, sort_keys=True), []
        ).append(str(run.root))
    if len(identities) > 1:
        problems.append(
            "run identities differ (representation/sampling/weights): "
            + "; ".join(
                f"{digest[:16]}… -> {roots}" for digest, roots in sorted(identities.items())
            )
        )

    owners: dict[str, list[_RunView]] = {video_id: [] for video_id in target}
    extras: dict[str, list[str]] = {}
    for run in runs:
        for video_id in sorted(run.videos):
            if video_id in owners:
                owners[video_id].append(run)
            else:
                extras.setdefault(str(run.root), []).append(video_id)
    conflicts = {video: views for video, views in owners.items() if len(views) > 1}
    missing = sorted(video for video, views in owners.items() if not views)
    duplicate_audit: dict[str, dict[str, Any]] = {}
    owner_by_video: dict[str, _RunView] = {}
    if conflicts and request.duplicate_policy == "fail":
        problems.append(
            "target videos claimed by more than one run: "
            + "; ".join(
                f"{video} -> {[str(view.root) for view in views]}"
                for video, views in sorted(conflicts.items())
            )
        )
    elif conflicts:
        preferred = Path(request.prefer_run).expanduser().resolve()
        preferred_run = next((run for run in runs if run.root == preferred), None)
        if preferred_run is None:
            problems.append(f"prefer_run is not one of the merge run roots: {preferred}")
        for video, views in sorted(conflicts.items()):
            kept = next((view for view in views if view.root == preferred), None)
            if kept is None:
                problems.append(
                    f"prefer_run does not cover duplicated video {video}: "
                    f"{[str(view.root) for view in views]}"
                )
                continue
            for dropped in views:
                if dropped is kept:
                    continue
                if not _rows_bytes_identical(kept, dropped, video):
                    problems.append(
                        f"duplicated video {video} bytes differ between {kept.root} and "
                        f"{dropped.root}; prefer_run only resolves byte-identical duplicates"
                    )
                    continue
                duplicate_audit[video] = {
                    "verdict": "duplicate_identical",
                    "kept_run": str(kept.root),
                    "dropped_run": str(dropped.root),
                    "kept_index_sha256": _video_index_sha256(kept, video),
                    "dropped_index_sha256": _video_index_sha256(dropped, video),
                    "arrays_sha256_match": True,
                }
    if missing:
        problems.append(f"target videos missing from every run: {missing}")
    if extras:
        problems.append(
            "runs contain videos outside the target list: "
            + "; ".join(f"{root} -> {videos}" for root, videos in sorted(extras.items()))
        )
    if problems:
        raise FeatureMergeError(
            "feature run merge rejected:\n  - " + "\n  - ".join(problems)
        )

    for video, views in owners.items():
        owner_by_video[video] = (
            next(run for run in views if str(run.root) == duplicate_audit[video]["kept_run"])
            if video in duplicate_audit
            else views[0]
        )

    spec_identity = runs[0].spec_identity
    merged_digest = _merged_view_digest(spec_identity, runs)
    canonical_spec = json.loads(json.dumps(spec_identity, ensure_ascii=False, sort_keys=True))
    canonical_spec["sampling"]["source_digest"] = merged_digest
    partial_rescue = {
        video_id: {
            "source_run": str(owner_by_video[video_id].root),
            "source_status": "partial_failed",
            "failures": [str(item) for item in owner_by_video[video_id].partial_failures],
        }
        for video_id in sorted(target)
        if owner_by_video[video_id].rescued_partial
    }
    merged = sorted(
        (row for run in runs for row in run.rows if owner_by_video[row.video_id] is run),
        key=lambda item: (
            item.encoder_fingerprint,
            item.video_id,
            item.clip_index,
            item.clip_id,
        ),
    )
    destination = _publish_view(
        output_root=request.output_root,
        run_id=request.run_id or new_run_id("feature-merge"),
        runs=runs,
        owner_by_video=owner_by_video,
        target=target,
        rows=merged,
        canonical_spec=canonical_spec,
        view_digest=merged_digest,
        digest_field="merged_view_digest",
        transport=request.transport,
        view_schema=MERGED_VIEW_SCHEMA,
        contract_schema=MERGE_SCHEMA,
        contract_filename="merge-contract.json",
        view_kind="merged_view",
        note=(
            "Merged multi-run feature view. This is not a native extraction run: "
            "consume it through the merge contract or the explicit v0 diagnostic "
            "interfaces, never as a formal/development dense extraction."
        ),
        extra_contract_fields={
            "duplicate_policy": request.duplicate_policy,
            "duplicate_audit": duplicate_audit,
            "partial_rescue": partial_rescue,
            **equivalence_fields,
        },
    )
    contract_path = destination / "merge-contract.json"
    return {
        "status": "completed",
        "run_dir": str(destination),
        "merged_view": True,
        "complete": True,
        "videos": len(target),
        "clips": len(merged),
        "transport": request.transport,
        "duplicate_policy": request.duplicate_policy,
        "duplicate_audit": duplicate_audit,
        "merged_view_digest": merged_digest,
        "encoder_fingerprints": sorted({run.fingerprint for run in runs}),
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
    }


def patch_view_contract(
    *,
    view_root: str | Path,
    output_name: str = "merge-contract.v2.json",
    original_role_lock_path: str | Path | None = None,
    original_role_lock_sha256: str | None = None,
    role_equivalence: str | None = None,
    authority_contract_path: str | Path | None = None,
    authority_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish a patched copy of an existing merge contract; never rewrites it.

    Two narrow, independently usable patches:

    * add ``original_role_lock`` (development-over-merged-view training);
    * correct the ``role_equivalence.declared`` value (e.g. a full-1610 view
      published with the fit label before the fulltrain declaration value
      existed) — requires the fulltrain value, an externally bound authority
      contract, and records ``role_equivalence_before`` in ``patched_from``.

    The original file is left byte-identical; the new file records
    ``patched_from`` with the original's SHA-256, and consumers bind the new
    file by its own path+SHA.
    """

    root = Path(view_root).expanduser().resolve()
    resolved_receipt = json.loads((root / "resolved.json").read_text(encoding="utf-8"))
    if resolved_receipt.get("schema") != MERGED_VIEW_SCHEMA:
        raise FeatureMergeError(
            f"contract patch supports merged views only: {root}"
        )
    contract_path = root / "merge-contract.json"
    if not contract_path.is_file():
        raise FeatureMergeError(f"merge contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != MERGE_SCHEMA
        or contract.get("status") != "ready"
        or contract.get("complete") is not True
    ):
        raise FeatureMergeError("merge contract is not a ready complete merge contract")
    declared = (contract.get("role_equivalence") or {}).get("declared")
    if declared not in ROLE_EQUIVALENCE_VALUES:
        raise FeatureMergeError(
            "contract patch refuses views without a recognized role-equivalence declaration"
        )
    if Path(output_name).name != output_name or output_name in {".", ".."}:
        raise FeatureMergeError("output_name must be a basename")
    output_path = root / output_name
    if output_path == contract_path or output_path.exists():
        raise FeatureMergeError(
            f"contract patch refuses to overwrite an existing contract: {output_path}"
        )
    if (original_role_lock_path is None) != (original_role_lock_sha256 is None):
        raise FeatureMergeError("original role lock path and SHA-256 must be provided together")
    if role_equivalence is not None and role_equivalence != ROLE_EQUIVALENCE_FULLTRAIN:
        raise FeatureMergeError("contract patch only supports the fulltrain role-equivalence target")
    if role_equivalence is not None and declared == ROLE_EQUIVALENCE_FULLTRAIN:
        raise FeatureMergeError("contract already declares the fulltrain role equivalence")
    if (authority_contract_path is None) != (authority_contract_sha256 is None):
        raise FeatureMergeError("authority contract path and SHA-256 must be provided together")
    if role_equivalence is not None and authority_contract_path is None:
        raise FeatureMergeError("role-equivalence patch requires the externally bound authority contract")
    if original_role_lock_path is None and role_equivalence is None:
        raise FeatureMergeError(
            "contract patch requires at least one patch (original role lock or role equivalence)"
        )
    patched: dict[str, Any] = {**contract}
    if original_role_lock_path is not None:
        lock_path = Path(original_role_lock_path).expanduser().resolve()
        if not lock_path.is_file():
            raise FeatureMergeError(f"original role lock is missing: {lock_path}")
        if sha256_file(lock_path) != original_role_lock_sha256:
            raise FeatureMergeError(
                f"original role lock SHA-256 differs from its external binding: {lock_path}"
            )
        patched["original_role_lock"] = {
            "path": str(lock_path),
            "sha256": original_role_lock_sha256,
        }
    if role_equivalence is not None:
        authority = Path(authority_contract_path).expanduser().resolve()
        if not authority.is_file() or sha256_file(authority) != authority_contract_sha256:
            raise FeatureMergeError(f"role-equivalence authority contract SHA differs: {authority}")
        equivalence = dict(contract["role_equivalence"])
        equivalence["declared"] = ROLE_EQUIVALENCE_FULLTRAIN
        equivalence["authority_contract"] = {
            "path": str(authority),
            "sha256": authority_contract_sha256,
        }
        patched["role_equivalence"] = equivalence
    patched["patched_from"] = {
        "path": str(contract_path),
        "sha256": sha256_file(contract_path),
        "reason": (
            "add original_role_lock for development-over-merged-view training"
            if role_equivalence is None
            else "correct role_equivalence declaration to engineering-shards-of-official-fulltrain"
        ),
        **({"role_equivalence_before": declared} if role_equivalence is not None else {}),
    }
    atomic_write_json(output_path, patched)
    return {
        "status": "completed",
        "contract": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "patched_from": dict(patched["patched_from"]),
        "original_role_lock": dict(patched.get("original_role_lock") or {}),
        "role_equivalence": dict(patched["role_equivalence"]),
    }


def run_feature_subset(request: FeatureSubsetRequest) -> dict[str, Any]:
    """Publish an integrity-checked subset view of one completed run."""

    if not isinstance(request, FeatureSubsetRequest):
        raise TypeError("request must be a FeatureSubsetRequest")
    target = set(request.target_video_ids)
    problems: list[str] = []
    run = _load_run(request.source_run_root, problems)
    if run is None:
        raise FeatureMergeError(
            "feature subset rejected:\n  - " + "\n  - ".join(problems)
        )
    missing = sorted(target - run.videos)
    if missing:
        raise FeatureMergeError(
            f"feature subset rejected: videos missing from the source run: {missing}"
        )
    canonical_spec = json.loads(
        json.dumps(run.spec_identity, ensure_ascii=False, sort_keys=True)
    )
    subset_digest = compute_encoder_fingerprint(
        {
            "icassp2027.feature-subset-view": {
                "canonical_spec": canonical_spec,
                "source_run": {
                    "resolved_sha256": run.resolved_sha256,
                    "encoder_fingerprint": run.fingerprint,
                },
                "target_video_ids": sorted(target),
            }
        }
    )
    canonical_spec["sampling"]["source_digest"] = subset_digest
    owner_by_video = {video: run for video in target}
    rows = sorted(
        (row for row in run.rows if row.video_id in target),
        key=lambda item: (
            item.encoder_fingerprint,
            item.video_id,
            item.clip_index,
            item.clip_id,
        ),
    )
    destination = _publish_view(
        output_root=request.output_root,
        run_id=request.run_id or new_run_id("feature-subset"),
        runs=[run],
        owner_by_video=owner_by_video,
        target=target,
        rows=rows,
        canonical_spec=canonical_spec,
        view_digest=subset_digest,
        digest_field="subset_view_digest",
        transport=request.transport,
        view_schema=SUBSET_VIEW_SCHEMA,
        contract_schema=SUBSET_CONTRACT_SCHEMA,
        contract_filename="subset-contract.json",
        view_kind="subset_view",
        note=(
            "Subset view of one completed extraction run. This is not a native "
            "extraction run: consume it through the subset contract or the v0 "
            "diagnostic interfaces."
        ),
        extra_contract_fields={
            "subset_of": {
                "root": str(run.root),
                "resolved_sha256": run.resolved_sha256,
                "index_sha256": run.index_sha256,
                "encoder_fingerprint": run.fingerprint,
            }
        },
    )
    contract_path = destination / "subset-contract.json"
    return {
        "status": "completed",
        "run_dir": str(destination),
        "subset_view": True,
        "complete": True,
        "videos": len(target),
        "clips": len(rows),
        "transport": request.transport,
        "subset_view_digest": subset_digest,
        "encoder_fingerprints": [run.fingerprint],
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
    }


def _publish_view(
    *,
    output_root: str,
    run_id: str,
    runs: list[_RunView],
    owner_by_video: Mapping[str, _RunView],
    target: set[str],
    rows: list[FeatureRecord],
    canonical_spec: Mapping[str, Any],
    view_digest: str,
    digest_field: str,
    transport: str,
    view_schema: str,
    contract_schema: str,
    contract_filename: str,
    view_kind: str,
    note: str,
    extra_contract_fields: Mapping[str, Any],
) -> Path:
    """Transport shards, re-verify every bundle, and publish a non-native view."""

    output_root_path = Path(output_root).expanduser().resolve()
    destination = output_root_path / run_id
    if transport == "hardlink_npz":
        from .feature_resume import _hardlink_destination_root

        destination = _hardlink_destination_root(destination, runs[0].root)
        for run in runs[1:]:
            _hardlink_destination_root(destination, run.root)
    for run in runs:
        if run.root == destination or destination.is_relative_to(run.root):
            raise FeatureMergeError(
                f"published view must not live inside a source run: {destination}"
            )
    if destination.exists():
        raise FileExistsError(f"feature view already exists: {destination}")
    output_root_path.mkdir(parents=True, exist_ok=True)

    try:
        destination.mkdir()
        for video_id in sorted(target):
            run = owner_by_video[video_id]
            _transport_shard(
                transport, run, video_id, destination / "shards" / video_id, destination
            )

        # Reopen every transported bundle through the new store before the
        # final index exists; consumers can never observe an unchecked view.
        view_store = FeatureStore(destination)
        for row in rows:
            try:
                view_store.load_bundle(row)
            except (OSError, ValueError) as exc:
                raise FeatureMergeError(
                    f"view bundle integrity check failed: {row.clip_id}: {exc}"
                ) from exc
        atomic_write_jsonl(destination / "index.jsonl", (row.to_dict() for row in rows))

        source_runs = [
            {
                "root": str(run.root),
                "resolved_sha256": run.resolved_sha256,
                "status_sha256": run.status_sha256,
                "index_sha256": run.index_sha256,
                "encoder_fingerprint": run.fingerprint,
                "videos": sorted(run.videos & target),
            }
            for run in sorted(runs, key=lambda item: str(item.root))
        ]
        evidence_videos = {
            item["video_id"]: item
            for run in runs
            for item in run.resolved.get("data_content_evidence", {}).get("videos", [])
            if isinstance(item, dict) and item.get("video_id") in target
        }
        video_provenance = {
            video_id: {
                "source_run": str(owner_by_video[video_id].root),
                "source_index_sha256": _video_index_sha256(owner_by_video[video_id], video_id),
                "record_count": sum(
                    1 for row in owner_by_video[video_id].rows if row.video_id == video_id
                ),
                "arrays_sha256": sorted(
                    {
                        reference.sha256
                        for row in owner_by_video[video_id].rows
                        if row.video_id == video_id
                        for reference in row.arrays.values()
                    }
                ),
                "content": (
                    {
                        "sha256": evidence_videos[video_id]["sha256"],
                        "size_bytes": evidence_videos[video_id]["size_bytes"],
                    }
                    if video_id in evidence_videos
                    else None
                ),
            }
            for video_id in sorted(target)
        }
        identity = {
            "spec": canonical_spec,
            "sampling_kind": runs[0].resolved["spec"].get("sampling_kind"),
            digest_field: view_digest,
            "encoder_fingerprints": sorted({run.fingerprint for run in runs}),
            "paper_identities": sorted(
                {json.dumps(run.paper_identity, ensure_ascii=False, sort_keys=True) for run in runs}
            ),
        }
        transport_meta = {
            "kind": transport,
            "shared_blob_inodes": transport == "hardlink_npz",
            "source_blob_bytes_preserved": True,
            "source_indexes_preserved": True,
            "independent_blob_backup": transport == "hardlink_npz",
        }
        resolved = {
            "schema": view_schema,
            "status": "completed",
            view_kind: True,
            "complete": True,
            "note": note,
            "identity": identity,
            "videos": len(target),
            "clips": len(rows),
            "transport": transport_meta,
            "source_runs": source_runs,
            "video_provenance": video_provenance,
            "created_at": utc_now_iso(),
        }
        atomic_write_json(destination / "resolved.json", resolved)
        atomic_write_json(
            destination / "status.json",
            {
                "status": "completed",
                "completed": True,
                "failures": [],
                "feature_root": str(destination),
                view_kind: True,
                "videos": len(target),
                "records_written_to_shards": len(rows),
            },
        )
        contract = {
            "schema": contract_schema,
            "status": "ready",
            "complete": True,
            "note": note,
            "videos": len(target),
            "clips": len(rows),
            "identity": identity,
            "target_video_ids": sorted(target),
            "transport": transport_meta,
            "feature_store": {
                "root": str(destination),
                **{
                    name: {"path": str(destination / filename), "sha256": sha256_file(destination / filename)}
                    for name, filename in (
                        ("resolved", "resolved.json"),
                        ("status", "status.json"),
                        ("index", "index.jsonl"),
                    )
                },
            },
            "source_runs": source_runs,
            "video_provenance": video_provenance,
            **dict(extra_contract_fields),
            "created_at": utc_now_iso(),
        }
        atomic_write_json(destination / contract_filename, contract)
    except BaseException:
        # Never leave a half-built view behind: without its contract the
        # directory is not a consumable view, but remove it outright so a
        # later retry cannot mistake partial blobs for a publish attempt.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


__all__ = [
    "FeatureMergeError",
    "FeatureRunMergeRequest",
    "FeatureSubsetRequest",
    "MERGED_VIEW_SCHEMA",
    "MERGE_SCHEMA",
    "SUBSET_CONTRACT_SCHEMA",
    "SUBSET_VIEW_SCHEMA",
    "VIEW_CONTRACT_SCHEMAS",
    "VIEW_RESOLVED_SCHEMAS",
    "run_feature_merge",
    "run_feature_subset",
    "patch_view_contract",
    "view_identity_digest",
]
