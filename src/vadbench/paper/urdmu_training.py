"""Fixed-budget UR-DMU training over verified, complete native dense features.

The 200-bin operation is performed once per video, before any optimizer step.
The author's model, loss and normal-first batch semantics are unchanged. This
module neither scores test data nor selects a checkpoint using test labels.

``run_mode="v0_partial_cache"`` is an explicit engineering small-sample
diagnostic: it consumes a complete-video-subset *merged feature view* (see
``vadbench.workflows.feature_merge``) bound by a v0 view contract, trains an
explicitly smaller budget, and records the actual video counts and the
declared evaluation overlap. It never impersonates a formal or development
run and cannot consume their complete dense extraction contracts.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.features_dataset import FeatureDataset
from vadbench.data.manifest import DatasetSplit, SupervisionScope, load_manifest_jsonl
from vadbench.engine.train import load_checkpoint, save_checkpoint
from vadbench.features import FeatureStore, atomic_write_json, compute_encoder_fingerprint
from vadbench.data.feature_contracts import RepresentationIdentity, SamplingIdentity
from vadbench.engine.coverage import feature_store_coverage as _coverage
from vadbench.workflows.feature_merge import (
    VIEW_CONTRACT_SCHEMAS,
    VIEW_RESOLVED_SCHEMAS,
    view_identity_digest,
)
from vadbench.paper.quality_export import _feature_contract
from vadbench.integrations.detectors.urdmu.backend import UPSTREAM_COMMIT, build_urdmu, build_urdmu_loss

V0_VIEW_SCHEMA = "icassp2027.v0-partial-cache-view/v1"
V0_DATA_ROLE = "engineering_v0_partial_cache"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

COMPONENTS = ("embedding", "selfatt", "Amemory", "Nmemory", "encoder_mu", "encoder_var", "cls_head")
AGGREGATION = {
    "kind": "author_equal_bin_mean",
    "bins": 200,
    "boundaries": "np.linspace(0,N,201).astype(np.int64)",
    "empty_bin": "existing_feature_at_left_boundary",
    "output_dtype": "float32",
    "crops": 1,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf8"))
    _require(isinstance(value, dict), "training input JSON must contain an object")
    return value


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve repository-relative inputs independently of the caller's cwd."""

    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    rooted = (_REPOSITORY_ROOT / value).resolve()
    return rooted if rooted.exists() else value.resolve()


def _bound(path: str | Path, digest: str, name: str) -> Path:
    value = Path(path).expanduser().resolve()
    _require(sha256_file(value) == digest, f"{name} SHA-256 differs from its external binding")
    return value


def author_temporal_bins(features: Any) -> np.ndarray:
    """Author equal-bin mean, retaining its empty-bin rule without uint16 wrap."""
    values = np.asarray(features, dtype=np.float32)
    _require(
        values.ndim == 2 and min(values.shape) > 0 and np.isfinite(values).all(),
        "UR-DMU aggregation requires a nonempty finite [N,D] dense sequence",
    )
    boundaries = np.linspace(0, values.shape[0], 201).astype(np.int64)
    output = np.empty((200, values.shape[1]), dtype=np.float32)
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        output[i] = values[left:right].mean(axis=0) if left < right else values[left]
    return output


@dataclass(frozen=True)
class URDMUTrainingRequest:
    dataset: str
    feature_store: str
    train_manifest: str
    source_contract_path: str
    source_contract_sha256: str
    upstream_dir: str
    output_root: str
    device: str
    seed: int = 0
    protocol_path: str = "projects/icassp2027/decisions/official-detector-protocol-v2.json"
    run_mode: Literal["formal", "development", "engineering", "v0_partial_cache"] = "formal"
    steps: int = 3000
    bags_per_class: int = 64
    extraction_contract_path: str | None = None
    extraction_contract_sha256: str | None = None
    feature_view_contract_path: str | None = None
    feature_view_contract_sha256: str | None = None
    dataset_root: str | None = None
    aggregation_cache: str | None = None
    aggregation_cache_receipt_sha256: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_path", str(_resolve_project_path(self.protocol_path)))
        _require(self.dataset in {"ucf_crime", "xd_violence"}, "unsupported UR-DMU dataset")
        _require(
            self.run_mode in {"formal", "development", "engineering", "v0_partial_cache"},
            "run_mode must be formal, development, engineering or v0_partial_cache",
        )
        if self.run_mode == "v0_partial_cache":
            _require(
                type(self.seed) is int and 0 <= self.seed <= 2**31 - 1,
                "v0_partial_cache seed must be an explicit non-negative integer",
            )
        else:
            _require(type(self.seed) is int and self.seed in {0, 1, 2}, "UR-DMU seed must be 0, 1 or 2")
        _require(
            type(self.steps) is int
            and self.steps > 0
            and type(self.bags_per_class) is int
            and self.bags_per_class > 0,
            "optimizer steps and bags per class must be positive integers",
        )
        if self.run_mode == "v0_partial_cache":
            # v0 is an engineering smoke/diagnostic: its budget must stay below
            # the frozen formal contract so a v0 artifact can never carry it.
            _require(
                self.steps < 3000 and self.bags_per_class < 64,
                "v0_partial_cache requires an explicit budget below the formal 3000-step 64+64 contract",
            )
            _require(
                self.extraction_contract_path is None and self.extraction_contract_sha256 is None,
                "v0_partial_cache consumes a merged feature view, not a dense extraction contract",
            )
        if self.run_mode in {"formal", "development"}:
            _require(
                self.steps == 3000 and self.bags_per_class == 64,
                "formal/development UR-DMU requires 3000 steps and 64 normal + 64 abnormal bags",
            )
            _require(
                (self.extraction_contract_path is None) == (self.extraction_contract_sha256 is None),
                "extraction contract path and SHA-256 must be provided together",
            )
            _require(
                bool(self.extraction_contract_path and self.extraction_contract_sha256)
                or bool(self.feature_view_contract_path and self.feature_view_contract_sha256),
                "formal/development UR-DMU requires an externally bound dense extraction "
                "contract or a bound feature view contract with an explicit role equivalence",
            )
        if self.aggregation_cache is not None:
            _require(
                bool(self.aggregation_cache_receipt_sha256),
                "reused 200-bin cache requires an externally bound receipt SHA",
            )
        _require(
            (self.feature_view_contract_path is None) == (self.feature_view_contract_sha256 is None),
            "feature view contract path and SHA-256 must be provided together",
        )
        _require(
            self.run_id is None
            or bool(self.run_id)
            and self.run_id not in {".", ".."}
            and not any(c in self.run_id for c in "/\\"),
            "run_id must be a basename",
        )


def _device(request: URDMUTrainingRequest) -> torch.device:
    device = torch.device(request.device)
    if device.type == "cpu":
        _require(
            os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"},
            "CPU UR-DMU caller must hide CUDA before importing torch",
        )
        _require(
            not torch.cuda.is_available() and not torch.cuda.is_initialized(),
            "CPU UR-DMU cannot checkpoint with visible or initialized CUDA devices",
        )
    elif device.type == "cuda":
        _require(
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and device.index in {None, 0},
            "UR-DMU requires exactly its selected GPU to be visible for checkpoint RNG capture",
        )
    else:
        raise ValueError("UR-DMU supports explicit CPU or single-visible CUDA execution")
    return device


def _protocol(request: URDMUTrainingRequest) -> dict[str, Any]:
    document = _json(request.protocol_path)
    _require(
        document.get("schema") == "icassp2027.official-detector-protocol/v2",
        "UR-DMU protocol schema differs",
    )
    expected = {
        "name": "UR-DMU",
        "commit": UPSTREAM_COMMIT,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 5e-5,
        "optimizer_steps": 3000,
        "temporal_training_segments": 200,
        "normal_bags_per_step": 64,
        "anomaly_bags_per_step": 64,
        "memory_units_normal": 60,
        "memory_units_anomaly": 60,
        "checkpoint_selection": "fixed_final_step_no_test_selection",
        "loss": "unchanged_official_AD_Loss",
    }
    _require(
        all(document.get("author_backend", {}).get(k) == v for k, v in expected.items()),
        "UR-DMU architecture/loss/training budget differs from the fixed protocol",
    )
    return document


def _data_role(request: URDMUTrainingRequest) -> str:
    return {
        "formal": "official-fulltrain-final",
        "development": "development-fit",
        "engineering": "engineering_only",
        "v0_partial_cache": V0_DATA_ROLE,
    }[request.run_mode]


def _source(
    request: URDMUTrainingRequest,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
    path = _bound(
        request.source_contract_path, request.source_contract_sha256, "training source contract"
    )
    actual = load_manifest_jsonl(request.train_manifest)
    if request.run_mode in {"formal", "development"}:
        from vadbench.data.official_training import load_official_training_view

        authoritative_records, receipt = load_official_training_view(
            path, request.source_contract_sha256, dataset_root=request.dataset_root
        )
        _require(
            receipt.get("dataset") == request.dataset,
            "official training dataset differs from request",
        )
        _require(
            request.run_mode != "formal"
            or [r.to_dict() for r in actual] == [r.to_dict() for r in authoritative_records],
            "training manifest differs from the authoritative complete official view",
        )
        if request.run_mode == "development":
            from vadbench.paper.official_extraction import _development_role_records

            records, lock = _development_role_records(authoritative_records, receipt, path, "fit")
            _require(
                [r.to_dict() for r in actual] == [r.to_dict() for r in records],
                "development UR-DMU requires the complete original fit manifest",
            )
            receipt = {**receipt, "development_role_lock": lock}
        else:
            records = authoritative_records
    else:
        receipt = _json(path)
        if request.run_mode == "v0_partial_cache":
            _require(
                receipt.get("schema") == V0_VIEW_SCHEMA
                and receipt.get("status") == "ready"
                and receipt.get("data_role") == V0_DATA_ROLE
                and receipt.get("dataset") == request.dataset,
                "v0_partial_cache training requires its explicit v0 partial-cache view contract",
            )
        else:
            _require(
                receipt.get("schema") == "urdmu.engineering-training-view/v1"
                and receipt.get("status") == "ready"
                and receipt.get("data_role") == "engineering_only",
                "engineering training requires its explicit synthetic/subset contract",
            )
        binding = receipt["training_manifest"]
        _require(
            _bound(binding["path"], binding["sha256"], "engineering/v0 manifest")
            == Path(request.train_manifest).resolve(),
            f"{request.run_mode} contract binds another manifest",
        )
        role = receipt["source_role_contract"]
        _bound(role["path"], role["sha256"], "engineering/v0 role contract")
        records = actual
    _require(
        records and all(r.split == DatasetSplit.TRAIN for r in records),
        "UR-DMU training accepts only train videos",
    )
    if request.run_mode == "v0_partial_cache":
        classes = {bool(r.is_anomaly) for r in records}
        _require(
            classes == {False, True},
            "v0_partial_cache requires real normal and anomalous videos in the subset view",
        )
    _require(
        all(
            a.scope == SupervisionScope.VIDEO and a.span is None
            for r in records
            for a in r.annotations
        ),
        "UR-DMU training cannot consume temporal supervision",
    )
    paths = [path, Path(request.train_manifest).resolve(), Path(request.protocol_path).resolve()]
    if request.run_mode in {"engineering", "v0_partial_cache"}:
        paths.append(Path(receipt["source_role_contract"]["path"]).resolve())
    sources = {str(p): sha256_file(p) for p in paths}
    if request.run_mode in {"formal", "development"}:
        sources.update(receipt["source_hashes"])
    return records, receipt, sources


def _dense_source(request, records, sources, source_receipt=None):
    root = Path(request.feature_store).expanduser().resolve()
    entry = {
        name: {"path": str(root / filename), "sha256": sha256_file(root / filename)}
        for name, filename in (
            ("resolved", "resolved.json"),
            ("status", "status.json"),
            ("index", "index.jsonl"),
        )
    }
    document = _json(root / "resolved.json")
    if request.extraction_contract_path is None:
        # The explicit role-equivalence path: a merged view of official-view
        # engineering shards stands in for a native dense extraction contract
        # only when the view contract declares the equivalence and every
        # machine-verifiable anchor holds (checked in the adapter). Native
        # engineering stores without any contract keep the legacy path below.
        if request.run_mode in {"formal", "development"}:
            _require(
                document.get("schema") in VIEW_RESOLVED_SCHEMAS,
                "formal/development training without a dense extraction contract requires a merged feature view",
            )
            return _merged_official_dense_source(
                request,
                records,
                sources,
                root,
                entry,
                None,
                None,
                source_receipt,
            )
    else:
        contract_path = _bound(
            request.extraction_contract_path,
            request.extraction_contract_sha256,
            "dense extraction contract",
        )
        contract = _json(contract_path)
        expected_role = _data_role(request)
        _require(
            contract.get("schema") == "icassp2027.official-dense-extraction/v1"
            and contract.get("status") == "ready"
            and contract.get("data_role") == expected_role
            and contract.get("dataset") == request.dataset,
            "dense extraction contract data role differs from the training mode",
        )
        if request.run_mode == "development":
            _require(contract.get("development_role") == "fit", "development UR-DMU can train only fit")
            lock = source_receipt.get("development_role_lock") if source_receipt else None
            _require(
                isinstance(lock, dict) and contract.get("original_role_lock") == lock,
                "development extraction role-lock differs from the authoritative fit source",
            )
        view = contract["training_view_contract"]
        _require(
            _bound(view["path"], view["sha256"], "extraction training view")
            == Path(request.source_contract_path).resolve()
            and view["sha256"] == request.source_contract_sha256,
            "extraction training view differs from requested source",
        )
        manifest = contract["training_manifest"]
        _require(
            _bound(manifest["path"], manifest["sha256"], "extraction training manifest")
            == Path(request.train_manifest).resolve(),
            "extraction manifest differs from requested training manifest",
        )
        protocol = contract["protocol"]
        _require(
            protocol["sha256"] == sha256_file(Path(request.protocol_path)),
            "dense extraction protocol differs from the training protocol",
        )
        protocol_path = _bound(protocol["path"], protocol["sha256"], "extraction protocol")
        sources[str(protocol_path)] = protocol["sha256"]
        sources[str(contract_path)] = sha256_file(contract_path)
        if document.get("schema") in VIEW_RESOLVED_SCHEMAS:
            # A merged/subset feature view may back formal/development training
            # only through the narrow, fully-bound adapter below; the native
            # contract remains mandatory and its role/lock/view/manifest
            # bindings above are unchanged.
            return _merged_official_dense_source(
                request,
                records,
                sources,
                root,
                entry,
                contract,
                contract_path,
                source_receipt,
            )
        _require(
            contract["feature_store"] == {"root": str(root), **entry},
            "extraction contract feature files differ from the supplied FeatureStore",
        )
    spec = document["spec"]
    _require(
        spec.get("sampling_kind") == "dense" and spec["sampling"].get("regime") == "test_dense",
        "UR-DMU requires native dense features; train_32 cannot be upsampled to claim dense coverage",
    )
    representation = RepresentationIdentity.from_mapping(spec["representation"])
    sampling = SamplingIdentity.from_mapping(spec["sampling"])
    _require(
        representation.output_dim in {768, 1024}, "UR-DMU requires pooled D768 or D1024 features"
    )
    coverage = _coverage(records, root, document["encoder_fingerprint"])
    proxy = {
        "evaluation_encoder_fingerprint": document["encoder_fingerprint"],
        "evaluation_representation": representation.to_dict(),
        "evaluation_representation_fingerprint": representation.fingerprint,
        "evaluation_sampling": sampling.to_dict(),
        "evaluation_sampling_fingerprint": sampling.fingerprint,
        "coverage": coverage,
    }
    _, _, runtime, windows, evidence, _ = _feature_contract(
        root,
        entry,
        proxy,
        records,
        sources,
        expected_video_count=len(records),
        feature_root_loader=lambda *_: root,
    )
    if request.run_mode in {"formal", "development"}:
        observed = {r["video_id"]: (r["sha256"], r["size_bytes"]) for r in evidence["videos"]}
        expected = {
            r.video_id: (r.metadata["content_sha256"], r.metadata["content_size_bytes"])
            for r in records
        }
        _require(
            observed == expected,
            "dense feature content differs from the independent official training authority",
        )
    return (
        document,
        representation,
        sampling,
        {
            "runtime": runtime,
            "windows_sha256": windows,
            "data_content_evidence": evidence,
            "coverage": coverage,
            "feature_files": entry,
        },
        None,
    )


def _load_view_sequences(
    *,
    store: FeatureStore,
    records: tuple[Any, ...],
    fingerprints: set[str],
    paper_identities: set[str],
    representation: RepresentationIdentity,
    provenance: Mapping[str, Any],
    context: str,
) -> list[dict[str, Any]]:
    """Load per-video pooled sequences from a merged/subset view, fully checked."""

    rows = list(store.iter_records())
    _require(rows, f"{context} index is empty")
    by_video: dict[str, list[Any]] = {record.video_id: [] for record in records}
    for row in rows:
        _require(
            row.video_id in by_video,
            f"{context} contains a video outside the manifest: {row.video_id}",
        )
        _require(
            row.encoder_fingerprint in fingerprints
            and json.dumps(row.metadata.get("paper_identity"), ensure_ascii=False, sort_keys=True)
            in paper_identities,
            f"{context} row identity is not covered by the view contract: {row.clip_id}",
        )
        by_video[row.video_id].append(row)
    sequences: list[dict[str, Any]] = []
    for record in records:
        video_rows = sorted(by_video[record.video_id], key=lambda item: item.clip_index)
        _require(
            [row.clip_index for row in video_rows] == list(range(len(video_rows))),
            f"{context} lacks complete contiguous windows for {record.video_id}",
        )
        declared_content = record.metadata.get("content_sha256")
        if declared_content is not None:
            content = (provenance.get(record.video_id) or {}).get("content")
            _require(
                isinstance(content, dict) and content.get("sha256") == declared_content,
                f"{context} content evidence differs for {record.video_id}",
            )
        vectors = []
        for row in video_rows:
            bundle = store.load_bundle(row)
            features = np.asarray(bundle["features"], dtype=np.float32)
            pooled = np.asarray(bundle["pooled"], dtype=np.float32)
            _require(
                features.shape == (1, representation.output_dim)
                and pooled.shape == (representation.output_dim,)
                and np.isfinite(features).all()
                and np.isfinite(pooled).all()
                and np.array_equal(features[0], pooled),
                f"{context} pooled layout is invalid: {row.clip_id}",
            )
            vectors.append(pooled)
        sequences.append(
            {
                "video_id": record.video_id,
                "video_label": int(record.is_anomaly),
                "features": np.stack(vectors),
            }
        )
    return sequences


def _canonical_view_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    identity = {key: value for key, value in spec.items() if key != "micro_batch_size"}
    sampling = dict(identity.get("sampling") or {})
    sampling.pop("source_digest", None)
    identity["sampling"] = sampling
    return json.loads(json.dumps(identity, ensure_ascii=False, sort_keys=True))


def _merged_official_dense_source(
    request: URDMUTrainingRequest,
    records: tuple[Any, ...],
    sources: dict[str, str],
    root: Path,
    entry: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    contract_path: Path | None,
    source_receipt: Mapping[str, Any] | None = None,
) -> tuple[Any, RepresentationIdentity, SamplingIdentity, dict[str, Any], list[dict[str, Any]]]:
    """Bind a merged/subset feature view for formal/development UR-DMU training.

    Two anchors are accepted, anything else fails closed:

    * ``contract`` given (native dense extraction contract): the v2 behaviour —
      view identity must equal the native store's spec, duplicate audit empty
      or all ``duplicate_identical``, target list exactly the official role
      view, per-file SHA bindings, sampler-exact windows.
    * ``contract`` null (explicit role-equivalence path): the view contract
      must declare ``role_equivalence: engineering-shards-of-official-fit``
      or ``engineering-shards-of-official-fulltrain`` (formal + complete
      member list only)
      bound to the same authoritative training view contract the request
      binds, and every constituent run's resolved receipt must still hash-match
      and reproduce the view's canonical identity. The declaration is only a
      label: coverage, identity and integrity are all machine-verified here.
    """

    from vadbench.data.dense_sampling import DenseSamplingPlan

    _require(
        bool(request.feature_view_contract_path and request.feature_view_contract_sha256),
        "formal/development training on a merged view requires an externally bound feature view contract",
    )
    view_contract_path = _bound(
        request.feature_view_contract_path,
        request.feature_view_contract_sha256,
        "merged feature view contract",
    )
    view_contract = _json(view_contract_path)
    _require(
        view_contract.get("schema") in VIEW_CONTRACT_SCHEMAS
        and view_contract.get("status") == "ready"
        and view_contract.get("complete") is True,
        "feature view contract is not a ready complete merge/subset contract",
    )
    store_binding = view_contract.get("feature_store") or {}
    _require(
        store_binding.get("root") == str(root),
        "feature view contract binds another feature store root",
    )
    for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl")):
        binding = store_binding.get(name) or {}
        bound = _bound(binding.get("path"), binding.get("sha256"), f"merged view {name}")
        _require(bound == root / filename, f"merged view {name} differs from the supplied store")
        _require(
            entry[name]["sha256"] == binding["sha256"],
            f"merged view {name} digest differs from the supplied store",
        )
    sources[str(view_contract_path)] = sha256_file(view_contract_path)

    identity = view_contract["identity"]
    view_spec = identity.get("spec") or {}
    duplicate_audit = view_contract.get("duplicate_audit") or {}
    _require(
        all(isinstance(item, Mapping) and item.get("verdict") == "duplicate_identical"
            for item in duplicate_audit.values()),
        "merged view duplicate audit contains a non-identical duplicate verdict",
    )
    equivalence: Mapping[str, Any] | None = None
    if contract is None:
        equivalence = view_contract.get("role_equivalence") or {}
        _require(
            isinstance(equivalence, Mapping)
            and equivalence.get("declared") in {
                "engineering-shards-of-official-fit",
                "engineering-shards-of-official-fulltrain",
            },
            "formal/development training without a dense extraction contract requires "
            "the view's explicit role equivalence declaration",
        )
        declared = equivalence.get("declared")
        if declared == "engineering-shards-of-official-fulltrain":
            # The fulltrain declaration names the complete official training
            # view: only formal mode on the complete member list may use it.
            # Coverage equality is the existing target==records check below;
            # here we pin the complete-member expectation (quarantine-aware).
            from vadbench.data.official_training import EXPECTED as _EXPECTED_MEMBERS

            _require(
                request.run_mode == "formal",
                "the fulltrain role equivalence is accepted on the formal path only",
            )
            accepted = None
            if source_receipt is not None:
                # XD's quarantine-aware official-training receipt records the
                # accepted count under ``quarantine_exemption``.  Keep the
                # top-level form for older engineering receipts, but prefer
                # the authoritative nested count when present.
                nested = source_receipt.get("quarantine_exemption")
                if isinstance(nested, Mapping) and isinstance(
                    nested.get("accepted_members"), int
                ):
                    accepted = nested["accepted_members"]
                elif isinstance(source_receipt.get("accepted_members"), int):
                    accepted = source_receipt["accepted_members"]
            expected_total = accepted if accepted is not None else _EXPECTED_MEMBERS[request.dataset][0]
            _require(
                len(records) == expected_total,
                "fulltrain role equivalence requires the complete official training view",
            )
        authority = equivalence.get("authority_contract") or {}
        authority_path = Path(authority.get("path", "")).expanduser().resolve()
        _require(
            isinstance(authority.get("sha256"), str) and authority_path.is_file()
            and sha256_file(authority_path) == authority["sha256"],
            "role equivalence authority contract SHA differs",
        )
        _require(
            authority_path == Path(request.source_contract_path).expanduser().resolve()
            and authority["sha256"] == request.source_contract_sha256,
            "role equivalence authority differs from the requested training source contract",
        )
        # Identity anchor: every constituent run's resolved receipt must still
        # hash-match the merge contract and reproduce the canonical identity.
        for source_run in view_contract.get("source_runs") or []:
            run_root = Path(source_run.get("root", "")).expanduser().resolve()
            run_resolved_path = run_root / "resolved.json"
            _require(
                run_resolved_path.is_file()
                and sha256_file(run_resolved_path) == source_run.get("resolved_sha256"),
                f"role equivalence source run receipt changed: {run_root}",
            )
            run_resolved = _json(run_resolved_path)
            _require(
                _canonical_view_spec(run_resolved.get("spec") or {})
                == _canonical_view_spec(view_spec),
                f"role equivalence source run identity differs from the view: {run_root}",
            )
        if request.run_mode == "development":
            lock = source_receipt.get("development_role_lock") if source_receipt else None
            _require(
                isinstance(lock, dict) and view_contract.get("original_role_lock") == lock,
                "development role lock differs from the authoritative fit source",
            )
        native_root = None
    else:
        native_binding = contract["feature_store"]
        native_root = Path(native_binding["root"]).expanduser().resolve()
        for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl")):
            binding = native_binding.get(name) or {}
            bound = _bound(binding.get("path"), binding.get("sha256"), f"native dense store {name}")
            _require(bound == native_root / filename, f"native dense store {name} is not bound")
        sources[str(native_root / "resolved.json")] = native_binding["resolved"]["sha256"]
        sources[str(native_root / "status.json")] = native_binding["status"]["sha256"]
        sources[str(native_root / "index.jsonl")] = native_binding["index"]["sha256"]
        native_resolved = _json(native_root / "resolved.json")
        native_spec = native_resolved.get("spec") or {}
        _require(
            _canonical_view_spec(native_spec) == _canonical_view_spec(view_spec),
            "merged view identity (representation/sampling/weights) differs from the bound dense extraction contract",
        )
    _require(
        set(view_contract.get("target_video_ids") or ()) == {record.video_id for record in records},
        "merged view target list differs from the official role view",
    )

    representation = RepresentationIdentity.from_mapping(view_spec["representation"])
    sampling = SamplingIdentity.from_mapping(view_spec["sampling"])
    _require(
        view_spec.get("sampling_kind") == "dense" and sampling.regime == "test_dense",
        "merged view is not a native dense sampling identity",
    )
    _require(representation.output_dim in {768, 1024}, "UR-DMU requires pooled D768 or D1024 features")
    fingerprints = set(identity.get("encoder_fingerprints") or ())
    paper_identities = set(identity.get("paper_identities") or ())
    _require(fingerprints and paper_identities, "merged view contract lacks run identity evidence")

    store = FeatureStore(root)
    rows = list(store.iter_records())
    _require(rows, "merged view index is empty")
    rows_by_video: dict[str, list[Any]] = {record.video_id: [] for record in records}
    for row in rows:
        _require(
            row.video_id in rows_by_video,
            f"merged view contains a video outside the training manifest: {row.video_id}",
        )
        _require(
            row.encoder_fingerprint in fingerprints
            and json.dumps(row.metadata.get("paper_identity"), ensure_ascii=False, sort_keys=True)
            in paper_identities,
            f"merged view row identity is not covered by the view contract: {row.clip_id}",
        )
        rows_by_video[row.video_id].append(row)
    sampler = DenseSamplingPlan(
        clip_frames=sampling.window["clip_frames"],
        frame_stride=sampling.stride["frame_stride"],
        window_stride=sampling.frame_selection["window_stride"],
        short_policy=sampling.frame_selection["short_video_policy"],
    )
    for record in records:
        actual = sorted(rows_by_video[record.video_id], key=lambda item: item.clip_index)
        expected = sampler.sample(record.num_frames)
        _require(
            len(actual) == len(expected),
            f"{record.video_id}: merged view lacks sampler windows",
        )
        for row, sample in zip(actual, expected, strict=True):
            _require(
                row.clip_index == sample.clip_index
                and row.frame_start == sample.score_frame_start
                and row.frame_end == sample.score_frame_end,
                f"{record.video_id}: merged view window differs from the declared sampler",
            )
            _require(
                math.isclose(row.start_s, sample.score_frame_start / record.fps)
                and math.isclose(row.end_s, sample.score_frame_end / record.fps),
                f"{record.video_id}: merged view timeline differs from manifest FPS",
            )
    sequences = _load_view_sequences(
        store=store,
        records=records,
        fingerprints=fingerprints,
        paper_identities=paper_identities,
        representation=representation,
        provenance=view_contract.get("video_provenance") or {},
        context="merged view",
    )
    digest = view_identity_digest(identity)
    document = {
        "schema": view_contract.get("schema"),
        "merged_view": True,
        "encoder_fingerprint": None,
        "merged_view_digest": digest,
        "identity": identity,
        "view_contract": {"path": str(view_contract_path), "sha256": sha256_file(view_contract_path)},
    }
    feature_receipt = {
        "merged_view": True,
        "merged_view_digest": digest,
        "view_contract": dict(document["view_contract"]),
        "native_contract": (
            None
            if contract_path is None
            else {"path": str(contract_path), "sha256": sha256_file(contract_path)}
        ),
        "role_equivalence": None if equivalence is None else dict(equivalence),
        "native_feature_store": None if native_root is None else str(native_root),
        "duplicate_audit": {
            video: {"verdict": item.get("verdict"), "kept_run": item.get("kept_run")}
            for video, item in sorted(duplicate_audit.items())
        },
        "encoder_fingerprints": sorted(fingerprints),
        "feature_files": dict(entry),
        "note": (
            "Formal/development UR-DMU consumed a merged/subset feature view bound to the "
            "native dense extraction contract; identity, duplicate audit, target list and "
            "every SHA binding were re-verified."
        ),
    }
    return document, representation, sampling, feature_receipt, sequences


def _v0_dense_source(
    request: URDMUTrainingRequest,
    records: tuple[Any, ...],
    sources: dict[str, str],
    source_receipt: Mapping[str, Any],
) -> tuple[Any, RepresentationIdentity, SamplingIdentity, dict[str, Any], list[dict[str, Any]]]:
    """Bind a merged multi-run feature view for the v0 engineering diagnostic.

    The merged view is not a native extraction run: rows keep their per-run
    encoder fingerprints, so membership, identity and bundle integrity are
    re-verified here against the externally bound merge contract instead of
    the formal dense-extraction contract.
    """

    root = Path(request.feature_store).expanduser().resolve()
    view_binding = source_receipt["feature_view"]
    contract_path = _bound(
        view_binding["path"], view_binding["sha256"], "v0 merged feature view contract"
    )
    contract = _json(contract_path)
    _require(
        contract.get("schema") in VIEW_CONTRACT_SCHEMAS
        and contract.get("status") == "ready"
        and contract.get("complete") is True,
        "v0 feature view contract is not a ready complete merge/subset contract",
    )
    store_binding = contract.get("feature_store") or {}
    _require(
        store_binding.get("root") == str(root),
        "v0 view contract binds another feature store root",
    )
    entry = {}
    for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl")):
        binding = store_binding.get(name) or {}
        bound = _bound(binding.get("path"), binding.get("sha256"), f"v0 merged view {name}")
        _require(bound == root / filename, f"v0 merged view {name} differs from the supplied store")
        entry[name] = {"path": str(root / filename), "sha256": binding["sha256"]}
    sources[str(contract_path)] = sha256_file(contract_path)
    for name in ("resolved", "status", "index"):
        sources[entry[name]["path"]] = entry[name]["sha256"]

    identity = contract["identity"]
    spec = identity["spec"]
    representation = RepresentationIdentity.from_mapping(spec["representation"])
    sampling = SamplingIdentity.from_mapping(spec["sampling"])
    _require(
        spec.get("sampling_kind") == "dense" and sampling.regime == "test_dense",
        "v0_partial_cache requires native dense features in the merged view",
    )
    _require(representation.output_dim in {768, 1024}, "UR-DMU requires pooled D768 or D1024 features")
    fingerprints = set(identity.get("encoder_fingerprints") or ())
    paper_identities = set(identity.get("paper_identities") or ())
    _require(fingerprints and paper_identities, "v0 view contract lacks run identity evidence")

    manifest_ids = {record.video_id for record in records}
    provenance = contract.get("video_provenance") or {}
    store = FeatureStore(root)
    sequences = _load_view_sequences(
        store=store,
        records=records,
        fingerprints=fingerprints,
        paper_identities=paper_identities,
        representation=representation,
        provenance=provenance,
        context="v0 merged view",
    )
    digest = view_identity_digest(identity)
    document = {
        "schema": "icassp2027.merged-feature-view/v1",
        "merged_view": True,
        "encoder_fingerprint": None,
        "merged_view_digest": digest,
        "identity": identity,
        "view_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
    }
    feature_receipt = {
        "merged_view": True,
        "merged_view_digest": digest,
        "view_contract": dict(document["view_contract"]),
        "encoder_fingerprints": sorted(fingerprints),
        "video_provenance": {
            video_id: {"source_run": (provenance.get(video_id) or {}).get("source_run")}
            for video_id in sorted(manifest_ids)
        },
        "feature_files": entry,
        "video_counts": {
            "total": len(records),
            "normal": sum(1 for record in records if not record.is_anomaly),
            "anomaly": sum(1 for record in records if record.is_anomaly),
        },
        "evaluation_overlap": source_receipt.get("evaluation_overlap"),
        "note": (
            "v0_partial_cache consumes a merged multi-run feature view. This is an "
            "engineering small-sample diagnostic, not a formal/development baseline."
        ),
    }
    return document, representation, sampling, feature_receipt, sequences


def _unchanged(sources):
    for path, digest in sources.items():
        _bound(path, digest, "training source artifact")


def _aggregate(
    request,
    records,
    document,
    representation,
    sources,
    run,
    *,
    sequences: list[dict[str, Any]] | None = None,
):
    source_fingerprint = document.get("merged_view_digest") or document["encoder_fingerprint"]
    key = compute_encoder_fingerprint(
        {
            "source_sha256": sources,
            "aggregation": AGGREGATION,
            "representation": representation.fingerprint,
            "encoder_fingerprint": source_fingerprint,
        }
    )
    labels = [int(r.is_anomaly) for r in records]
    expected = {
        "cache_key": key,
        "video_ids": [r.video_id for r in records],
        "labels": labels,
        "shape": [len(records), 200, representation.output_dim],
        "aggregation": AGGREGATION,
        "data_role": _data_role(request),
    }
    if request.aggregation_cache is not None:
        root = Path(request.aggregation_cache).resolve()
        _bound(
            root / "receipt.json", request.aggregation_cache_receipt_sha256, "200-bin cache receipt"
        )
        receipt = _json(root / "receipt.json")
        _require(
            all(receipt.get(k) == v for k, v in expected.items()),
            "200-bin cache does not match the verified dense source",
        )
        _bound(root / "bags.npy", receipt["bags_sha256"], "200-bin cache")
        atomic_write_json(
            run / "aggregation-cache.json",
            {
                "reused": True,
                "root": str(root),
                "receipt_sha256": sha256_file(root / "receipt.json"),
            },
        )
    else:
        root = run / "aggregation"
        root.mkdir()
        cache = np.lib.format.open_memmap(
            root / "bags.npy", mode="w+", dtype=np.float32, shape=tuple(expected["shape"])
        )
        if sequences is None:
            dataset = FeatureDataset(
                request.feature_store,
                records,
                encoder_fingerprint=document["encoder_fingerprint"],
                supervision="weak",
                feature_level="clip",
                split="train",
                require_all_features=True,
                cache_sequences=False,
            )
            _require(len(dataset) == len(records), "dense training FeatureDataset membership changed")
            sequences = [dataset[index] for index in range(len(dataset))]
        else:
            _require(
                len(sequences) == len(records), "v0 sequence view membership changed"
            )
        lengths = []
        for index, record in enumerate(records):
            sequence = sequences[index]
            _require(
                sequence["video_id"] == record.video_id
                and sequence["video_label"] == labels[index],
                "dense FeatureDataset order or weak label differs",
            )
            _require(
                sequence["features"].shape[1] == representation.output_dim,
                "dense pooled dimension differs",
            )
            cache[index] = author_temporal_bins(sequence["features"])
            lengths.append(len(sequence["features"]))
        cache.flush()
        del cache
        receipt = {
            **expected,
            "schema_version": 1,
            "source_sha256": dict(sources),
            "source_dense_lengths": lengths,
            "bags_sha256": sha256_file(root / "bags.npy"),
        }
        atomic_write_json(root / "receipt.json", receipt)
        atomic_write_json(
            run / "aggregation-cache.json",
            {
                "reused": False,
                "root": str(root),
                "receipt_sha256": sha256_file(root / "receipt.json"),
            },
        )
    values = np.load(root / "bags.npy", allow_pickle=False)
    _require(
        values.dtype == np.float32
        and list(values.shape) == expected["shape"]
        and np.isfinite(values).all(),
        "200-bin cache values are invalid",
    )
    sources[str(root / "receipt.json")] = sha256_file(root / "receipt.json")
    sources[str(root / "bags.npy")] = receipt["bags_sha256"]
    return values, np.asarray(labels, dtype=np.int64), root, receipt


class _Bags(Dataset):
    def __init__(self, values, indices):
        self.values, self.indices = values, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return torch.from_numpy(self.values[self.indices[index]])


def _cycle(loader):
    while True:
        yield from loader


def run_urdmu_training(request: URDMUTrainingRequest) -> dict[str, Any]:
    """Train one seed and publish only the fixed final, QA-bound checkpoint."""
    device = _device(request)
    run = Path(request.output_root).expanduser().resolve() / (
        request.run_id or new_run_id("urdmu-training")
    )
    run.mkdir(parents=True, exist_ok=False)
    with record_stage(
        run,
        "urdmu_training",
        config=request.__dict__,
        inputs={
            "train_manifest": request.train_manifest,
            "training_view_contract": request.source_contract_path,
            "protocol": request.protocol_path,
            "extraction_contract": request.extraction_contract_path,
        },
        project_root=Path.cwd(),
    ):
        protocol = _protocol(request)
        records, source_receipt, sources = _source(request)
        sequences = None
        if request.run_mode == "v0_partial_cache":
            document, representation, sampling, feature_receipt, sequences = _v0_dense_source(
                request, records, sources, source_receipt
            )
        else:
            document, representation, sampling, feature_receipt, sequences = _dense_source(
                request, records, sources, source_receipt
            )
        if request.run_mode in {"formal", "development"}:
            _require(
                representation.backbone.runtime_id
                in protocol["datasets"][request.dataset]["encoders"],
                "encoder is outside the current official protocol",
            )
        values, labels, cache_root, cache_receipt = _aggregate(
            request, records, document, representation, sources, run, sequences=sequences
        )
        _unchanged(sources)
        normal, anomaly = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        _require(
            min(len(normal), len(anomaly)) >= request.bags_per_class,
            "both label groups must contain a complete drop_last batch",
        )
        random.seed(request.seed)
        np.random.seed(request.seed)
        torch.manual_seed(request.seed)
        model, backend_receipt = build_urdmu(
            representation.output_dim, request.upstream_dir, mode="Train"
        )
        criterion, loss_receipt = build_urdmu_loss(request.upstream_dir)
        model, criterion = model.to(device), criterion.to(device)
        original = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-5
        )
        loaders = [
            DataLoader(
                _Bags(values, indices),
                batch_size=request.bags_per_class,
                shuffle=True,
                drop_last=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(request.seed + offset),
            )
            for offset, indices in enumerate((normal, anomaly))
        ]
        streams = [_cycle(loader) for loader in loaders]
        fixed_batch = None
        nonzero_steps = 0
        with (run / "training-steps.jsonl").open("x", encoding="utf8") as log:
            for step in range(1, request.steps + 1):
                inputs = torch.cat((next(streams[0]), next(streams[1])), 0).to(device)
                targets = torch.cat(
                    (torch.zeros(request.bags_per_class), torch.ones(request.bags_per_class))
                ).to(device)
                if fixed_batch is None:
                    fixed_batch = inputs.detach().clone()
                model.flag = "Train"
                model.train()
                result = model(inputs)
                cost, losses = criterion(result, targets)
                _require(
                    all(torch.isfinite(value).all().item() for value in result.values())
                    and torch.isfinite(cost).all().item(),
                    "UR-DMU produced nonfinite training outputs/loss",
                )
                optimizer.zero_grad(set_to_none=True)
                cost.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                _require(
                    gradients and all(torch.isfinite(g).all().item() for g in gradients),
                    "UR-DMU gradients are missing or nonfinite",
                )
                nonzero = sum(bool(torch.count_nonzero(g).item()) for g in gradients)
                _require(nonzero > 0, "UR-DMU optimizer step has no nonzero gradients")
                optimizer.step()
                nonzero_steps += 1
                metrics = {
                    name: float(value.detach().item())
                    for name, value in {**losses, "distance": result["distance"]}.items()
                }
                _require(
                    all(np.isfinite(v) for v in metrics.values()),
                    "UR-DMU loss components are nonfinite",
                )
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "losses": metrics,
                            "nonzero_gradient_parameters": nonzero,
                            "normal_bags": request.bags_per_class,
                            "abnormal_bags": request.bags_per_class,
                            "normal_first": True,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                log.flush()
        changed = {
            component: sum(
                not torch.equal(original[name], p.detach().cpu())
                for name, p in model.named_parameters()
                if name.startswith(component + ".")
            )
            for component in COMPONENTS
        }
        _require(
            all(count > 0 for count in changed.values()),
            "UR-DMU did not update all seven required components",
        )
        model.flag = "Test"
        model.eval()
        with torch.inference_mode():
            reference = model(fixed_batch)["frame"].detach().clone()
        _require(torch.isfinite(reference).all().item(), "UR-DMU final eval output is nonfinite")
        _unchanged(sources)
        _device(request)  # save_checkpoint must not initialize any unintended GPU.
        metadata = {
            "schema": "urdmu.training/v1",
            "status": "completed_training",
            "run_mode": request.run_mode,
            "dataset": request.dataset,
            "data_role": _data_role(request),
            "development_role": "fit" if request.run_mode == "development" else None,
            "checkpoint_role": "dense_reference"
            if representation.reducer.get("name") == "identity"
            else "refit_head",
            "source_sha256": dict(sources),
            "source_training_view": source_receipt,
            "feature_source": feature_receipt,
            "representation": representation.to_dict(),
            "representation_fingerprint": representation.fingerprint,
            "sampling": sampling.to_dict(),
            "sampling_fingerprint": sampling.fingerprint,
            "sampling_regime_note": "test_dense names the historical dense algorithm; this run consumes train records only",
            "encoder_fingerprint": document["encoder_fingerprint"] or document.get("merged_view_digest"),
            "merged_view_digest": document.get("merged_view_digest"),
            "v0_diagnostic": request.run_mode == "v0_partial_cache",
            "note": (
                "v0_partial_cache is an engineering small-sample diagnostic over a merged "
                "feature view; it is not a formal or development baseline"
                if request.run_mode == "v0_partial_cache"
                else None
            ),
            "video_counts": {
                "total": len(records),
                "normal": int(np.count_nonzero(labels == 0)),
                "anomaly": int(np.count_nonzero(labels == 1)),
            },
            "evaluation_overlap": source_receipt.get("evaluation_overlap"),
            "aggregation_cache": {
                "root": str(cache_root),
                "receipt_sha256": sha256_file(cache_root / "receipt.json"),
                "bags_sha256": cache_receipt["bags_sha256"],
            },
            "backend": backend_receipt,
            "loss_backend": loss_receipt,
            "protocol_sha256": sha256_file(Path(request.protocol_path)),
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 5e-5,
            },
            "steps": request.steps,
            "bags_per_class": request.bags_per_class,
            "seed": request.seed,
            "class_loader_seeds": [request.seed, request.seed + 1],
            "nonzero_gradient_steps": nonzero_steps,
            "changed_parameters_by_component": changed,
            "checkpoint_selection": "fixed_final_step_no_test_selection",
            "runtime": {
                "python": sys.executable,
                "python_version": sys.version,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "vadbench_file": sys.modules["vadbench"].__file__,
                "training_module_file": __file__,
                "training_module_sha256": sha256_file(Path(__file__)),
            },
        }
        checkpoint = run / "checkpoints/final.pt"
        artifact = save_checkpoint(
            checkpoint, model, optimizer=optimizer, step=request.steps, metadata=metadata
        )
        restored, _ = build_urdmu(representation.output_dim, request.upstream_dir, mode="Test")
        restored = restored.to(device)
        loaded = load_checkpoint(
            checkpoint, restored, map_location=device, strict=True, verify=True
        )
        restored.flag = "Test"
        restored.eval()
        with torch.inference_mode():
            output = restored(fixed_batch)["frame"]
        _require(
            torch.equal(reference, output),
            "UR-DMU fixed-batch final checkpoint reload is not exact",
        )
        qa = {
            "status": "passed",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": artifact.sha256,
                "step": request.steps,
            },
            "nonzero_gradient_steps": nonzero_steps,
            "changed_parameters_by_component": changed,
            "reload_parity": {
                "exact_equal": True,
                "max_abs_difference": 0.0,
                "shape": list(output.shape),
                "flag": "Test",
                "training": False,
            },
            "strict_reload": {
                "missing_keys": loaded["missing_keys"],
                "unexpected_keys": loaded["unexpected_keys"],
            },
            "fixed_batch_source": "first optimizer batch; no evaluation/selection data",
        }
        atomic_write_json(run / "training_qa.json", qa)
        result = {
            "status": "completed",
            "run_dir": str(run),
            "run_mode": request.run_mode,
            "data_role": _data_role(request),
            "development_role": "fit" if request.run_mode == "development" else None,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": artifact.sha256,
            "training_qa_sha256": sha256_file(run / "training_qa.json"),
            "aggregation_cache": str(cache_root),
            "aggregation_cache_receipt_sha256": sha256_file(cache_root / "receipt.json"),
            "optimizer_steps": request.steps,
            "official_model_scores_read": False,
            "v0_diagnostic": request.run_mode == "v0_partial_cache",
            "video_counts": {
                "total": len(records),
                "normal": int(np.count_nonzero(labels == 0)),
                "anomaly": int(np.count_nonzero(labels == 1)),
            },
            "evaluation_overlap": source_receipt.get("evaluation_overlap"),
        }
        atomic_write_json(run / "result.json", result)
    return result
