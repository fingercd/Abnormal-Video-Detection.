"""Fixed-budget UR-DMU training over verified, complete native dense features.

The 200-bin operation is performed once per video, before any optimizer step.
The author's model, loss and normal-first batch semantics are unchanged. This
module neither scores test data nor selects a checkpoint using test labels.
"""

from __future__ import annotations

import json
import os
import random
import sys
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
from vadbench.features import atomic_write_json, compute_encoder_fingerprint
from vadbench.paper.compatibility import RepresentationIdentity, SamplingIdentity
from vadbench.paper.evaluation import _coverage
from vadbench.paper.quality_export import _feature_contract
from vadbench.paper.urdmu_backend import UPSTREAM_COMMIT, build_urdmu, build_urdmu_loss

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
    run_mode: Literal["formal", "engineering"] = "formal"
    steps: int = 3000
    bags_per_class: int = 64
    extraction_contract_path: str | None = None
    extraction_contract_sha256: str | None = None
    dataset_root: str | None = None
    aggregation_cache: str | None = None
    aggregation_cache_receipt_sha256: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require(self.dataset in {"ucf_crime", "xd_violence"}, "unsupported UR-DMU dataset")
        _require(
            self.run_mode in {"formal", "engineering"}, "run_mode must be formal or engineering"
        )
        _require(type(self.seed) is int and self.seed in {0, 1, 2}, "UR-DMU seed must be 0, 1 or 2")
        _require(
            type(self.steps) is int
            and self.steps > 0
            and type(self.bags_per_class) is int
            and self.bags_per_class > 0,
            "optimizer steps and bags per class must be positive integers",
        )
        if self.run_mode == "formal":
            _require(
                self.steps == 3000 and self.bags_per_class == 64,
                "formal UR-DMU requires 3000 steps and 64 normal + 64 abnormal bags",
            )
            _require(
                bool(self.extraction_contract_path and self.extraction_contract_sha256),
                "formal UR-DMU requires the externally bound official dense extraction contract",
            )
        if self.aggregation_cache is not None:
            _require(
                bool(self.aggregation_cache_receipt_sha256),
                "reused 200-bin cache requires an externally bound receipt SHA",
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


def _source(
    request: URDMUTrainingRequest,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, str]]:
    path = _bound(
        request.source_contract_path, request.source_contract_sha256, "training source contract"
    )
    actual = load_manifest_jsonl(request.train_manifest)
    if request.run_mode == "formal":
        from vadbench.data.official_training import load_official_training_view

        records, receipt = load_official_training_view(
            path, request.source_contract_sha256, dataset_root=request.dataset_root
        )
        _require(
            receipt.get("dataset") == request.dataset,
            "official training dataset differs from request",
        )
        _require(
            [r.to_dict() for r in actual] == [r.to_dict() for r in records],
            "training manifest differs from the authoritative complete official view",
        )
    else:
        receipt = _json(path)
        _require(
            receipt.get("schema") == "urdmu.engineering-training-view/v1"
            and receipt.get("status") == "ready"
            and receipt.get("data_role") == "engineering_only",
            "engineering training requires its explicit synthetic/subset contract",
        )
        binding = receipt["training_manifest"]
        _require(
            _bound(binding["path"], binding["sha256"], "engineering manifest")
            == Path(request.train_manifest).resolve(),
            "engineering contract binds another manifest",
        )
        role = receipt["source_role_contract"]
        _bound(role["path"], role["sha256"], "engineering role contract")
        records = actual
    _require(
        records and all(r.split == DatasetSplit.TRAIN for r in records),
        "UR-DMU training accepts only train videos",
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
    if request.run_mode == "engineering":
        paths.append(Path(receipt["source_role_contract"]["path"]).resolve())
    sources = {str(p): sha256_file(p) for p in paths}
    if request.run_mode == "formal":
        sources.update(receipt["source_hashes"])
    return records, receipt, sources


def _dense_source(request, records, sources):
    root = Path(request.feature_store).expanduser().resolve()
    entry = {
        name: {"path": str(root / filename), "sha256": sha256_file(root / filename)}
        for name, filename in (
            ("resolved", "resolved.json"),
            ("status", "status.json"),
            ("index", "index.jsonl"),
        )
    }
    if request.extraction_contract_path is not None:
        contract_path = _bound(
            request.extraction_contract_path,
            request.extraction_contract_sha256,
            "dense extraction contract",
        )
        contract = _json(contract_path)
        _require(
            contract.get("schema") == "icassp2027.official-dense-extraction/v1"
            and contract.get("status") == "ready"
            and contract.get("data_role") == "official-fulltrain-final"
            and contract.get("dataset") == request.dataset,
            "dense extraction contract is not the complete official training view",
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
        _require(
            contract["feature_store"] == {"root": str(root), **entry},
            "extraction contract feature files differ from the supplied FeatureStore",
        )
        protocol = contract["protocol"]
        _require(
            protocol["sha256"] == sha256_file(Path(request.protocol_path)),
            "dense extraction protocol differs from the training protocol",
        )
        protocol_path = _bound(protocol["path"], protocol["sha256"], "extraction protocol")
        sources[str(protocol_path)] = protocol["sha256"]
        sources[str(contract_path)] = sha256_file(contract_path)
    document = _json(root / "resolved.json")
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
    if request.run_mode == "formal":
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
    )


def _unchanged(sources):
    for path, digest in sources.items():
        _bound(path, digest, "training source artifact")


def _aggregate(request, records, document, representation, sources, run):
    key = compute_encoder_fingerprint(
        {
            "source_sha256": sources,
            "aggregation": AGGREGATION,
            "representation": representation.fingerprint,
            "encoder_fingerprint": document["encoder_fingerprint"],
        }
    )
    labels = [int(r.is_anomaly) for r in records]
    expected = {
        "cache_key": key,
        "video_ids": [r.video_id for r in records],
        "labels": labels,
        "shape": [len(records), 200, representation.output_dim],
        "aggregation": AGGREGATION,
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
        lengths = []
        for index, record in enumerate(records):
            sequence = dataset[index]
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
            "data_role": "official-fulltrain-final"
            if request.run_mode == "formal"
            else "engineering_only",
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
        document, representation, sampling, feature_receipt = _dense_source(
            request, records, sources
        )
        if request.run_mode == "formal":
            _require(
                representation.backbone.runtime_id
                in protocol["datasets"][request.dataset]["encoders"],
                "encoder is outside the current official protocol",
            )
        values, labels, cache_root, cache_receipt = _aggregate(
            request, records, document, representation, sources, run
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
            "data_role": "official-fulltrain-final"
            if request.run_mode == "formal"
            else "engineering_only",
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
            "encoder_fingerprint": document["encoder_fingerprint"],
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
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": artifact.sha256,
            "training_qa_sha256": sha256_file(run / "training_qa.json"),
            "aggregation_cache": str(cache_root),
            "aggregation_cache_receipt_sha256": sha256_file(cache_root / "receipt.json"),
            "optimizer_steps": request.steps,
            "official_model_scores_read": False,
        }
        atomic_write_json(run / "result.json", result)
    return result
