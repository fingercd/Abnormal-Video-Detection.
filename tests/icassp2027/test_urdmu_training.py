"""Real UR-DMU CPU optimization over synthetic, SHA-verified dense NPZ features."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import VideoManifestRecord, load_manifest_jsonl, write_manifest_jsonl
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.paper import urdmu_training as training
from vadbench.paper.compatibility import (
    BackboneIdentity,
    RepresentationIdentity,
    SamplingIdentity,
    feature_cache_key,
)
from vadbench.paper.extraction import (
    _semantic_runtime_identity,
    _verified_code_digest,
    _verified_weights_digest,
    make_data_content_evidence,
    make_sampling_identity,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


@pytest.fixture
def source(tmp_path, monkeypatch, request):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    per_class = getattr(request, "param", 2)
    for i in range(2 * per_class):
        name = f"synthetic-{i}"
        (raw / f"{name}.mp4").write_bytes(f"synthetic bytes {i}".encode())
        records.append(
            VideoManifestRecord(
                video_id=name,
                path=f"{name}.mp4",
                split="train",
                category="Normal" if i < per_class else "Anomaly",
                is_anomaly=i >= per_class,
                num_frames=96,
                fps=24.0,
                duration_seconds=4.0,
                metadata={
                    "synthetic_only": True,
                    "content_sha256": sha256_file(raw / f"{name}.mp4"),
                    "content_size_bytes": (raw / f"{name}.mp4").stat().st_size,
                },
            )
        )
    manifest = write_manifest_jsonl(records, tmp_path / "train.jsonl")
    role = _write(tmp_path / "roles.json", {r.video_id: "engineering_train" for r in records})
    contract = _write(
        tmp_path / "contract.json",
        {
            "schema": "urdmu.engineering-training-view/v1",
            "status": "ready",
            "data_role": "engineering_only",
            "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "source_role_contract": {"path": str(role), "sha256": sha256_file(role)},
        },
    )
    verified = {
        "adapter": "toy",
        "constructor": {"clip_frames": 16},
        "checkpoint": {"sha256": {"fixture-native": "a" * 64}},
    }
    runtime = {
        "adapter_type": "fixture.Adapter",
        "adapter_library_version": None,
        "encoder_type": "fixture.Encoder",
        "encoder_library_version": None,
        "capabilities": {"fixed_num_frames": 16, "supports_fixed_clip": True},
        "properties": {"pooling": "mean"},
        "runtime_configurations": {"model_config": {"hidden_size": 768}},
        "implementation_files": {
            "model": {
                "role": "model",
                "module": "fixture",
                "path": str(tmp_path / "native.py"),
                "sha256": "b" * 64,
            }
        },
        "loaded_library_versions": {"torch": str(torch.__version__)},
    }
    representation = RepresentationIdentity(
        BackboneIdentity(
            "toy",
            _verified_weights_digest(verified),
            _verified_code_digest(verified, runtime),
            {"profile": "fixture"},
            {"kind": "mean"},
        ),
        {"name": "identity"},
        768,
        "float32",
        {"kind": "native"},
    )
    sampling = make_sampling_identity(
        records,
        dataset_root=raw,
        sampling_kind="dense",
        clip_frames=16,
        frame_stride=2,
        window_stride=16,
    )
    runtime.update(
        runtime_id="toy",
        representation_fingerprint=representation.fingerprint,
        sampling_fingerprint=sampling.fingerprint,
        source_data_digest=sampling.source_digest,
        declared_readout=dict(representation.backbone.readout),
        verified_encoder_identity=verified,
    )
    fingerprint = compute_encoder_fingerprint(
        {"paper_pooled_cache": _semantic_runtime_identity(runtime)}
    )
    paper = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    root = tmp_path / "features"
    store = FeatureStore(root)
    rng = np.random.default_rng(817)
    count = 0
    for record in records:
        for sample in DenseSamplingPlan(16, 2, 16).sample(record.num_frames):
            vector = rng.normal(0, 0.3, 768).astype(np.float32)
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:clip-{sample.clip_index:06d}",
                clip_index=sample.clip_index,
                encoder_fingerprint=fingerprint,
                features=vector[None, :],
                pooled=vector,
                start_s=sample.score_frame_start / 24.0,
                end_s=sample.score_frame_end / 24.0,
                frame_start=sample.score_frame_start,
                frame_end=sample.score_frame_end,
                metadata={
                    "paper_identity": paper,
                    "sampling": {
                        "kind": "dense",
                        "clip_index": sample.clip_index,
                        "actual_frame_stride": 2,
                        "end_anchored": sample.end_anchored,
                        "frame_indices": list(sample.frame_indices),
                        "valid_mask": list(sample.valid_mask),
                    },
                    "source_video": {
                        "manifest_path": record.path,
                        "actual_num_frames": 96,
                        "actual_fps": 24.0,
                        "width": 64,
                        "height": 48,
                        "content_sha256": sha256_file(record.resolve_path(raw)),
                        "content_size_bytes": record.resolve_path(raw).stat().st_size,
                    },
                },
            )
            count += 1
    _write(
        root / "resolved.json",
        {
            "spec": {
                "representation": representation.to_dict(),
                "sampling": sampling.to_dict(),
                "sampling_kind": "dense",
            },
            "runtime": runtime,
            "data_content_evidence": make_data_content_evidence(records, dataset_root=raw),
            "encoder_fingerprint": fingerprint,
            "paper_identity": paper,
        },
    )
    _write(
        root / "status.json",
        {
            "status": "completed",
            "completed": True,
            "failures": [],
            "feature_root": str(root.resolve()),
            "encoder_fingerprint": fingerprint,
            "records_written_to_shards": count,
        },
    )
    return training.URDMUTrainingRequest(
        dataset="ucf_crime",
        feature_store=str(root),
        train_manifest=str(manifest),
        source_contract_path=str(contract),
        source_contract_sha256=sha256_file(contract),
        upstream_dir="not-accessed-until-training",
        output_root=str(tmp_path / "runs"),
        device="cpu",
        run_mode="engineering",
        steps=2,
        bags_per_class=per_class,
        run_id="smoke",
    )


@pytest.fixture
def upstream():
    root = Path(
        os.environ.get(
            "URDMU_UPSTREAM_DIR", "outputs/icassp2027/research/author-recipes-20260918/UR-DMU"
        )
    ).resolve()
    if not root.is_dir():
        pytest.skip("pinned external UR-DMU checkout unavailable")
    extra = None
    if importlib.util.find_spec("einops") is None:
        site = Path("C:/Users/lenovo/anaconda3/envs/pytorch/Lib/site-packages")
        if not (site / "einops").is_dir():
            pytest.skip("an existing real einops installation is required")
        extra = str(site)
        sys.path.append(extra)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield root
    finally:
        torch.set_num_threads(threads)
        if extra is not None:
            sys.path.remove(extra)


@pytest.mark.parametrize("length", [1, 2, 199, 200, 201, 65536, 70001])
def test_author_bins_use_int64_and_exact_empty_bin_rule(length):
    features = np.arange(length, dtype=np.float32)[:, None]
    output = training.author_temporal_bins(features)
    edges = np.linspace(0, length, 201).astype(np.int64)
    expected = np.asarray(
        [
            features[a:b].mean(axis=0) if a < b else features[a]
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float32,
    )
    assert output.shape == (200, 1) and np.array_equal(output, expected)
    if length > 65535:
        assert output[-1, 0] > 65000


def test_formal_budget_cannot_be_replaced_by_engineering_smoke(source):
    with pytest.raises(ValueError, match="3000 steps"):
        replace(source, run_mode="formal")
    with pytest.raises(ValueError, match="extraction contract"):
        replace(source, run_mode="formal", steps=3000, bags_per_class=64)


def test_development_budget_cannot_be_replaced_by_engineering_smoke(source):
    with pytest.raises(ValueError, match="3000 steps"):
        replace(source, run_mode="development")
    with pytest.raises(ValueError, match="extraction contract"):
        replace(source, run_mode="development", steps=3000, bags_per_class=64)


def test_development_source_requires_exact_complete_fit_role(source, monkeypatch):
    from vadbench.data import official_training

    full = list(load_manifest_jsonl(source.train_manifest))
    roles = {full[0].video_id: "fit", full[1].video_id: "select", full[2].video_id: "fit", full[3].video_id: "confirm"}
    full = [replace(record, metadata={**record.metadata, "original_role": roles[record.video_id]}) for record in full]
    fit_manifest = write_manifest_jsonl((full[0], full[2]), Path(source.feature_store).parent / "fit.jsonl")
    lock = Path(source.source_contract_path).parent / "roles.json"
    _write(lock, {"partitions": roles})

    def load(path, expected_sha256, **_kwargs):
        assert Path(path).resolve() == Path(source.source_contract_path).resolve()
        assert expected_sha256 == source.source_contract_sha256
        return tuple(full), {
            "dataset": source.dataset,
            "inputs": {"role_lock": {"path": lock.name, "sha256": sha256_file(lock)}},
            "source_hashes": {},
        }

    monkeypatch.setattr(official_training, "load_official_training_view", load)
    request = replace(
        source,
        run_mode="development",
        train_manifest=str(fit_manifest),
        steps=3000,
        bags_per_class=64,
        extraction_contract_path=str(lock),
        extraction_contract_sha256=sha256_file(lock),
    )
    records, receipt, _ = training._source(request)
    assert [record.video_id for record in records] == [full[0].video_id, full[2].video_id]
    assert receipt["development_role_lock"] == {"path": str(lock.resolve()), "sha256": sha256_file(lock)}
    wrong = write_manifest_jsonl((full[0], full[1]), Path(source.feature_store).parent / "wrong.jsonl")
    with pytest.raises(ValueError, match="complete original fit"):
        training._source(replace(request, train_manifest=str(wrong)))


def test_uniform32_source_is_rejected_before_training(source):
    path = Path(source.feature_store) / "resolved.json"
    value = json.loads(path.read_text())
    value["spec"]["sampling_kind"] = "uniform_full"
    value["spec"]["sampling"]["regime"] = "train_32"
    _write(path, value)
    with pytest.raises(ValueError, match="train_32 cannot be upsampled"):
        training.run_urdmu_training(source)


@pytest.mark.parametrize("mutation", ["missing_window", "changed_indices"])
def test_dense_coverage_and_actual_native_windows_are_mandatory(source, mutation):
    path = Path(source.feature_store) / "index.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_window":
        rows.pop(0)
    else:
        rows[0]["metadata"]["sampling"]["frame_indices"][0] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    with pytest.raises(ValueError, match="uncovered|frame_indices|sampler window"):
        training.run_urdmu_training(source)


def test_cpu_guard_rejects_visible_cuda_before_checkpoint_access(source, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CPU guard queried visible CUDA")
    )
    with pytest.raises(ValueError, match="hide CUDA"):
        training.run_urdmu_training(source)


def test_real_two_step_training_updates_all_components_and_exactly_reloads(
    source, upstream, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU checkpoint initialized CUDA RNG")
    )
    request = replace(source, upstream_dir=str(upstream))
    result = training.run_urdmu_training(request)
    assert result["status"] == "completed" and result["run_mode"] == "engineering"
    root = Path(result["run_dir"])
    qa = json.loads((root / "training_qa.json").read_text())
    assert qa["checkpoint"]["sha256"] == sha256_file(Path(result["checkpoint_path"]))
    assert qa["nonzero_gradient_steps"] == 2 and all(
        qa["changed_parameters_by_component"][c] > 0 for c in training.COMPONENTS
    )
    assert qa["reload_parity"]["exact_equal"] and qa["reload_parity"]["flag"] == "Test"
    assert qa["reload_parity"]["training"] is False and qa["reload_parity"]["shape"] == [4, 200]
    steps = [json.loads(line) for line in (root / "training-steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in steps] == [1, 2]
    assert all(
        r["normal_first"]
        and r["normal_bags"] == r["abnormal_bags"] == 2
        and "distance" in r["losses"]
        for r in steps
    )
    assert len(list((root / "checkpoints").glob("*.pt"))) == 1
    cache = json.loads((Path(result["aggregation_cache"]) / "receipt.json").read_text())
    assert cache["shape"] == [4, 200, 768] and cache["data_role"] == "engineering_only"
    assert cache["source_dense_lengths"] == [6] * 4
    # A second seed consumes the same preaggregated verified cache, without ever
    # re-reading dense NPZ sequences during its optimizer loop.
    monkeypatch.setattr(
        FeatureStore,
        "load_bundle",
        lambda *a, **k: pytest.fail("dense features reloaded after aggregation cache reuse"),
    )
    repeated = training.run_urdmu_training(
        replace(
            request,
            seed=1,
            steps=1,
            run_id="seed1",
            aggregation_cache=result["aggregation_cache"],
            aggregation_cache_receipt_sha256=result["aggregation_cache_receipt_sha256"],
        )
    )
    assert repeated["aggregation_cache"] == result["aggregation_cache"]


def test_npz_content_tamper_cannot_enter_training_cache(source):
    row = next(FeatureStore(source.feature_store).iter_records())
    reference = row.arrays["pooled"]
    (Path(source.feature_store) / reference.path).write_bytes(b"tampered NPZ")
    with pytest.raises((ValueError, OSError)):
        training.run_urdmu_training(source)


@pytest.mark.parametrize("source", [1], indirect=True)
def test_one_normal_one_anomaly_native_acceptance_batch_runs_two_real_steps(source, upstream):
    result = training.run_urdmu_training(replace(source, upstream_dir=str(upstream)))
    qa = json.loads((Path(result["run_dir"]) / "training_qa.json").read_text())
    assert qa["checkpoint"]["step"] == 2 and qa["reload_parity"]["shape"] == [2, 200]
    assert all(qa["changed_parameters_by_component"][c] > 0 for c in training.COMPONENTS)


def test_reused_aggregation_receipt_requires_the_original_external_pin(source):
    records, _, sources = training._source(source)
    document, representation, _, _ = training._dense_source(source, records, sources)
    preparation = Path(source.output_root) / "cache-preparation"
    preparation.mkdir(parents=True)
    _, _, cache_root, _ = training._aggregate(
        source, records, document, representation, sources, preparation
    )
    receipt_path = cache_root / "receipt.json"
    expected = sha256_file(receipt_path)
    altered = json.loads(receipt_path.read_text())
    altered["bags_sha256"] = "f" * 64
    _write(receipt_path, altered)
    with pytest.raises(ValueError, match="cache receipt SHA"):
        training.run_urdmu_training(
            replace(
                source,
                steps=1,
                run_id="tampered-cache",
                aggregation_cache=str(cache_root),
                aggregation_cache_receipt_sha256=expected,
            )
        )


def test_temporal_annotations_cannot_enter_even_engineering_training(source):
    from vadbench.data.manifest import SupervisionAnnotation, TemporalSpan, load_manifest_jsonl

    rows = list(load_manifest_jsonl(source.train_manifest))
    rows[2] = replace(
        rows[2],
        annotations=(
            SupervisionAnnotation(scope="frame", is_anomaly=True, span=TemporalSpan(3, 8, "frame")),
        ),
    )
    write_manifest_jsonl(rows, source.train_manifest)
    path = Path(source.source_contract_path)
    contract = json.loads(path.read_text())
    contract["training_manifest"]["sha256"] = sha256_file(Path(source.train_manifest))
    _write(path, contract)
    with pytest.raises(ValueError, match="temporal supervision"):
        training.run_urdmu_training(replace(source, source_contract_sha256=sha256_file(path)))


def _extraction_contract(request):
    root = Path(request.feature_store)
    contract = root.parent / "extraction-contract.json"

    def entry(path):
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    _write(
        contract,
        {
            "schema": "icassp2027.official-dense-extraction/v1",
            "status": "ready",
            "data_role": "official-fulltrain-final",
            "dataset": request.dataset,
            "training_view_contract": entry(Path(request.source_contract_path)),
            "training_manifest": entry(Path(request.train_manifest)),
            "protocol": entry(Path(request.protocol_path)),
            "feature_store": {
                "root": str(root),
                **{
                    k: entry(root / v)
                    for k, v in (
                        ("resolved", "resolved.json"),
                        ("status", "status.json"),
                        ("index", "index.jsonl"),
                    )
                },
            },
        },
    )
    return replace(
        request,
        run_mode="formal",
        steps=3000,
        bags_per_class=64,
        extraction_contract_path=str(contract),
        extraction_contract_sha256=sha256_file(contract),
    )


def test_formal_path_calls_authoritative_training_view_without_subset_fallback(source, monkeypatch):
    from vadbench.data import official_training

    calls = []

    def reject(path, digest, **kwargs):
        calls.append((str(path), digest))
        raise ValueError("independent complete official view rejected")

    monkeypatch.setattr(official_training, "load_official_training_view", reject)
    with pytest.raises(ValueError, match="independent complete official view rejected"):
        training.run_urdmu_training(_extraction_contract(source))
    assert calls == [
        (str(Path(source.source_contract_path).resolve()), source.source_contract_sha256)
    ]


def test_coherent_dense_raw_sha_tamper_rejected_against_independent_authority(source):
    from vadbench.data.manifest import load_manifest_jsonl

    root = Path(source.feature_store)
    document = json.loads((root / "resolved.json").read_text())
    evidence = document["data_content_evidence"]
    target = evidence["videos"][0]["video_id"]
    evidence["videos"][0]["sha256"] = "f" * 64
    evidence["source_digest"] = compute_encoder_fingerprint(
        {
            "canonical_manifest": evidence["canonical_manifest_sha256"],
            "video_contents": evidence["videos"],
        }
    )
    sampling = replace(
        SamplingIdentity.from_mapping(document["spec"]["sampling"]),
        source_digest=evidence["source_digest"],
    )
    representation = RepresentationIdentity.from_mapping(document["spec"]["representation"])
    document["spec"]["sampling"] = sampling.to_dict()
    document["runtime"]["sampling_fingerprint"] = sampling.fingerprint
    document["runtime"]["source_data_digest"] = evidence["source_digest"]
    fingerprint = compute_encoder_fingerprint(
        {"paper_pooled_cache": _semantic_runtime_identity(document["runtime"])}
    )
    document["encoder_fingerprint"] = fingerprint
    document["paper_identity"] = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    _write(root / "resolved.json", document)
    status = json.loads((root / "status.json").read_text())
    status["encoder_fingerprint"] = fingerprint
    _write(root / "status.json", status)
    rows = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    for row in rows:
        row["encoder_fingerprint"] = fingerprint
        row["metadata"]["paper_identity"] = document["paper_identity"]
        if row["video_id"] == target:
            row["metadata"]["source_video"]["content_sha256"] = "f" * 64
    (root / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8"
    )
    with pytest.raises(ValueError, match="independent official training authority"):
        training._dense_source(
            _extraction_contract(source), load_manifest_jsonl(source.train_manifest), {}
        )
