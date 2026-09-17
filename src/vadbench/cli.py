"""VADBench 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vadbench import __version__
from vadbench.artifacts import ArtifactStore, PredictionRecord, file_identity, record_stage
from vadbench.benchmark_plan import run_benchmark_plan
from vadbench.checkpoints import (
    CheckpointError,
    fetch_checkpoint,
    load_checkpoint_registry,
    verify_checkpoint,
)
from vadbench.config import ConfigError, load_experiment, load_yaml
from vadbench.data.audit import audit_ucf_crime_dataset
from vadbench.data.enrich import enrich_video_info
from vadbench.data.labels import LabelProjectionError
from vadbench.data.manifest import ManifestError, load_manifest_jsonl
from vadbench.data.ucf_crime import (
    UCFCrimeImportResult,
    import_ucf_crime,
    write_ucf_crime_manifests,
)
from vadbench.data.video import iter_fixed_segment_batches, iter_streaming_chunk_batches
from vadbench.doctor import diagnostics_json
from vadbench.features import FeatureStore, atomic_write_json
from vadbench.orchestration import (
    compression_from_experiment,
    create_encoder_from_experiment,
    encoder_identity,
    iter_microbatches,
)
from vadbench.registry import ENCODER_REGISTRY, RegistryError
from vadbench.smoke import run_encoder_smoke_v2, write_smoke_result_v2

DEFAULT_REGISTRY = Path("registry/checkpoints.yaml")


def evaluate_manifest_predictions(*args: Any, **kwargs: Any) -> Any:
    """Load frame-level evaluation code only when the command needs it."""

    from vadbench.engine.evaluate import evaluate_manifest_predictions as implementation

    return implementation(*args, **kwargs)


def _feature_extraction_engine(*args: Any, **kwargs: Any) -> Any:
    """Construct the extraction engine without importing torch during CLI discovery."""

    from vadbench.engine.extract import FeatureExtractionEngine

    return FeatureExtractionEngine(*args, **kwargs)


def predict_feature_head(*args: Any, **kwargs: Any) -> Any:
    """Load the prediction stack only when ``predict`` executes."""

    from vadbench.engine.predict import predict_feature_head as implementation

    return implementation(*args, **kwargs)


def train_feature_head(*args: Any, **kwargs: Any) -> Any:
    """Load the training stack only when ``train`` executes."""

    from vadbench.engine.runner import train_feature_head as implementation

    return implementation(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vadbench",
        description="视频编码器、UCF-Crime 训练评测与缓存压缩实验框架",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="只读检查运行环境和目录")
    doctor.add_argument("--project-root", default=".")
    doctor.set_defaults(handler=_doctor)

    config = sub.add_parser("config", help="实验配置操作")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_sub.add_parser("validate", help="校验实验 YAML")
    config_validate.add_argument("path")
    config_validate.add_argument("--defaults")
    config_validate.set_defaults(handler=_config_validate)

    encoders = sub.add_parser("encoders", help="查看可插拔 encoder 及其真实能力")
    encoders_sub = encoders.add_subparsers(dest="encoders_command", required=True)
    encoders_list = encoders_sub.add_parser("list", help="列出已注册 encoder")
    encoders_list.set_defaults(handler=_encoders_list)
    encoders_inspect = encoders_sub.add_parser("inspect", help="查看 encoder capability")
    encoders_inspect.add_argument("encoder_id")
    encoders_inspect.set_defaults(handler=_encoders_inspect)

    integrations = sub.add_parser("integrations", help="统一模型接入目录、预检与矩阵冒烟")
    integrations_sub = integrations.add_subparsers(dest="integrations_command", required=True)
    integrations_list = integrations_sub.add_parser("list", help="列出25项接入目标")
    integrations_list.set_defaults(handler=_integrations_list)

    integrations_preflight = integrations_sub.add_parser(
        "preflight", help="只读预检配置、锁和本地资产"
    )
    integrations_preflight.add_argument(
        "--id", "--encoder", dest="integration_ids", action="append"
    )
    integrations_preflight.add_argument("--family", action="append")
    integrations_preflight.add_argument("--run-mode", action="append")
    integrations_preflight.add_argument("--runtime", action="append")
    integrations_preflight.add_argument("--limit", type=int)
    integrations_preflight.set_defaults(handler=_integrations_preflight)

    integrations_smoke = integrations_sub.add_parser(
        "smoke", help="用当前视频运行一个或指定的真实接入目标"
    )
    integrations_smoke.add_argument("--video", required=True)
    integrations_smoke.add_argument("--id", "--encoder", dest="integration_ids", action="append")
    integrations_smoke.add_argument("--device")
    integrations_smoke.add_argument("--output-root", default="outputs/encoder-integration")
    integrations_smoke.add_argument("--limit", type=int, default=1)
    integrations_smoke.add_argument("--skip-preflight", action="store_true")
    integrations_smoke.set_defaults(handler=_integrations_smoke)

    integrations_matrix = integrations_sub.add_parser(
        "matrix", help="生成25项预检矩阵，使用--execute才运行真实前向"
    )
    integrations_matrix.add_argument("--video", required=True)
    integrations_matrix.add_argument("-c", "--config")
    integrations_matrix.add_argument("--id", "--encoder", dest="integration_ids", action="append")
    integrations_matrix.add_argument("--family", action="append")
    integrations_matrix.add_argument("--run-mode", action="append")
    integrations_matrix.add_argument("--runtime", action="append")
    integrations_matrix.add_argument("--device")
    integrations_matrix.add_argument("--output-root", default="outputs/encoder-integration")
    integrations_matrix.add_argument("--matrix-path")
    integrations_matrix.add_argument("--limit", type=int)
    integrations_matrix.add_argument("--execute", action="store_true")
    integrations_matrix.add_argument("--skip-preflight", action="store_true")
    integrations_matrix.set_defaults(handler=_integrations_matrix)

    manifest = sub.add_parser("manifest", help="数据清单导入与校验")
    manifest_sub = manifest.add_subparsers(dest="manifest_command", required=True)

    manifest_import = manifest_sub.add_parser(
        "import-ucf", help="从 UCF-Crime 官方 split/annotation 构造清单"
    )
    manifest_import.add_argument("--dataset-root", required=True)
    manifest_import.add_argument("--train-split", action="append", required=True)
    manifest_import.add_argument("--temporal-annotations", required=True)
    manifest_import.add_argument("--test-split", action="append")
    manifest_import.add_argument("--uca-captions", action="append")
    manifest_import.add_argument("--output-dir", required=True)
    manifest_import.add_argument("--require-files", action="store_true")
    manifest_import.add_argument(
        "--probe-video-info",
        action="store_true",
        help="读取实际视频，补齐 num_frames/fps/duration（正式帧级评测建议启用）",
    )
    manifest_import.set_defaults(handler=_manifest_import_ucf)

    manifest_validate = manifest_sub.add_parser("validate", help="校验 manifest JSONL")
    manifest_validate.add_argument("path")
    manifest_validate.add_argument("--dataset-root")
    manifest_validate.add_argument("--require-files", action="store_true")
    manifest_validate.set_defaults(handler=_manifest_validate)

    manifest_enrich = manifest_sub.add_parser("enrich", help="从实际视频补齐容器元数据")
    manifest_enrich.add_argument("path")
    manifest_enrich.add_argument("--dataset-root", required=True)
    manifest_enrich.add_argument("--output", required=True)
    manifest_enrich.set_defaults(handler=_manifest_enrich)

    manifest_audit = manifest_sub.add_parser(
        "audit-ucf", help="审计真实 UCF-Crime 文件、容器与官方 1610/290 协议"
    )
    manifest_audit.add_argument("--dataset-root", required=True)
    manifest_audit.add_argument("--train-manifest", required=True)
    manifest_audit.add_argument("--test-manifest", required=True)
    manifest_audit.add_argument("--output", required=True)
    manifest_audit.add_argument(
        "--deep-hash",
        action="store_true",
        help="逐文件读取并计算 SHA256；默认只 stat 与 probe 容器",
    )
    manifest_audit.set_defaults(handler=_manifest_audit_ucf)

    weights = sub.add_parser("weights", help="权重注册、下载和校验")
    weights.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    weights_sub = weights.add_subparsers(dest="weights_command", required=True)

    weights_list = weights_sub.add_parser("list", help="列出已注册权重")
    weights_list.set_defaults(handler=_weights_list)

    weights_verify = weights_sub.add_parser("verify", help="校验本地权重 SHA256")
    weights_verify.add_argument("checkpoint_id")
    weights_verify.add_argument("path")
    weights_verify.set_defaults(handler=_weights_verify)

    weights_fetch = weights_sub.add_parser("fetch", help="下载固定 revision 的权重")
    weights_fetch.add_argument("checkpoint_id")
    weights_fetch.add_argument("path")
    weights_fetch.add_argument("--accept-license", required=True)
    weights_fetch.add_argument("--local-files-only", action="store_true")
    weights_fetch.set_defaults(handler=_weights_fetch)

    evaluate = sub.add_parser("evaluate", help="按标准帧级协议评测预测 JSONL")
    evaluate.add_argument("-c", "--config", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--output")
    evaluate.add_argument(
        "--protocol", choices=("official", "subset", "generic"), default="official"
    )
    evaluate.add_argument("--dataset-audit", help="official 模式所需的已通过数据审计 JSON")
    evaluate.set_defaults(handler=_evaluate)

    extract = sub.add_parser("extract", help="从 manifest 抽取标准化 encoder 特征")
    extract.add_argument("-c", "--config", required=True)
    extract.add_argument("--manifest")
    extract.add_argument("--split", choices=("train", "test"), default="train")
    extract.add_argument("--output")
    extract.add_argument("--limit-videos", type=int)
    extract.set_defaults(handler=_extract)

    train = sub.add_parser("train", help="用标准特征仓训练 MIL/时序检测头")
    train.add_argument("-c", "--config", required=True)
    train.add_argument("--features", required=True)
    train.add_argument("--train-manifest")
    train.add_argument("--validation-manifest")
    train.add_argument("--output")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--device")
    train.set_defaults(handler=_train)

    predict = sub.add_parser("predict", help="从冻结特征和已验证 checkpoint 生成标准预测")
    predict.add_argument("-c", "--config", required=True)
    predict.add_argument("--features", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--manifest")
    predict.add_argument("--output")
    predict.add_argument("--device")
    predict.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="允许非完整帧覆盖；正式 UCF 评测不得启用",
    )
    predict.set_defaults(handler=_predict)

    smoke = sub.add_parser("smoke", help="用真实权重对一个视频执行 encoder 冒烟")
    smoke.add_argument("-c", "--config", required=True)
    smoke.add_argument("--video", required=True)
    smoke.add_argument("--encoder")
    smoke.add_argument("--device")
    smoke.add_argument("--chunk-frames", type=int)
    smoke.add_argument("--kv-size", type=int)
    smoke.add_argument(
        "--native-compression-mode",
        choices=("off", "predict", "static_pseudo"),
    )
    smoke.add_argument("--chunks", type=int, default=2)
    smoke.add_argument("--output")
    smoke.set_defaults(handler=_smoke)

    benchmark = sub.add_parser("benchmark", help="运行逐 case 加载的固定/流式性能基准")
    benchmark.add_argument("-c", "--config", required=True)
    benchmark.add_argument("--video")
    benchmark.add_argument("--device")
    benchmark.add_argument("--warmup", type=int)
    benchmark.add_argument("--repeat", type=int)
    benchmark.add_argument("--output")
    benchmark.set_defaults(handler=_benchmark)
    return parser


def _doctor(args: argparse.Namespace) -> int:
    print(diagnostics_json(args.project_root))
    return 0


def _config_validate(args: argparse.Namespace) -> int:
    config = load_experiment(args.path, defaults=args.defaults)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def _capability_payload(capabilities) -> dict[str, object]:
    return {
        "supports_fixed_clip": capabilities.supports_fixed_clip,
        "supports_streaming": capabilities.supports_streaming,
        "supports_training": capabilities.supports_training,
        "cache_kinds": sorted(item.value for item in capabilities.cache_kinds),
        "cache_access": capabilities.cache_access,
        "fixed_num_frames": capabilities.fixed_num_frames,
        "min_frames": capabilities.min_frames,
        "max_frames": capabilities.max_frames,
    }


def _encoders_list(args: argparse.Namespace) -> int:
    del args
    payload = [
        {
            "id": spec.name,
            "capabilities": _capability_payload(spec.capabilities),
            "metadata": dict(spec.metadata),
        }
        for spec in ENCODER_REGISTRY.specs()
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _encoders_inspect(args: argparse.Namespace) -> int:
    spec = ENCODER_REGISTRY.get_spec(args.encoder_id)
    print(
        json.dumps(
            {
                "id": spec.name,
                "target": spec.target_path,
                "capabilities": _capability_payload(spec.capabilities),
                "metadata": dict(spec.metadata),
                "default_kwargs": dict(spec.default_kwargs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _integrations_list(args: argparse.Namespace) -> int:
    del args
    from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG

    payload = []
    for record in DEFAULT_INTEGRATION_CATALOG.integrations:
        payload.append(
            {
                "id": record.id,
                "display_name": record.display_name,
                "family": record.family,
                "status": record.status,
                "backend": record.backend,
                "run_mode": record.run_mode,
                "feature_stage": record.feature_stage,
                "runtime": record.environment.runtime,
                "environment_profile": record.environment.profile,
                "definition": record.definition,
                "checkpoint": record.checkpoint.registry_id,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _integration_filter_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "integration_ids": getattr(args, "integration_ids", None),
        "families": getattr(args, "family", None),
        "run_modes": getattr(args, "run_mode", None),
        "runtimes": getattr(args, "runtime", None),
        "limit": getattr(args, "limit", None),
    }


def _integrations_preflight(args: argparse.Namespace) -> int:
    from vadbench.engine.integration_matrix import preflight_integrations

    results = preflight_integrations(
        project_root=Path.cwd(),
        **_integration_filter_args(args),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _integrations_smoke(args: argparse.Namespace) -> int:
    from vadbench.engine.integration_matrix import run_integration_matrix

    config: dict[str, object] = {}
    if args.device:
        config["encoder"] = {"device": args.device}
    result = run_integration_matrix(
        config=config,
        video_path=args.video,
        project_root=Path.cwd(),
        output_root=args.output_root,
        skip_preflight=args.skip_preflight,
        write_results=True,
        **_integration_filter_args(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _integrations_matrix(args: argparse.Namespace) -> int:
    from vadbench.engine.integration_matrix import run_integration_matrix

    config = args.config
    if config is None:
        config = {}
    elif isinstance(config, (str, Path)):
        config = load_yaml(config)
    if args.device:
        config = dict(config)
        encoder = dict(config.get("encoder", {}))
        encoder["device"] = args.device
        config["encoder"] = encoder
    result = run_integration_matrix(
        config=config,
        video_path=args.video,
        project_root=Path.cwd(),
        output_root=args.output_root,
        matrix_path=args.matrix_path,
        skip_preflight=args.skip_preflight,
        write_results=True,
        execute=args.execute,
        **_integration_filter_args(args),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _manifest_import_ucf(args: argparse.Namespace) -> int:
    result = import_ucf_crime(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        temporal_annotations=args.temporal_annotations,
        test_split=args.test_split,
        uca_captions=args.uca_captions,
        require_files=args.require_files,
    )
    if args.probe_video_info:
        result = UCFCrimeImportResult(
            train=enrich_video_info(result.train, args.dataset_root),
            test=enrich_video_info(result.test, args.dataset_root),
        )
    train_path, test_path = write_ucf_crime_manifests(result, args.output_dir)
    print(
        json.dumps(
            {
                "train": {"path": str(train_path), "videos": len(result.train)},
                "test": {"path": str(test_path), "videos": len(result.test)},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _manifest_validate(args: argparse.Namespace) -> int:
    records = load_manifest_jsonl(
        args.path,
        dataset_root=args.dataset_root,
        require_files=args.require_files,
    )
    splits: dict[str, int] = {}
    scopes: dict[str, int] = {}
    for record in records:
        splits[record.split.value] = splits.get(record.split.value, 0) + 1
        for scope in record.supervision_scopes:
            scopes[scope.value] = scopes.get(scope.value, 0) + 1
    print(
        json.dumps(
            {
                "path": str(Path(args.path).resolve()),
                "videos": len(records),
                "splits": splits,
                "scopes": scopes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _manifest_enrich(args: argparse.Namespace) -> int:
    from vadbench.data.manifest import write_manifest_jsonl

    records = load_manifest_jsonl(args.path, dataset_root=args.dataset_root, require_files=True)
    enriched = enrich_video_info(records, args.dataset_root)
    output = write_manifest_jsonl(
        enriched,
        args.output,
        dataset_root=args.dataset_root,
        require_files=True,
    )
    print(json.dumps({"path": str(output), "videos": len(enriched)}, ensure_ascii=False, indent=2))
    return 0


def _manifest_audit_ucf(args: argparse.Namespace) -> int:
    report = audit_ucf_crime_dataset(
        args.dataset_root,
        args.train_manifest,
        args.test_manifest,
        deep_hash=args.deep_hash,
    )
    output = Path(args.output)
    atomic_write_json(output, report)
    summary = {
        "status": report["status"],
        "passed": report["passed"],
        "output": str(output.resolve()),
        "observed": report["observed"],
        "files": report["files"],
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


def _load_spec(args: argparse.Namespace):
    specs = load_checkpoint_registry(args.registry)
    try:
        return specs[args.checkpoint_id]
    except KeyError as exc:
        raise CheckpointError(f"未知 checkpoint：{args.checkpoint_id}") from exc


def _weights_list(args: argparse.Namespace) -> int:
    specs = load_checkpoint_registry(args.registry)
    payload = [
        {
            "id": spec.id,
            "adapter": spec.adapter,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "notes": spec.notes,
        }
        for spec in specs.values()
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _weights_verify(args: argparse.Namespace) -> int:
    actual = verify_checkpoint(_load_spec(args), args.path)
    print(json.dumps(actual, ensure_ascii=False, indent=2))
    return 0


def _weights_fetch(args: argparse.Namespace) -> int:
    path = fetch_checkpoint(
        _load_spec(args),
        args.path,
        accepted_license=args.accept_license,
        local_files_only=args.local_files_only,
    )
    print(path)
    return 0


def _load_prediction_jsonl(path: str | Path) -> tuple[PredictionRecord, ...]:
    result: list[PredictionRecord] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
                result.append(PredictionRecord.from_dict(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: 非法 prediction record：{exc}") from exc
    if not result:
        raise ValueError(f"prediction JSONL 为空：{source}")
    return tuple(result)


def _evaluate(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    manifest_path = args.manifest or config["dataset"]["test_manifest"]
    output_path = Path(
        args.output
        or (
            Path(config["output"]["root"])
            / config["output"]["run_name"]
            / "evaluation/metrics.json"
        )
    )
    with record_stage(
        output_path.parent,
        "evaluate",
        config={**config, "protocol": args.protocol},
        inputs={
            "manifest": manifest_path,
            "predictions": args.predictions,
            "audit": args.dataset_audit,
        },
    ) as stage:
        result = evaluate_manifest_predictions(
            _load_prediction_jsonl(args.predictions),
            manifest_path,
            protocol=args.protocol,
            audit_report=args.dataset_audit,
        )
        metrics = result.to_dict()
        atomic_write_json(output_path, metrics)
        np.savez_compressed(output_path.with_name("frame_scores.npz"), **result.frame_scores)
        stage["outputs"] = {"metrics": file_identity(output_path)}
    print(
        json.dumps({"metrics": metrics, "output": str(output_path)}, ensure_ascii=False, indent=2)
    )
    return 0


def _extraction_batches(records, config):
    """Use the same sampling implementation for fixed clips and streamed MIL segments."""
    sampler = config.get("sampler", {})
    streaming = config.get("streaming", {})
    if sampler.get("kind", "uniform_segments") == "chronological_stream":
        if not streaming.get("enabled"):
            raise ValueError("chronological_stream requires streaming.enabled=true")
        return iter_streaming_chunk_batches(
            records,
            config["dataset"]["root"],
            chunk_frames=int(streaming["chunk_frames"]),
            sample_fps=streaming.get("sample_fps"),
        )
    if sampler.get("kind", "uniform_segments") != "uniform_segments":
        raise ValueError("sampler.kind must be uniform_segments or chronological_stream")
    return iter_fixed_segment_batches(
        records,
        config["dataset"]["root"],
        num_segments=int(sampler.get("segments_per_video", 32)),
        clip_frames=int(sampler.get("clip_frames", 16)),
        frame_stride=int(sampler.get("frame_stride", 2)),
        position=str(sampler.get("position", "center")),
        seed=sampler.get("seed"),
    )


def _extract(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    manifest_path = args.manifest or config["dataset"][f"{args.split}_manifest"]
    project_root = Path.cwd().resolve()
    run_name = str(config["output"]["run_name"])
    run_dir = (Path(args.output or config["output"]["root"]) / run_name).resolve()
    with record_stage(
        run_dir, "extract", config=config, inputs={"manifest": manifest_path}
    ) as stage:
        records = load_manifest_jsonl(manifest_path)
        if args.limit_videos is not None:
            if args.limit_videos <= 0:
                raise ValueError("--limit-videos 必须大于 0")
            records = records[: args.limit_videos]
        if not records:
            raise ValueError("没有可抽取的视频")
        adapter, definition = create_encoder_from_experiment(config, project_root=project_root)
        resolved_identity = definition.get("identity") or encoder_identity(
            definition, project_root=project_root
        )
        fingerprint_manifest = {
            "encoder": resolved_identity,
            "sampler": config.get("sampler", {}),
            "streaming": config.get("streaming", {}),
        }
        artifact_store = ArtifactStore(run_dir, run_id=stage["run_id"])
        feature_store = FeatureStore(run_dir / "features")
        engine = _feature_extraction_engine(
            adapter=adapter,
            manifest=fingerprint_manifest,
            feature_store=feature_store,
            artifact_store=artifact_store,
            train=False,
        )
        stage["encoder_fingerprint"] = engine.encoder_fingerprint
        stage["resolved_encoder"] = definition
        if config.get("streaming", {}).get("enabled", False):
            compression = compression_from_experiment(config)
            extracted = []
            for record in records:
                chunks = iter_microbatches(iter(_extraction_batches((record,), config)), 1)
                extracted.extend(
                    engine.extract_stream(
                        chunks,
                        video_id=record.video_id,
                        compression=compression,
                    )
                )
        else:
            batches = _extraction_batches(records, config)
            batch_size = int(config["encoder"].get("micro_batch_size", 4))
            extracted = engine.extract(iter_microbatches(iter(batches), batch_size))
        payload = {
            "run_dir": str(run_dir),
            "videos": len(records),
            "feature_records": len(extracted),
            "encoder_fingerprint": engine.encoder_fingerprint,
            "feature_index": str(feature_store.index_path),
            "cache_telemetry": str(artifact_store.cache_telemetry_path),
        }
        stage["outputs"] = {"feature_index": file_identity(feature_store.index_path)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps 必须大于 0")
        config = {**config, "training": {**config.get("training", {}), "max_steps": args.max_steps}}
    train_path = args.train_manifest or config["dataset"]["train_manifest"]
    validation_path = args.validation_manifest or config["dataset"].get("validation_manifest")
    output_dir = Path(
        args.output or (Path(config["output"]["root"]) / config["output"]["run_name"] / "training")
    )
    with record_stage(
        output_dir,
        "train",
        config=config,
        inputs={
            "train_manifest": train_path,
            "validation_manifest": validation_path,
            "feature_index": Path(args.features) / "index.jsonl",
        },
    ) as stage:
        result = train_feature_head(
            config,
            feature_store=args.features,
            train_manifest=train_path,
            validation_manifest=validation_path,
            output_dir=output_dir,
            device=args.device,
        )
        payload = result.to_dict()
        stage["outputs"] = payload
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    manifest_path = args.manifest or config["dataset"]["test_manifest"]
    output = Path(
        args.output
        or (
            Path(config["output"]["root"])
            / config["output"]["run_name"]
            / "predictions/predictions.jsonl"
        )
    )
    with record_stage(
        output.parent,
        "predict",
        config=config,
        inputs={
            "manifest": manifest_path,
            "checkpoint": args.checkpoint,
            "feature_index": Path(args.features) / "index.jsonl",
        },
    ) as stage:
        records = predict_feature_head(
            config,
            args.features,
            manifest_path,
            args.checkpoint,
            output,
            device=args.device,
            strict_coverage=not args.allow_incomplete_coverage,
        )
        stage["outputs"] = {"predictions": file_identity(output)}
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "prediction_records": len(records),
                "videos": len({item.video_id for item in records}),
                "strict_coverage": not args.allow_incomplete_coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    if args.encoder:
        config = {**config, "encoder": {**config["encoder"], "adapter": args.encoder}}
    params = dict(config["encoder"].get("params", {}))
    if args.device:
        params["device"] = args.device
    if args.kv_size is not None:
        if args.kv_size <= 0:
            raise ValueError("--kv-size 必须大于 0")
        params["kv_size"] = args.kv_size
    if args.native_compression_mode:
        params["native_compression_mode"] = args.native_compression_mode
    if params:
        config = {**config, "encoder": {**config["encoder"], "params": params}}
    if args.chunk_frames is not None:
        if args.chunk_frames <= 0:
            raise ValueError("--chunk-frames 必须大于 0")
        config = {
            **config,
            "streaming": {**config.get("streaming", {}), "chunk_frames": args.chunk_frames},
        }
    result = run_encoder_smoke_v2(
        config,
        args.video,
        project_root=Path.cwd(),
        max_chunks=args.chunks,
    )
    output = args.output or (
        Path(config["output"]["root"])
        / config["output"]["run_name"]
        / "smoke"
        / f"{config['encoder']['adapter']}.json"
    )
    output_path = write_smoke_result_v2(result, output)
    print(json.dumps({**result, "output": str(output_path)}, ensure_ascii=False, indent=2))
    return {"smoke_pass": 0, "blocked": 2}.get(result.get("status"), 1)


def _benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark_plan(
        args.config,
        project_root=Path.cwd(),
        video=args.video,
        device=args.device,
        warmup=args.warmup,
        repeat_count=args.repeat,
        output=args.output,
    )
    plan = load_yaml(args.config)
    output = Path(args.output or plan["benchmark"]["output"]).resolve()
    print(
        json.dumps(
            {
                "output": str(output),
                "comparison": result["comparison"],
                "cases": [
                    {
                        "name": case["name"],
                        "mode": case["mode"],
                        "aggregate": case["aggregate"],
                        "accuracy_eligibility": case["accuracy_eligibility"],
                    }
                    for case in result["cases"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        ConfigError,
        CheckpointError,
        LabelProjectionError,
        ManifestError,
        RegistryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
