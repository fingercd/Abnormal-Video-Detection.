"""Run one frozen pooled-feature paper detector experiment."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.controller import DetectionExperimentRequest, run_detection_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--project", default="projects/icassp2027/profile.yaml")
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reducer", choices=("identity", "global_uniform", "paired_random", "pair_linear"), default="identity")
    parser.add_argument("--reducer-seed", type=int, default=0)
    parser.add_argument("--calibration-run", help="completed fit-only gate calibration directory; pair_linear only")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--short-policy", choices=("strict", "stride1_if_needed"), default="strict")
    parser.add_argument(
        "--dense-window-stride",
        type=int,
        help="source-frame step; defaults to half the native window span",
    )
    parser.add_argument("--output-dim", type=int, required=True)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--processor-tensor-type", choices=("pt", "np"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluate-official-test", action="store_true")
    parser.add_argument("--method-frozen", action="store_true")
    parser.add_argument("--audit-report")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-source-run", help="reuse strictly validated complete video shards from this prior detection run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = DetectionExperimentRequest(
        encoder=args.encoder,
        device=args.device,
        dataset_root=args.dataset_root,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        evaluation_manifest=args.evaluation_manifest,
        output_root=args.output_root,
        project=args.project,
        reducer=args.reducer,
        reducer_seed=args.reducer_seed,
        calibration_run=args.calibration_run,
        frame_stride=args.frame_stride,
        short_policy=args.short_policy,
        dense_window_stride=args.dense_window_stride,
        output_dim=args.output_dim,
        precision=args.precision,
        processor_tensor_type=args.processor_tensor_type,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_mode="official" if args.evaluate_official_test else "engineering",
        method_frozen=args.method_frozen,
        audit_report=args.audit_report,
        run_id=args.run_id,
        resume_source_run=args.resume_source_run,
    )
    result = run_detection_experiment(request)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
