"""Write provenance-bound UR-DMU predictions without opening official frame truth."""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ucf_crime", "xd_violence"), required=True)
    parser.add_argument("--phase", choices=("development_video", "official_frame"), required=True)
    parser.add_argument("--mode", choices=("direct_insert", "refit_head"), required=True)
    for name in (
        "trained-run",
        "feature-store",
        "evaluation-manifest",
        "upstream-dir",
        "output-root",
        "device",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--freeze-path")
    parser.add_argument("--audit-report")
    parser.add_argument("--feature-contract-path")
    parser.add_argument("--feature-contract-sha256")
    parser.add_argument("--max-attention-workspace-bytes", type=int)
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        help="explicitly opt into the verified exact all-key UR-DMU Test attention runner",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.phase == "official_frame":
        if not args.freeze_path or not args.audit_report:
            parser.error("official_frame requires --freeze-path and --audit-report")
    elif args.freeze_path or args.audit_report:
        parser.error("development_video cannot accept --freeze-path or --audit-report")
    elif not args.feature_contract_path or not args.feature_contract_sha256:
        parser.error("development_video requires --feature-contract-path and --feature-contract-sha256")
    if args.device.split(":", 1)[0] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Import after CPU visibility is fixed, because UR-DMU checkpoint loading
    # records the runtime and must not initialize an unintended CUDA device.
    from vadbench.paper.urdmu_evaluation import URDMUEvaluationRequest, run_urdmu_evaluation

    result = run_urdmu_evaluation(URDMUEvaluationRequest(**vars(args)))
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": result.run_dir,
                "predictions": result.predictions,
                "phase": result.phase,
                "video_metrics": result.video_metrics,
                "official_frame_scores_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
