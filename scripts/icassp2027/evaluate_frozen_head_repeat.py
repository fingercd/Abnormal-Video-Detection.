"""Score a frozen seed-1/2 head using a sealed completed seed-0 test FeatureStore."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.repeat_evaluation import (
    FrozenRepeatEvaluationRequest,
    run_frozen_repeat_ucf_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--repeat-head-run", required=True)
    parser.add_argument("--source-evaluation-run", required=True)
    parser.add_argument(
        "--dense-head-repeat-run", help="matching seed dense head repeat for optional direct_insert"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--freeze-path", default="projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    )
    parser.add_argument(
        "--role-lock-path",
        default="outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json",
    )
    parser.add_argument(
        "--head-data-contract-path",
        default="projects/icassp2027/decisions/head-data-contract-v1.json",
    )
    parser.add_argument(
        "--source-manifest-root",
        default="outputs/icassp2027/assets/full-ucf-head-data-contract-20260918",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    result = run_frozen_repeat_ucf_evaluation(
        FrozenRepeatEvaluationRequest(
            encoder=args.encoder,
            repeat_head_run=args.repeat_head_run,
            source_evaluation_run=args.source_evaluation_run,
            dense_head_repeat_run=args.dense_head_repeat_run,
            output_root=args.output_root,
            device=args.device,
            freeze_path=args.freeze_path,
            role_lock_path=args.role_lock_path,
            head_data_contract_path=args.head_data_contract_path,
            source_manifest_root=args.source_manifest_root,
            run_id=args.run_id,
        )
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
