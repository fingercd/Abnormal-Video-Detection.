"""Evaluate frozen UCF detector heads on the sealed official test split."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.evaluation import FrozenEvaluationRequest, run_frozen_ucf_evaluation

_SEALED_BASE = "/users/fotile/icassp2027-runs/code-cd39a5d/outputs/icassp2027/assets/ucf-decoded-reconciliation-20260918T081000Z/result/sealed-test"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--method-source-run",
        required=True,
        help="completed frozen controller run for the method refit head",
    )
    parser.add_argument(
        "--dense-source-run",
        help="completed frozen identity controller run for secondary direct_insert",
    )
    parser.add_argument(
        "--calibration-run",
        help="completed frozen pair_linear calibration run; required by that method",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-manifest", default=_SEALED_BASE + "/test-decoded-reconciled.jsonl")
    parser.add_argument("--audit-report", default=_SEALED_BASE + "/dataset-audit-report.json")
    parser.add_argument("--project", default="projects/icassp2027/profile.yaml")
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
    parser.add_argument("--processor-tensor-type", choices=("pt", "np"))
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = FrozenEvaluationRequest(
        encoder=args.encoder,
        device=args.device,
        dataset_root=args.dataset_root,
        test_manifest=args.test_manifest,
        audit_report=args.audit_report,
        method_source_run=args.method_source_run,
        dense_source_run=args.dense_source_run,
        calibration_run=args.calibration_run,
        output_root=args.output_root,
        project=args.project,
        freeze_path=args.freeze_path,
        role_lock_path=args.role_lock_path,
        head_data_contract_path=args.head_data_contract_path,
        source_manifest_root=args.source_manifest_root,
        processor_tensor_type=args.processor_tensor_type,
        run_id=args.run_id,
    )
    print(json.dumps(run_frozen_ucf_evaluation(request).__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
