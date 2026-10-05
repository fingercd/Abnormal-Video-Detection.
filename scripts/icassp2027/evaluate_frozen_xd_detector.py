"""Evaluate frozen XD heads only after a full externally sealed raw-coordinate audit."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.xd_evaluation import FrozenXDEvaluationRequest, run_frozen_xd_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=("videomaev2", "timesformer"))
    for name in (
        "device",
        "dataset-root",
        "method-source-run",
        "output-root",
        "test-manifest",
        "canonical-metadata",
        "ground-truth",
        "raw-coordinate-receipt-sha256",
        "head-data-contract-path",
        "head-data-contract-sha256",
        "role-lock-path",
        "original-role-lock-path",
        "source-manifest-root",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--audit-report", required=True, help="sealed XD raw-coordinate receipt; not a UCF audit"
    )
    parser.add_argument("--dense-source-run")
    parser.add_argument(
        "--calibration-run", help="corresponding frozen UCF pair_linear gate; no XD refit"
    )
    parser.add_argument("--project", default="projects/icassp2027/profile.yaml")
    parser.add_argument(
        "--freeze-path", default="projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    )
    parser.add_argument(
        "--scope-path", default="projects/icassp2027/decisions/xd-validation-scope-v1.json"
    )
    parser.add_argument("--processor-tensor-type", choices=("np", "pt"))
    parser.add_argument("--resume-source-evaluation-run")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_frozen_xd_evaluation(FrozenXDEvaluationRequest(**vars(args)))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
