"""Score a permitted XD repeat head from the sealed seed-0 XD FeatureStore."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.xd_repeat_evaluation import (
    FrozenXDRepeatEvaluationRequest,
    run_frozen_xd_repeat_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=("videomaev2", "timesformer"))
    parser.add_argument("--repeat-head-run", required=True)
    parser.add_argument("--source-evaluation-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dense-head-repeat-run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = run_frozen_xd_repeat_evaluation(FrozenXDRepeatEvaluationRequest(**vars(args)))
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
