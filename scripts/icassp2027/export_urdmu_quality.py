"""Export paired frame quality from two provenance-bound UR-DMU prediction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.workflows.urdmu_quality_export import compare_prediction_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ucf_crime", "xd_violence"), required=True)
    parser.add_argument("--dense-predictions", required=True)
    parser.add_argument("--method-predictions", required=True)
    parser.add_argument("--truth-jsonl", required=True)
    parser.add_argument("--source", action="append", default=[], help="additional provenance file to SHA-bind")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare_prediction_runs(
        dense_predictions=args.dense_predictions,
        method_predictions=args.method_predictions,
        truth_jsonl=args.truth_jsonl,
        dataset=args.dataset,
        source_paths=args.source,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(output), "metric": result["metric"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
