"""Compare completed sealed XD RGB evaluation runs without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.artifacts import record_stage
from vadbench.features import atomic_write_json
from vadbench.paper.xd_quality_export import compare_frozen_xd_quality, summarize_xd_quality_seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-run",
        action="append",
        type=Path,
        required=True,
        help="completed dense official evaluation run; repeat in matched-seed order",
    )
    parser.add_argument(
        "--method-run",
        action="append",
        type=Path,
        required=True,
        help="completed method official evaluation run paired with --dense-run",
    )
    parser.add_argument("--mode", choices=("refit_head", "direct_insert"), default="refit_head")
    parser.add_argument("--output", type=Path, required=True, help="new comparison directory")
    args = parser.parse_args(argv)
    if len(args.dense_run) != len(args.method_run) or len(args.dense_run) not in (1, 3):
        parser.error("provide one paired seed or the complete three paired seeds 0/1/2")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("comparison output must be a new directory")
    pairs = list(zip(args.dense_run, args.method_run, strict=True))
    config = {
        "mode": args.mode,
        "pairs": [{"dense": str(d.resolve()), "method": str(m.resolve())} for d, m in pairs],
        "metric": "frame_pr_auc",
        "model_inference": False,
    }
    inputs = {
        f"{role}_{i}_completed_result": path.resolve() / "result.json"
        for i, pair in enumerate(pairs)
        for role, path in zip(("dense", "method"), pair, strict=True)
    }
    output.mkdir(parents=True)
    with record_stage(output, "frozen_xd_quality_comparison", config=config, inputs=inputs):
        results = [
            compare_frozen_xd_quality(dense, method, mode=args.mode) for dense, method in pairs
        ]
        summary = summarize_xd_quality_seeds(results)
        atomic_write_json(output / "comparisons.json", results)
        atomic_write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "head_seeds": summary["head_seeds"],
                "n_videos_per_seed": summary["n_videos_per_seed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
