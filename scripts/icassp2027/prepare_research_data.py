"""Create an isolated, W-safe UCF-Crime research-data preparation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.research.data_prepare import (
    CohortLimits,
    prepare_ucf_research_data,
    write_prepared_research_data,
)


def _source_groups(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("--source-groups 必须是 video_id 到 source_id 的 JSON 对象")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--temporal-annotations", type=Path, required=True)
    parser.add_argument("--test-split", type=Path)
    parser.add_argument("--source-groups", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202709)
    parser.add_argument("--windows-per-video", type=int, default=8)
    parser.add_argument("--debug-per-label", type=int, default=4)
    parser.add_argument("--explore-per-label", type=int, default=64)
    parser.add_argument("--confirm-per-label", type=int, default=32)
    parser.add_argument("--select-per-label", type=int, default=32)
    args = parser.parse_args()

    result = prepare_ucf_research_data(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        temporal_annotations=args.temporal_annotations,
        test_split=args.test_split,
        source_groups=_source_groups(args.source_groups),
        limits=CohortLimits(
            debug_per_label=args.debug_per_label,
            explore_per_label=args.explore_per_label,
            confirm_per_label=args.confirm_per_label,
            select_per_label=args.select_per_label,
        ),
        windows_per_video=args.windows_per_video,
        seed=args.seed,
    )
    written = write_prepared_research_data(result, args.output_dir)
    print(
        json.dumps(
            {
                **result.receipt(),
                "output_dir": str(args.output_dir.resolve()),
                "artifacts": {name: str(path) for name, path in sorted(written.items())},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
