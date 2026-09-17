"""Prepare full-list XD raw-video weak-label roles from ZIP directory metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.data.xd_violence import (
    load_xd_archive_members,
    prepare_xd_data,
    write_prepared_xd_data,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="six *.cd files plus their central_plan.json; no test annotations",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="raw files laid out as <root>/<archive_volume>/<archive_member>",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202709)
    args = parser.parse_args()
    prepared = prepare_xd_data(
        load_xd_archive_members(args.metadata_dir), dataset_root=args.dataset_root, seed=args.seed
    )
    artifacts = write_prepared_xd_data(prepared, args.output_dir)
    summary = prepared.receipt()
    summary.pop("official_train_test_shared_name_source_groups")
    print(
        json.dumps(
            {
                **summary,
                "artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
