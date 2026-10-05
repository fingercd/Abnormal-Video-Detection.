"""Build an explicit, integrity-checked subset view from one completed run.

The subset view transports (hardlink/copy) the requested videos' shards from
a completed extraction run into a new non-native view with its own contract.
It exists so v0/debug consumers can read a verified slice without re-running
an encoder; it never replaces a native extraction run identity.
"""

from __future__ import annotations

import argparse
import json


def _video_list(path: str) -> tuple[str, ...]:
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SystemExit(f"--video-list must be a JSON string array or one ID per line: {path}")
    return tuple(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="completed extraction run root")
    parser.add_argument("--video-list", required=True, help="subset video IDs (JSON array or text lines)")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--transport",
        choices=("hardlink_npz", "copy"),
        default="hardlink_npz",
        help="hardlink_npz shares blob inodes on one filesystem and is the only no-copy mode",
    )
    args = parser.parse_args(argv)
    from vadbench.workflows.feature_merge import FeatureSubsetRequest, run_feature_subset

    result = run_feature_subset(
        FeatureSubsetRequest(
            source_run_root=args.source_run,
            target_video_ids=_video_list(args.video_list),
            output_root=args.output_root,
            run_id=args.run_id,
            transport=args.transport,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
