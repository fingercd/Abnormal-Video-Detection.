"""Merge completed per-GPU extraction runs into one auditable FeatureStore view.

The target video list is a JSON file (a JSON array of video ID strings) or a
plain-text file with one video ID per line.  Every listed video must appear in
exactly one completed run; the merge fails closed with a full problem listing
otherwise.  The result is a new directory (merged view) containing the final
index, a merged-view resolved receipt and a merge contract with per-video
provenance; it is consumed through the v0 diagnostic interfaces, never as a
native formal/development extraction run.
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
    parser.add_argument("--video-list", required=True, help="target video IDs (JSON array or text lines)")
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="completed extraction run root; repeatable, each run covers a disjoint subset",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--transport",
        choices=("hardlink_npz", "copy"),
        default="hardlink_npz",
        help="hardlink_npz shares blob inodes on one filesystem and is the only no-copy mode",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("fail", "prefer_run"),
        default="fail",
        help="prefer_run resolves byte-identical duplicates only; divergent bytes always fail",
    )
    parser.add_argument("--prefer-run", help="run root kept for duplicated videos with prefer_run")
    parser.add_argument(
        "--allow-partial",
        action="append",
        default=[],
        help="run root allowed to contribute complete per-video shards from a partial run; repeatable",
    )
    parser.add_argument(
        "--role-equivalence",
        choices=(
            "engineering-shards-of-official-fit",
            "engineering-shards-of-official-fulltrain",
        ),
        help="explicit role equivalence declaration recorded in the merge contract",
    )
    parser.add_argument("--authority-contract-path", help="authoritative training view contract path")
    parser.add_argument("--authority-contract-sha256", help="authoritative training view contract SHA-256")
    parser.add_argument(
        "--original-role-lock-path",
        help="original role lock path; defaults to sources/role_lock.json under the authority contract dir",
    )
    parser.add_argument("--original-role-lock-sha256", help="original role lock SHA-256 (required to record it)")
    args = parser.parse_args(argv)
    from vadbench.workflows.feature_merge import FeatureRunMergeRequest, run_feature_merge

    result = run_feature_merge(
        FeatureRunMergeRequest(
            target_video_ids=_video_list(args.video_list),
            run_roots=tuple(args.run_root),
            output_root=args.output_root,
            run_id=args.run_id,
            transport=args.transport,
            duplicate_policy=args.duplicate_policy,
            prefer_run=args.prefer_run,
            allow_partial=tuple(args.allow_partial),
            role_equivalence=args.role_equivalence,
            authority_contract_path=args.authority_contract_path,
            authority_contract_sha256=args.authority_contract_sha256,
            original_role_lock_path=args.original_role_lock_path,
            original_role_lock_sha256=args.original_role_lock_sha256,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
