"""Publish a patched copy of an existing merge contract; never rewrites it.

For already-published merged views that predate role-lock recording (their
merge-contract.json lacks ``original_role_lock``), this writes a new
``merge-contract.v2.json`` next to the original with the lock binding added
and a ``patched_from`` SHA chain to the untouched original. Consumers bind the
new file by its own path+SHA. Preferred alternative when practical: re-run
merge_feature_runs.py with --original-role-lock-*, which republishes cleanly.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", required=True, help="published merged view directory")
    parser.add_argument("--original-role-lock-path")
    parser.add_argument("--original-role-lock-sha256")
    parser.add_argument(
        "--role-equivalence",
        choices=("engineering-shards-of-official-fulltrain",),
        help="correct the declaration value (records role_equivalence_before); requires --authority-contract-*",
    )
    parser.add_argument("--authority-contract-path")
    parser.add_argument("--authority-contract-sha256")
    parser.add_argument("--output-name", default="merge-contract.v2.json")
    args = parser.parse_args(argv)
    from vadbench.workflows.feature_merge import patch_view_contract

    result = patch_view_contract(
        view_root=args.view_root,
        output_name=args.output_name,
        original_role_lock_path=args.original_role_lock_path,
        original_role_lock_sha256=args.original_role_lock_sha256,
        role_equivalence=args.role_equivalence,
        authority_contract_path=args.authority_contract_path,
        authority_contract_sha256=args.authority_contract_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
