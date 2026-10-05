"""Materialize fixed XD head inputs, or report blocked incomplete targets."""

from __future__ import annotations

import argparse
import json

from vadbench.data.xd_materialization import (
    audit_xd_ready_manifest,
    freeze_xd_canary_rule,
    materialize_xd_heads,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation", choices=("materialize", "audit-ready"), default="materialize"
    )
    for argument in (
        "plan",
        "original-lock",
        "full-manifest",
        "ready-manifest",
        "ready-receipt",
        "scope",
        "dataset-root",
        "output-dir",
    ):
        parser.add_argument("--" + argument, required=True)
    for argument in (
        "quarantine",
        "method-freeze",
        "canary-rule",
        "ready-audit",
        "ready-audit-sha256",
    ):
        parser.add_argument("--" + argument)
    args = parser.parse_args()
    if args.operation == "audit-ready":
        result = audit_xd_ready_manifest(
            plan_path=args.plan,
            scope_path=args.scope,
            original_lock_path=args.original_lock,
            full_manifest_path=args.full_manifest,
            ready_manifest_path=args.ready_manifest,
            upstream_receipt_path=args.ready_receipt,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if any(
        getattr(args, name) is None
        for name in (
            "quarantine",
            "method_freeze",
            "canary_rule",
            "ready_audit",
            "ready_audit_sha256",
        )
    ):
        parser.error(
            "materialize requires quarantine, method-freeze, canary-rule and an externally frozen ready-audit/ready-audit-sha256"
        )
    freeze_xd_canary_rule(args.plan, args.scope, args.canary_rule)
    result = materialize_xd_heads(
        plan_path=args.plan,
        original_lock_path=args.original_lock,
        full_manifest_path=args.full_manifest,
        ready_manifest_path=args.ready_manifest,
        ready_receipt_path=args.ready_receipt,
        quarantine_path=args.quarantine,
        scope_path=args.scope,
        method_freeze_path=args.method_freeze,
        canary_rule_path=args.canary_rule,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        ready_audit_path=args.ready_audit,
        ready_audit_sha256=args.ready_audit_sha256,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "head_training_ready",
                    "head_encoders",
                    "targets",
                    "canary",
                    "head_data_contract",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["head_training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
