"""Train one permitted XD seed-1/2 detector-head repeat on frozen features."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.xd_head_repeats import FrozenXDHeadRepeatRequest, run_frozen_xd_head_repeat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=("videomaev2", "timesformer"))
    parser.add_argument("--source-controller-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2))
    parser.add_argument("--role-lock", required=True)
    parser.add_argument("--head-data-contract", required=True)
    parser.add_argument("--head-data-contract-sha256", required=True)
    parser.add_argument("--source-manifest-root", required=True)
    parser.add_argument("--original-role-lock", required=True)
    parser.add_argument(
        "--scope", default="projects/icassp2027/decisions/xd-validation-scope-v1.json"
    )
    parser.add_argument(
        "--freeze", default="projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    result = run_frozen_xd_head_repeat(
        FrozenXDHeadRepeatRequest(
            encoder=args.encoder,
            source_controller_run=args.source_controller_run,
            output_root=args.output_root,
            seed=args.seed,
            role_lock_path=args.role_lock,
            head_data_contract_path=args.head_data_contract,
            head_data_contract_sha256=args.head_data_contract_sha256,
            source_manifest_root=args.source_manifest_root,
            original_role_lock_path=args.original_role_lock,
            scope_path=args.scope,
            freeze_path=args.freeze,
            device=args.device,
            run_id=args.run_id,
        )
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
