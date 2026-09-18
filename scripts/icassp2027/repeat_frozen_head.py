"""Train one frozen seed-1/2 head repeat from a verified controller FeatureStore."""

from __future__ import annotations

import argparse
import json

from vadbench.paper.head_repeats import FrozenHeadRepeatRequest, run_frozen_head_repeat


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--source-controller-run", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", required=True, type=int, choices=(1, 2))
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--freeze-path", default="projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    )
    p.add_argument(
        "--role-lock-path",
        default="outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json",
    )
    p.add_argument(
        "--head-data-contract-path",
        default="projects/icassp2027/decisions/head-data-contract-v1.json",
    )
    p.add_argument(
        "--source-manifest-root",
        default="outputs/icassp2027/assets/full-ucf-head-data-contract-20260918",
    )
    p.add_argument("--run-id")
    a = p.parse_args(argv)
    request = FrozenHeadRepeatRequest(
        encoder=a.encoder,
        source_controller_run=a.source_controller_run,
        output_root=a.output_root,
        seed=a.seed,
        freeze_path=a.freeze_path,
        role_lock_path=a.role_lock_path,
        head_data_contract_path=a.head_data_contract_path,
        source_manifest_root=a.source_manifest_root,
        device=a.device,
        run_id=a.run_id,
    )
    print(json.dumps(run_frozen_head_repeat(request).__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
