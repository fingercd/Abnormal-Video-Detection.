"""Train the unchanged UR-DMU backend from strictly verified native dense features."""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "feature-store",
        "train-manifest",
        "source-contract-path",
        "source-contract-sha256",
        "upstream-dir",
        "output-root",
        "device",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--run-mode", choices=("formal", "engineering"), default="formal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--bags-per-class", type=int, default=64)
    parser.add_argument(
        "--protocol-path",
        default="projects/icassp2027/decisions/official-detector-protocol-v2.json",
    )
    for name in (
        "extraction-contract-path",
        "extraction-contract-sha256",
        "dataset-root",
        "aggregation-cache",
        "aggregation-cache-receipt-sha256",
        "run-id",
    ):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    if args.device.split(":", 1)[0] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Importing vadbench can import torch; CPU visibility must be fixed first.
    from vadbench.paper.urdmu_training import URDMUTrainingRequest, run_urdmu_training

    result = run_urdmu_training(URDMUTrainingRequest(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
