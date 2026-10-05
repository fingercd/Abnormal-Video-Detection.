"""Extract complete native dense training features under an explicit full-train contract."""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--training-contract-path", required=True)
    parser.add_argument("--training-contract-sha256", required=True)
    parser.add_argument("--protocol-path", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project", default="projects/icassp2027/profile.yaml")
    parser.add_argument("--reducer", choices=(
        "identity", "global_uniform", "paired_random", "pair_linear",
        "pair_select", "pair_fixed", "pair_random_member", "pair_reverse",
        "group_uniform", "group_random",
    ), default="identity")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        choices=(0.8, 0.6, 0.4),
        help="frozen three-tier budget for the selection reducers (pair_select/group_uniform/group_random)",
    )
    parser.add_argument("--calibration-run")
    parser.add_argument("--processor-tensor-type", choices=("pt", "np"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume-source")
    parser.add_argument("--engineering-video-id", action="append", default=[])
    parser.add_argument("--development-role", choices=("fit", "select"))
    parser.add_argument(
        "--test-manifest-path",
        help="sealed official test manifest; evaluation-only extraction, mutually exclusive with roles/subsets",
    )
    parser.add_argument("--test-manifest-sha256")
    parser.add_argument(
        "--test-dataset-root",
        help="root under which sealed test manifest paths resolve; never assumed to be the training root",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 < args.cuda_memory_fraction <= 1:
        parser.error("cuda-memory-fraction must be in (0, 1]")
    if (args.test_manifest_path is None) != (args.test_manifest_sha256 is None):
        parser.error("--test-manifest-path and --test-manifest-sha256 must be given together")
    if args.test_manifest_path is not None and (
        args.development_role is not None or args.engineering_video_id
    ):
        parser.error("sealed test extraction cannot be combined with --development-role or --engineering-video-id")
    if args.test_dataset_root is not None and args.test_manifest_path is None:
        parser.error("--test-dataset-root requires --test-manifest-path")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    if args.device != "cpu":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=args.device)
    from vadbench.paper.official_extraction import (
        OfficialDenseExtractionRequest,
        run_official_dense_extraction,
    )

    values = vars(args).copy()
    values.pop("cuda_memory_fraction")
    values["engineering_video_ids"] = tuple(values.pop("engineering_video_id"))
    result = run_official_dense_extraction(OfficialDenseExtractionRequest(**values))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
