"""Prepare the audited UR-DMU 200-bin cache without starting optimizer steps.

This is a CPU-side preparation wrapper for a verified dense/merged view.  It
delegates source, identity, membership and 200-bin construction to the same
private APIs used by ``run_urdmu_training``; it does not implement another
cache format or relax any source binding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "feature-store",
        "train-manifest",
        "source-contract-path",
        "source-contract-sha256",
        "upstream-dir",
        "output-root",
        "protocol-path",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--feature-view-contract-path")
    parser.add_argument("--feature-view-contract-sha256")
    parser.add_argument("--extraction-contract-path")
    parser.add_argument("--extraction-contract-sha256")
    parser.add_argument("--dataset-root")
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vadbench.artifacts import new_run_id, record_stage
    from vadbench.checkpoints import sha256_file
    from vadbench.features import atomic_write_json
    from vadbench.paper.urdmu_training import (
        URDMUTrainingRequest,
        _aggregate,
        _dense_source,
        _source,
    )

    request = URDMUTrainingRequest(
        **vars(args),
        device="cpu",
        run_mode="formal",
        steps=3000,
        bags_per_class=64,
    )
    run = Path(request.output_root).expanduser().resolve() / (
        request.run_id or new_run_id("urdmu-aggregation")
    )
    run.mkdir(parents=True, exist_ok=False)
    with record_stage(
        run,
        "urdmu_aggregation",
        config=request.__dict__,
        inputs={
            "train_manifest": request.train_manifest,
            "training_view_contract": request.source_contract_path,
            "feature_store": request.feature_store,
            "protocol": request.protocol_path,
            "feature_view_contract": request.feature_view_contract_path,
            "extraction_contract": request.extraction_contract_path,
        },
        project_root=Path.cwd(),
    ):
        records, source_receipt, sources = _source(request)
        document, representation, _sampling, _feature_receipt, sequences = _dense_source(
            request, records, sources, source_receipt
        )
        _values, labels, cache_root, cache_receipt = _aggregate(
            request,
            records,
            document,
            representation,
            sources,
            run,
            sequences=sequences,
        )
        result = {
            "status": "completed",
            "run_dir": str(run),
            "cache_root": str(cache_root),
            "cache_receipt_sha256": sha256_file(cache_root / "receipt.json"),
            "bags_sha256": cache_receipt["bags_sha256"],
            "videos": len(records),
            "normal_videos": int((labels == 0).sum()),
            "anomaly_videos": int((labels == 1).sum()),
            "shape": list(_values.shape),
            "optimizer_steps_started": False,
            "source_sha256": dict(sources),
        }
        atomic_write_json(run / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
