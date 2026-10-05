"""Create reversible navigation links for inventoried ICASSP features.

Links preserve the original run directories and relative FeatureStore paths.
Only frozen matrix bindings are labelled formal; other links remain candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _label(ratio: float | None) -> str:
    return f"keep_{round((1.0 if ratio is None else ratio) * 100):03d}"


def _source(row: dict[str, object]) -> Path:
    source = row["source"]
    assert isinstance(source, dict)
    return Path(str(source["path"]))


def plan(
    asset_root: Path, features: list[dict[str, object]],
    bindings: list[dict[str, object]],
) -> list[dict[str, str]]:
    planned: dict[str, dict[str, str]] = {}

    def add(destination: Path, source: Path, role: str) -> None:
        key = str(destination)
        record = {"destination": key, "source": str(source), "role": role}
        if key in planned and planned[key]["source"] != record["source"]:
            raise ValueError(f"collision: {key}")
        planned[key] = record

    for row in features:
        source = _source(row)
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        add(asset_root / "features" / "_discovered" / str(row["family"])
            / str(row["role"]) / str(row["dataset"]) / f"{digest}-{source.name}",
            source, "discovered_unverified")

        if row["role"] == "merged_view" and row["split"] == "train" and "merged-full" in source.parts:
            add(asset_root / "features" / "video_vit" / str(row["dataset"]) / "train"
                / str(row["encoder"]) / "dense" / "keep_100" / "run",
                source, "full_train_candidate")
        elif row["role"] == "budget_sweep_candidate":
            add(asset_root / "features" / "video_vit" / str(row["dataset"]) / "test"
                / str(row["encoder"]) / "pair_select" / _label(row["keep_ratio"]) / "run",
                source, "budget_sweep_candidate")
        elif row["role"] == "published_head_input_candidate":
            add(asset_root / "features" / "clip_dsanet" / str(row["dataset"]) / "test"
                / "clip_vit_b16" / str(row["method"]) / _label(row["keep_ratio"]) / "run",
                source, "published_head_input_candidate")
        elif row["role"] == "local_head_attempt_input":
            add(asset_root / "features" / "clip_dsanet" / str(row["dataset"]) / "train"
                / "clip_vit_b16" / "dense" / source.name / "run",
                source, "local_head_attempt_input")

    for row in bindings:
        contract = row["feature_contract"]
        assert isinstance(contract, dict)
        source = Path(str(contract["path"])).parent
        add(asset_root / "features" / "video_vit" / str(row["dataset"]) / "test"
            / str(row["encoder"]) / str(row["method"]) / _label(row["keep_ratio"]) / "run",
            source, "frozen_six_cell_binding")

    return [planned[key] for key in sorted(planned)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--feature-inventory", required=True)
    parser.add_argument("--formal-bindings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    asset_root = Path(args.asset_root).resolve()
    if not (asset_root / "features").is_dir():
        parser.error("asset root must already contain features/")
    records = plan(asset_root, _read_jsonl(Path(args.feature_inventory)),
                   _read_jsonl(Path(args.formal_bindings)))
    for record in records:
        destination = Path(record["destination"])
        source = Path(record["source"])
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"existing link points elsewhere: {destination}")
        elif destination.exists():
            raise ValueError(f"destination already exists: {destination}")
    if args.apply:
        output = Path(args.output)
        if output.exists():
            parser.error(f"output already exists: {output}")
        for record in records:
            destination = Path(record["destination"])
            if not destination.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(record["source"], destination, target_is_directory=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                          for record in records), encoding="utf-8")
    print(json.dumps({"views": len(records), "applied": args.apply,
                      "formal": sum(x["role"] == "frozen_six_cell_binding" for x in records)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
