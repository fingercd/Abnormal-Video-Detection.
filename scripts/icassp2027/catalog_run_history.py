"""Catalog run-level receipts in legacy VAD roots without traversing blobs.

This inventories persisted run metadata, including failed and historical
attempts. It does not reclassify a successful process as scientifically valid.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket


MARKERS = {
    "result.json", "status.json", "summary.json", "resolved.json", "resolved_reducer.json",
    "training_qa.json", "extraction-contract.json", "merge-contract.json", "receipt.json",
    "completion.json", "head-qa.json", "sealed_features.json", "sealed_test_features.json",
}
SKIP = {".git", "__pycache__", ".pytest_cache", "blobs", "blocks", "shards", "features",
        "weights", "checkpoints", "receipts", "vendor", "external", "external-v2",
        "analysis_runtime", "analysis-src-r01"}


def family(path: Path) -> str:
    value = str(path)
    if "dsanet-extension" in value:
        return "clip_dsanet"
    if any(x in value for x in ("screen-64", "idea", "/v0", "probe", "batch1")):
        return "development"
    if any(x in value for x in ("efficiency", "benchmark", "batch-scaling")):
        return "benchmark"
    if "icassp2027" in value:
        return "video_vit_or_research"
    return "legacy_framework"


def collect(roots: list[Path], frozen: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    records, scanned = {}, []
    for root in roots:
        if not root.is_dir():
            scanned.append({"root": str(root), "status": "missing"})
            continue
        dirs_seen = 0
        for base, dirs, files in os.walk(root, followlinks=False):
            directory = Path(base)
            dirs[:] = sorted(name for name in dirs if name not in SKIP
                             and not name.startswith("code-")
                             and not (directory / name).is_symlink())
            dirs_seen += 1
            found = sorted(set(files) & MARKERS)
            if not found or str(directory) in records:
                continue
            markers, payloads = {}, {}
            for name in found:
                path = directory / name
                size = path.stat().st_size
                item = {"path": str(path), "bytes": size}
                if size <= 1024 * 1024:
                    data = path.read_bytes()
                    item["sha256"] = hashlib.sha256(data).hexdigest()
                    try:
                        value = json.loads(data)
                        if isinstance(value, dict):
                            payloads[name] = value
                    except (ValueError, UnicodeError):
                        item["parse_status"] = "invalid_json"
                markers[name] = item
            statuses = {}
            for name in ("result.json", "status.json", "summary.json", "completion.json", "receipt.json"):
                data = payloads.get(name, {})
                if "status" in data:
                    statuses[name] = data["status"]
                elif "completed" in data:
                    statuses[name] = {"completed": data["completed"]}
            selected = frozen.get(str(directory), [])
            suspicious = any(word in directory.name for word in ("failed", "partial", "stale", "cancelled", "quarantine"))
            run_ids = sorted({data[key] for data in payloads.values() for key in ('run_id', 'run_name')
                              if isinstance(data.get(key), str) and data[key]})
            records[str(directory)] = {
                "run_directory": str(directory), "discovery_root": str(root), "family": family(directory),
                "metadata_statuses": statuses, "markers": markers,
                "observed_run_ids": run_ids,
                "frozen_matrix_roles": selected,
                "catalog_role": "frozen_matrix_bound" if selected else "unverified_historical_attempt" if suspicious else "unverified_run",
                "scientific_qa_revalidated": False,
            }
        scanned.append({"root": str(root), "status": "scanned", "directories_visited": dirs_seen})
    return [records[k] for k in sorted(records)], scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    assets = Path(args.asset_root)
    output = Path(args.output_dir)
    if output.exists():
        parser.error("output already exists")
    old_runs = Path('/users/fotile/icassp2027-runs')
    roots = [Path('/users/fotile/VAD/outputs')]
    roots.extend(sorted(Path('/data2/localdisk').glob('fotile-icassp2027-*')))
    roots.extend(p for p in sorted(old_runs.iterdir()) if p.is_dir()
                 and not p.name.startswith('code-') and p.name not in {'project', 'vendor', 'incoming', 'transfer', 'env-hooks'})
    roots.extend(p / 'outputs' for p in sorted(old_runs.glob('code-*')) if (p / 'outputs').is_dir())
    roots.append(old_runs / 'project/outputs')
    frozen = {}
    for line in (assets/'catalog/formal_feature_bindings-20260925.jsonl').read_text().splitlines():
        row = json.loads(line)
        for key in ('feature_contract', 'head_checkpoint', 'prediction'):
            if key not in row:
                continue
            path = Path(row[key]['path'])
            directory = path.parent.parent if key == 'head_checkpoint' else path.parent
            frozen.setdefault(str(directory), []).append(':'.join((row['dataset'], row['encoder'], row['method'], key)))
    records, scanned = collect(roots, frozen)
    output.mkdir(parents=True)
    (output/'runs.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n' for x in records), encoding='utf-8')
    by_id = {}
    for record in records:
        for run_id in record['observed_run_ids']:
            by_id.setdefault(run_id, []).append(record['run_directory'])
    repeated = [{'run_id': key, 'directories': paths, 'status': 'possible_copies_not_content_verified'}
                for key, paths in sorted(by_id.items()) if len(paths) > 1]
    (output/'repeated_run_ids.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n'
                                              for x in repeated), encoding='utf-8')
    summary = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'host': socket.gethostname(),
               'directories_with_receipts': len(records), 'distinct_observed_run_ids': len(by_id),
               'repeated_run_ids': len(repeated), 'families': dict(Counter(x['family'] for x in records)),
               'roles': dict(Counter(x['catalog_role'] for x in records)), 'roots': scanned,
               'skipped_subdirectories': sorted(SKIP),
               'scope': 'run_receipts_only_not_blob_or_scientific_reaudit'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('directories_with_receipts', 'distinct_observed_run_ids',
                     'repeated_run_ids', 'families', 'roles')}, sort_keys=True))


if __name__ == '__main__':
    main()
