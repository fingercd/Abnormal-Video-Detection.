"""Verify frozen six-cell feature paths without reading predictions or scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(row: dict[str, object], *, full_index_sha: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "dataset": row["dataset"], "encoder": row["encoder"], "method": row["method"],
        "checks": {}, "issues": [],
    }
    checks = result["checks"]
    issues = result["issues"]
    assert isinstance(checks, dict) and isinstance(issues, list)
    contract_decl = row["feature_contract"]
    index_decl = row["feature_index"]
    assert isinstance(contract_decl, dict) and isinstance(index_decl, dict)
    contract = Path(str(contract_decl["path"]))
    index = Path(str(index_decl["path"]))

    if not contract.is_file():
        issues.append("contract_missing")
    else:
        checks["contract_bytes"] = contract.stat().st_size
        actual = sha256(contract)
        checks["contract_sha_matches"] = actual == contract_decl.get("sha256")
        if not checks["contract_sha_matches"]:
            issues.append("contract_sha_mismatch")

    if not index.is_file():
        issues.append("index_missing")
    else:
        checks["index_bytes"] = index.stat().st_size
        checks["index_expected_sha256"] = index_decl.get("sha256")
        checks["index_sha_checked"] = full_index_sha
        if full_index_sha:
            checks["index_sha_matches"] = sha256(index) == index_decl.get("sha256")
            if not checks["index_sha_matches"]:
                issues.append("index_sha_mismatch")
        with index.open("r", encoding="utf-8") as source:
            first = source.readline()
        if not first:
            issues.append("index_empty")
        else:
            try:
                item = json.loads(first)
                relative = item["feature_path"]
                blob = Path(relative)
                if not blob.is_absolute():
                    blob = index.parent / blob
                if not blob.is_file():
                    issues.append("first_blob_missing")
                else:
                    checks["first_blob_bytes"] = blob.stat().st_size
                    expected_blob_sha = item.get("arrays", {}).get("features", {}).get("sha256")
                    checks["first_blob_sha_matches"] = sha256(blob) == expected_blob_sha
                    if not checks["first_blob_sha_matches"]:
                        issues.append("first_blob_sha_mismatch")
                    with zipfile.ZipFile(blob) as archive:
                        bad_member = archive.testzip()
                    checks["first_blob_zip_crc_ok"] = bad_member is None
                    if bad_member is not None:
                        issues.append("first_blob_zip_crc_failed")
            except (KeyError, ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                issues.append(f"first_record_invalid:{type(exc).__name__}")

    for name in ("head_checkpoint", "prediction"):
        declaration = row.get(name)
        if isinstance(declaration, dict):
            path = Path(str(declaration["path"]))
            checks[f"{name}_exists"] = path.is_file()
            if not path.is_file():
                issues.append(f"{name}_missing")
    result["status"] = (
        "index_verified_sample_blob_verified" if full_index_sha else "sample_verified_index_sha_pending"
    ) if not issues else "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-index-sha", action="store_true")
    args = parser.parse_args()
    bindings = Path(args.bindings)
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    rows = [json.loads(line) for line in bindings.read_text(encoding="utf-8").splitlines()]
    results = [verify(row, full_index_sha=args.full_index_sha) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                              for item in results), encoding="utf-8")
    print(json.dumps({"total": len(results), "verified": sum(
        item["status"] != "failed" for item in results), "full_index_sha": args.full_index_sha,
        "failed": sum(item["status"] == "failed" for item in results),
        "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
