"""Read-only raw-content comparison against a pre-refactor worktree baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.drive
        or "\\" in value
        or any(p in {"..", ".", ""} for p in value.split("/"))
        or any(c in value for c in "*?[]")
    ):
        raise ValueError(f"expected an exact repository-relative file path: {value!r}")
    forbidden = {
        "weights",
        "weights-v2",
        "data",
        "external",
        "external-v2",
        "outputs",
        ".venv",
        ".encoder-envs",
        ".git",
        "VAD_Idea",
    }
    if posix.parts[0] in forbidden or any(
        p.startswith((".env", ".venv"))
        or p.lower() in {"credentials", "secrets", "id_rsa", "id_ed25519"}
        for p in posix.parts
    ):
        raise ValueError(f"baseline must not read protected asset/secret roots: {value}")
    return Path(*posix.parts)


def check(root: Path, baseline: dict, allow_changed: list[str]) -> dict:
    root = root.resolve()
    allowed = {relative_path(value).as_posix() for value in allow_changed}
    records = baseline["files"]
    if allowed - records.keys():
        raise ValueError("allow-changed must identify files present in the original baseline")
    report = {
        "changed": [],
        "missing": [],
        "symlink": [],
        "allowed_changed": [],
        "checked_files": 0,
    }
    for relative, expected in records.items():
        rel = relative_path(relative)
        target = root / rel
        if any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            report["symlink"].append(relative)
        elif not target.is_file():
            report["missing"].append(relative)
        else:
            with target.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                report["allowed_changed" if relative in allowed else "changed"].append(relative)
        report["checked_files"] += 1
    report["passed"] = not any(report[key] for key in ("changed", "missing", "symlink"))
    report["asset_scope"] = (
        "content hashes cover code/config/doc files only; large assets remain outside write scope"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--allow-changed", action="append", default=[])
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = check(args.root, baseline, args.allow_changed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
