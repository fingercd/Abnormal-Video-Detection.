"""按 integrations/*/upstream.lock.yaml 获取或校验上游代码。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def sync_lock(lock_path: Path, root: Path, *, verify_only: bool) -> tuple[str, str]:
    data = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    integration = str(data["integration"])
    repository = str(data["source"]["repository"])
    commit = str(data["source"]["commit"])
    checkout = root / "external" / ("hermes" if integration.startswith("hermes") else integration)
    if not (checkout / ".git").is_dir():
        if verify_only:
            raise RuntimeError(f"missing checkout: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", repository, str(checkout))
    if not verify_only:
        _run("git", "fetch", "origin", commit, cwd=checkout)
        _run("git", "checkout", "--detach", commit, cwd=checkout)
    actual = _run("git", "rev-parse", "HEAD", cwd=checkout)
    if actual != commit:
        raise RuntimeError(f"{integration}: expected {commit}, got {actual}")
    return integration, actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    locks = sorted((root / "integrations").glob("*/upstream.lock.yaml"))
    if not locks:
        raise SystemExit("no upstream locks found")
    for lock in locks:
        name, commit = sync_lock(lock, root, verify_only=args.verify_only)
        print(f"{name}\t{commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
