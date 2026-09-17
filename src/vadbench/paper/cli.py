"""Paper-specific entry point; existing public CLI and model catalog remain intact."""

from __future__ import annotations

import argparse
import json
import sys

from vadbench.paper.profile import load_project
from vadbench.paper.resolve import resolve_probe, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vadbench.paper")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "probe", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--project", required=True, help="Project profile YAML")
        command.add_argument("--root", help="Project root for a nonstandard profile location")
        if name == "probe":
            command.add_argument("--suite", required=True)
        if name == "verify":
            command.add_argument("--encoder", required=True)
            command.add_argument("--video", required=True)
        if name in {"probe", "verify"}:
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="Read configuration only; no model loading or writes",
            )
            command.add_argument(
                "--device", default="cpu", help="Explicit execution device (default: cpu)"
            )
    args = parser.parse_args(argv)
    try:
        project = load_project(args.project, root=args.root)
        if args.command == "status":
            result = status(project)
        elif args.command == "verify":
            from vadbench.paper.stages import run_verification, verify_plan

            result = verify_plan(project, args.encoder, args.video, args.device)
            if not args.dry_run:
                result = run_verification(project, result)
        else:
            result = resolve_probe(project, args.suite)
            if not args.dry_run:
                from vadbench.paper.stages import run_probe

                result = run_probe(project, result, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"paper: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
