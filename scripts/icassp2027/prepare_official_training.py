"""Prepare authoritative official training views; no encoder or training calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.data.official_training import audit_ucf_official_training, materialize


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-ucf", help="independent CPU streaming byte and container audit")
    prepare = commands.add_parser("materialize", help="new official view or blocked inventory")
    for command in (audit, prepare):
        command.add_argument("--full-manifest", type=Path, required=True,
                             help="UCF official Anomaly_Train.txt, or XD official train.full.jsonl")
        command.add_argument("--role-lock", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        for role in ("fit", "confirm", "select"):
            command.add_argument("--" + role, type=Path, required=command is audit)
    audit.add_argument("--full-manifest-sha256", required=True)
    audit.add_argument("--role-lock-sha256", required=True)
    audit.add_argument("--provider-metadata", type=Path, required=True)
    audit.add_argument("--provider-metadata-sha256", required=True)
    prepare.add_argument("--dataset", choices=("ucf", "xd", "ucf_crime", "xd_violence"), required=True)
    prepare.add_argument("--evidence", type=Path, required=True)
    prepare.add_argument("--evidence-sha256", required=True)
    prepare.add_argument("--ready-manifest", type=Path)
    prepare.add_argument("--ready-receipt", type=Path)
    prepare.add_argument("--provider-metadata", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    options = vars(parser().parse_args(argv))
    command = options.pop("command")
    receipt = (audit_ucf_official_training if command == "audit-ucf" else materialize)(**options)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["state"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
