"""Run one stage of the frozen DSANet CLIP workflow."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Sequence


STAGES = {
    "extract": "_extract_runtime",
    "score": "_score_runtime",
    "quality": "_quality_runtime",
}


def stage_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m vadbench.workflows.dsanet {stage}")
    if stage == "extract":
        parser.description = "Extract frozen OpenAI CLIP features with optional PairSelect."
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--bridge", type=Path, help="Legacy replay bridge file; defaults to vadbench's CLIP bridge")
        parser.add_argument("--clip-repo", type=Path, required=True)
        parser.add_argument("--weights", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--gpu", type=int, required=True)
        parser.add_argument("--keep-ratio", type=float, choices=[0.8, 0.6, 0.4])
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--limit-train-videos", type=int, default=0)
        parser.add_argument("--shard-index", type=int, default=0)
        parser.add_argument("--shards", type=int, default=1)
        parser.add_argument("--resume", action="store_true")
    elif stage == "score":
        parser.description = "Score sealed features with one fixed DSANet checkpoint; no frame truth input."
        parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
        parser.add_argument("--upstream-src", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--features", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
    elif stage == "quality":
        parser.description = "Compare sealed DSANet scores with frame truth."
        parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
        parser.add_argument("--scores-root", type=Path, required=True)
        parser.add_argument("--truth", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--checkpoint-origin", choices=["published", "local_full_train"], required=True)
        parser.add_argument("--tiers", nargs="+", choices=["0.8", "0.6", "0.4"], default=["0.8", "0.6", "0.4"])
    else:
        raise ValueError(f"unknown DSANet stage: {stage}")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    top = argparse.ArgumentParser(prog="python -m vadbench.workflows.dsanet")
    top.add_argument("stage", choices=STAGES, help="extract, score, or quality")
    if not args or args[0] in ("-h", "--help"):
        top.print_help()
        return
    stage = top.parse_args(args[:1]).stage
    options = stage_parser(stage).parse_args(args[1:])
    import_module(f".{STAGES[stage]}", package=__package__).run(options)


if __name__ == "__main__":
    main()
