"""Audit all 800 raw XD videos and preregistered label-coordinate candidates on CPU."""

from __future__ import annotations

import argparse
import json

from vadbench.research.xd_raw_alignment import (
    RawAlignmentRequest,
    XDAlignmentError,
    run_raw_alignment_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "raw-root",
        "test-index",
        "canonical-metadata",
        "prior-alignment-root",
        "method-freeze",
        "volume-receipt",
        "output-root",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--cpu-lock-path",
        required=True,
        help="shared host CPU lease; acquired only after the full raw volume is ready",
    )
    parser.add_argument(
        "--resume-source-run",
        help="prior immutable/interrupted audit; reused probes are reverified and separately counted",
    )
    parser.add_argument(
        "--wait-for-ready",
        action="store_true",
        help="preregister now, then wait for the supervisor's full raw-volume ready receipt",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_raw_alignment_audit(RawAlignmentRequest(**vars(args)))
    except Exception as exc:
        # Do not expose a path containing a test identity, source interval or GT.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": str(exc)
                    if isinstance(exc, XDAlignmentError)
                    else type(exc).__name__,
                }
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "sealed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
