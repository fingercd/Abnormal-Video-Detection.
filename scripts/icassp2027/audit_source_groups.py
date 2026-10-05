"""Audit frozen videos for near-duplicate candidates without changing source groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vadbench.data.manifest import VideoManifestRecord, load_manifest_jsonl
from vadbench.research.source_audit import (
    SourceAuditConfig,
    audit_source_candidates,
    write_contact_sheets,
    write_overview_contact_sheet,
    write_source_audit,
)


def _mapping(path: Path | None, *, identity: bool = False) -> dict[str, str] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("映射文件必须是 JSON 对象")
    if identity:
        return {
            video_id: value["sha256"]
            for video_id, value in raw.items()
            if isinstance(video_id, str)
            and isinstance(value, dict)
            and isinstance(value.get("sha256"), str)
        }
    if not all(isinstance(video_id, str) and isinstance(value, str) for video_id, value in raw.items()):
        raise ValueError("source 映射必须是 video_id 到 source_id 的字符串对象")
    return raw


def _records_from_cohort(path: Path, dataset_root: Path) -> tuple[VideoManifestRecord, ...]:
    """Resolve a frozen cohort's IDs locally without consulting its labels."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    video_ids = list(dict.fromkeys(row["video_id"] for row in rows))
    by_stem: dict[str, list[Path]] = {}
    for video in dataset_root.rglob("*.mp4"):
        by_stem.setdefault(video.stem, []).append(video)
    records = []
    for video_id in video_ids:
        matches = by_stem.get(video_id, [])
        if len(matches) != 1:
            raise ValueError(f"cohort video_id={video_id!r} 在 dataset root 中应恰好对应一个视频")
        relative = matches[0].relative_to(dataset_root).as_posix()
        # The audit never reads this placeholder label/category; it only needs the frozen ID and path.
        records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=relative,
                split="train",
                category="audit_only",
                is_anomaly=False,
            )
        )
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path)
    input_group.add_argument("--cohort", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--known-identities", type=Path)
    parser.add_argument("--verified-sources", type=Path)
    parser.add_argument("--samples-per-video", type=int, default=9)
    parser.add_argument("--max-hamming-distance", type=int, default=6)
    parser.add_argument("--min-matching-frames", type=int, default=4)
    parser.add_argument("--min-consecutive-matches", type=int, default=3)
    parser.add_argument("--max-contact-sheets", type=int, default=20)
    args = parser.parse_args()

    records = (
        load_manifest_jsonl(args.manifest, dataset_root=args.dataset_root, require_files=True)
        if args.manifest is not None
        else _records_from_cohort(args.cohort, args.dataset_root)
    )
    result = audit_source_candidates(
        records,
        args.dataset_root,
        known_sha256=_mapping(args.known_identities, identity=True),
        verified_sources=_mapping(args.verified_sources),
        config=SourceAuditConfig(
            samples_per_video=args.samples_per_video,
            max_hamming_distance=args.max_hamming_distance,
            min_matching_frames=args.min_matching_frames,
            min_consecutive_matches=args.min_consecutive_matches,
            max_contact_sheets=args.max_contact_sheets,
        ),
    )
    audit_path = write_source_audit(result, args.output_dir)
    sheets = (
        write_contact_sheets(result, records, args.dataset_root, args.output_dir / "contact-sheets")
        if result.candidates
        else []
    )
    overview = write_overview_contact_sheet(
        result, records, args.dataset_root, args.output_dir / "overview-contact-sheet.png"
    )
    print(
        json.dumps(
            {
                "audit": str(audit_path.resolve()),
                "video_count": len(result.videos),
                "candidate_pair_count": len(result.candidates),
                "contact_sheets": [str(path.resolve()) for path in sheets],
                "overview_contact_sheet": str(overview.resolve()),
                "automatic_source_group_updates": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
