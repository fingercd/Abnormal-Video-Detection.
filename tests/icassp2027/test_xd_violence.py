from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from vadbench.data.manifest import ManifestError, load_manifest_jsonl
from vadbench.data.xd_violence import (
    XDArchiveInventory,
    XDArchiveMember,
    XDImportError,
    load_xd_archive_members,
    parse_xd_labels,
    prepare_xd_data,
    write_prepared_xd_data,
)

VOLUME_COUNTS = {
    "train_0001_1004": 1004,
    "train_1005_2004": 1000,
    "train_2005_2804": 800,
    "train_2805_3319": 515,
    "train_3320_3954": 635,
    "test_videos": 800,
}
TRAIN_VOLUMES = tuple(name for name in VOLUME_COUNTS if name != "test_videos")


def _member_path(volume: str, index: int) -> str:
    if volume == "train_0001_1004" and index in {1, 2}:
        token = "B1" if index == 1 else "A"
        return f"Videos/train/shared_source__#{index:03d}_label_{token}.mp4"
    return f"Videos/{volume}/plain_{index:04d}_label_A.mp4"


def _inventory(
    *, remove: tuple[str, int] | None = None, shared_train_test: bool = False
) -> XDArchiveInventory:
    members: list[XDArchiveMember] = []
    for volume, count in VOLUME_COUNTS.items():
        for index in range(1, count + 1):
            if remove == (volume, index):
                continue
            members.append(
                XDArchiveMember(
                    volume=volume,
                    path=_member_path(volume, index),
                    crc32=index,
                    compressed_size=100 + index,
                    uncompressed_size=200 + index,
                )
            )
    if shared_train_test:
        shared = "Videos/collision/shared__001_label_B1.mp4"
        train_index = next(
            index for index, member in enumerate(members) if member.volume == "train_0001_1004"
        )
        test_index = next(
            index for index, member in enumerate(members) if member.volume == "test_videos"
        )
        members[train_index] = replace(members[train_index], path=shared)
        members[test_index] = replace(members[test_index], path=shared)
    return XDArchiveInventory(
        members=tuple(members),
        central_sha256={name: hashlib.sha256(name.encode()).hexdigest() for name in VOLUME_COUNTS},
    )


def _opaque_id(member_path: str) -> str:
    relative = member_path.removeprefix("Videos/").rsplit(".", 1)[0].casefold()
    return "xd-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()


def _touch_member(root: Path, member: XDArchiveMember) -> None:
    path = root / member.volume / member.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * member.uncompressed_size)


def _central_directory(paths: list[str]) -> bytes:
    entries = []
    for index, path in enumerate(paths):
        name = path.encode("utf-8")
        entries.append(
            struct.pack(
                "<4s6H3I5H2I",
                b"PK\x01\x02",
                20,
                20,
                0,
                0,
                0,
                0,
                index,
                100 + index,
                200 + index,
                len(name),
                0,
                0,
                0,
                0,
                0,
                0,
            )
            + name
        )
    return b"".join(entries)


def _write_central_metadata(
    metadata_dir: Path, paths_by_volume: dict[str, list[str]]
) -> dict[str, bytes]:
    metadata_dir.mkdir()
    central_bytes = {volume: _central_directory(paths) for volume, paths in paths_by_volume.items()}
    plan = {
        "volumes": [
            {
                "volume": volume,
                "status": "need_central",
                "central_directory_bytes": len(central_bytes[volume]),
            }
            for volume in VOLUME_COUNTS
        ]
    }
    (metadata_dir / "central_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for volume, data in central_bytes.items():
        (metadata_dir / f"{volume}.cd").write_bytes(data)
    return central_bytes


def test_parse_xd_labels_preserves_valid_multilabel_tokens_and_rejects_invalid_forms() -> None:
    assert parse_xd_labels("Videos/train/clip__001_label_B1-B4-0.mp4") == ("B1", "B4", "0")
    assert parse_xd_labels("Videos/train/normal_label_A.mp4") == ("A",)

    for path in (
        "Videos/train/clip_label_A-B1.mp4",
        "Videos/train/clip_label_0-B1.mp4",
        "Videos/train/clip_label_B1-B1.mp4",
        "Videos/train/clip_label_B3.mp4",
        "Videos/train/clip_without_a_label.mp4",
    ):
        with pytest.raises(XDImportError):
            parse_xd_labels(path)


def test_loader_preserves_central_member_identity_and_digest(tmp_path: Path) -> None:
    paths_by_volume = {
        volume: [_member_path(volume, index) for index in range(1, count + 1)]
        for volume, count in VOLUME_COUNTS.items()
    }
    central_bytes = _write_central_metadata(tmp_path / "metadata", paths_by_volume)

    inventory = load_xd_archive_members(tmp_path / "metadata")

    assert {member.volume for member in inventory.members} == set(VOLUME_COUNTS)
    assert {
        volume: sum(member.volume == volume for member in inventory.members)
        for volume in VOLUME_COUNTS
    } == VOLUME_COUNTS
    assert inventory.central_sha256 == {
        volume: hashlib.sha256(data).hexdigest() for volume, data in central_bytes.items()
    }
    matching = next(
        member
        for member in inventory.members
        if member.path.endswith("shared_source__#001_label_B1.mp4")
    )
    assert matching.uncompressed_size == 200
    assert matching.compressed_size == 100


@pytest.mark.parametrize(
    "path_by_volume, plan_length_delta",
    [
        ({"train_0001_1004": ["Videos/../escape_label_A.mp4"]}, 0),
        ({"train_0001_1004": ["Videos/train/valid_label_A.mp4"]}, 1),
    ],
)
def test_loader_rejects_path_traversal_and_planned_central_length_mismatch(
    tmp_path: Path, path_by_volume: dict[str, list[str]], plan_length_delta: int
) -> None:
    paths = {volume: [] for volume in VOLUME_COUNTS}
    paths.update(path_by_volume)
    metadata = tmp_path / "metadata"
    central_bytes = _write_central_metadata(metadata, paths)
    plan_path = metadata / "central_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["volumes"][0]["central_directory_bytes"] += plan_length_delta
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert central_bytes["train_0001_1004"]

    with pytest.raises(XDImportError):
        load_xd_archive_members(metadata)


def test_prepare_requires_full_official_volume_counts() -> None:
    incomplete = _inventory(remove=("train_2805_3319", 515))

    with pytest.raises(XDImportError):
        prepare_xd_data(incomplete, Path("not-used"))


def test_prepare_uses_complete_member_paths_keeps_sources_together_and_freezes_roles(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    dataset_root = tmp_path / "dataset"
    first, second = inventory.members[0], inventory.members[1]
    _touch_member(dataset_root, first)
    _touch_member(dataset_root, second)

    before = prepare_xd_data(inventory, dataset_root)
    (dataset_root / second.volume / second.path).unlink()
    after = prepare_xd_data(inventory, dataset_root)

    assert len(before.train) == 3954
    assert before.partition_lock == after.partition_lock
    assert {record.video_id for record in before.available_train} - {
        record.video_id for record in after.available_train
    } == {_opaque_id(second.path)}
    assert _opaque_id(first.path) in {record.video_id for record in after.available_train}
    first_record = next(
        record for record in before.train if record.video_id == _opaque_id(first.path)
    )
    assert first_record.resolve_path(dataset_root) == dataset_root / first.volume / first.path
    shared_ids = [_opaque_id(_member_path("train_0001_1004", index)) for index in (1, 2)]
    assert len({before.partition_lock[video_id] for video_id in shared_ids}) == 1
    assert {record.is_anomaly for record in before.train if record.video_id in shared_ids} == {
        False,
        True,
    }
    unknown_ids = [
        _opaque_id(member.path)
        for member in inventory.members
        if member.volume in TRAIN_VOLUMES and "__" not in Path(member.path).name
    ]
    assert unknown_ids
    assert {before.partition_lock[video_id] for video_id in unknown_ids} == {"fit"}
    receipt = before.receipt()
    assert receipt["source_verified"] == "unknown"
    assert receipt["derived_group_missing_train_videos"] > 0


def test_prepare_allows_duplicate_basenames_in_different_directories(tmp_path: Path) -> None:
    inventory = _inventory()
    members = list(inventory.members)
    first = next(
        index for index, member in enumerate(members) if member.volume == "train_1005_2004"
    )
    second = next(
        index for index, member in enumerate(members) if member.volume == "train_2005_2804"
    )
    members[first] = replace(members[first], path="Videos/one/same_label_B1.mp4")
    members[second] = replace(members[second], path="Videos/two/same_label_B1.mp4")
    inventory = replace(inventory, members=tuple(members))

    prepared = prepare_xd_data(inventory, tmp_path / "dataset")

    assert {
        _opaque_id("Videos/one/same_label_B1.mp4"),
        _opaque_id("Videos/two/same_label_B1.mp4"),
    }.issubset({record.video_id for record in prepared.train})


def test_prepare_rejects_a_member_reused_between_train_and_test(tmp_path: Path) -> None:
    with pytest.raises(XDImportError):
        prepare_xd_data(_inventory(shared_train_test=True), tmp_path / "dataset")


def test_test_index_contains_member_identity_only_and_writer_refuses_replacement(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    dataset_root = tmp_path / "dataset"
    _touch_member(dataset_root, inventory.members[0])
    prepared = prepare_xd_data(inventory, dataset_root)

    assert len(prepared.test_index) == 800
    forbidden = {
        "annotations",
        "category",
        "ground_truth",
        "is_anomaly",
        "label_tokens",
        "raw_label_tokens",
        "temporal_label",
        "temporal_truth",
        "weak_label",
    }
    for item in prepared.test_index:
        assert forbidden.isdisjoint(item)

    output_dir = tmp_path / "prepared"
    written = write_prepared_xd_data(prepared, output_dir)
    assert written
    assert all(path.is_file() for path in written.values())
    with pytest.raises(ManifestError):
        load_manifest_jsonl(written["test.identity"])
    with pytest.raises(FileExistsError):
        write_prepared_xd_data(prepared, output_dir)
