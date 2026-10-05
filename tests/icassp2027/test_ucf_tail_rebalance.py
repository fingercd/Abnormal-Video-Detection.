"""UCF tail replan must preserve completed videos and reject a live writer."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "work/dsanet-extension-20260923-r01/plan_ucf_tail_rebalance.py"
)
spec = importlib.util.spec_from_file_location("ucf_tail_rebalance", SOURCE)
assert spec and spec.loader
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    master = tmp_path / "ucf_train.jsonl"
    campaign = tmp_path / "old"
    rows = []
    pending_indices = {0, 1, 4, 5, 8, 9, 12, 13}
    for index in range(1610):
        video_id = f"v{index:04d}"
        video = tmp_path / f"{video_id}.mp4"
        if index in pending_indices:
            video.write_bytes(b"x" * (index + 1))
        rows.append({"video_id": video_id, "raw_path": str(video),
                     "dataset": "ucf", "role": "train", "label": "Normal",
                     "crop_ids": list(range(10))})
    master.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (campaign / "state").mkdir(parents=True)
    for shard in range(4):
        root = campaign / f"ucf-shard{shard}"
        (root / "receipts").mkdir(parents=True)
        (campaign / "state" / f"ucf-shard{shard}.pid").write_text("99999999")
        members = rows[shard::4]
        for row in members:
            if int(row["video_id"][1:]) not in pending_indices:
                (root / "receipts" / f"{row['video_id']}.json").write_text("{}")
        if shard in (2, 3):
            (root / "summary.json").write_text(
                json.dumps({"status": "completed", "videos": len(members)})
            )
    return master, campaign


def test_tail_plan_is_disjoint_and_refuses_live_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    master, campaign = make_fixture(tmp_path)
    output = tmp_path / "new"
    args = ["plan_ucf_tail_rebalance.py", "--master", str(master),
            "--old-campaign", str(campaign), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    (campaign / "state/ucf-shard0.pid").write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match="still alive"):
        planner.main()
    assert not output.exists()

    (campaign / "state/ucf-shard0.pid").write_text("99999999")
    planner.main()
    receipt = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert receipt["old_completed_videos"] == 1602
    assert receipt["new_pending_videos"] == 8
    assigned = [json.loads(line)["video_id"] for lane in range(4)
                for line in (output / f"ucf-part{lane}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assigned) == len(set(assigned)) == 8
    assert set(assigned) == {f"v{index:04d}" for index in (0, 1, 4, 5, 8, 9, 12, 13)}
