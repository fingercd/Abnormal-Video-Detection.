"""A damaged final checkpoint must stop the eight-way scoring handoff."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "work/dsanet-extension-20260923-r01/verify_four_heads.py"
)
spec = importlib.util.spec_from_file_location("dsanet_head_gate", SOURCE)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_head_gate_requires_unchanged_four_fixed_final_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_root = tmp_path / "csv"
    csv_root.mkdir()
    root = tmp_path / "heads"
    root.mkdir()
    for dataset, count in (("ucf", 16100), ("xd", 39500)):
        csv_path = csv_root / f"{dataset}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("path", "label"))
            writer.writerows((f"/features/{dataset}-{index}.npy", "Normal") for index in range(count))
        (csv_root / f"{dataset}.json").write_text(
            json.dumps({"status": "completed", "feature_rows": count,
                        "csv_sha256": gate.sha256(csv_path)}), encoding="utf-8"
        )
        for seed in (234, 235):
            head = root / f"{dataset}-seed{seed}"
            head.mkdir()
            final = head / "final.pt"
            final.write_bytes(f"{dataset}-{seed}-fixed-final".encode())
            params = {"total_parameters": 100, "trainable_parameters": 20,
                      "frozen_parameters": 80}
            (head / "resolved.json").write_text(
                json.dumps({"dataset": dataset, "seed": seed,
                            "full_train_rows": count, "smoke_one_step": False,
                            "train_csv": str(csv_path), **params}), encoding="utf-8"
            )
            (head / "completed.json").write_text(
                json.dumps({"status": "completed", "epoch": 10,
                            "checkpoint_selection": "fixed_final_epoch_no_test_access",
                            "checkpoint_sha256": gate.sha256(final),
                            "checkpoint_bytes": final.stat().st_size, **params}), encoding="utf-8"
            )

    args = ["verify_four_heads.py", "--root", str(root), "--csv-root", str(csv_root),
            "--output", str(tmp_path / "passed.json")]
    monkeypatch.setattr(sys, "argv", args)
    gate.main()
    assert json.loads((tmp_path / "passed.json").read_text())["status"] == "four_fixed_final_heads_ready"

    (root / "xd-seed235/final.pt").write_bytes(b"changed after training")
    args[-1] = str(tmp_path / "rejected.json")
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="changed"):
        gate.main()
    assert not (tmp_path / "rejected.json").exists()
