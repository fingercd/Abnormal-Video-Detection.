"""Real CPU head training on synthetic NPZ features; no encoder or real GT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import test_xd_quality_export as quality_fixture
import torch

from vadbench.paper import xd_evaluation as xd
from vadbench.paper import xd_head_repeats as repeat

pipeline = quality_fixture.pipeline
synthetic_dataset = quality_fixture.synthetic_dataset
_bounded_cpu_threads = quality_fixture._bounded_cpu_threads
pair = quality_fixture.pair


def _materialize_features(source):
    """Replace fixture-only array references with real per-clip NPZ bundles."""
    labels = {
        r.video_id: int(r.is_anomaly)
        for r in quality_fixture.load_manifest_jsonl(source.train_manifest)
    }
    labels.update(
        {
            r.video_id: int(r.is_anomaly)
            for r in quality_fixture.load_manifest_jsonl(source.validation_manifest)
        }
    )
    for folder in (source.train_feature_store, source.validation_feature_store):
        index = folder / "index.jsonl"
        rows = [json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
        for i, row in enumerate(rows):
            value = np.asarray(
                [labels[row["video_id"]], row["clip_index"] / 32, 0.25, -0.5], dtype=np.float32
            )
            path = folder / f"synthetic-{i:03d}.npz"
            np.savez(path, features=value[None], pooled=value)
            for name, reference in row["arrays"].items():
                array = value[None] if name == "features" else value
                reference.update(
                    path=path.name,
                    sha256=quality_fixture._sha(path),
                    shape=list(array.shape),
                    dtype=array.dtype.str,
                    nbytes=array.nbytes,
                )
        quality_fixture.helper._write_index(index, rows)


def test_xd_repeat_writer_trains_twenty_epochs_and_actual_loader_accepts_qa(pair, tmp_path):
    _materialize_features(pair.dense_head)
    source = xd.load_frozen_xd_detector_source(pair.dense_head.root, **pair.source_kwargs)
    request = repeat.FrozenXDHeadRepeatRequest(
        encoder="toy",
        source_controller_run=str(source.root),
        output_root=str(tmp_path / "trained"),
        seed=1,
        role_lock_path=str(pair.data.lock),
        head_data_contract_path=str(pair.data.contract),
        head_data_contract_sha256=quality_fixture._sha(pair.data.contract),
        source_manifest_root=str(pair.data.source_root),
        original_role_lock_path=pair.request.original_role_lock_path,
        scope_path=pair.request.scope_path,
        freeze_path=str(pair.data.freeze),
        device="cpu",
        run_id="actual-training",
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = repeat.run_frozen_xd_head_repeat(request)
    finally:
        torch.set_num_threads(previous_threads)
    loaded = repeat.load_frozen_xd_repeat_head_source(result.run_dir, **pair.source_kwargs)
    assert loaded.training_identity.seed == 1
    assert loaded.checkpoint_metadata["config"]["epochs"] == 20
    assert loaded.checkpoint_metadata["config"]["batch_size"] == 16
    assert loaded.checkpoint_metadata["config"]["learning_rate"] == 0.001
    assert loaded.checkpoint_metadata["config"]["expected_clips"] == 32
    qa = json.loads((Path(result.run_dir) / "head/training_qa.json").read_text(encoding="utf8"))
    assert qa["status"] == "passed" and qa["nonzero_gradient_steps"] > 0
    assert qa["checkpoint"]["epoch"] == 20
    assert qa["changed_parameter_count"] > 0 and qa["parameter_delta_l2"] > 0
    assert qa["reload_parity"]["outputs"]["snippet_logits"]["exact_equal"]
    assert qa["reload_parity"]["outputs"]["video_logits"]["exact_equal"]
    assert result.checkpoint_sha256 == quality_fixture._sha(loaded.checkpoint)
    loaded.verify_unchanged()
    # Fail the actual produced artifact after it has been accepted once.
    path = Path(result.run_dir) / "head/training_qa.json"
    qa["status"] = "failed"
    quality_fixture._write(path, qa)
    with pytest.raises(ValueError, match="QA"):
        repeat.load_frozen_xd_repeat_head_source(result.run_dir, **pair.source_kwargs)
