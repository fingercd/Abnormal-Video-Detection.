"""Compatibility gates for moved quality, timing, and UR-DMU modules."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


MOVED = {
    "quality_comparison": "vadbench.workflows.quality_comparison",
    "urdmu_quality_export": "vadbench.workflows.urdmu_quality_export",
    "efficiency": "vadbench.workflows.efficiency",
    "video_efficiency": "vadbench.workflows.video_efficiency",
    "urdmu_backend": "vadbench.integrations.detectors.urdmu.backend",
    "urdmu_inference": "vadbench.integrations.detectors.urdmu.inference",
}


@pytest.mark.parametrize("legacy,target", MOVED.items())
def test_legacy_and_canonical_imports_share_module_and_patch(legacy, target, monkeypatch):
    old = importlib.import_module(f"vadbench.paper.{legacy}")
    current = importlib.import_module(target)
    assert old is current
    assert old.__name__ == target
    marker = object()
    monkeypatch.setattr(old, "_compatibility_probe", marker, raising=False)
    assert current._compatibility_probe is marker


def test_author_attention_class_uses_canonical_backend_module():
    pytest.importorskip("torch")
    pytest.importorskip("einops")
    from vadbench.integrations.detectors.urdmu.backend import build_urdmu
    from vadbench.integrations.detectors.urdmu.inference import _iter_attention_modules

    upstream = Path(
        os.environ.get(
            "URDMU_UPSTREAM_DIR",
            Path(__file__).parents[1]
            / "outputs/icassp2027/research/author-recipes-20260918/UR-DMU",
        )
    ).resolve()
    if not upstream.is_dir():
        pytest.skip("pinned external UR-DMU checkout is unavailable")
    model, _ = build_urdmu(768, upstream, mode="Test")
    attention = list(_iter_attention_modules(model))
    assert len(attention) == 2
    assert all(
        type(module).__module__ == "vadbench.integrations.detectors.urdmu.backend"
        for _, module in attention
    )
