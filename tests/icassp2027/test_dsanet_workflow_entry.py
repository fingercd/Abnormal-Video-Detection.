"""The DSANet workflow CLI should stay light and preserve stage boundaries."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vadbench.token_reduction.bridges import clip
from vadbench.workflows.dsanet import __main__ as cli
from vadbench.workflows.dsanet._bridge import bridge_symbols


LEGACY_LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/icassp2027/dsanet_legacy.py"


def _legacy_module():
    spec = importlib.util.spec_from_file_location("dsanet_legacy_test", LEGACY_LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("args", [["--help"], ["extract", "--help"], ["score", "--help"], ["quality", "--help"]])
def test_help_does_not_load_runtime(args, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "import_module", lambda *a, **k: pytest.fail("runtime imported by help"))
    if args[0] == "--help":
        cli.main(args)
    else:
        with pytest.raises(SystemExit, match="0"):
            cli.main(args)
    assert "usage:" in capsys.readouterr().out


def test_extract_passes_paths_and_uses_core_bridge_by_default(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "import_module", lambda name, package: SimpleNamespace(run=calls.append))
    cli.main([
        "extract", "--manifest", "manifest.jsonl", "--clip-repo", "openai-clip",
        "--weights", "ViT-B-16.pt", "--output", "features", "--gpu", "3",
        "--keep-ratio", "0.6", "--workers", "1",
    ])
    options = calls.pop()
    assert options.manifest == Path("manifest.jsonl")
    assert options.gpu == 3 and options.keep_ratio == 0.6 and options.workers == 1
    assert options.bridge is None
    bridge_type, loader, path = bridge_symbols(options.bridge)
    assert bridge_type is clip.ClipVisionBridge
    assert loader is clip.load_openai_clip
    assert path.resolve() == Path(clip.__file__).resolve()


def test_score_passes_fixed_inputs_without_truth(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "import_module", lambda name, package: SimpleNamespace(run=calls.append))
    cli.main([
        "score", "--dataset", "xd", "--upstream-src", "DSANet",
        "--manifest", "xd_test.jsonl", "--features", "xd-dense",
        "--checkpoint", "model_xd.pth", "--output", "xd-scores",
    ])
    options = calls.pop()
    assert options.dataset == "xd" and options.features == Path("xd-dense")
    assert options.checkpoint == Path("model_xd.pth")
    assert not hasattr(options, "truth")
    with pytest.raises(SystemExit, match="2"):
        cli.main([
            "score", "--dataset", "xd", "--upstream-src", "DSANet",
            "--manifest", "xd_test.jsonl", "--features", "xd-dense",
            "--checkpoint", "model_xd.pth", "--output", "xd-scores",
            "--truth", "truth.jsonl",
        ])


def test_quality_passes_sealed_inputs(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "import_module", lambda name, package: SimpleNamespace(run=calls.append))
    cli.main([
        "quality", "--dataset", "ucf", "--scores-root", "scores",
        "--truth", "ucf.jsonl", "--output", "quality",
        "--checkpoint-origin", "published", "--tiers", "0.8", "0.4",
    ])
    options = calls.pop()
    assert options.truth == Path("ucf.jsonl")
    assert options.tiers == ["0.8", "0.4"]


def test_legacy_launcher_injects_current_bridge_only_when_needed() -> None:
    legacy = _legacy_module()
    args = legacy._stage_args(["extract", "--manifest", "source.jsonl"])
    assert args[1:3] == ["--bridge", str(legacy.CORE_BRIDGE)]
    assert legacy.CORE_BRIDGE == Path(clip.__file__).resolve()
    custom = ["extract", "--bridge", "historical_clip_bridge.py"]
    assert legacy._stage_args(custom) == custom
    assert legacy._stage_args(["score", "--dataset", "ucf"]) == ["score", "--dataset", "ucf"]


@pytest.mark.parametrize("stage", [None, "extract", "score"])
def test_legacy_help_avoids_vadbench_top_level_import(stage) -> None:
    # -S ensures the help path only needs the standard library; check in a fresh process.
    code = (
        "import runpy,sys\n"
        "path=sys.argv[1]\n"
        "sys.argv=[path]+sys.argv[2:]\n"
        "try: runpy.run_path(path, run_name='__main__')\n"
        "except SystemExit as exc: assert exc.code == 0, exc.code\n"
        "assert 'vadbench' not in sys.modules\n"
        "assert '_vadbench_dsanet_legacy.__main__' in sys.modules\n"
    )
    args = [sys.executable, "-S", "-c", code, str(LEGACY_LAUNCHER)]
    if stage:
        args.append(stage)
    result = subprocess.run([*args, "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_legacy_quality_requires_modern_python(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        _legacy_module().main(["quality", "--help"])
    assert "Python 3.10+" in capsys.readouterr().err
