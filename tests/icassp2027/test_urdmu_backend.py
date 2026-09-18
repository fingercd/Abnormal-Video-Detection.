"""Real pinned UR-DMU asset integration; synthetic inputs are not paper results.

These tests skip when the external checkout or checkpoints are unavailable in
CI. They neither vendor a substitute implementation nor modify the author clone.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vadbench.paper import urdmu_backend as backend  # noqa: E402


def test_upstream_lf_and_crlf_checkouts_execute_identical_definitions(upstream, tmp_path):
    model_lf = model_crlf = None
    receipts = []
    for ending in ("lf", "crlf"):
        directory = tmp_path / ending
        directory.mkdir()
        # Read-only Git metadata reference; the loader only runs rev-parse.
        (directory / ".git").write_text(f"gitdir: {(upstream / '.git').as_posix()}\n")
        for name, digest in backend.SOURCE_SHA256.items():
            content = (upstream / name).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(content).hexdigest() == digest
            (directory / name).write_bytes(content if ending == "lf" else content.replace(b"\n", b"\r\n"))
        torch.manual_seed(17)
        model, receipt = backend.build_urdmu(768, directory, mode="Test")
        receipts.append(receipt)
        if ending == "lf":
            model_lf = model
        else:
            model_crlf = model
    assert receipts[0]["source_sha256"] == receipts[1]["source_sha256"]
    assert receipts[0]["raw_source_sha256"] != receipts[1]["raw_source_sha256"]
    assert receipts[0]["executed_definition_ast_sha256"] == receipts[1]["executed_definition_ast_sha256"]
    with torch.inference_mode():
        value = torch.randn(1, 16, 768)
        assert torch.equal(model_lf(value)["frame"], model_crlf(value)["frame"])


@pytest.fixture(scope="module")
def upstream():
    root = Path(
        os.environ.get(
            "URDMU_UPSTREAM_DIR",
            str(
                Path(__file__).parents[2]
                / "outputs/icassp2027/research/author-recipes-20260918/UR-DMU"
            ),
        )
    ).resolve()
    if not root.is_dir():
        pytest.skip("pinned external UR-DMU checkout is not available")
    # Torch above is already imported from the active project interpreter.
    # Reuse only an existing einops installation for this process, never install.
    dependency_site = Path("C:/Users/lenovo/anaconda3/envs/pytorch/Lib/site-packages")
    extra_path = None
    if importlib.util.find_spec("einops") is None:
        if not (dependency_site / "einops").is_dir():
            pytest.skip("real einops is unavailable in this execution environment")
        extra_path = str(dependency_site)
        sys.path.append(extra_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield root
    finally:
        torch.set_num_threads(previous_threads)
        if extra_path is not None:
            sys.path.remove(extra_path)


@pytest.mark.parametrize("checkpoint_name", ["ucf_trans_2022.pkl", "xd_trans_2022.pkl"])
def test_author_checkpoints_strict_finite_eval_and_state_dict_reload(
    upstream, checkpoint_name, tmp_path
):
    checkpoint = upstream / "models" / checkpoint_name
    if not checkpoint.is_file():
        pytest.skip(f"external author checkpoint unavailable: {checkpoint_name}")
    watched = ("model", "memory", "translayer", "utils", "train", "visdom", "ipdb")
    before_modules = {name: sys.modules.get(name) for name in watched}
    model, receipt = backend.build_urdmu(1024, upstream, mode="Test", checkpoint=checkpoint)
    assert {name: sys.modules.get(name) for name in watched} == before_modules
    assert not model.training and model.flag == "Test"
    assert receipt["upstream_commit"] == backend.UPSTREAM_COMMIT
    assert receipt["source_sha256"] == backend.SOURCE_SHA256
    assert receipt["checkpoint"]["weights_only"] and receipt["checkpoint"]["strict"]
    assert receipt["checkpoint"]["sha256"] == backend.OFFICIAL_CHECKPOINT_SHA256[checkpoint_name]
    assert receipt["checkpoint"]["missing_keys"] == receipt["checkpoint"]["unexpected_keys"] == []
    assert len(model.state_dict()) == 34
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    torch.manual_seed(20260918)
    inputs = torch.randn(1, 2, 32, 1024)
    with torch.inference_mode():
        prediction = model(inputs)["frame"]
        separately = model(inputs.reshape(2, 32, 1024))["frame"].mean(0, keepdim=True)
    assert prediction.shape == (1, 32)
    assert torch.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    torch.testing.assert_close(prediction, separately, atol=1e-7, rtol=1e-6)
    path = tmp_path / "reloaded-state.pt"
    torch.save(model.state_dict(), path)
    restored, _ = backend.build_urdmu(1024, upstream, mode="Test")
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        assert torch.equal(prediction, restored(inputs)["frame"])


@pytest.mark.parametrize("input_dim", [768, 1024])
def test_original_train_forward_all_losses_backward_and_optimizer(upstream, input_dim, monkeypatch):
    def forbidden_cuda(*args, **kwargs):
        pytest.fail("reviewed UR-DMU definitions attempted a hard-coded Tensor.cuda allocation")

    monkeypatch.setattr(torch.Tensor, "cuda", forbidden_cuda)
    torch.manual_seed(20260918)
    model, receipt = backend.build_urdmu(input_dim, upstream, mode="Train")
    criterion, loss_receipt = backend.build_urdmu_loss(upstream)
    assert model.training and model.flag == "Train"
    assert model.Amemory.memory_block.shape == model.Nmemory.memory_block.shape == (60, 512)
    assert len(receipt["device_patches"]) == 6
    assert receipt["device_patches"] == loss_receipt["device_patches"]
    assert [p["file"] for p in receipt["device_patches"]].count("translayer.py") == 2
    assert [p["file"] for p in receipt["device_patches"]].count("train.py") == 4
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    # The first half is normal, the second half abnormal, as the author's Train path requires.
    inputs = torch.randn(4, 32, input_dim)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    result = model(inputs)
    cost, losses = criterion(result, labels)
    assert set(losses) == {
        "total_loss",
        "att_loss",
        "N_Aatt",
        "A_loss",
        "N_loss",
        "A_Nloss",
        "triplet",
        "kl_loss",
    }
    assert result["frame"].shape == (4, 32)
    assert all(torch.isfinite(value).all() for value in result.values())
    assert all(torch.isfinite(value).all() for value in losses.values())
    expected = (
        losses["att_loss"]
        + 0.1 * (losses["A_loss"] + losses["N_Aatt"] + losses["N_loss"] + losses["A_Nloss"])
        + 0.1 * losses["triplet"]
        + 0.001 * losses["kl_loss"]
        + 0.0001 * result["distance"]
    )
    assert torch.equal(cost, expected)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.zero_grad()
    cost.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    components = (
        "embedding",
        "selfatt",
        "Amemory",
        "Nmemory",
        "encoder_mu",
        "encoder_var",
        "cls_head",
    )
    for component in components:
        assert any(
            parameter.grad.abs().sum() > 0
            for name, parameter in model.named_parameters()
            if name.startswith(component + ".")
        ), f"no gradient reached {component}"
    optimizer.step()
    for component in components:
        assert any(
            not torch.equal(original[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if name.startswith(component + ".")
        ), f"optimizer did not update {component}"


def test_changed_source_bytes_fail_before_any_definition_executes(upstream, monkeypatch):
    read_bytes = Path.read_bytes

    def changed_bytes(path):
        content = read_bytes(path)
        return content + b"\n# injected mutation" if path == upstream / "train.py" else content

    # Simulate a changed external file without modifying the read-only clone.
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    monkeypatch.setattr(
        backend.ast, "parse", lambda *a, **k: pytest.fail("unverified source was parsed/executed")
    )
    with pytest.raises(ValueError, match="upstream SHA-256 mismatch: train.py"):
        backend.build_urdmu(768, upstream)


@pytest.mark.parametrize("dimension", [True, 512, 768.0])
def test_unsupported_input_dimensions_fail_before_external_access(dimension, tmp_path):
    with pytest.raises(ValueError, match="input_dim"):
        backend.build_urdmu(dimension, tmp_path / "does-not-exist")


def test_unknown_mode_is_not_silently_treated_as_test(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        backend.build_urdmu(768, tmp_path / "does-not-exist", mode="typo")


def test_unknown_checkpoint_is_not_treated_as_official(upstream, tmp_path):
    checkpoint = tmp_path / "unknown.pt"
    checkpoint.write_bytes(b"not an author checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        backend.build_urdmu(1024, upstream, checkpoint=checkpoint)
