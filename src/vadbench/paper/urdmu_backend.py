"""Load the pinned, external UR-DMU network and original AD_Loss definitions.

This is a backend candidate until the author feature assets have been reproduced.
No third-party code is vendored, imported as a module, or written back. Only the
reviewed definitions below execute after all five source files and Git HEAD pass
their fixed identity checks. The six original CUDA constant allocations follow
the current input device; architecture, parameter names and loss terms are intact.

Train input is the author's normal-first, abnormal-second balanced batch, with
shape [B,T,D] or [B,crops,T,D]. Changing an existing model between training and
inference requires BOTH ``model.flag = 'Train'/'Test'`` and ``train()/eval()``;
these are separate controls in the original implementation. Serialize state_dict,
not the dynamically created external Python classes.
"""

from __future__ import annotations

import ast
import hashlib
import io
import math
import subprocess
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

UPSTREAM_COMMIT = "40cfdf5f8bebbc3b59373935f9d6ca00e2f32bbc"
SOURCE_SHA256 = {
    "utils.py": "2df18a9898a53d2269a61706ad0f17b62f9f566e403077f42f9af8fca451a34f",
    "memory.py": "707f33164bb721bab9767f50b5fe053183341ca634f0e4339837f15a00c6f2de",
    "translayer.py": "d2c1596effc7ecd5d0940fcc01845f8a18b2d97a5cc4cdd4726cd08928ace1bf",
    "model.py": "6cebc33270dd1f28384747a2b45536b26753d18f518509250a8956d9f40d26d5",
    "train.py": "c5d30ffc7dc401925d5d00c983aa32bf3add77c050b582e3d87b61698d2b9901",
}
OFFICIAL_CHECKPOINT_SHA256 = {
    "ucf_trans_2022.pkl": "2a64562df08a4142b19067fe46f07c6fdaf9923de36ded1b827489d6196429cd",
    "xd_trans_2022.pkl": "f30e9fa84d9eb680b7b13d0c5d7406b11e53dcdf37f3827fcc8899d2448f3d9b",
}
_DEFINITIONS = {
    "utils.py": ("norm",),
    "memory.py": ("Memory_Unit",),
    "translayer.py": ("PreNorm", "FeedForward", "Attention", "Transformer"),
    "model.py": ("Temporal", "ADCLS_head", "WSAD"),
    "train.py": ("AD_Loss",),
}
LOSS_COEFFICIENTS = {
    "att_loss": 1.0,
    "A_loss": 0.1,
    "N_Aatt": 0.1,
    "N_loss": 0.1,
    "A_Nloss": 0.1,
    "triplet": 0.1,
    "kl_loss": 0.001,
    "distance": 0.0001,
}


class _InputDeviceConstants(ast.NodeTransformer):
    """Adapt only the reviewed no-argument Tensor.cuda calls in selected code."""

    def __init__(self, filename: str, input_name: str):
        self.filename = filename
        self.input_name = input_name
        self.receipts: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            if node.args or node.keywords:
                raise ValueError("unreviewed UR-DMU CUDA call signature")
            changed = ast.Call(
                func=ast.Attribute(value=node.func.value, attr="to", ctx=ast.Load()),
                args=[
                    ast.Attribute(
                        value=ast.Name(id=self.input_name, ctx=ast.Load()),
                        attr="device",
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            )
            self.receipts.append(
                {
                    "file": self.filename,
                    "line": node.lineno,
                    "before": ast.unparse(node),
                    "after": ast.unparse(changed),
                }
            )
            return ast.copy_location(changed, node)
        return self.generic_visit(node)


def _load_definitions(upstream_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(upstream_dir).expanduser().resolve()
    # Keep and execute these exact checked bytes, avoiding a hash/read race.
    sources, raw_hashes = {}, {}
    for name, expected in SOURCE_SHA256.items():
        content = (root / name).read_bytes()
        raw_hashes[name] = hashlib.sha256(content).hexdigest()
        # Git's Windows checkout may use CRLF. Pin exact upstream Git blob
        # content after this single newline conversion; no whitespace/code
        # changes are accepted. Execute these same checked bytes.
        canonical = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise ValueError(f"UR-DMU upstream SHA-256 mismatch: {name}")
        sources[name] = canonical.decode("utf8")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("UR-DMU upstream Git HEAD differs from the pinned commit")

    # A real installed einops is required. Environment/path configuration belongs
    # to the caller; this module never installs packages or changes sys.path.
    import einops

    namespace = {
        "__name__": __name__,
        "torch": torch,
        "nn": nn,
        "Module": nn.Module,
        "math": math,
        "rearrange": einops.rearrange,
    }
    patches = []
    definition_hashes = {}
    for name, names in _DEFINITIONS.items():
        tree = ast.parse(sources[name], filename=str(root / name))
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        if tuple(node.name for node in selected) != names:
            raise ValueError(f"UR-DMU reviewed definition inventory differs: {name}")
        module = ast.Module(body=selected, type_ignores=[])
        if name in {"translayer.py", "train.py"}:
            transform = _InputDeviceConstants(name, "x" if name == "translayer.py" else "att")
            module = transform.visit(module)
            expected_calls = 2 if name == "translayer.py" else 4
            if len(transform.receipts) != expected_calls:
                raise ValueError(f"UR-DMU reviewed device allocation count differs: {name}")
            patches.extend(transform.receipts)
        ast.fix_missing_locations(module)
        definition_hashes[name] = hashlib.sha256(
            ast.dump(module, include_attributes=False).encode("utf8")
        ).hexdigest()
        exec(compile(module, str(root / name), "exec"), namespace)
    return namespace, {
        "schema_version": 1,
        "backend": "urdmu",
        "status": "candidate_pending_author_feature_asset_reproduction",
        "upstream_dir": str(root),
        "upstream_commit": commit,
        "source_sha256": dict(SOURCE_SHA256),
        "source_hash_basis": "exact_upstream_git_blob_after_CRLF_to_LF_only",
        "raw_source_sha256": raw_hashes,
        "selected_definitions": {name: list(names) for name, names in _DEFINITIONS.items()},
        "executed_definition_ast_sha256": definition_hashes,
        "device_patches": patches,
        "third_party_files_modified": False,
        "module_imports_from_upstream": False,
        "torch_version": str(torch.__version__),
        "einops_version": einops.__version__,
        "einops_file": str(Path(einops.__file__).resolve()),
        "checkpoint_serialization": "state_dict_only",
    }


def build_urdmu(
    input_dim: int,
    upstream_dir: str | Path,
    *,
    mode: Literal["Train", "Test"] = "Train",
    checkpoint: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the original 60+60-memory WSAD; optionally load pinned author weights.

    Both author checkpoints have input dimension 1024. Newly trained 768/1024
    checkpoints should be restored through normal strict state_dict loading,
    with the training controller's own artifact SHA binding.
    """
    if type(input_dim) is not int or input_dim not in {768, 1024}:
        raise ValueError("UR-DMU input_dim must be 768 or 1024")
    if mode not in {"Train", "Test"}:
        raise ValueError("UR-DMU mode must be Train or Test")
    namespace, receipt = _load_definitions(upstream_dir)
    model = namespace["WSAD"](input_dim, flag=mode, a_nums=60, n_nums=60)
    model.train(mode == "Train")
    checkpoint_receipt = None
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        names = [
            name for name, expected in OFFICIAL_CHECKPOINT_SHA256.items() if expected == digest
        ]
        if not names:
            raise ValueError("UR-DMU checkpoint SHA-256 differs from both pinned author assets")
        state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        loaded = model.load_state_dict(state, strict=True)
        checkpoint_receipt = {
            "path": str(path),
            "sha256": digest,
            "author_asset": names[0],
            "weights_only": True,
            "strict": True,
            "missing_keys": list(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys),
        }
    return model, {
        **receipt,
        "input_dim": input_dim,
        "initial_mode": mode,
        "anomaly_memory_slots": 60,
        "normal_memory_slots": 60,
        "checkpoint": checkpoint_receipt,
    }


def build_urdmu_loss(upstream_dir: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Return original AD_Loss with unchanged coefficients and logging keys.

    Forward returns ``(total, losses)``. As upstream, the distance regularizer
    contributes to total but is exposed in the network result, not losses dict.
    """
    namespace, receipt = _load_definitions(upstream_dir)
    return namespace["AD_Loss"](), {**receipt, "loss_coefficients": dict(LOSS_COEFFICIENTS)}


__all__ = ["build_urdmu", "build_urdmu_loss"]
