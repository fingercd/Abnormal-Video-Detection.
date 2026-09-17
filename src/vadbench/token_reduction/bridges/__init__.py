"""Instance-scoped observation bridges for the four active encoders.

These bridges locate modules on an already loaded model.  They never replace a
global ``forward`` method and they intentionally expose no variable-length
execution path in this first identity-only stage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch.nn as nn


class BridgeUnsupportedError(RuntimeError):
    """Raised when a loaded model does not expose a verified bridge path."""


@dataclass(frozen=True)
class EncoderBridgeReceipt:
    """Evidence captured while locating a concrete loaded-model bridge."""

    encoder_id: str
    model_type: str
    block_path: str
    block_count: int
    layout: str | None
    position_contract: str | None
    has_cls: bool | None
    special_token_indices: tuple[int, ...] | None
    probe_available: bool
    reduction_ready: bool
    supports_grad: bool | None
    notes: tuple[str, ...] = ()

    @property
    def reduction_available(self) -> bool:
        """Compatibility spelling used in the encoder-scope document."""

        return self.reduction_ready


def _at_path(root: Any, path: tuple[str, ...]) -> Any | None:
    value = root
    for name in path:
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def _unwrap_loaded_model(encoder_id: str, loaded: Any) -> nn.Module:
    """Unwrap only the repository's known adapter/wrapper seams."""

    if isinstance(loaded, nn.Module):
        return loaded
    if encoder_id == "videomaev2":
        encoder = getattr(loaded, "encoder", None)
        if isinstance(encoder, nn.Module):
            return encoder
    elif encoder_id in {"timesformer", "videomae"}:
        model = getattr(loaded, "model", None)
        if isinstance(model, nn.Module):
            return model
    elif encoder_id == "vjepa2":
        worker = getattr(loaded, "encoder", loaded)
        if worker is None and hasattr(loaded, "bridge"):
            worker = loaded.bridge.encoder
        model = getattr(worker, "model", None)
        if isinstance(model, nn.Module):
            return model
    raise BridgeUnsupportedError(
        f"{encoder_id} bridge 需要已加载的 torch.nn.Module，未找到已知 adapter/wrapper 的模型对象"
    )


_BLOCK_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "videomaev2": (
        ("backbone", "videomae", "encoder", "layer"),
        ("backbone", "encoder", "layer"),
        ("backbone", "model", "blocks"),
        ("backbone", "blocks"),
        ("videomae", "encoder", "layer"),
        ("encoder", "layer"),
        ("model", "blocks"),
        ("blocks",),
    ),
    "timesformer": (
        ("timesformer", "encoder", "layer"),
        ("encoder", "layer"),
        ("model", "encoder", "layer"),
    ),
    "videomae": (
        ("videomae", "encoder", "layer"),
        ("encoder", "layer"),
        ("model", "encoder", "layer"),
    ),
    # The bridge deliberately never walks a ``predictor`` path.  Current
    # V-JEPA adapters invoke get_vision_features, so only a named vision
    # encoder is eligible for observation.
    "vjepa2": (
        ("encoder", "layer"),
        ("vision_model", "encoder", "layer"),
        ("vision_model", "blocks"),
        ("vision_encoder", "encoder", "layer"),
        ("vision_encoder", "blocks"),
    ),
}


def _find_blocks(encoder_id: str, root: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in _BLOCK_PATHS[encoder_id]:
        candidate = _at_path(root, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return ".".join(path), candidate
    supported = ", ".join(".".join(path) for path in _BLOCK_PATHS[encoder_id])
    raise BridgeUnsupportedError(
        f"{encoder_id} 未在已加载模型上找到可验证的 block 路径；仅支持：{supported}"
    )


def _cls_metadata(root: nn.Module) -> tuple[bool | None, tuple[int, ...] | None]:
    for path in (
        ("cls_token",),
        ("embeddings", "cls_token"),
        ("videomae", "embeddings", "cls_token"),
        ("timesformer", "embeddings", "cls_token"),
    ):
        if _at_path(root, path) is not None:
            return True, (0,)
    config = getattr(root, "config", None)
    for name in ("use_cls_token", "add_cls_token"):
        value = getattr(config, name, None)
        if isinstance(value, bool):
            return value, (0,) if value else ()
    # No class-token assertion may be inferred from a sequence's first entry.
    return None, None


def _first_module(block: nn.Module, paths: tuple[tuple[str, ...], ...]) -> nn.Module | None:
    for path in paths:
        candidate = _at_path(block, path)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


class EncoderBridge:
    """Read-only, concrete-model bridge with named modules for ProbeCollector."""

    def __init__(self, encoder_id: str, loaded_model: Any) -> None:
        if encoder_id not in _BLOCK_PATHS:
            allowed = ", ".join(sorted(_BLOCK_PATHS))
            raise BridgeUnsupportedError(f"未知 active encoder={encoder_id!r}；允许：{allowed}")
        self._encoder_id = encoder_id
        self._root = _unwrap_loaded_model(encoder_id, loaded_model)
        self._block_path, self._blocks = _find_blocks(encoder_id, self._root)
        has_cls, special_indices = _cls_metadata(self._root)
        self._block_owner = _at_path(self._root, tuple(self._block_path.split(".")[:-1]))
        if encoder_id == "videomaev2" and hasattr(self._block_owner, "patch_embed"):
            # This native backbone embeds patches only; runtime sequence length
            # is checked against the actual Conv3d output by TubeletGeometry.
            has_cls = hasattr(self._block_owner, "cls_token")
            special_indices = (0,) if has_cls else ()
        layout = (
            getattr(getattr(self._root, "config", None), "attention_type", "unverified")
            if encoder_id == "timesformer"
            else "unverified"
        )
        position_contract = (
            "no-ragged-reduction-path-established"
            if encoder_id == "timesformer"
            else "position-handling-unverified"
        )
        notes = [
            "仅暴露实例级只读 observation sites；没有重写模型 forward。",
            "当前仅 identity baseline；尚未验证真实变长后续计算，reduction_ready=false。",
        ]
        supports_grad = False if encoder_id == "vjepa2" else None
        if encoder_id == "vjepa2":
            notes.append(
                "现有 V-JEPA 2 wrapper 在 get_vision_features 外使用 torch.no_grad；"
                "本 bridge 不提供训练内部插件路径。"
            )
        self._receipt = EncoderBridgeReceipt(
            encoder_id=encoder_id,
            model_type=f"{type(self._root).__module__}.{type(self._root).__qualname__}",
            block_path=self._block_path,
            block_count=len(self._blocks),
            layout=layout,
            position_contract=position_contract,
            has_cls=has_cls,
            special_token_indices=special_indices,
            probe_available=True,
            reduction_ready=False,
            supports_grad=supports_grad,
            notes=tuple(notes),
        )

    def receipt(self) -> EncoderBridgeReceipt:
        """Return the immutable result of actual model-path resolution."""

        return self._receipt

    @property
    def model(self) -> nn.Module:
        return self._root

    def geometry(self, frame_indices: Any, valid_mask: Any) -> Any:
        """Observe verified native geometry; unsupported layouts fail explicitly."""
        from .geometry import TubeletGeometry

        if self._encoder_id != "videomaev2" or not hasattr(self._block_owner, "patch_embed"):
            raise BridgeUnsupportedError(
                "runtime tubelet geometry currently supports native VideoMAEv2 only"
            )
        if self._receipt.has_cls:
            raise BridgeUnsupportedError("native VideoMAEv2 geometry expects patch-only sequences")
        return TubeletGeometry(self._block_owner.patch_embed, frame_indices, valid_mask)

    def architecture(self) -> dict[str, Any]:
        """Describe actual modules, including operations that bypass module hooks."""
        import inspect
        from dataclasses import asdict
        from pathlib import Path

        from vadbench.hashing import sha256_file

        result = asdict(self.receipt())
        sources = {}
        for module in (self._root, self._block_owner, self._blocks[0]):
            source = Path(inspect.getfile(type(module)))
            if source.is_file():
                sources[str(source)] = sha256_file(source)
        result["implementation_sources_sha256"] = sources
        result["blocks"] = []
        for index, block in enumerate(self._blocks):
            attention = _first_module(block, (("attn",), ("attention",), ("self_attn",)))
            inner_attention = getattr(attention, "attention", attention)
            result["blocks"].append(
                {
                    "index": index,
                    "path": f"{self._block_path}.{index}",
                    "class": f"{type(block).__module__}.{type(block).__qualname__}",
                    "num_heads": getattr(
                        inner_attention,
                        "num_heads",
                        getattr(inner_attention, "num_attention_heads", None),
                    ),
                    "qkv_paths": [
                        f"{self._block_path}.{index}.{path}"
                        for path, _ in block.named_modules()
                        if path.rsplit(".", 1)[-1] in {"qkv", "query", "key", "value"}
                    ],
                    "qkv_note": "Native VideoMAEv2 uses functional linear; qkv module hook does not execute."
                    if self._encoder_id == "videomaev2"
                    else "Module paths only; position processing must be verified separately.",
                }
            )
        position = getattr(self._block_owner, "pos_embed", None)
        result["position_shape"] = list(position.shape) if position is not None else None
        result["position_trainable"] = (
            bool(getattr(position, "requires_grad", False)) if position is not None else None
        )
        result["native_readout"] = (
            "mean_then_fc_norm"
            if getattr(self._block_owner, "fc_norm", None) is not None
            else "unverified"
        )
        result["gradient_validation"] = "not_performed"
        return result

    def observation_sites(self, depths: Iterable[int] | None = None) -> Mapping[str, nn.Module]:
        """Return readable hook sites without changing model execution.

        The values are live modules from this bridge's *specific* loaded model.
        Callers own all hook handles and must remove them in ``finally`` or a
        context manager.  Names ending in ``.output`` mean forward-hook output,
        never an unobservable block input.
        """

        selected = range(len(self._blocks)) if depths is None else tuple(depths)
        sites: dict[str, nn.Module] = {}
        patch = getattr(self._block_owner, "patch_embed", None)
        if isinstance(patch, nn.Module):
            sites["embedding.output"] = patch
        for depth in selected:
            if type(depth) is not int or not 0 <= depth < len(self._blocks):
                raise BridgeUnsupportedError(
                    f"{self._encoder_id} 不存在 block depth={depth!r}；范围为 0..{len(self._blocks) - 1}"
                )
            block = self._blocks[depth]
            prefix = f"block.{depth}"
            sites[f"{prefix}.input"] = block
            sites[f"{prefix}.output"] = block
            for name in ("norm1", "layernorm_before", "layernorm1"):
                module = _first_module(block, ((name,),))
                if module is not None:
                    sites[f"{prefix}.norm1"] = module
                    break
            attention = _first_module(block, (("attn",), ("attention",), ("self_attn",)))
            if attention is not None:
                attention_prefix = (
                    f"{prefix}.spatial.attn"
                    if self._encoder_id == "timesformer"
                    else f"{prefix}.attn"
                )
                sites[f"{attention_prefix}.output"] = attention
                probabilities = getattr(attention, "attn_drop", None)
                if probabilities is None:
                    probabilities = _at_path(attention, ("attention", "attn_drop"))
                if isinstance(probabilities, nn.Module):
                    sites[f"{attention_prefix}.probs.input"] = probabilities
            if self._encoder_id == "timesformer":
                temporal = getattr(block, "temporal_attention", None)
                if isinstance(temporal, nn.Module):
                    sites[f"{prefix}.temporal.attn.output"] = temporal
                    temporal_probs = _at_path(temporal, ("attention", "attn_drop"))
                    if isinstance(temporal_probs, nn.Module):
                        sites[f"{prefix}.temporal.attn.probs.input"] = temporal_probs
            qkv = _first_module(
                block,
                (
                    ("attn", "qkv"),
                    ("attention", "qkv"),
                    ("attention", "self", "qkv"),
                    ("attention", "attention", "qkv"),
                ),
            )
            if qkv is not None and self._encoder_id != "videomaev2":
                sites[f"{prefix}.attn.qkv"] = qkv
            for name in ("norm2", "layernorm_after", "layernorm2"):
                module = _first_module(block, ((name,),))
                if module is not None:
                    sites[f"{prefix}.mid.input"] = module
                    sites[f"{prefix}.norm2"] = module
                    break
            mlp = _first_module(block, (("mlp",), ("output", "dropout")))
            if mlp is not None:
                sites[f"{prefix}.mlp.output"] = mlp
        return MappingProxyType(sites)


def create_observation_bridge(encoder_id: str, loaded_model: Any) -> EncoderBridge:
    """Locate a bridge on an already-loaded active model or its known adapter."""

    return EncoderBridge(encoder_id, loaded_model)


__all__ = [
    "BridgeUnsupportedError",
    "EncoderBridge",
    "EncoderBridgeReceipt",
    "create_observation_bridge",
]
