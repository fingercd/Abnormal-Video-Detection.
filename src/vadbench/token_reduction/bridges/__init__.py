"""Instance-scoped observation bridges for the four active encoders.

These bridges locate modules on an already loaded model.  They never replace a
global ``forward`` method.  Explicit indexed interventions are separate from
observation and do not supply a selector or establish method quality.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch.nn as nn

from .indexed import (
    IndexedInterventionError,
    IndexedTokenIntervention,
    identity_indices,
    indexed_gather,
)


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
        if encoder_id == "videomaev2":
            native_patch = _first_module(
                self._root,
                (
                    ("backbone", "patch_embed"),
                    ("backbone", "videomae", "patch_embed"),
                    ("backbone", "model", "patch_embed"),
                    ("patch_embed",),
                ),
            )
            if native_patch is not None:
                # This native backbone embeds patches only; runtime sequence
                # length is checked against its actual Conv3d output below.
                has_cls, special_indices = False, ()
        if encoder_id in {"videomae", "vjepa2"}:
            # Both supported native base encoders are patch-only.  This is a
            # concrete model-family contract, not an inference from token 0.
            has_cls, special_indices = False, ()
        layout = (
            getattr(getattr(self._root, "config", None), "attention_type", "unverified")
            if encoder_id == "timesformer"
            else "unverified"
        )
        position_contract = {
            "timesformer": "absolute-spatial-and-temporal-position; no-ragged-reduction-path-established",
            "videomae": "fixed-sincos-position-before-encoder; no-ragged-reduction-path-established",
            "vjepa2": "three-axis-RoPE-on-QK-before-attention-backend; no-ragged-reduction-path-established",
        }.get(encoder_id, "position-handling-unverified")
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

    def block_output_tensor(self, depth: int, output: Any) -> Any:
        """Return the tensor slot that an identity hook may replace.

        HF TimeSformer and V-JEPA2 blocks return ``(hidden_states,
        attention)``.  Their attention side output remains part of native
        control flow and must survive an identity observation hook unchanged.
        """
        self._validate_depth(depth)
        tensor = output[0] if isinstance(output, tuple) and output else output
        if not hasattr(tensor, "ndim") or tensor.ndim != 3:
            raise BridgeUnsupportedError(
                f"{self._encoder_id} block.{depth} 输出不含可验证的 [B,N,D] hidden-state tensor"
            )
        return tensor

    def replace_block_output_tensor(
        self, depth: int, output: Any, tokens: Any, *, allow_sequence_shrink: bool = False
    ) -> Any:
        """Replace only block hidden states and preserve native tuple fields.

        Identity parity keeps the exact shape.  The separate explicit-index
        experiment may shorten only the sequence axis; it cannot change batch
        size or channel width.
        """
        previous = self.block_output_tensor(depth, output)
        shape = tuple(getattr(tokens, "shape", ()))
        previous_shape = tuple(getattr(previous, "shape", ()))
        valid_shrink = (
            allow_sequence_shrink
            and len(shape) == 3
            and shape[0] == previous_shape[0]
            and shape[2] == previous_shape[2]
            and 0 < shape[1] <= previous_shape[1]
        )
        if shape != previous_shape and not valid_shrink:
            raise BridgeUnsupportedError(
                "identity hook replacement shape must equal the native block hidden state"
            )
        if isinstance(output, tuple):
            return (tokens, *output[1:])
        return tokens

    def _validate_depth(self, depth: int) -> None:
        if type(depth) is not int or not 0 <= depth < len(self._blocks):
            raise BridgeUnsupportedError(
                f"{self._encoder_id} 不存在 block depth={depth!r}；范围为 0..{len(self._blocks) - 1}"
            )

    def probe_token_metadata(
        self, geometry: Any, sites: Mapping[str, nn.Module] | None = None
    ) -> Any:
        """Build collector metadata from a completed native geometry observer.

        This narrow observation API is intentionally separate from deployment
        ``TokenLayout``.  Calling it before the observed forward, or passing a
        bridge-owned geometry from another model, fails rather than guessing.
        """
        from vadbench.research.collectors import ProbeSiteMetadata, ProbeTokenMetadata

        layout = getattr(geometry, "layout", None)
        receipt = getattr(geometry, "receipt", None)
        if layout is None or not isinstance(receipt, dict):
            raise BridgeUnsupportedError(
                "native geometry must complete before probe metadata is requested"
            )
        source = layout.provenance.get("coordinate_source")
        if not isinstance(source, str):
            raise BridgeUnsupportedError("native geometry lacks coordinate provenance")
        site_metadata: dict[str, Any] = {}
        if self._encoder_id == "timesformer":
            grid = receipt.get("grid")
            if (
                not isinstance(grid, list)
                or len(grid) != 3
                or not all(type(value) is int and value > 0 for value in grid)
            ):
                raise BridgeUnsupportedError("TimeSformer geometry lacks verified [T,H,W] grid")
            frames, grid_h, grid_w = grid
            for site in sites or self.observation_sites():
                if ".temporal.attn.projection." in site:
                    site_metadata[site] = ProbeSiteMetadata(
                        "timesformer_patch_global", frames, grid_h * grid_w
                    )
                elif ".temporal." in site:
                    site_metadata[site] = ProbeSiteMetadata(
                        "timesformer_temporal", frames, grid_h * grid_w
                    )
                elif ".spatial." in site:
                    site_metadata[site] = ProbeSiteMetadata(
                        "timesformer_spatial", frames, grid_h * grid_w
                    )
        return ProbeTokenMetadata(
            valid_mask=layout.valid_mask,
            coordinates=layout.source_coordinates,
            coordinate_source=source,
            special_token_indices=self._receipt.special_token_indices,
            has_cls=self._receipt.has_cls,
            site_metadata=site_metadata,
        )

    def geometry(self, frame_indices: Any, valid_mask: Any) -> Any:
        """Return a native-layout observer for this concrete loaded model.

        The result only observes actual patch/embedding execution.  It does not
        infer a grid from a token count, and it deliberately has no reduction
        execution semantics.
        """
        from .geometry import TimeSformerGeometry, TubeletGeometry

        if self._encoder_id == "timesformer":
            embeddings = _at_path(self._root, ("embeddings",))
            first = self._blocks[0]
            temporal_norm = getattr(first, "temporal_layernorm", None)
            if not isinstance(embeddings, nn.Module) or not isinstance(temporal_norm, nn.Module):
                raise BridgeUnsupportedError(
                    "TimeSformer geometry requires actual divided_space_time embeddings and temporal_layernorm"
                )
            if self._receipt.layout != "divided_space_time":
                raise BridgeUnsupportedError(
                    "TimeSformer geometry currently verifies divided_space_time only"
                )
            return TimeSformerGeometry(embeddings, temporal_norm, frame_indices, valid_mask)
        patch = _first_module(
            self._root,
            (
                ("backbone", "patch_embed"),
                ("backbone", "videomae", "patch_embed"),
                ("backbone", "model", "patch_embed"),
                ("embeddings", "patch_embeddings"),
                ("encoder", "embeddings", "patch_embeddings"),
                ("embeddings", "patch_embed"),
                ("patch_embed",),
            ),
        )
        if not isinstance(patch, nn.Module):
            raise BridgeUnsupportedError(
                f"{self._encoder_id} 未找到具有实际 Conv3d 的 native patch embedding"
            )
        if self._receipt.has_cls:
            raise BridgeUnsupportedError(
                f"{self._encoder_id} 的当前 geometry 预期 patch-only sequence"
            )
        return TubeletGeometry(patch, frame_indices, valid_mask)

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
        if position is None:
            position = _at_path(self._root, ("embeddings", "position_embeddings"))
        result["position_shape"] = list(position.shape) if position is not None else None
        result["position_trainable"] = (
            bool(getattr(position, "requires_grad", False)) if position is not None else None
        )
        if self._encoder_id == "videomae":
            result["native_readout"] = (
                "HF base model returns token sequence; adapter pooling is recorded separately"
            )
        elif self._encoder_id == "timesformer":
            result["native_readout"] = (
                "HF base model returns CLS-bearing sequence; adapter pooling is recorded separately"
            )
        elif self._encoder_id == "vjepa2":
            result["native_readout"] = (
                "official get_vision_features output; adapter-specific readout must be recorded"
            )
        else:
            result["native_readout"] = (
                "mean_then_fc_norm"
                if getattr(self._block_owner, "fc_norm", None) is not None
                else "unverified"
            )
        if self._encoder_id == "vjepa2":
            result["position_application"] = (
                "three-axis RoPE is applied to projected Q and K inside native attention before the attention backend"
            )
        elif self._encoder_id == "videomae":
            result["position_application"] = "fixed sin-cos embedding is added before the encoder"
        elif self._encoder_id == "timesformer":
            result["position_application"] = (
                "absolute spatial then temporal embeddings are added before divided attention"
            )
        result["gradient_validation"] = "not_performed"
        result["attention_backend"] = getattr(
            getattr(self._root, "config", None), "_attn_implementation", None
        )
        result["observation_site_semantics"] = (
            {
                "block.input": "native block input before its first normalization",
                "temporal.attn.pre_norm.input": "TimeSformer temporal branch input before temporal LayerNorm",
                "temporal.norm1": "TimeSformer temporal LayerNorm output",
                "temporal.attn.pre_projection.output": "TimeSformer temporal attention projection before DropPath/temporal_dense; activation only",
                "temporal.attn.projection.output": "TimeSformer temporal_dense output after temporal DropPath, before residual add",
                "spatial.attn.pre_norm.input": "TimeSformer per-frame spatial branch input before LayerNorm",
                "spatial.attn.output": "TimeSformer spatial attention projection before DropPath/residual",
                "mlp.pre_norm.input": "input to post-attention MLP LayerNorm",
                "mlp.pre_residual.output": "MLP branch output before residual; V-JEPA2 is before DropPath",
                "block.output": "full native block output after residual(s)",
            }
            if self._encoder_id == "timesformer"
            else {
                "block.input": "native block input before first normalization",
                "norm1": "first LayerNorm output before attention",
                "attn.output": "attention projection before residual; V-JEPA2 is before DropPath",
                "mlp.pre_norm.input": "input to post-attention MLP LayerNorm",
                "mlp.pre_residual.output": "MLP branch output before residual; V-JEPA2 is before DropPath",
                "block.output": "full native block output after residual(s)",
            }
        )
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
        patch = _first_module(
            self._root,
            (
                ("backbone", "patch_embed"),
                ("backbone", "videomae", "patch_embed"),
                ("backbone", "model", "patch_embed"),
                ("embeddings", "patch_embeddings"),
                ("encoder", "embeddings", "patch_embeddings"),
                ("embeddings", "patch_embed"),
                ("patch_embed",),
            ),
        )
        if self._encoder_id == "timesformer":
            embeddings = _at_path(self._root, ("embeddings",))
            if not isinstance(embeddings, nn.Module):
                raise BridgeUnsupportedError("TimeSformer 未找到完整 embeddings module")
            # PatchEmbeddings returns [B*T,P,D], which is an internal
            # execution batch and cannot be rendered against B clip IDs.
            # The complete embedding module has actual CLS, absolute spatial
            # and temporal positions, and verified [B,1+P*T,D] layout.
            sites["embedding.post_position.output"] = embeddings
        elif isinstance(patch, nn.Module):
            sites["embedding.output"] = patch
        for depth in selected:
            self._validate_depth(depth)
            block = self._blocks[depth]
            prefix = f"block.{depth}"
            sites[f"{prefix}.input"] = block
            sites[f"{prefix}.output"] = block
            if self._encoder_id == "timesformer":
                temporal_norm = getattr(block, "temporal_layernorm", None)
                if isinstance(temporal_norm, nn.Module):
                    sites[f"{prefix}.temporal.attn.pre_norm.input"] = temporal_norm
                    sites[f"{prefix}.temporal.norm1"] = temporal_norm
                spatial_norm = getattr(block, "layernorm_before", None)
                if isinstance(spatial_norm, nn.Module):
                    sites[f"{prefix}.spatial.attn.pre_norm.input"] = spatial_norm
                    sites[f"{prefix}.spatial.norm1"] = spatial_norm
            else:
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
                    # This is before ``temporal_dense`` and the residual.  It
                    # is useful activation evidence, but never a P16 branch
                    # update proxy.
                    sites[f"{prefix}.temporal.attn.pre_projection.output"] = temporal
                    temporal_probs = _at_path(temporal, ("attention", "attn_drop"))
                    if isinstance(temporal_probs, nn.Module):
                        sites[f"{prefix}.temporal.attn.probs.input"] = temporal_probs
                    temporal_qkv = _first_module(temporal, (("attention", "qkv"), ("qkv",)))
                    if temporal_qkv is not None:
                        sites[f"{prefix}.temporal.attn.qkv"] = temporal_qkv
                temporal_projection = getattr(block, "temporal_dense", None)
                if isinstance(temporal_projection, nn.Module):
                    # The native input has already passed DropPath; this is
                    # the actual temporal branch projection, still before its
                    # add to patch tokens.
                    sites[f"{prefix}.temporal.attn.projection.output"] = temporal_projection
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
                qkv_prefix = (
                    f"{prefix}.spatial.attn"
                    if self._encoder_id == "timesformer"
                    else f"{prefix}.attn"
                )
                sites[f"{qkv_prefix}.qkv"] = qkv
            if self._encoder_id in {"videomae", "vjepa2"}:
                inner = _first_module(
                    block,
                    (("attention",),)
                    if self._encoder_id == "vjepa2"
                    else (("attention", "attention"), ("attn", "attention")),
                )
                if inner is not None:
                    sites[f"{prefix}.attn.probs.output"] = inner
                    for projection in ("query", "key", "value"):
                        module = getattr(inner, projection, None)
                        if isinstance(module, nn.Module):
                            sites[f"{prefix}.attn.{projection}"] = module
            for name in ("norm2", "layernorm_after", "layernorm2"):
                module = _first_module(block, ((name,),))
                if module is not None:
                    sites[f"{prefix}.mlp.pre_norm.input"] = module
                    sites[f"{prefix}.norm2"] = module
                    break
            mlp = _first_module(block, (("mlp",), ("output", "dropout")))
            if mlp is not None:
                # For HF VideoMAE/TimeSformer this is the output dropout
                # before the residual inside the surrounding output/block;
                # for V-JEPA2 it is MLP output before DropPath.  The name
                # makes that boundary explicit instead of claiming a block
                # residual output is a branch update.
                sites[f"{prefix}.mlp.pre_residual.output"] = mlp
        return MappingProxyType(sites)


def create_observation_bridge(encoder_id: str, loaded_model: Any) -> EncoderBridge:
    """Locate a bridge on an already-loaded active model or its known adapter."""

    return EncoderBridge(encoder_id, loaded_model)


__all__ = [
    "BridgeUnsupportedError",
    "EncoderBridge",
    "EncoderBridgeReceipt",
    "IndexedInterventionError",
    "IndexedTokenIntervention",
    "create_observation_bridge",
    "identity_indices",
    "indexed_gather",
]
