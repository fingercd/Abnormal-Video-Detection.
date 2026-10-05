"""Concrete bridge for the OpenAI CLIP ViT-B/16 visual tower.

The repository contains several anomaly-detection projects that import the
same OpenAI CLIP visual tower.  This module deliberately bridges the visual
model, rather than any of those detector heads.  It is usable with an already
loaded ``openai/CLIP`` model and keeps all optional CLIP imports out of the
normal VADBench import path.

The official OpenAI implementation represents the vision transformer sequence
as ``[N, B, D]`` inside ``visual.transformer``.  Its class token and absolute
position embedding are created once before ``resblocks``.  Consequently a
pre-hook on a verified block can replace the sequence with a shorter sequence;
the following blocks and the native ``CLS -> ln_post -> proj`` readout then
run on the shorter sequence.  The bridge records this fact explicitly and
refuses to guess a layout for a different model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence


class ClipBridgeError(RuntimeError):
    """Raised when a loaded CLIP visual tower does not satisfy this bridge."""


@dataclass(frozen=True)
class ClipVisionReceipt:
    """Resolved facts from one concrete CLIP visual tower."""

    encoder_id: str
    model_type: str
    visual_path: str
    input_resolution: int
    patch_size: int
    grid_h: int
    grid_w: int
    hidden_dim: int
    block_count: int
    attention_heads: int
    projection_dim: int
    sequence_length: int
    has_cls: bool
    cls_index: int
    position_shape: tuple[int, ...]
    position_contract: str
    native_readout: str
    native_preprocess_contract: str
    attention_output_contract: str
    supports_intermediate_hook: bool
    supports_suffix_sequence_shrink: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClipHookResult:
    """Output of one CLIP forward with an optional block-input hook."""

    pooled: Any
    block_input: Any | None
    block_output: Any | None
    attention_output: Any | None
    input_sequence_length: int | None
    output_sequence_length: int | None
    depth: int | None
    replaced: bool
    selection_indices: Any | None = None


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise ClipBridgeError("CLIP bridge 需要具有 shape 的参数或张量")
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ClipBridgeError(f"无法读取 CLIP shape={raw!r}") from exc


def _module_path(root: Any, path: Sequence[str]) -> Any | None:
    value = root
    for name in path:
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def load_openai_clip(
    repo: str | Path,
    weights: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, Callable[[Any], Any], dict[str, str]]:
    """Load the pinned OpenAI CLIP implementation without installing it.

    ``repo`` must contain the official ``clip`` package and ``weights`` must
    be an already verified local checkpoint.  The returned identity is used in
    probe receipts; no network download is attempted.
    """

    import sys

    repo_path = Path(repo).expanduser().resolve()
    weight_path = Path(weights).expanduser().resolve()
    if not repo_path.is_dir():
        raise ClipBridgeError(f"CLIP repo 不存在：{repo_path}")
    if not weight_path.is_file():
        raise ClipBridgeError(f"CLIP checkpoint 不存在：{weight_path}")
    source_path = repo_path / "clip"
    if not source_path.is_dir():
        raise ClipBridgeError(f"CLIP repo 缺少 clip package：{source_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        clip_module = import_module("clip")
        model, preprocess = clip_module.load(str(weight_path), device=device, jit=False)
    except Exception as exc:  # pragma: no cover - depends on optional remote runtime
        raise ClipBridgeError(f"加载 OpenAI CLIP ViT-B/16 失败：{exc}") from exc
    return model, preprocess, {"repo": str(repo_path), "weights": str(weight_path)}


class ClipVisionBridge:
    """Bridge a verified OpenAI CLIP visual tower and its native readout."""

    def __init__(self, loaded_model: Any, *, encoder_id: str = "openai_clip_vit_b16") -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ClipBridgeError("CLIP bridge 需要 torch") from exc

        model = loaded_model
        visual = getattr(model, "visual", None)
        if visual is None:
            # Accept a bare visual tower for unit tests and controlled callers,
            # while still requiring every native module below.
            visual = model
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        conv1 = getattr(visual, "conv1", None)
        position = getattr(visual, "positional_embedding", None)
        class_embedding = getattr(visual, "class_embedding", None)
        ln_pre = getattr(visual, "ln_pre", None)
        ln_post = getattr(visual, "ln_post", None)
        projection = getattr(visual, "proj", None)
        if not isinstance(visual, nn.Module) or not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise ClipBridgeError("未找到官方 CLIP visual.transformer.resblocks ModuleList/Sequential")
        if len(blocks) <= 0:
            raise ClipBridgeError("CLIP visual transformer 没有 resblocks")
        if not isinstance(conv1, nn.Conv2d):
            raise ClipBridgeError("CLIP visual.conv1 不是二维 patch Conv2d")
        if position is None or class_embedding is None or not isinstance(ln_pre, nn.Module):
            raise ClipBridgeError("CLIP visual 缺少 class/position/ln_pre，无法确认真实 token 语义")
        if not isinstance(ln_post, nn.Module) or projection is None:
            raise ClipBridgeError("CLIP visual 缺少 native ln_post/proj readout")

        input_resolution = int(getattr(visual, "input_resolution", 0))
        patch_kernel = tuple(int(item) for item in conv1.kernel_size)
        patch_stride = tuple(int(item) for item in conv1.stride)
        if input_resolution <= 0 or patch_kernel[0] != patch_kernel[1] or patch_stride != patch_kernel:
            raise ClipBridgeError("CLIP visual 的输入分辨率/patch stride 不满足固定 ViT patch 契约")
        patch_size = patch_kernel[0]
        if input_resolution % patch_size:
            raise ClipBridgeError("CLIP input_resolution 不能整除 patch_size")
        grid_h = grid_w = input_resolution // patch_size
        position_shape = _shape(position)
        if position_shape != (grid_h * grid_w + 1, int(conv1.out_channels)):
            raise ClipBridgeError(
                "CLIP positional_embedding shape 与 conv1 的真实 grid/hidden dim 不一致"
            )
        cls_shape = _shape(class_embedding)
        hidden_dim = int(conv1.out_channels)
        if cls_shape != (hidden_dim,):
            raise ClipBridgeError("CLIP class_embedding 不是一个 hidden_dim 向量")
        projection_shape = _shape(projection)
        if len(projection_shape) != 2 or projection_shape[0] != hidden_dim:
            raise ClipBridgeError("CLIP visual.proj 不是 [hidden_dim, projection_dim]")
        first_attn = getattr(blocks[0], "attn", None)
        attention_heads = int(getattr(first_attn, "num_heads", 0))
        if attention_heads <= 0:
            raise ClipBridgeError("CLIP 首个 ResidualAttentionBlock 缺少可验证 num_heads")
        # Official visual attention has no fixed-length attention mask.  A
        # non-null mask would make suffix shortening unsafe for this bridge.
        for block_index, block in enumerate(blocks):
            attention = getattr(block, "attn", None)
            if attention is None or getattr(block, "ln_1", None) is None:
                raise ClipBridgeError(f"CLIP resblock[{block_index}] 不是官方 ResidualAttentionBlock")
            if getattr(block, "attn_mask", None) is not None:
                raise ClipBridgeError("当前 CLIP visual block 带固定 attn_mask，不能宣称可缩短序列")
            external_scales = tuple(
                name
                for name in ("gamma_1", "gamma_2", "ls1", "ls2", "layer_scale_1", "layer_scale_2")
                if getattr(block, name, None) is not None
            )
            if external_scales:
                raise ClipBridgeError(
                    f"CLIP resblock[{block_index}] 有未建模的外置 gamma/layerscale={external_scales}"
                )

        self.model = model
        self.visual = visual
        self.blocks = blocks
        self._torch = torch
        self._pair_tables: dict[str, Any] = {}
        self._receipt = ClipVisionReceipt(
            encoder_id=encoder_id,
            model_type=f"{type(visual).__module__}.{type(visual).__qualname__}",
            visual_path="visual" if visual is not model else "<bare-visual>",
            input_resolution=input_resolution,
            patch_size=patch_size,
            grid_h=grid_h,
            grid_w=grid_w,
            hidden_dim=hidden_dim,
            block_count=len(blocks),
            attention_heads=attention_heads,
            projection_dim=projection_shape[1],
            sequence_length=grid_h * grid_w + 1,
            has_cls=True,
            cls_index=0,
            position_shape=position_shape,
            position_contract=(
                "absolute positional_embedding is added once after patch projection and CLS concat, "
                "then ln_pre; no relative/RoPE path in the verified visual tower"
            ),
            native_readout="ln_post(sequence[:, 0, :]) @ visual.proj",
            native_preprocess_contract=(
                "official clip.load preprocess: Resize(short=input_resolution, bicubic) -> "
                "CenterCrop(input_resolution) -> RGB tensor -> CLIP mean/std normalization"
            ),
            attention_output_contract=(
                "block.attn module hook returns MultiheadAttention output[0] as [N,B,D]; "
                "official ResidualAttentionBlock applies it directly in x + attention output, "
                "with no external gamma/layerscale"
            ),
            supports_intermediate_hook=True,
            supports_suffix_sequence_shrink=True,
            notes=(
                "block hooks observe [N,B,D] inside the official transformer; bridge exposes [N,B,D] unchanged",
                "a shortened sequence keeps the real CLS at index 0 and runs all suffix blocks at N_out",
            ),
        )

    def receipt(self) -> ClipVisionReceipt:
        return self._receipt

    def architecture(self) -> dict[str, Any]:
        """Return JSON-safe architecture and position metadata."""

        result = asdict(self.receipt())
        result["position_shape"] = list(self.receipt().position_shape)
        result["patch_coordinates"] = self.token_coordinates().tolist()
        result["block_paths"] = [f"visual.transformer.resblocks.{i}" for i in range(self.receipt().block_count)]
        return result

    def token_coordinates(self) -> Any:
        """Return ``[CLS, patch row-major]`` coordinates with ``(-1,-1)`` for CLS."""

        import torch

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.receipt().grid_h),
                torch.arange(self.receipt().grid_w),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        cls = torch.tensor([[-1, -1]], dtype=torch.long)
        return torch.cat((cls, grid.to(dtype=torch.long)), dim=0)

    def _validate_depth(self, depth: int) -> None:
        if type(depth) is not int or not 0 <= depth < len(self.blocks):
            raise ClipBridgeError(f"CLIP block depth 必须在 [0,{len(self.blocks) - 1}]，得到 {depth!r}")

    def _validate_indices(self, indices: Any, *, device: Any) -> Any:
        torch = self._torch
        value = torch.as_tensor(indices, dtype=torch.long, device=device)
        if value.ndim != 1 or value.numel() <= 0:
            raise ClipBridgeError("CLIP token indices 必须是一维非空序列")
        if int(value[0].item()) != 0:
            raise ClipBridgeError("CLIP token intervention 必须保留真实 CLS index 0")
        if int(value.min().item()) < 0 or int(value.max().item()) >= self.receipt().sequence_length:
            raise ClipBridgeError("CLIP token indices 越过 native sequence")
        if value.unique().numel() != value.numel():
            raise ClipBridgeError("CLIP token indices 不能重复")
        native_length = self.receipt().sequence_length
        if value.numel() == native_length:
            native_indices = torch.arange(native_length, dtype=torch.long, device=device)
            if not torch.equal(value, native_indices):
                raise ClipBridgeError(
                    "完整 CLIP identity gather 必须保持 native sequence 的原顺序"
                )
            return value
        if value.numel() > native_length:
            raise ClipBridgeError("CLIP token indices 不能超过 native sequence 长度")
        return value

    def _pair_table(self, device: Any) -> Any:
        """Cache the native horizontal-pair table for the verified patch grid."""

        key = str(device)
        table = self._pair_tables.get(key)
        if table is None:
            height, width = self.receipt().grid_h, self.receipt().grid_w
            if width % 2:
                raise ClipBridgeError("CLIP PairSelect 需要偶数 patch 宽度")
            grid = self._torch.arange(1, height * width + 1, device=device).reshape(height, width)
            table = grid.reshape(height, width // 2, 2).reshape(-1, 2)
            self._pair_tables[key] = table
        return table

    def _pair_select(self, tokens: Any, keep_ratio: float) -> tuple[Any, Any]:
        """Apply the frozen five-pair quota and member-priority rule per image.

        ``tokens`` is the input of visual block 6, hence the output of the
        first six dense blocks. All ranking and gathering remain on its GPU.
        """

        if keep_ratio not in (0.80, 0.60, 0.40):
            raise ClipBridgeError("CLIP PairSelect keep_ratio 必须为 0.80/0.60/0.40")
        torch = self._torch
        table = self._pair_table(tokens.device)
        batch, width = int(tokens.shape[1]), int(tokens.shape[2])
        hidden = tokens.permute(1, 0, 2)
        members = hidden[:, table.reshape(-1)].reshape(batch, table.shape[0], 2, width)
        member_norms = members.float().norm(dim=-1)
        pair_scores = member_norms.amax(dim=-1)
        better_is_right = member_norms[:, :, 1] > member_norms[:, :, 0]
        first = table[:, 0].unsqueeze(0).expand(batch, -1)
        second = table[:, 1].unsqueeze(0).expand(batch, -1)
        better = torch.where(better_is_right, second, first)
        worse = torch.where(better_is_right, first, second)

        full_groups, remainder = divmod(int(table.shape[0]), 5)
        quota = int(10 * keep_ratio + 0.5)
        selected = []
        if full_groups:
            scores = pair_scores[:, : full_groups * 5].reshape(batch, full_groups, 5)
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            good = better[:, : full_groups * 5].reshape(batch, full_groups, 5).gather(2, order)
            other = worse[:, : full_groups * 5].reshape(batch, full_groups, 5).gather(2, order)
            priority = torch.cat((good, other), dim=-1)
            selected.append(priority[:, :, :quota].reshape(batch, -1))
        if remainder:
            start = full_groups * 5
            order = torch.argsort(pair_scores[:, start:], dim=1, descending=True, stable=True)
            good = better[:, start:].gather(1, order)
            other = worse[:, start:].gather(1, order)
            tail_quota = int(2 * remainder * keep_ratio + 0.5)
            selected.append(torch.cat((good, other), dim=1)[:, :tail_quota])
        patch_indices = torch.cat(selected, dim=1).sort(dim=1).values
        indices = torch.cat(
            (torch.zeros((batch, 1), dtype=torch.long, device=tokens.device), patch_indices), dim=1
        )
        reduced = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, width))
        return reduced.permute(1, 0, 2), indices

    def forward(
        self,
        images: Any,
        *,
        capture_depth: int | None = None,
        keep_indices: Any | None = None,
        keep_ratio: float | None = None,
        capture_attention: bool = False,
    ) -> ClipHookResult:
        """Run native ``encode_image`` with an optional real block-input edit.

        ``keep_indices`` is applied at the input of ``capture_depth``.  The
        selected block and all later transformer blocks consequently receive a
        shorter sequence.  No labels or detector output are visible to this
        method.
        """

        if capture_depth is None and (keep_indices is not None or keep_ratio is not None):
            raise ClipBridgeError("CLIP token intervention 需要同时指定 capture_depth")
        if keep_indices is not None and keep_ratio is not None:
            raise ClipBridgeError("keep_indices 和 keep_ratio 不能同时使用")
        if capture_depth is None and capture_attention:
            raise ClipBridgeError("capture_attention 需要同时指定 capture_depth")
        if capture_depth is not None:
            self._validate_depth(capture_depth)
        captures: dict[str, Any] = {}
        handles: list[Any] = []

        if capture_depth is not None:
            block = self.blocks[capture_depth]

            def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
                if not args:
                    raise ClipBridgeError("CLIP block hook 没有输入 tuple")
                tokens = args[0]
                if not hasattr(tokens, "shape") or len(tokens.shape) != 3:
                    raise ClipBridgeError("CLIP block input 必须是 [N,B,D]")
                if int(tokens.shape[0]) != self.receipt().sequence_length:
                    raise ClipBridgeError(
                        f"CLIP block input sequence={int(tokens.shape[0])} 与 native "
                        f"{self.receipt().sequence_length} 不一致；未静默猜测布局"
                    )
                if keep_indices is None and keep_ratio is None:
                    captures["input"] = tokens
                    return args
                if keep_ratio is None:
                    indices = self._validate_indices(keep_indices, device=tokens.device)
                    reduced = tokens.index_select(0, indices)
                else:
                    reduced, indices = self._pair_select(tokens, keep_ratio)
                captures["replacement_indices"] = indices
                captures["input"] = reduced
                return (reduced, *args[1:])

            def output_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
                    raise ClipBridgeError("CLIP block output 必须是 [N,B,D]")
                captures["output"] = tensor
                return output

            def attention_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
                    raise ClipBridgeError("CLIP attention output 必须是 [N,B,D]")
                captures["attention_output"] = tensor
                return output

            handles.append(block.register_forward_pre_hook(pre_hook))
            handles.append(block.register_forward_hook(output_hook))
            if capture_attention:
                handles.append(block.attn.register_forward_hook(attention_hook))
        try:
            encode_image = getattr(self.model, "encode_image", None)
            if not callable(encode_image):
                pooled = self.visual(images)
            else:
                pooled = encode_image(images)
        finally:
            for hook_handle in handles:
                hook_handle.remove()
        block_input = captures.get("input")
        block_output = captures.get("output")
        return ClipHookResult(
            pooled=pooled,
            block_input=block_input,
            block_output=block_output,
            attention_output=captures.get("attention_output"),
            input_sequence_length=None if block_input is None else int(block_input.shape[0]),
            output_sequence_length=None if block_output is None else int(block_output.shape[0]),
            depth=capture_depth,
            replaced=keep_indices is not None or keep_ratio is not None,
            selection_indices=captures.get("replacement_indices"),
        )


def fixed_local_keep_indices(
    grid_h: int = 14,
    grid_w: int = 14,
    *,
    group_h: int = 2,
    group_w: int = 2,
    keep_per_group: int = 2,
) -> Any:
    """Build deterministic ``CLS + 2-of-4`` local patch indices.

    This is a neutral intervention control.  It does not inspect labels,
    detector logits, future layers, or pre-extracted features.
    """

    import torch

    if any(type(item) is not int or item <= 0 for item in (grid_h, grid_w, group_h, group_w)):
        raise ClipBridgeError("CLIP patch grid/group 必须是正整数")
    if grid_h % group_h or grid_w % group_w:
        raise ClipBridgeError("CLIP patch grid 必须能整除 local group")
    group_size = group_h * group_w
    if type(keep_per_group) is not int or not 0 < keep_per_group < group_size:
        raise ClipBridgeError("keep_per_group 必须在 1..group_size-1")
    kept = [0]
    for group_row in range(0, grid_h, group_h):
        for group_col in range(0, grid_w, group_w):
            local = []
            for row in range(group_h):
                for col in range(group_w):
                    local.append(1 + (group_row + row) * grid_w + group_col + col)
            kept.extend(local[:keep_per_group])
    return torch.tensor(kept, dtype=torch.long)


__all__ = [
    "ClipBridgeError",
    "ClipHookResult",
    "ClipVisionBridge",
    "ClipVisionReceipt",
    "fixed_local_keep_indices",
    "load_openai_clip",
]
