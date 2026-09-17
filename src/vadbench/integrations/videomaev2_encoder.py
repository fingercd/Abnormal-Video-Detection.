"""VideoMAE v2 encoder shared by VADBench and the retained legacy prototype."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class VideoMAEv2EncoderConfig:
    """Configuration for the OpenGVLab VideoMAE v2 Hugging Face model."""

    model_name: str = "OpenGVLab/VideoMAEv2-Base"
    image_size: int = 224
    num_frames: int = 16
    use_half: bool = True
    pooling: str = "auto"


class VideoMAEv2Encoder(nn.Module):
    """Load and run a stateless fixed-clip VideoMAE v2 backbone."""

    def __init__(self, cfg: VideoMAEv2EncoderConfig, device: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(cfg.use_half and self.device_str.startswith("cuda"))
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("缺少 transformers。请安装 `vadbench[videomaev2]`") from exc

        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        self._patch_transformers5_tied_weights(config, cfg.model_name)
        try:
            from transformers import VideoMAEImageProcessor

            self.processor = VideoMAEImageProcessor.from_pretrained(
                cfg.model_name, trust_remote_code=True
            )
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(
                cfg.model_name, trust_remote_code=True
            )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        self.backbone.to(self.device_str)
        meta_params = [
            name for name, parameter in self.backbone.named_parameters() if parameter.is_meta
        ]
        meta_buffers = [name for name, buffer in self.backbone.named_buffers() if buffer.is_meta]
        if meta_params or meta_buffers:
            raise RuntimeError(
                "VideoMAE v2 backbone 加载后仍存在 meta tensor，无法推理。\n"
                f"meta parameters: {meta_params}\nmeta buffers: {meta_buffers}"
            )
        self._fix_meta_tensors(self.backbone)
        hidden_size = (
            getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None) or 768
        )
        self.embedding_dim = int(hidden_size)

    @staticmethod
    def _patch_transformers5_tied_weights(config: Any, model_name: str) -> None:
        try:
            auto_map = getattr(config, "auto_map", None) or {}
            model_cls_path = auto_map.get("AutoModel")
            if not model_cls_path:
                return
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            cls = get_class_from_dynamic_module(model_cls_path, model_name)
            if not hasattr(cls, "all_tied_weights_keys"):

                @property
                def _all_tied_weights_keys(self: Any) -> Any:
                    tied = getattr(self, "_tied_weights_keys", {})
                    return tied if tied is not None else {}

                cls.all_tied_weights_keys = _all_tied_weights_keys
        except Exception:
            return

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        table = np.array(
            [
                [position / np.power(10000, 2 * (index // 2) / d_hid) for index in range(d_hid)]
                for position in range(n_position)
            ]
        )
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.tensor(table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    def _fix_meta_tensors(self, module: nn.Module) -> None:
        for name, value in list(module.__dict__.items()):
            if not isinstance(value, torch.Tensor) or not value.is_meta:
                continue
            if name == "pos_embed":
                patch_embed = getattr(module, "patch_embed", None)
                embed_dim = getattr(module, "embed_dim", None)
                num_patches = getattr(patch_embed, "num_patches", None)
                if embed_dim is not None and num_patches is not None:
                    tensor = self._get_sinusoid_encoding_table(num_patches, embed_dim)
                    setattr(module, name, tensor.to(self.device_str))
                    continue
            elif name == "cls_token":
                embed_dim = getattr(module, "embed_dim", None)
                if embed_dim is not None:
                    tensor = torch.zeros(1, 1, embed_dim)
                    nn.init.trunc_normal_(tensor, std=0.02)
                    setattr(module, name, tensor.to(self.device_str))
                    continue
            setattr(module, name, torch.zeros_like(value, device=self.device_str))
        for child in module.children():
            self._fix_meta_tensors(child)

    def _get_encoder_layers(self) -> nn.ModuleList | None:
        def at_path(root: nn.Module, path: tuple[str, ...]) -> nn.Module | None:
            current: Any = root
            for name in path:
                if not hasattr(current, name):
                    return None
                current = getattr(current, name)
            return current if isinstance(current, nn.Module) else None

        def looks_like_blocks(module: nn.Module | None) -> nn.ModuleList | None:
            if not isinstance(module, nn.ModuleList) or len(module) == 0:
                return None
            first = module[0]
            return (
                module
                if any(hasattr(first, name) for name in ("attn", "mlp", "norm1", "norm2"))
                else None
            )

        for path in (
            ("videomae", "encoder", "layer"),
            ("encoder", "layer"),
            ("model", "blocks"),
            ("blocks",),
        ):
            layers = looks_like_blocks(at_path(self.backbone, path))
            if layers is not None:
                return layers
        candidates = [
            layers
            for _, module in self.backbone.named_modules()
            if (layers := looks_like_blocks(module)) is not None
        ]
        return max(candidates, key=len, default=None)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        self.freeze_backbone()
        layers = self._get_encoder_layers()
        if layers is None or n <= 0:
            return
        for index in range(len(layers) - min(int(n), len(layers)), len(layers)):
            for parameter in layers[index].parameters():
                parameter.requires_grad = True

    def trainable_param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        elif not enabled and hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def _pool(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            tensor = outputs.float()
            if tensor.dim() == 1:
                return tensor.unsqueeze(0)
            if tensor.dim() == 2:
                return tensor
            return tensor.mean(dim=tuple(range(1, tensor.dim())))
        pooling = (self.cfg.pooling or "auto").lower().strip()
        if pooling in {"auto", "pooler"} and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output.float()
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            if hidden.dim() == 3:
                return (
                    hidden[:, 0].float()
                    if pooling in {"auto", "cls"}
                    else hidden.mean(dim=1).float()
                )
            return hidden.float().mean(dim=tuple(range(1, hidden.dim())))
        if hasattr(outputs, "__dict__"):
            for value in outputs.__dict__.values():
                if isinstance(value, torch.Tensor):
                    return value.mean(dim=tuple(range(1, value.dim()))).float()
        raise RuntimeError("无法从模型输出中取出 embedding")

    def _tensor_from_rgb_lists(self, clips: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
        prepared: list[list[np.ndarray]] = []
        for clip in clips:
            frames = list(clip)
            if len(frames) != self.cfg.num_frames:
                raise ValueError(f"每段需 {self.cfg.num_frames} 帧，当前 {len(frames)}")
            prepared.append([np.asarray(frame, dtype=np.uint8) for frame in frames])
        try:
            inputs = self.processor(prepared, return_tensors="pt")
        except TypeError:
            inputs = self.processor(videos=prepared, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 5 and pixel_values.shape[1] != 3 and pixel_values.shape[2] == 3:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        return pixel_values

    def forward(self, clips: Any) -> torch.Tensor:
        pixel_values = (
            clips.to(self.device_str)
            if isinstance(clips, torch.Tensor)
            else self._tensor_from_rgb_lists(clips).to(self.device_str)
        )
        return self._pool(self.backbone(pixel_values=pixel_values.float()))


__all__ = ["VideoMAEv2Encoder", "VideoMAEv2EncoderConfig"]
