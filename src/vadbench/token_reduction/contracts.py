"""Contracts shared by token-reduction policies and encoder bridges.

The package deliberately contains deployment-time tensor metadata only.  It
does not import the research collectors and its context rejects information
that would let a reducer use labels, video identifiers, or future activations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch


class TokenReductionContractError(ValueError):
    """Raised when token metadata cannot safely describe a reduction."""


def _require_tensor(value: Any, name: str, *, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TokenReductionContractError(f"{name} 必须是 torch.Tensor")
    if value.ndim != dimensions:
        raise TokenReductionContractError(
            f"{name} 必须是 {dimensions} 维，实际 shape={tuple(value.shape)}"
        )
    return value


def _forbid_sensitive_keys(value: Any, *, path: str = "metadata") -> None:
    """Reject metadata that would make a reducer label- or video-aware."""

    forbidden = (
        "label",
        "video",
        "category",
        "future",
        "teacher",
        "anomaly",
        "filename",
        "path",
        "truth",
    )
    if isinstance(value, Mapping):
        for key, nested in value.items():
            text = str(key).lower()
            if any(marker in text for marker in forbidden):
                raise TokenReductionContractError(
                    f"{path} 不得包含标签、视频标识或未来张量字段：{key!r}"
                )
            _forbid_sensitive_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _forbid_sensitive_keys(nested, path=f"{path}[{index}]")


@dataclass(frozen=True)
class TokenLayout:
    """Per-token provenance needed by a reducer without inferring a grid.

    ``source_coordinates`` has shape ``[B, N, 3]`` and uses the bridge's
    documented ``(t, h, w)`` coordinate system.  A missing tensor means that
    the bridge could not establish real coordinates; callers must not replace
    it with an approximate uniform mapping.  Invalid positions carry zero mass
    and ``original_token_ids == -1``.
    """

    valid_mask: torch.Tensor
    original_token_ids: torch.Tensor
    special_token_mask: torch.Tensor
    mass: torch.Tensor
    source_coordinates: torch.Tensor | None
    position_contract: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_mask = _require_tensor(self.valid_mask, "valid_mask", dimensions=2)
        original_ids = _require_tensor(self.original_token_ids, "original_token_ids", dimensions=2)
        special_mask = _require_tensor(self.special_token_mask, "special_token_mask", dimensions=2)
        mass = _require_tensor(self.mass, "mass", dimensions=2)
        shape = tuple(valid_mask.shape)
        if tuple(original_ids.shape) != shape or tuple(special_mask.shape) != shape:
            raise TokenReductionContractError(
                "token layout mask 和 original_token_ids shape 必须一致"
            )
        if tuple(mass.shape) != shape:
            raise TokenReductionContractError("mass shape 必须与 valid_mask 一致")
        if valid_mask.dtype is not torch.bool or special_mask.dtype is not torch.bool:
            raise TokenReductionContractError("valid_mask 与 special_token_mask 必须为 bool")
        if original_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TokenReductionContractError("original_token_ids 必须为整数 tensor")
        if not torch.is_floating_point(mass):
            raise TokenReductionContractError("mass 必须为浮点 tensor")
        if not torch.isfinite(mass).all() or (mass < 0).any():
            raise TokenReductionContractError("mass 必须是有限的非负数")
        if (special_mask & ~valid_mask).any():
            raise TokenReductionContractError("special token 必须也是有效 token")
        if (original_ids[~valid_mask] != -1).any():
            raise TokenReductionContractError("无效 token 的 original_token_ids 必须为 -1")
        if (mass[~valid_mask] != 0).any():
            raise TokenReductionContractError("无效 token 的 mass 必须为 0")
        if (mass[valid_mask] <= 0).any():
            raise TokenReductionContractError("有效 token 的 mass 必须为正数")
        if (original_ids[valid_mask] < 0).any():
            raise TokenReductionContractError("有效 token IDs 必须非负")
        for row, mask in enumerate(valid_mask):
            row_ids = original_ids[row, mask]
            if row_ids.numel() != torch.unique(row_ids).numel():
                raise TokenReductionContractError("每个样本的有效 original_token_ids 必须唯一")
        if self.source_coordinates is not None:
            coordinates = _require_tensor(
                self.source_coordinates, "source_coordinates", dimensions=3
            )
            if tuple(coordinates.shape[:2]) != shape or coordinates.shape[2] != 3:
                raise TokenReductionContractError(
                    "source_coordinates 必须为 [B, N, 3] 的真实 (t, h, w) 坐标"
                )
        if not isinstance(self.position_contract, str) or not self.position_contract.strip():
            raise TokenReductionContractError("position_contract 必须为非空字符串")
        if not isinstance(self.provenance, Mapping):
            raise TokenReductionContractError("provenance 必须为 Mapping")
        _forbid_sensitive_keys(self.provenance, path="provenance")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def batch_size(self) -> int:
        return int(self.valid_mask.shape[0])

    @property
    def token_capacity(self) -> int:
        return int(self.valid_mask.shape[1])

    @property
    def valid_token_counts(self) -> torch.Tensor:
        return self.valid_mask.sum(dim=1, dtype=torch.int64)

    def to(self, device: Any) -> TokenLayout:
        return replace(
            self,
            valid_mask=self.valid_mask.to(device),
            original_token_ids=self.original_token_ids.to(device),
            special_token_mask=self.special_token_mask.to(device),
            mass=self.mass.to(device),
            source_coordinates=None
            if self.source_coordinates is None
            else self.source_coordinates.to(device),
        )


@dataclass(frozen=True)
class ReductionContext:
    """Information available at the current reduction point only."""

    layer_depth: int
    budget: int | None = None
    same_block_qkv: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.layer_depth) is not int or self.layer_depth < 0:
            raise TokenReductionContractError("layer_depth 必须是非负整数")
        if self.budget is not None and (type(self.budget) is not int or self.budget <= 0):
            raise TokenReductionContractError("budget 必须为正整数或 None")
        if self.same_block_qkv is not None:
            _require_tensor(self.same_block_qkv, "same_block_qkv", dimensions=3)
        if not isinstance(self.metadata, Mapping):
            raise TokenReductionContractError("metadata 必须为 Mapping")
        _forbid_sensitive_keys(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ReductionResult:
    """Reduced tokens plus a sparse, CSR-like member mapping.

    ``member_offsets`` is ``[B, N_out + 1]`` and ``member_token_ids`` stores
    one compact original-token id per output slot for the identity reducer.
    Invalid output slots have an unchanged offset and member id ``-1``.  Future
    reducers may store several ids per output by widening the final dimension.
    """

    tokens: torch.Tensor
    layout: TokenLayout
    member_offsets: torch.Tensor
    member_token_ids: torch.Tensor
    input_token_counts: torch.Tensor
    output_token_counts: torch.Tensor
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = _require_tensor(self.tokens, "tokens", dimensions=3)
        batch_size, output_capacity, _ = tokens.shape
        if self.layout.batch_size != batch_size or self.layout.token_capacity != output_capacity:
            raise TokenReductionContractError("tokens 与输出 TokenLayout shape 必须一致")
        offsets = _require_tensor(self.member_offsets, "member_offsets", dimensions=2)
        members = _require_tensor(self.member_token_ids, "member_token_ids", dimensions=2)
        input_counts = _require_tensor(self.input_token_counts, "input_token_counts", dimensions=1)
        output_counts = _require_tensor(
            self.output_token_counts, "output_token_counts", dimensions=1
        )
        if tuple(offsets.shape) != (batch_size, output_capacity + 1):
            raise TokenReductionContractError("member_offsets 必须为 [B, N_out + 1]")
        if members.shape[0] != batch_size:
            raise TokenReductionContractError("member_token_ids 必须为 [B, M]")
        if tuple(input_counts.shape) != (batch_size,) or tuple(output_counts.shape) != (
            batch_size,
        ):
            raise TokenReductionContractError("input/output_token_counts 必须为 [B]")
        if not torch.all(offsets[:, 1:] >= offsets[:, :-1]):
            raise TokenReductionContractError("member_offsets 必须单调不减")
        if (
            offsets.dtype != torch.int64
            or members.dtype != torch.int64
            or (offsets[:, 0] != 0).any()
            or (offsets[:, -1] > members.shape[1]).any()
        ):
            raise TokenReductionContractError("CSR member offsets/IDs 必须为 int64 且界限正确")
        if not torch.equal(output_counts.to(torch.int64), self.layout.valid_token_counts):
            raise TokenReductionContractError("output_token_counts 必须匹配 layout.valid_mask")
        for row in range(batch_size):
            length = int(offsets[row, -1])
            if (members[row, :length] < 0).any() or (members[row, length:] != -1).any():
                raise TokenReductionContractError("CSR 有效成员需连续存放，尾部 padding 为 -1")
        if not isinstance(self.telemetry, Mapping):
            raise TokenReductionContractError("telemetry 必须为 Mapping")
        _forbid_sensitive_keys(self.telemetry, path="telemetry")
        object.__setattr__(self, "telemetry", MappingProxyType(dict(self.telemetry)))


@runtime_checkable
class TokenReducer(Protocol):
    """A label-free policy operating on a single currently available layer."""

    def reduce(
        self, tokens: torch.Tensor, layout: TokenLayout, context: ReductionContext
    ) -> ReductionResult:
        """Return new tokens, layout, sparse membership, and telemetry."""


__all__ = [
    "ReductionContext",
    "ReductionResult",
    "TokenLayout",
    "TokenReducer",
    "TokenReductionContractError",
]
