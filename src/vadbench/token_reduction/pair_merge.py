"""Verified horizontal pair merging at one native encoder depth.

The merge is deliberately local: it sees only the current block output and a
``TokenLayout`` whose real coordinates were observed by an encoder bridge.
It has no labels, video identity, future activations, or dense-teacher input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch
from torch import nn

from .contracts import TokenLayout


class PairMergeError(ValueError):
    """Raised when a native layout cannot support fixed horizontal pairing."""


def _require_long(value: torch.Tensor, name: str, dimensions: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
        raise PairMergeError(f"{name} 必须是 {dimensions} 维 tensor")
    if value.dtype != torch.int64:
        raise PairMergeError(f"{name} 必须是 int64 tensor")


@dataclass(frozen=True)
class PairMergeSpec:
    """Static native pair table and first-member output anchors.

    ``pair_indices`` is ``[B, P, 2]`` and identifies the left/right source
    tokens for every output patch slot.  ``gate_group_ids`` lets TimeSformer
    tie all time slots of a spatial trajectory to one pair weight; for the
    other active encoders every pair is its own group.  The suffix always sees
    ``output_indices``: special tokens plus the first member of each pair.
    """

    encoder_id: str
    input_token_count: int
    output_indices: torch.Tensor
    anchor_indices: torch.Tensor
    anchor_slots: torch.Tensor
    pair_indices: torch.Tensor
    gate_group_ids: torch.Tensor
    trajectory_length: int
    output_layout: TokenLayout
    input_mass: torch.Tensor
    input_topology_sha256: str

    def __post_init__(self) -> None:
        if self.encoder_id not in {"videomaev2", "timesformer", "videomae", "vjepa2"}:
            raise PairMergeError(f"不支持的 active encoder={self.encoder_id!r}")
        if type(self.input_token_count) is not int or self.input_token_count <= 0:
            raise PairMergeError("input_token_count 必须为正整数")
        _require_long(self.output_indices, "output_indices", 2)
        _require_long(self.anchor_indices, "anchor_indices", 2)
        _require_long(self.anchor_slots, "anchor_slots", 2)
        _require_long(self.pair_indices, "pair_indices", 3)
        _require_long(self.gate_group_ids, "gate_group_ids", 2)
        batch, pairs, pair_width = self.pair_indices.shape
        if pair_width != 2 or self.anchor_indices.shape != (batch, pairs):
            raise PairMergeError("pair_indices 必须为 [B,P,2] 且 anchor_indices 为 [B,P]")
        if self.anchor_slots.shape != (batch, pairs) or self.gate_group_ids.shape != (batch, pairs):
            raise PairMergeError("anchor_slots 与 gate_group_ids 必须为 [B,P]")
        if type(self.trajectory_length) is not int or self.trajectory_length <= 0:
            raise PairMergeError("trajectory_length 必须为正整数")
        if pairs % self.trajectory_length:
            raise PairMergeError("pair 数必须被 trajectory_length 整除")
        if self.output_indices.shape[0] != batch:
            raise PairMergeError("output_indices batch 必须与 pair_indices 一致")
        if self.output_layout.batch_size != batch or self.output_layout.token_capacity != self.output_indices.shape[1]:
            raise PairMergeError("output_layout 必须与 output_indices 对齐")
        if (
            not isinstance(self.input_mass, torch.Tensor)
            or self.input_mass.ndim != 1
            or not torch.is_floating_point(self.input_mass)
        ):
            raise PairMergeError("input_mass 必须为浮点 [B] tensor")
        if self.input_mass.shape != (batch,):
            raise PairMergeError("input_mass 必须为 [B]")
        for name, value in (
            ("output_indices", self.output_indices),
            ("anchor_indices", self.anchor_indices),
            ("pair_indices", self.pair_indices),
        ):
            if (value < 0).any() or (value >= self.input_token_count).any():
                raise PairMergeError(f"{name} 含超出 native token 范围的索引")
        if (self.anchor_slots < 0).any() or (self.anchor_slots >= self.output_indices.shape[1]).any():
            raise PairMergeError("anchor_slots 含超出输出范围的索引")
        if (self.gate_group_ids < 0).any():
            raise PairMergeError("gate_group_ids 必须非负")
        if not torch.equal(
            self.output_indices.gather(1, self.anchor_slots), self.anchor_indices
        ):
            raise PairMergeError("anchor_slots 必须指向对应的 first-member anchor")
        if not torch.allclose(self.output_layout.mass.sum(dim=1), self.input_mass.to(self.output_layout.mass)):
            raise PairMergeError("merge 后 mass 总和必须等于原 valid mass")

    def receipt(self) -> dict[str, object]:
        """Describe merge mass and the explicit approximate position choice."""

        pair_mass = self.output_layout.mass.gather(1, self.anchor_slots)
        special = self.output_layout.special_token_mask
        return {
            "operator": "horizontal_pair_weighted_mean",
            "position_anchor": "first_member_native_index",
            "position_semantics": "approximate_anchor_position",
            "pair_mass_rule": "sum_of_pair_members",
            "pair_mass_min": float(pair_mass.min().item()),
            "pair_mass_max": float(pair_mass.max().item()),
            "special_mass_preserved": bool(
                torch.equal(
                    self.output_layout.mass[special], torch.ones_like(self.output_layout.mass[special])
                )
            ),
            "input_valid_mass": self.input_mass.tolist(),
            "output_valid_mass": self.output_layout.mass.sum(dim=1).tolist(),
            "input_topology_sha256": self.input_topology_sha256,
        }

    def with_layout(self, layout: TokenLayout) -> PairMergeSpec:
        """Reuse a verified topology with another batch of the same geometry.

        This only rebuilds vectorized output metadata, including member mass.
        It deliberately does not scan Python coordinate dictionaries again;
        callers obtain it through :class:`PairMergePlanCache` only after the
        same native bridge geometry has already been verified once.
        """

        if _topology_sha256(layout) != self.input_topology_sha256:
            raise PairMergeError("cached pair plan 与当前 layout 的真实 topology 不一致")
        return replace(
            self,
            output_layout=_output_layout(
                layout, self.output_indices, self.anchor_slots, self.pair_indices
            ),
            input_mass=layout.mass.sum(dim=1),
        )

    def to(self, device: torch.device | str) -> PairMergeSpec:
        """Move the fixed plan once at setup; the forward hot path does no copy."""

        return replace(
            self,
            output_indices=self.output_indices.to(device),
            anchor_indices=self.anchor_indices.to(device),
            anchor_slots=self.anchor_slots.to(device),
            pair_indices=self.pair_indices.to(device),
            gate_group_ids=self.gate_group_ids.to(device),
            output_layout=self.output_layout.to(device),
            input_mass=self.input_mass.to(device),
        )


def _grid(layout: TokenLayout) -> tuple[int, int, int]:
    grid = layout.provenance.get("grid")
    if not isinstance(grid, list) or len(grid) != 3 or not all(
        type(value) is int and value > 0 for value in grid
    ):
        raise PairMergeError("pair merge 需要已验证的 [T,H,W] grid")
    return tuple(grid)


def _locations(layout: TokenLayout, batch: int) -> dict[tuple[int, int, int], int]:
    assert layout.source_coordinates is not None
    locations: dict[tuple[int, int, int], int] = {}
    for index in torch.nonzero(~layout.special_token_mask[batch], as_tuple=False).flatten().tolist():
        key = tuple(int(value) for value in layout.source_coordinates[batch, index].tolist())
        if key in locations:
            raise PairMergeError("真实坐标不能唯一确定水平相邻 pair")
        locations[key] = index
    return locations


def _output_layout(
    layout: TokenLayout,
    output_indices: torch.Tensor,
    anchor_slots: torch.Tensor,
    pair_indices: torch.Tensor,
) -> TokenLayout:
    """Build output provenance with pair member mass summed at each anchor."""

    output_special = layout.special_token_mask.gather(1, output_indices)
    output_coordinates = layout.source_coordinates.gather(
        1, output_indices.unsqueeze(-1).expand(-1, -1, 3)
    )
    output_ids = layout.original_token_ids.gather(1, output_indices)
    output_mass = layout.mass.gather(1, output_indices).clone()
    member_mass = layout.mass.gather(1, pair_indices.reshape(layout.batch_size, -1))
    member_mass = member_mass.reshape(pair_indices.shape).sum(dim=-1)
    output_mass.scatter_(1, anchor_slots, member_mass)
    return TokenLayout(
        valid_mask=torch.ones_like(output_indices, dtype=torch.bool),
        original_token_ids=output_ids,
        special_token_mask=output_special,
        mass=output_mass,
        source_coordinates=output_coordinates,
        position_contract=layout.position_contract,
        provenance={
            **dict(layout.provenance),
            "pair_merge": {
                "operator": "horizontal_pair_weighted_mean",
                "anchor": "first_member_native_index",
                "position_semantics": "approximate_anchor_position",
                "mass": "sum_of_pair_members",
            },
        },
    )


def _cache_key(encoder_id: str, layout: TokenLayout) -> tuple[object, ...]:
    coordinate_source = layout.provenance.get("coordinate_source")
    if not isinstance(coordinate_source, str) or not coordinate_source.startswith("verified-"):
        raise PairMergeError("pair plan cache 只接受 bridge 已验证的坐标来源")
    return (
        encoder_id,
        layout.position_contract,
        coordinate_source,
        _grid(layout),
        layout.batch_size,
        layout.token_capacity,
        _topology_sha256(layout),
    )


def _topology_sha256(layout: TokenLayout) -> str:
    """Bind setup-time native coordinates/IDs/masks, not video/frame identity.

    Mass is intentionally separate: ``with_layout`` recomputes output mass
    for the same verified pairing. This CPU digest belongs in setup only.
    """
    if layout.source_coordinates is None:
        raise PairMergeError("pair merge topology 缺少已验证的真实 token 坐标")
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "position_contract": layout.position_contract,
        "coordinate_source": layout.provenance.get("coordinate_source"),
        "grid": _grid(layout),
    }, sort_keys=True).encode("utf-8"))
    for name in ("source_coordinates", "original_token_ids", "special_token_mask", "valid_mask"):
        tensor = getattr(layout, name).detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class PairMergePlanCache:
    """Cache coordinate-validated pair topology by bridge geometry, never video ID."""

    def __init__(self) -> None:
        self._plans: dict[tuple[object, ...], PairMergeSpec] = {}

    def resolve(self, encoder_id: str, layout: TokenLayout) -> PairMergeSpec:
        key = _cache_key(encoder_id, layout)
        plan = self._plans.get(key)
        if plan is None:
            plan = horizontal_pair_merge_spec(encoder_id, layout)
            self._plans[key] = plan
            return plan
        return plan.with_layout(layout)


def horizontal_pair_merge_spec(encoder_id: str, layout: TokenLayout) -> PairMergeSpec:
    """Build a deterministic left-anchor horizontal pair plan from real coordinates."""

    if layout.source_coordinates is None:
        raise PairMergeError("pair merge 缺少已验证的真实 token 坐标")
    if bool((~layout.valid_mask).any()):
        raise PairMergeError("pair merge 不支持 padding 或 ragged layout")
    frames, height, width = _grid(layout)
    if width % 2:
        raise PairMergeError("pair merge 要求偶数 grid width")
    if encoder_id not in {"videomaev2", "timesformer", "videomae", "vjepa2"}:
        raise PairMergeError(f"不支持的 active encoder={encoder_id!r}")
    if encoder_id == "timesformer":
        special = layout.special_token_mask
        if not bool(special[:, 0].all()) or bool(special.sum(dim=1).ne(1).any()):
            raise PairMergeError("TimeSformer pair merge 需要唯一 index-0 CLS")
        # Divided spatial attention reshapes retained locations by the native
        # patch width, so half-width coverage is valid only for even height.
        if height % 2:
            raise PairMergeError("TimeSformer pair merge 要求偶数 grid height 以保留 native patch-width")
    elif bool(layout.special_token_mask.any()):
        raise PairMergeError(f"{encoder_id} 当前 pair merge 仅支持 patch-only layout")

    expected_patches = frames * height * width
    patch_count = layout.token_capacity - int(layout.special_token_mask[0].sum())
    if patch_count != expected_patches:
        raise PairMergeError("layout token 数与已验证 [T,H,W] grid 不一致")

    output_rows: list[torch.Tensor] = []
    anchor_rows: list[torch.Tensor] = []
    slot_rows: list[torch.Tensor] = []
    pair_rows: list[torch.Tensor] = []
    group_rows: list[torch.Tensor] = []
    for batch in range(layout.batch_size):
        locations = _locations(layout, batch)
        pairs: list[tuple[int, int]] = []
        groups: list[int] = []
        if encoder_id == "timesformer":
            # Public TimeSformer order is spatial-major/time-minor.  Each
            # pair's score must be shared over its complete time trajectory.
            group = 0
            for row in range(height):
                for column in range(0, width, 2):
                    for time in range(frames):
                        left = locations.get((time, row, column))
                        right = locations.get((time, row, column + 1))
                        if left is None or right is None:
                            raise PairMergeError("layout 缺少完整 TimeSformer 相邻轨迹")
                        pairs.append((left, right))
                        groups.append(group)
                    group += 1
        else:
            group = 0
            for time in range(frames):
                for row in range(height):
                    for column in range(0, width, 2):
                        left = locations.get((time, row, column))
                        right = locations.get((time, row, column + 1))
                        if left is None or right is None:
                            raise PairMergeError("layout 缺少完整水平相邻 pair")
                        pairs.append((left, right))
                        groups.append(group)
                        group += 1
        pair_tensor = torch.tensor(pairs, dtype=torch.int64)
        anchors = pair_tensor[:, 0]
        specials = torch.nonzero(layout.special_token_mask[batch], as_tuple=False).flatten().to(torch.int64)
        output = torch.sort(torch.cat((specials, anchors))).values
        output_rows.append(output)
        anchor_rows.append(anchors)
        slot_rows.append(torch.searchsorted(output, anchors.contiguous()))
        pair_rows.append(pair_tensor)
        group_rows.append(torch.tensor(groups, dtype=torch.int64))
    output_indices = torch.stack(output_rows)
    anchor_slots = torch.stack(slot_rows)
    output_layout = _output_layout(layout, output_indices, anchor_slots, torch.stack(pair_rows))
    return PairMergeSpec(
        encoder_id=encoder_id,
        input_token_count=layout.token_capacity,
        output_indices=output_indices,
        anchor_indices=torch.stack(anchor_rows),
        anchor_slots=anchor_slots,
        pair_indices=torch.stack(pair_rows),
        gate_group_ids=torch.stack(group_rows),
        trajectory_length=frames if encoder_id == "timesformer" else 1,
        output_layout=output_layout,
        input_mass=layout.mass.sum(dim=1),
        input_topology_sha256=_topology_sha256(layout),
    )


class PairLinearGate(nn.Module):
    """Bias-free shared linear score for the two members of every pair."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, pairs: torch.Tensor, *, trajectory_length: int = 1) -> torch.Tensor:
        if pairs.ndim != 4 or pairs.shape[2] != 2 or pairs.shape[-1] != self.weight.numel():
            raise PairMergeError("gate pairs 必须为 [B,P,2,D] 且 D 与 weight 一致")
        if type(trajectory_length) is not int or trajectory_length <= 0:
            raise PairMergeError("trajectory_length 必须为正整数")
        if pairs.shape[1] % trajectory_length:
            raise PairMergeError("pair 数必须被 trajectory_length 整除")
        logits = torch.einsum("bpmd,d->bpm", pairs, self.weight)
        if trajectory_length == 1:
            return torch.softmax(logits, dim=-1)
        groups = logits.reshape(
            logits.shape[0], logits.shape[1] // trajectory_length, trajectory_length, 2
        )
        tied = groups.mean(dim=2, keepdim=True).expand_as(groups).reshape_as(logits)
        return torch.softmax(tied, dim=-1)


class PairWeightedMerge(nn.Module):
    """Mean or soft linear pair merge with identical train/inference semantics."""

    def __init__(self, dim: int, gate: PairLinearGate | None = None) -> None:
        super().__init__()
        if gate is not None and gate.weight.numel() != dim:
            raise ValueError("gate width 必须与 merge dim 一致")
        self.dim = dim
        self.gate = gate

    def forward(self, hidden: torch.Tensor, spec: PairMergeSpec) -> torch.Tensor:
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[:2] != (spec.output_indices.shape[0], spec.input_token_count)
            or hidden.shape[-1] != self.dim
        ):
            raise PairMergeError("hidden 必须与 PairMergeSpec 对齐为 [B,N,D]")
        if not torch.is_floating_point(hidden):
            raise PairMergeError("hidden 必须为浮点 tensor")
        if spec.output_indices.device != hidden.device:
            raise PairMergeError("PairMergeSpec 必须在初始化时通过 .to(hidden.device) 移至执行设备")
        pairs = spec.pair_indices
        members = hidden.gather(
            1, pairs.reshape(hidden.shape[0], -1).unsqueeze(-1).expand(-1, -1, self.dim)
        ).reshape(hidden.shape[0], pairs.shape[1], 2, self.dim)
        if self.gate is None:
            merged = members.mean(dim=2)
        else:
            weights = self.gate(members, trajectory_length=spec.trajectory_length)
            # Match torch.mean's float32 accumulation for low-precision inputs.
            # Multiplying in fp16 first would round small members before their
            # sum and break exact zero-gate/mean parity; casts preserve gradients.
            accumulation_dtype = (
                torch.float32 if hidden.dtype in (torch.float16, torch.bfloat16) else hidden.dtype
            )
            merged = (
                members.to(accumulation_dtype) * weights.unsqueeze(-1).to(accumulation_dtype)
            ).sum(dim=2).to(hidden.dtype)
        output_indices = spec.output_indices
        output = hidden.gather(1, output_indices.unsqueeze(-1).expand(-1, -1, self.dim))
        slots = spec.anchor_slots
        return output.scatter(1, slots.unsqueeze(-1).expand(-1, -1, self.dim), merged)


__all__ = [
    "PairLinearGate",
    "PairMergePlanCache",
    "PairMergeError",
    "PairMergeSpec",
    "PairWeightedMerge",
    "horizontal_pair_merge_spec",
]
