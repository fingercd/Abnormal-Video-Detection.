"""Bounded, read-only PyTorch hooks and first-pass probe summaries.

The collector observes modules supplied by an encoder bridge.  It never changes
hook return values, stores no GPU tensors, and has no label argument.  Missing
layout, mask, coordinate, CLS, or attention evidence is represented as an
explicit unavailable probe row rather than an inferred approximation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np


class ProbeCollectionError(ValueError):
    """A caller supplied invalid observation metadata or sites."""


@dataclass(frozen=True, slots=True)
class ProbeLimits:
    """Explicit collection limits; defaults are the protocol's lightweight mode."""

    max_observations: int = 512
    max_sampled_tokens: int = 256
    max_attention_queries: int = 64
    coverage_representatives: tuple[int, ...] = (1, 4, 16)
    redundancy_similarity: float = 0.9

    def __post_init__(self) -> None:
        for name in ("max_observations", "max_sampled_tokens", "max_attention_queries"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ProbeCollectionError(f"{name} 必须是正整数")
        if not self.coverage_representatives or any(
            not isinstance(value, int) or value <= 0 for value in self.coverage_representatives
        ):
            raise ProbeCollectionError("coverage_representatives 必须是正整数序列")
        if (
            not isinstance(self.redundancy_similarity, (int, float))
            or not -1 <= self.redundancy_similarity <= 1
        ):
            raise ProbeCollectionError("redundancy_similarity 必须在 [-1, 1]")


@dataclass(frozen=True, slots=True)
class ProbeTokenMetadata:
    """Actual token validity and provenance supplied by the bridge/caller.

    ``valid_mask`` is mandatory for sequence statistics: a collector cannot
    derive token padding from a padded frame batch.  Coordinates use actual
    ``(time, height, width)`` indices in the observed token layout and need an
    explicit source string; a guessed uniform timeline is rejected.
    """

    valid_mask: Any | None = None
    coordinates: Any | None = None
    coordinate_source: str | None = None
    special_token_indices: tuple[int, ...] | None = None
    has_cls: bool | None = None

    def __post_init__(self) -> None:
        if self.coordinates is not None:
            if not isinstance(self.coordinate_source, str) or not self.coordinate_source.strip():
                raise ProbeCollectionError(
                    "coordinates 需要非空 coordinate_source，不能猜测坐标来源"
                )
        elif self.coordinate_source is not None:
            raise ProbeCollectionError("coordinate_source 只能在提供 coordinates 时使用")
        if self.special_token_indices is not None:
            if any(not isinstance(value, int) or value < 0 for value in self.special_token_indices):
                raise ProbeCollectionError("special_token_indices 必须是非负整数")
            if len(set(self.special_token_indices)) != len(self.special_token_indices):
                raise ProbeCollectionError("special_token_indices 不可重复")
        if self.has_cls is False and self.special_token_indices:
            raise ProbeCollectionError("has_cls=False 时不能提供 CLS/special token 索引")
        if self.has_cls is True and not self.special_token_indices:
            raise ProbeCollectionError("has_cls=True requires the actual CLS index")


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    """One CPU-only, JSON-ready observation summary from a module hook."""

    site: str
    layer_index: int | None
    sublayer_kind: str | None
    batch_size: int | None
    token_count: int | None
    dtype: str
    rows: tuple[dict[str, Any], ...]

    def to_rows(
        self,
        *,
        run_id: str,
        encoder_id: str,
        checkpoint_digest: str | None,
        clip_ids: Sequence[str],
        video_ids: Sequence[str],
        partitions: Sequence[str] | None = None,
        backend: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Render protocol-shaped rows without joining labels.

        ``partitions`` is accepted as an identity field only.  Weak/temporal
        labels are intentionally absent; use :func:`research.labels.join_probe_rows`
        after collection.
        """

        if self.batch_size is not None and (
            len(clip_ids) != self.batch_size or len(video_ids) != self.batch_size
        ):
            raise ProbeCollectionError("clip_ids/video_ids 数量必须匹配观察 batch")
        if (
            partitions is not None
            and self.batch_size is not None
            and len(partitions) != self.batch_size
        ):
            raise ProbeCollectionError("partitions 数量必须匹配观察 batch")
        output: list[dict[str, Any]] = []
        for row in self.rows:
            batch_index = row.get("batch_index")
            if (
                not isinstance(batch_index, int)
                or self.batch_size is None
                or not 0 <= batch_index < self.batch_size
            ):
                continue
            result = {
                "run_id": run_id,
                "encoder_id": encoder_id,
                "checkpoint_digest": checkpoint_digest,
                "video_id": video_ids[batch_index],
                "clip_id": clip_ids[batch_index],
                "partition": None if partitions is None else partitions[batch_index],
                "layer_index": self.layer_index,
                "site": self.site,
                "sublayer_kind": self.sublayer_kind,
                "head_id": row.get("head_id"),
                "probe_id": row["probe_id"],
                "statistic_name": row["statistic_name"],
                "statistic_value": row.get("statistic_value"),
                "status": row["status"],
                "detail": row.get("detail"),
                "num_valid_tokens": row.get("num_valid_tokens"),
                "sampled_token_count": row.get("sampled_token_count"),
                "query_count": row.get("query_count"),
                "backend": backend,
                "dtype": self.dtype,
            }
            output.append(result)
        return tuple(output)


def _as_numpy(value: Any) -> np.ndarray | None:
    """Detach a tensor promptly; return None for non-tensor module outputs."""

    if isinstance(value, np.ndarray):
        return np.asarray(value)
    detach = getattr(value, "detach", None)
    if not callable(detach):
        return None
    detached = detach()
    cpu = getattr(detached, "cpu", None)
    if not callable(cpu):
        return None
    converted = cpu()
    if str(converted.dtype) == "torch.bfloat16":
        converted = converted.float()
    numpy = getattr(converted, "numpy", None)
    return None if not callable(numpy) else np.asarray(numpy())


def _first_tensor(value: Any) -> Any | None:
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and callable(getattr(value, "detach", None))
    ):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _site_parts(site: str) -> tuple[int | None, str | None]:
    pieces = site.split(".")
    layer_index: int | None = None
    for number, piece in enumerate(pieces[:-1]):
        if piece == "block" and pieces[number + 1].isdigit():
            layer_index = int(pieces[number + 1])
            break
    if any(piece in {"attn", "attention"} for piece in pieces):
        return layer_index, "attention"
    if "mlp" in pieces:
        return layer_index, "mlp"
    if pieces[-1] in {"input", "output", "norm1", "norm", "qkv", "probs"}:
        return layer_index, pieces[-1]
    return layer_index, None


def _even_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)]


def _spatiotemporal_sample(
    valid: np.ndarray, coordinates: np.ndarray | None, maximum: int
) -> np.ndarray:
    if coordinates is None or len(valid) <= maximum:
        return _even_sample(valid, maximum)
    times = np.unique(coordinates[valid, 0])
    spatial = np.unique(coordinates[valid, 1:], axis=0)
    spatial_budget = maximum // len(times)
    if spatial_budget == 0:
        return _even_sample(valid, maximum)
    # Keep the same selected spatial tracks over time; paired adjacent spatial
    # anchors also retain local-neighbour statistics in the bounded sample.
    anchors = _even_sample(np.arange(len(spatial)), max(1, spatial_budget // 2))
    chosen = np.unique(np.concatenate((anchors, np.minimum(anchors + 1, len(spatial) - 1))))[
        :spatial_budget
    ]
    chosen_spatial = spatial[chosen]
    mask = (coordinates[valid, None, 1:] == chosen_spatial[None]).all(axis=-1).any(axis=-1)
    return valid[mask][:maximum]


def _quantile(values: np.ndarray, percentile: float) -> float:
    return float(np.quantile(values, percentile))


def _gini(probabilities: np.ndarray) -> float:
    if len(probabilities) <= 1:
        return 0.0
    ordered = np.sort(probabilities)
    count = len(ordered)
    return float(
        (2 * np.arange(1, count + 1) @ ordered) / (count * ordered.sum()) - (count + 1) / count
    )


@dataclass(slots=True)
class ProbeCollector(AbstractContextManager["ProbeCollector"]):
    """Context-managed read-only module hooks for first-pass probes.

    Sites ending in ``.input`` use a forward pre-hook; every other site uses a
    forward hook.  Both hooks return ``None``.  The object may be reused only
    after leaving the previous context.
    """

    observation_sites: Mapping[str, Any]
    token_metadata: ProbeTokenMetadata
    limits: ProbeLimits = field(default_factory=ProbeLimits)
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)
    _observations: list[ProbeObservation] = field(default_factory=list, init=False, repr=False)
    _input_norms: dict[tuple[int | None, str], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    _active: bool = field(default=False, init=False, repr=False)
    dropped_observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_sites, Mapping) or not self.observation_sites:
            raise ProbeCollectionError("observation_sites 必须是非空 site→nn.Module 映射")
        if not isinstance(self.token_metadata, ProbeTokenMetadata):
            raise ProbeCollectionError("token_metadata 必须是 ProbeTokenMetadata")
        for site, module in self.observation_sites.items():
            if not isinstance(site, str) or not site:
                raise ProbeCollectionError("observation site 名称必须是非空字符串")
            if not callable(getattr(module, "register_forward_hook", None)):
                raise ProbeCollectionError(f"{site} 不是可注册 PyTorch hook 的模块")

    @property
    def observations(self) -> tuple[ProbeObservation, ...]:
        return tuple(self._observations)

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> ProbeCollector:
        if self._active:
            raise ProbeCollectionError("ProbeCollector 已激活")
        self._observations.clear()
        self._input_norms.clear()
        self.dropped_observations = 0
        try:
            for site, module in self.observation_sites.items():
                if site.endswith(".input"):
                    handle = module.register_forward_pre_hook(self._make_pre_hook(site))
                else:
                    handle = module.register_forward_hook(self._make_forward_hook(site))
                self._handles.append(handle)
        except Exception:
            self.close()
            raise
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Remove every hook, including after a failing model forward."""

        handles, self._handles = self._handles, []
        self._active = False
        for handle in handles:
            handle.remove()
        self._input_norms.clear()

    def run(self, forward: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run a forward under hooks and guarantee cleanup even when it raises."""

        with self:
            return forward(*args, **kwargs)

    def _make_pre_hook(self, site: str) -> Callable[[Any, tuple[Any, ...]], None]:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = _first_tensor(inputs)
            if value is not None:
                self._capture(site, value)
            return None

        return hook

    def _make_forward_hook(self, site: str) -> Callable[[Any, tuple[Any, ...], Any], None]:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = _first_tensor(output)
            if value is not None:
                self._capture(site, value)
            return None

        return hook

    def _capture(self, site: str, value: Any) -> None:
        if len(self._observations) >= self.limits.max_observations:
            self.dropped_observations += 1
            return
        if len(value.shape) == 4 and "probs" in site:
            # Sample query rows on the originating device, retaining all keys.
            # Never make a CPU copy of the complete N x N attention tensor.
            ids = _even_sample(np.arange(value.shape[2]), self.limits.max_attention_queries)
            sampled = value[:, :, ids.tolist(), :]
            array = _as_numpy(sampled)
            layer_index, _ = _site_parts(site)
            self._observations.append(self._summarize_attention(site, layer_index, array, ids))
            return
        array = _as_numpy(value)
        if array is None:
            return
        layer_index, sublayer_kind = _site_parts(site)
        if array.ndim == 3:
            observation = self._summarize_tokens(site, layer_index, sublayer_kind, array)
        elif array.ndim == 4 and sublayer_kind == "attention":
            observation = self._summarize_attention(
                site, layer_index, array, np.arange(array.shape[2])
            )
        else:
            observation = ProbeObservation(
                site=site,
                layer_index=layer_index,
                sublayer_kind=sublayer_kind,
                batch_size=int(array.shape[0]) if array.ndim else None,
                token_count=None,
                dtype=str(array.dtype),
                rows=(
                    _unavailable(
                        0,
                        "P01",
                        "tensor_layout",
                        "site output is not [B,N,D] or actual attention [B,H,Q,K]",
                    ),
                ),
            )
        self._observations.append(observation)

    def _sequence_mask(self, array: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        mask = self.token_metadata.valid_mask
        if mask is None:
            return None, "valid_mask is required; collector will not infer token padding"
        converted = _as_numpy(mask)
        if converted is None:
            converted = np.asarray(mask)
        if converted.shape != array.shape[:2]:
            return (
                None,
                f"valid_mask shape {converted.shape} does not match tokens {array.shape[:2]}",
            )
        if converted.dtype != np.bool_:
            return None, "valid_mask must be boolean"
        if (
            self.token_metadata.special_token_indices
            and max(self.token_metadata.special_token_indices) >= array.shape[1]
        ):
            return None, "special token index exceeds observed token count"
        return converted.copy(), None

    def _coordinates(self, array: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        coordinates = self.token_metadata.coordinates
        if coordinates is None:
            return None, "actual token coordinates were not supplied"
        converted = _as_numpy(coordinates)
        if converted is None:
            converted = np.asarray(coordinates)
        if converted.shape != (*array.shape[:2], 3):
            return (
                None,
                f"coordinates shape {converted.shape} does not match [B,N,3]={(*array.shape[:2], 3)}",
            )
        if not np.issubdtype(converted.dtype, np.integer):
            return None, "coordinates must be integer (t,h,w) token indices"
        return converted, None

    def _patch_mask(self, mask: np.ndarray) -> np.ndarray:
        result = mask.copy()
        special = self.token_metadata.special_token_indices
        if special is not None:
            for index in special:
                if index < result.shape[1]:
                    result[:, index] = False
        return result

    def _summarize_tokens(
        self, site: str, layer_index: int | None, sublayer_kind: str | None, array: np.ndarray
    ) -> ProbeObservation:
        batch, tokens, dimension = array.shape
        mask, error = self._sequence_mask(array)
        if error:
            return self._all_unavailable(
                site, layer_index, sublayer_kind, batch, tokens, str(array.dtype), error
            )
        if self.token_metadata.special_token_indices is None:
            return self._all_unavailable(
                site,
                layer_index,
                sublayer_kind,
                batch,
                tokens,
                str(array.dtype),
                "special token identity is unverified",
            )
        assert mask is not None
        patch_mask = self._patch_mask(mask)
        coordinates, coordinate_error = self._coordinates(array)
        rows: list[dict[str, Any]] = []
        for batch_index in range(batch):
            valid = np.flatnonzero(patch_mask[batch_index])
            if not len(valid):
                rows.extend(
                    _unavailable(
                        batch_index,
                        probe,
                        "no_valid_patch_tokens",
                        "padding/special-token mask left no patch token",
                    )
                    for probe in ("P01", "P02", "P04", "P07", "P16")
                )
                continue
            sampled = _spatiotemporal_sample(
                valid,
                None if coordinates is None else coordinates[batch_index],
                self.limits.max_sampled_tokens,
            )
            values = np.asarray(array[batch_index, sampled], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                rows.extend(
                    _unavailable(
                        batch_index,
                        probe,
                        "nonfinite",
                        "observed token tensor contains NaN/Inf",
                        len(valid),
                        len(sampled),
                    )
                    for probe in ("P01", "P02", "P04", "P07", "P16")
                )
                continue
            rows.extend(self._p01(batch_index, values, len(valid), len(sampled), dimension))
            rows.extend(self._p02(batch_index, values, len(valid), len(sampled)))
            if coordinate_error:
                rows.append(
                    _unavailable(
                        batch_index,
                        "P04",
                        "coordinates",
                        coordinate_error,
                        len(valid),
                        len(sampled),
                    )
                )
                rows.append(
                    _unavailable(
                        batch_index,
                        "P07",
                        "coordinates",
                        coordinate_error,
                        len(valid),
                        len(sampled),
                    )
                )
            else:
                assert coordinates is not None
                rows.extend(
                    self._p04(
                        batch_index, values, sampled, coordinates[batch_index, sampled], len(valid)
                    )
                )
                rows.extend(
                    self._p07(
                        batch_index, values, sampled, coordinates[batch_index, sampled], len(valid)
                    )
                )
            rows.extend(self._p16_activation(batch_index, values, len(valid), len(sampled)))
        mean_norms = np.linalg.norm(np.asarray(array, dtype=np.float64), axis=-1)
        if site.endswith(".mid.input"):
            self._input_norms[(layer_index, "mlp")] = mean_norms
        elif site.endswith(".input"):
            self._input_norms[(layer_index, "attention")] = mean_norms
        elif sublayer_kind in {"attention", "mlp"}:
            incoming = self._input_norms.get((layer_index, sublayer_kind))
            if incoming is None or incoming.shape != mean_norms.shape:
                for index in range(batch):
                    rows.append(
                        _unavailable(
                            index,
                            "P16",
                            "branch_update_ratio",
                            "matching block input was not observed",
                        )
                    )
            else:
                for index in range(batch):
                    usable = (
                        patch_mask[index]
                        & np.isfinite(incoming[index])
                        & np.isfinite(mean_norms[index])
                    )
                    if not usable.any() or np.linalg.norm(incoming[index, usable]) == 0:
                        rows.append(
                            _unavailable(
                                index, "P16", "branch_update_ratio", "zero/invalid block input norm"
                            )
                        )
                    else:
                        ratio = float(
                            np.linalg.norm(mean_norms[index, usable])
                            / np.linalg.norm(incoming[index, usable])
                        )
                        rows.append(
                            _available(
                                index,
                                "P16",
                                f"{sublayer_kind}_update_to_input_norm_ratio",
                                ratio,
                                int(usable.sum()),
                                int(usable.sum()),
                            )
                        )
        return ProbeObservation(
            site, layer_index, sublayer_kind, batch, tokens, str(array.dtype), tuple(rows)
        )

    def _all_unavailable(
        self,
        site: str,
        layer_index: int | None,
        sublayer_kind: str | None,
        batch: int,
        tokens: int,
        dtype: str,
        detail: str,
    ) -> ProbeObservation:
        rows = tuple(
            _unavailable(index, probe, "input_validation", detail)
            for index in range(batch)
            for probe in ("P01", "P02", "P04", "P07", "P16")
        )
        return ProbeObservation(site, layer_index, sublayer_kind, batch, tokens, dtype, rows)

    def _p01(
        self, index: int, values: np.ndarray, valid_count: int, sampled_count: int, dimension: int
    ) -> list[dict[str, Any]]:
        norms = np.linalg.norm(values, axis=-1) / sqrt(dimension)
        median = _quantile(norms, 0.5)
        p95 = _quantile(norms, 0.95)
        results = [
            _available(index, "P01", "activation_norm_median", median, valid_count, sampled_count),
            _available(
                index,
                "P01",
                "activation_norm_iqr",
                _quantile(norms, 0.75) - _quantile(norms, 0.25),
                valid_count,
                sampled_count,
            ),
            _available(index, "P01", "activation_norm_p95", p95, valid_count, sampled_count),
            _available(
                index,
                "P01",
                "token_channel_variance_mean",
                float(values.var(axis=0).mean()),
                valid_count,
                sampled_count,
            ),
        ]
        if median == 0:
            results.append(
                _unavailable(
                    index,
                    "P01",
                    "activation_norm_p95_over_p50",
                    "degenerate_p50",
                    valid_count=valid_count,
                    sampled_count=sampled_count,
                )
            )
        else:
            results.append(
                _available(
                    index,
                    "P01",
                    "activation_norm_p95_over_p50",
                    p95 / median,
                    valid_count,
                    sampled_count,
                )
            )
        return results

    def _p02(
        self, index: int, values: np.ndarray, valid_count: int, sampled_count: int
    ) -> list[dict[str, Any]]:
        if len(values) < 2:
            return [
                _unavailable(
                    index,
                    "P02",
                    "effective_rank",
                    "requires_at_least_two_tokens",
                    valid_count,
                    sampled_count,
                )
            ]
        centred = values - values.mean(axis=0, keepdims=True)
        gram = centred @ centred.T if centred.shape[0] <= centred.shape[1] else centred.T @ centred
        energy = np.maximum(np.linalg.eigvalsh(gram), 0)[::-1]
        total = float(energy.sum())
        if total == 0:
            return [
                _available(
                    index,
                    "P02",
                    "effective_rank",
                    0.0,
                    valid_count,
                    sampled_count,
                    detail="degenerate_zero_centered_tokens",
                ),
                _available(
                    index,
                    "P02",
                    "rank90",
                    0.0,
                    valid_count,
                    sampled_count,
                    detail="degenerate_zero_centered_tokens",
                ),
                _available(
                    index,
                    "P02",
                    "effective_rank_normalized",
                    0.0,
                    valid_count,
                    sampled_count,
                    detail="degenerate_zero_centered_tokens",
                ),
            ]
        probabilities = energy / total
        positive = probabilities[probabilities > 0]
        rank = float(np.exp(-(positive * np.log(positive)).sum()))
        rank90 = float(np.searchsorted(np.cumsum(probabilities), 0.9, side="left") + 1)
        maximum = min(len(values) - 1, values.shape[1])
        return [
            _available(index, "P02", "effective_rank", rank, valid_count, sampled_count),
            _available(index, "P02", "rank90", rank90, valid_count, sampled_count),
            _available(
                index,
                "P02",
                "effective_rank_normalized",
                rank / maximum,
                valid_count,
                sampled_count,
            ),
        ]

    def _p04(
        self,
        index: int,
        values: np.ndarray,
        sampled: np.ndarray,
        coordinates: np.ndarray,
        valid_count: int,
    ) -> list[dict[str, Any]]:
        count = len(values)
        if count < 2:
            return [
                _unavailable(
                    index, "P04", "similarity", "requires_at_least_two_tokens", valid_count, count
                )
            ]
        scale = np.linalg.norm(values, axis=1, keepdims=True)
        nonzero = scale[:, 0] > 0
        if nonzero.sum() < 2:
            return [
                _unavailable(index, "P04", "similarity", "zero_norm_tokens", valid_count, count)
            ]
        normalized = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        similarity = normalized @ normalized.T
        time = coordinates[:, 0]
        spatial = coordinates[:, 1:]
        results: list[dict[str, Any]] = []
        categories = {
            "same_frame_local": (time[:, None] == time[None, :])
            & (np.abs(spatial[:, None] - spatial[None, :]).sum(axis=-1) == 1),
            "same_frame_nonlocal": (time[:, None] == time[None, :])
            & (np.abs(spatial[:, None] - spatial[None, :]).sum(axis=-1) > 1),
            "adjacent_time_corresponding": (np.abs(time[:, None] - time[None, :]) == 1)
            & np.all(spatial[:, None] == spatial[None, :], axis=-1),
        }
        for name, pair_mask in categories.items():
            pair_mask &= np.triu(np.ones((count, count), dtype=bool), k=1)
            pair_mask &= nonzero[:, None] & nonzero[None, :]
            values_for_pairs = similarity[pair_mask]
            if not len(values_for_pairs):
                results.append(
                    _unavailable(
                        index,
                        "P04",
                        f"{name}_cosine_mean",
                        "no_matching_real_coordinate_pairs",
                        valid_count,
                        count,
                    )
                )
            else:
                results.append(
                    _available(
                        index,
                        "P04",
                        f"{name}_cosine_mean",
                        float(values_for_pairs.mean()),
                        valid_count,
                        count,
                    )
                )
        off_diagonal = similarity[
            (~np.eye(count, dtype=bool)) & nonzero[:, None] & nonzero[None, :]
        ]
        results.append(
            _available(
                index,
                "P04",
                "redundancy_fraction",
                float((off_diagonal >= self.limits.redundancy_similarity).mean()),
                valid_count,
                count,
            )
        )
        for selected_count in sorted({min(k, count) for k in self.limits.coverage_representatives}):
            selected = np.linspace(0, count - 1, selected_count, dtype=np.int64)
            coverage = similarity[nonzero][:, selected].max(axis=1).mean()
            results.append(
                _available(
                    index,
                    "P04",
                    f"uniform_representative_coverage_k{selected_count}",
                    float(coverage),
                    valid_count,
                    count,
                )
            )
        return results

    def _p07(
        self,
        index: int,
        values: np.ndarray,
        sampled: np.ndarray,
        coordinates: np.ndarray,
        valid_count: int,
    ) -> list[dict[str, Any]]:
        count = len(values)
        if count < 2:
            return [
                _unavailable(
                    index,
                    "P07",
                    "temporal_change",
                    "requires_at_least_two_tokens",
                    valid_count,
                    count,
                )
            ]
        scale = np.linalg.norm(values, axis=1, keepdims=True)
        if (scale[:, 0] > 0).sum() < 2:
            return [
                _unavailable(
                    index, "P07", "temporal_change", "zero_norm_tokens", valid_count, count
                )
            ]
        normalized = np.divide(values, scale, out=np.zeros_like(values), where=scale != 0)
        similarity = normalized @ normalized.T
        time, spatial = coordinates[:, 0], coordinates[:, 1:]
        direct = (np.abs(time[:, None] - time[None, :]) == 1) & np.all(
            spatial[:, None] == spatial[None, :], axis=-1
        )
        direct &= np.triu(np.ones((count, count), dtype=bool), k=1)
        direct &= (scale[:, 0] > 0)[:, None] & (scale[:, 0] > 0)[None, :]
        direct_values = 1 - similarity[direct]
        results: list[dict[str, Any]] = []
        if len(direct_values):
            results.append(
                _available(
                    index,
                    "P07",
                    "same_position_adjacent_time_change_median",
                    float(np.median(direct_values)),
                    valid_count,
                    count,
                )
            )
            results.append(
                _available(
                    index,
                    "P07",
                    "same_position_adjacent_time_change_p90",
                    float(np.quantile(direct_values, 0.9)),
                    valid_count,
                    count,
                )
            )
        else:
            results.append(
                _unavailable(
                    index,
                    "P07",
                    "same_position_adjacent_time_change_median",
                    "no_adjacent_temporal_pairs",
                    valid_count,
                    count,
                )
            )
        aligned: list[float] = []
        for source in range(count):
            candidates = np.flatnonzero(
                (np.abs(time - time[source]) == 1)
                & (np.abs(spatial - spatial[source]).sum(axis=1) <= 1)
            )
            candidates = candidates[scale[candidates, 0] > 0]
            if len(candidates) and scale[source, 0] > 0:
                aligned.append(float(1 - similarity[source, candidates].max()))
        if aligned:
            results.append(
                _available(
                    index,
                    "P07",
                    "local_aligned_adjacent_time_change_median",
                    float(np.median(aligned)),
                    valid_count,
                    count,
                )
            )
        else:
            results.append(
                _unavailable(
                    index,
                    "P07",
                    "local_aligned_adjacent_time_change_median",
                    "no_local_temporal_candidates",
                    valid_count,
                    count,
                )
            )
        return results

    def _p16_activation(
        self, index: int, values: np.ndarray, valid_count: int, sampled_count: int
    ) -> list[dict[str, Any]]:
        norms = np.linalg.norm(values, axis=-1) / sqrt(values.shape[-1])
        return [
            _available(
                index,
                "P16",
                "activation_norm_median",
                float(np.median(norms)),
                valid_count,
                sampled_count,
            )
        ]

    def _summarize_attention(
        self, site: str, layer_index: int | None, attention: np.ndarray, query_ids: np.ndarray
    ) -> ProbeObservation:
        batch, heads, queries, keys = attention.shape
        rows: list[dict[str, Any]] = []
        mask_reference = np.empty((batch, keys, 1), dtype=np.float32)
        mask, error = self._sequence_mask(mask_reference)
        if error or len(query_ids) != queries or np.any(query_ids >= keys):
            detail = error or "attention query IDs do not map to token key axis"
            rows.extend(
                _unavailable(index, probe, "attention_validation", detail)
                for index in range(batch)
                for probe in ("P10", "P11", "P13")
            )
            return ProbeObservation(
                site, layer_index, "attention", batch, keys, str(attention.dtype), tuple(rows)
            )
        assert mask is not None
        if not np.all(np.isfinite(attention)):
            rows.extend(
                _unavailable(index, probe, "nonfinite", "attention contains NaN/Inf")
                for index in range(batch)
                for probe in ("P10", "P11", "P13")
            )
            return ProbeObservation(
                site, layer_index, "attention", batch, keys, str(attention.dtype), tuple(rows)
            )
        for batch_index in range(batch):
            valid = np.flatnonzero(mask[batch_index])
            if len(valid) < 2:
                rows.extend(
                    _unavailable(
                        batch_index,
                        probe,
                        "insufficient_valid_tokens",
                        "attention requires at least two valid tokens",
                        len(valid),
                        0,
                    )
                    for probe in ("P10", "P11", "P13")
                )
                continue
            query_slots = np.flatnonzero(np.isin(query_ids, valid))
            query_slots = _even_sample(query_slots, self.limits.max_attention_queries)
            selected_queries = query_ids[query_slots]
            if not len(selected_queries):
                rows.extend(
                    _unavailable(
                        batch_index,
                        probe,
                        "no_valid_queries",
                        "sampled query rows contain no valid tokens",
                    )
                    for probe in ("P10", "P11", "P13")
                )
                continue
            for head in range(heads):
                full_rows = np.asarray(attention[batch_index, head, query_slots], dtype=np.float64)
                if np.any(full_rows < 0) or not np.allclose(
                    full_rows.sum(axis=-1), 1.0, atol=2e-3, rtol=2e-3
                ):
                    rows.extend(
                        _unavailable(
                            batch_index,
                            probe,
                            "invalid_attention_probabilities",
                            "native pre-dropout attention rows must sum to one",
                        )
                        for probe in ("P10", "P11", "P13")
                    )
                    continue
                matrix = full_rows[:, valid]
                row_sum = matrix.sum(axis=1, keepdims=True)
                if np.any(matrix < 0) or np.any(row_sum <= 0):
                    rows.extend(
                        _unavailable(
                            batch_index,
                            probe,
                            "invalid_attention_probabilities",
                            "attention rows must be non-negative with positive mass",
                            len(valid),
                            len(valid),
                            len(selected_queries),
                            head,
                        )
                        for probe in ("P10", "P11")
                    )
                    continue
                probabilities = matrix / row_sum
                detail = "sampled queries, all valid keys; conditional on valid keys when padding is present"
                log_probabilities = np.log(np.maximum(probabilities, np.finfo(float).tiny))
                entropy = -((probabilities * log_probabilities).sum(axis=1) / np.log(len(valid)))
                top = np.sort(probabilities, axis=1)[:, -min(4, len(valid)) :].sum(axis=1)
                rows.append(
                    _available(
                        batch_index,
                        "P10",
                        "outgoing_attention_entropy_mean",
                        float(entropy.mean()),
                        len(valid),
                        len(valid),
                        len(selected_queries),
                        head,
                        detail,
                    )
                )
                rows.append(
                    _available(
                        batch_index,
                        "P10",
                        "outgoing_attention_top4_mass_mean",
                        float(top.mean()),
                        len(valid),
                        len(valid),
                        len(selected_queries),
                        head,
                    )
                )
                incoming = probabilities.mean(axis=0)
                in_entropy = -float(
                    (incoming * np.log(np.maximum(incoming, np.finfo(float).tiny))).sum()
                    / np.log(len(valid))
                )
                rows.append(
                    _available(
                        batch_index,
                        "P11",
                        "sampled_query_incoming_attention_entropy",
                        in_entropy,
                        len(valid),
                        len(valid),
                        len(selected_queries),
                        head,
                        detail,
                    )
                )
                rows.append(
                    _available(
                        batch_index,
                        "P11",
                        "sampled_query_incoming_attention_top4_mass",
                        float(np.sort(incoming)[-min(4, len(valid)) :].sum()),
                        len(valid),
                        len(valid),
                        len(selected_queries),
                        head,
                    )
                )
                rows.append(
                    _available(
                        batch_index,
                        "P11",
                        "sampled_query_incoming_attention_gini",
                        _gini(incoming),
                        len(valid),
                        len(valid),
                        len(selected_queries),
                        head,
                    )
                )
                rows.extend(
                    self._p13_from_attention(
                        batch_index, head, probabilities, valid, selected_queries
                    )
                )
        return ProbeObservation(
            site, layer_index, "attention", batch, keys, str(attention.dtype), tuple(rows)
        )

    def _p13_from_attention(
        self,
        index: int,
        head: int,
        probabilities: np.ndarray,
        valid: np.ndarray,
        selected_queries: np.ndarray,
    ) -> list[dict[str, Any]]:
        if self.token_metadata.has_cls is False:
            return [
                _not_applicable(
                    index,
                    "P13",
                    "cls_attention",
                    "bridge receipt states that this encoder has no CLS",
                    len(valid),
                    len(valid),
                    len(selected_queries),
                    head,
                )
            ]
        special = self.token_metadata.special_token_indices
        if self.token_metadata.has_cls is not True or not special:
            return [
                _unavailable(
                    index,
                    "P13",
                    "cls_attention",
                    "real CLS identity is unavailable",
                    len(valid),
                    len(valid),
                    len(selected_queries),
                    head,
                )
            ]
        cls = special[0]
        matches = np.flatnonzero(selected_queries == cls)
        key_positions = np.flatnonzero(valid != cls)
        if not len(matches) or not len(key_positions):
            return [
                _unavailable(
                    index,
                    "P13",
                    "cls_to_patch_attention_entropy",
                    "sampled attention does not include CLS query or patch keys",
                    len(valid),
                    len(valid),
                    len(selected_queries),
                    head,
                )
            ]
        row = probabilities[matches[0], key_positions]
        row = row / row.sum() if row.sum() else row
        if row.sum() <= 0:
            return [
                _unavailable(
                    index,
                    "P13",
                    "cls_to_patch_attention_entropy",
                    "CLS row has zero patch mass",
                    len(valid),
                    len(valid),
                    len(selected_queries),
                    head,
                )
            ]
        entropy = (
            -float((row * np.log(np.maximum(row, np.finfo(float).tiny))).sum() / np.log(len(row)))
            if len(row) > 1
            else 0.0
        )
        return [
            _available(
                index,
                "P13",
                "cls_to_patch_attention_entropy",
                entropy,
                len(valid),
                len(valid),
                len(selected_queries),
                head,
            ),
            _available(
                index,
                "P13",
                "cls_to_patch_attention_top4_mass",
                float(np.sort(row)[-min(4, len(row)) :].sum()),
                len(valid),
                len(valid),
                len(selected_queries),
                head,
            ),
        ]


def _available(
    batch_index: int,
    probe: str,
    statistic: str,
    value: float,
    valid_count: int | None = None,
    sampled_count: int | None = None,
    query_count: int | None = None,
    head_id: int | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "probe_id": probe,
        "statistic_name": statistic,
        "statistic_value": float(value),
        "status": "available",
        "detail": detail,
        "num_valid_tokens": valid_count,
        "sampled_token_count": sampled_count,
        "query_count": query_count,
        "head_id": head_id,
    }


def _unavailable(
    batch_index: int,
    probe: str,
    statistic: str,
    detail: str,
    valid_count: int | None = None,
    sampled_count: int | None = None,
    query_count: int | None = None,
    head_id: int | None = None,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "probe_id": probe,
        "statistic_name": statistic,
        "statistic_value": None,
        "status": "unavailable",
        "detail": detail,
        "num_valid_tokens": valid_count,
        "sampled_token_count": sampled_count,
        "query_count": query_count,
        "head_id": head_id,
    }


def _not_applicable(
    batch_index: int,
    probe: str,
    statistic: str,
    detail: str,
    valid_count: int | None = None,
    sampled_count: int | None = None,
    query_count: int | None = None,
    head_id: int | None = None,
) -> dict[str, Any]:
    row = _unavailable(
        batch_index, probe, statistic, detail, valid_count, sampled_count, query_count, head_id
    )
    row["status"] = "not_applicable"
    return row
