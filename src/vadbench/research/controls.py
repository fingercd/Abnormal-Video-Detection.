"""Label-free input controls for normal--abnormal probe analysis.

The functions in this module run beside decoded :class:`~vadbench.contracts.ClipBatch`
objects, before any encoder invocation.  They describe only the supplied valid
frames; they do not inspect labels, infer scenes, estimate optical flow, or
enter a deployment reducer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from vadbench.contracts import ClipBatch


class ControlError(ValueError):
    """A control sidecar or its frozen calibration is malformed."""


CONTROL_BIN_FIELDS = frozenset({"motion_bin", "brightness_bin"})
CONTROL_IDENTITY_FIELDS = ("encoder_id", "input_sampling_id")


def _as_numpy(value: Any, name: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return np.asarray(numpy())
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - ClipBatch validates ordinary arrays
        raise ControlError(f"无法读取 {name}") from exc


def _downsample_indices(length: int, maximum: int) -> np.ndarray:
    target = min(length, maximum)
    return np.rint(np.linspace(0, length - 1, target)).astype(np.int64)


def raw_clip_controls(
    batch: ClipBatch,
    *,
    clip_ids: Sequence[str] | None = None,
    low_resolution: tuple[int, int] = (32, 32),
) -> tuple[dict[str, Any], ...]:
    """Return finite, label-free controls for each clip in a decoded batch.

    ``adjacent_frame_mad_per_s_mean`` is RGB mean absolute difference of
    spatially downsampled *supplied adjacent valid frames*, divided by their
    actual timestamp gap.  It is explicitly a frame-change proxy, **not**
    optical flow or a scene/motion classification.
    """

    if (
        not isinstance(low_resolution, tuple)
        or len(low_resolution) != 2
        or any(type(value) is not int or value <= 0 for value in low_resolution)
    ):
        raise ControlError("low_resolution 必须是两个正整数 (height, width)")
    if clip_ids is not None and (
        len(clip_ids) != batch.batch_size
        or any(not isinstance(value, str) or not value for value in clip_ids)
    ):
        raise ControlError("clip_ids 必须为与 batch 对齐的非空字符串序列")

    frames = _as_numpy(batch.frames, "frames")
    times = _as_numpy(batch.timestamps_s, "timestamps_s").astype(np.float64, copy=False)
    valid = (
        np.ones((batch.batch_size, batch.num_frames), dtype=bool)
        if batch.valid_mask is None
        else _as_numpy(batch.valid_mask, "valid_mask").astype(bool, copy=False)
    )
    indices = (
        None if batch.frame_indices is None else _as_numpy(batch.frame_indices, "frame_indices")
    )
    height, width = batch.spatial_size
    row_indices = _downsample_indices(height, low_resolution[0])
    column_indices = _downsample_indices(width, low_resolution[1])
    records: list[dict[str, Any]] = []
    for item in range(batch.batch_size):
        count = int(valid[item].sum())
        clip_frames = frames[item, :count].astype(np.float32) / 255.0
        clip_times = times[item, :count]
        luminance = (
            0.2126 * clip_frames[..., 0]
            + 0.7152 * clip_frames[..., 1]
            + 0.0722 * clip_frames[..., 2]
        )
        reduced = clip_frames[:, row_indices][:, :, column_indices, :]
        pair_mad = (
            np.mean(np.abs(np.diff(reduced, axis=0)), axis=(1, 2, 3)) if count > 1 else np.array([])
        )
        gaps = np.diff(clip_times)
        usable = gaps > 0
        rates = pair_mad[usable] / gaps[usable] if len(pair_mad) else np.array([])
        record = {
            "video_id": batch.video_ids[item],
            "clip_id": None if clip_ids is None else clip_ids[item],
            "batch_index": item,
            "brightness_mean": float(luminance.mean()),
            "brightness_std": float(luminance.std(ddof=0)),
            "adjacent_frame_mad_mean": None if not len(pair_mad) else float(pair_mad.mean()),
            "adjacent_frame_mad_per_s_mean": None if not len(rates) else float(rates.mean()),
            "motion_status": "available" if len(rates) else "unavailable",
            "motion_detail": (
                "low-resolution RGB adjacent-frame mean absolute difference per actual dt; not optical flow"
                if len(rates)
                else "fewer than two positive-dt valid frames; frame-change rate unavailable"
            ),
            "adjacent_valid_pair_count": int(len(pair_mad)),
            "positive_dt_pair_count": int(usable.sum()),
            "zero_dt_pair_count": int((~usable).sum()),
            "valid_frame_count": count,
            "padded_frame_count": int(batch.num_frames - count),
            "padding_fraction": float((batch.num_frames - count) / batch.num_frames),
            "source_start_s": float(clip_times[0]),
            "source_end_s": float(clip_times[-1]),
            "source_duration_s": float(clip_times[-1] - clip_times[0]),
            "decoded_frame_height": height,
            "decoded_frame_width": width,
            "resolution_source": "ClipBatch.frames decoded BTHWC resolution; original-video resolution is not inferred",
            "low_resolution_height": int(len(row_indices)),
            "low_resolution_width": int(len(column_indices)),
        }
        if indices is not None:
            record["source_frame_first"] = int(indices[item, 0])
            record["source_frame_last"] = int(indices[item, count - 1])
        else:
            record["source_frame_first"] = None
            record["source_frame_last"] = None
        records.append(record)
    return tuple(records)


@dataclass(frozen=True, slots=True)
class ControlBinCalibration:
    """Frozen normal-fit tertiles for applying control strata to other roles."""

    brightness_thresholds: tuple[float, float]
    motion_thresholds: tuple[float, float]
    fit_normal_video_counts: dict[str, int]
    encoder_id: str
    input_sampling_id: str
    control_input_sha256: str | None = None
    fit_normal_video_ids: tuple[str, ...] = ()
    source_partition: str = "fit"
    quantiles: tuple[float, float] = (1 / 3, 2 / 3)
    video_aggregation: str = "mean available clips per video"

    def __post_init__(self) -> None:
        for name in ("brightness_thresholds", "motion_thresholds"):
            values = getattr(self, name)
            if len(values) != 2 or not all(isfinite(float(value)) for value in values):
                raise ControlError(f"{name} 必须为两个有限阈值")
            if values[0] > values[1]:
                raise ControlError(f"{name} 必须递增")
        if self.source_partition != "fit":
            raise ControlError("控制量阈值只能从 fit 分区的正常视频拟合")
        if any(value < 3 for value in self.fit_normal_video_counts.values()):
            raise ControlError("每个控制量至少需要三个 fit 正常独立视频")
        if not isinstance(self.encoder_id, str) or not self.encoder_id:
            raise ControlError("校准必须绑定一个非空 encoder_id")
        if not isinstance(self.input_sampling_id, str) or not self.input_sampling_id:
            raise ControlError("校准必须绑定一个非空 input_sampling_id")
        if self.control_input_sha256 is not None and (
            not isinstance(self.control_input_sha256, str) or len(self.control_input_sha256) != 64
        ):
            raise ControlError("control_input_sha256 必须为 SHA-256 十六进制摘要或 null")
        if not self.fit_normal_video_ids or any(
            not isinstance(value, str) or not value for value in self.fit_normal_video_ids
        ):
            raise ControlError("校准必须保留实际使用的 fit 正常 video_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "control-bin-calibration-v1",
            "encoder_id": self.encoder_id,
            "input_sampling_id": self.input_sampling_id,
            "control_input_sha256": self.control_input_sha256,
            "source_partition": self.source_partition,
            "fit_label": 0,
            "video_aggregation": self.video_aggregation,
            "quantiles": list(self.quantiles),
            "thresholds": {
                "brightness_bin": list(self.brightness_thresholds),
                "motion_bin": list(self.motion_thresholds),
            },
            "fit_normal_video_counts": dict(self.fit_normal_video_counts),
            "fit_normal_video_ids": list(self.fit_normal_video_ids),
            "motion_definition": "low-resolution RGB adjacent-frame MAD per actual dt; not optical flow",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ControlBinCalibration:
        try:
            thresholds = value["thresholds"]
            return cls(
                brightness_thresholds=tuple(float(item) for item in thresholds["brightness_bin"]),
                motion_thresholds=tuple(float(item) for item in thresholds["motion_bin"]),
                fit_normal_video_counts={
                    str(key): int(item) for key, item in value["fit_normal_video_counts"].items()
                },
                encoder_id=str(value["encoder_id"]),
                input_sampling_id=str(value["input_sampling_id"]),
                control_input_sha256=value.get("control_input_sha256"),
                fit_normal_video_ids=tuple(str(item) for item in value["fit_normal_video_ids"]),
                source_partition=str(value.get("source_partition", "fit")),
                quantiles=tuple(float(item) for item in value.get("quantiles", (1 / 3, 2 / 3))),
                video_aggregation=str(
                    value.get("video_aggregation", "mean available clips per video")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlError("控制量 calibration 文件格式不合法") from exc


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def aggregate_control_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Aggregate raw clip controls to one row per ``(encoder_id, video_id)``.

    The encoder and explicit input-sampling identity are part of the key's
    contract: native preprocessing/sampling from different encoders cannot be
    mixed when fitting a normal control calibration.
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        encoder_id, video_id = raw.get("encoder_id"), raw.get("video_id")
        if not isinstance(encoder_id, str) or not encoder_id:
            raise ControlError("raw control 行必须含非空 encoder_id")
        if not isinstance(video_id, str) or not video_id:
            raise ControlError("raw control 行必须含非空 video_id")
        groups[(encoder_id, video_id)].append(dict(raw))
    output: list[dict[str, Any]] = []
    for (encoder_id, video_id), members in sorted(groups.items()):
        metadata: dict[str, Any] = {}
        for field in ("partition", "weak_label", "input_sampling_id"):
            values = {row.get(field) for row in members}
            if len(values) != 1:
                raise ControlError(f"同一 encoder/video 的 {field} 不一致：{encoder_id}/{video_id}")
            metadata[field] = values.pop()
        if not isinstance(metadata["input_sampling_id"], str) or not metadata["input_sampling_id"]:
            raise ControlError(
                "raw control 行必须有 input_sampling_id；不能以不同 encoder 的原生采样混合拟合"
            )
        numeric = {
            field: [
                value for row in members if (value := _finite_number(row.get(field))) is not None
            ]
            for field in (
                "brightness_mean",
                "brightness_std",
                "adjacent_frame_mad_mean",
                "adjacent_frame_mad_per_s_mean",
                "source_duration_s",
                "padding_fraction",
            )
        }
        result = {
            "encoder_id": encoder_id,
            "video_id": video_id,
            **metadata,
            "control_clip_count": len(members),
            "motion_available_clip_count": len(numeric["adjacent_frame_mad_per_s_mean"]),
            "control_aggregation": "mean available clip controls per encoder/video",
        }
        for field, values in numeric.items():
            result[field] = None if not values else float(np.mean(values))
        output.append(result)
    return tuple(output)


def calibrate_control_bins(
    rows: Iterable[Mapping[str, Any]],
    *,
    encoder_id: str,
    input_sampling_id: str,
    control_input_sha256: str | None = None,
) -> ControlBinCalibration:
    """Fit brightness/motion tertiles from fit-partition normal videos only."""

    metric_fields = {
        "brightness_bin": "brightness_mean",
        "motion_bin": "adjacent_frame_mad_per_s_mean",
    }
    if not isinstance(encoder_id, str) or not encoder_id:
        raise ControlError("encoder_id 必须是非空字符串")
    if not isinstance(input_sampling_id, str) or not input_sampling_id:
        raise ControlError("input_sampling_id 必须是非空字符串")
    per_video: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("encoder_id") != encoder_id:
            continue
        if row.get("input_sampling_id") != input_sampling_id:
            raise ControlError("同一 encoder 的控制量含多个 input_sampling_id，拒绝混合拟合")
        if row.get("partition") != "fit" or row.get("weak_label") != 0:
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ControlError("fit 正常控制量行必须含非空 video_id")
        for bin_field, metric in metric_fields.items():
            value = _finite_number(row.get(metric))
            if value is not None:
                per_video[video_id][bin_field].append(value)
    thresholds: dict[str, tuple[float, float]] = {}
    counts: dict[str, int] = {}
    for bin_field in metric_fields:
        values = np.asarray(
            [
                float(np.mean(metrics[bin_field]))
                for metrics in per_video.values()
                if metrics[bin_field]
            ],
            dtype=float,
        )
        counts[bin_field] = int(len(values))
        if len(values) < 3:
            raise ControlError(f"{bin_field} 至少需要三个 fit 正常独立视频，实际 {len(values)}")
        lower, upper = np.quantile(values, (1 / 3, 2 / 3))
        thresholds[bin_field] = (float(lower), float(upper))
    return ControlBinCalibration(
        brightness_thresholds=thresholds["brightness_bin"],
        motion_thresholds=thresholds["motion_bin"],
        fit_normal_video_counts=counts,
        encoder_id=encoder_id,
        input_sampling_id=input_sampling_id,
        control_input_sha256=control_input_sha256,
        fit_normal_video_ids=tuple(sorted(per_video)),
    )


def _tertile(value: float | None, thresholds: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "mid"
    return "high"


def apply_control_bins(
    rows: Iterable[Mapping[str, Any]], calibration: ControlBinCalibration | Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Apply frozen fit-normal thresholds without refitting or relabelling rows."""

    frozen = (
        calibration
        if isinstance(calibration, ControlBinCalibration)
        else ControlBinCalibration.from_mapping(calibration)
    )
    result: list[dict[str, Any]] = []
    thresholds = {
        "brightness_bin": ("brightness_mean", frozen.brightness_thresholds),
        "motion_bin": ("adjacent_frame_mad_per_s_mean", frozen.motion_thresholds),
    }
    for row in rows:
        updated = dict(row)
        if updated.get("encoder_id") != frozen.encoder_id:
            raise ControlError("control calibration encoder_id 与待应用行不一致")
        if updated.get("input_sampling_id") != frozen.input_sampling_id:
            raise ControlError("control calibration input_sampling_id 与待应用行不一致")
        for bin_field, (metric, limits) in thresholds.items():
            assigned = _tertile(_finite_number(updated.get(metric)), limits)
            existing = updated.get(bin_field)
            if existing not in {None, "", "unknown", assigned}:
                raise ControlError(f"{bin_field} 已有冲突值，拒绝静默覆盖")
            updated[bin_field] = assigned
        updated["control_bin_calibration"] = frozen.as_dict()["schema"]
        result.append(updated)
    return tuple(result)
