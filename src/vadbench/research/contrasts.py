"""Video-level normal--abnormal contrasts for probe-summary exports.

This module deliberately sits on the analysis side of the research boundary.
It consumes already label-joined ``probe_summary.jsonl`` rows, reduces repeated
clip observations to one value per video, and treats videos (or pre-defined
matched groups) as the resampling units.  It has no encoder or reducer imports.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np


class ContrastError(ValueError):
    """The supplied analysis export or frozen definition is malformed."""


SIGNATURE_FIELDS = (
    "encoder_id",
    "layer_index",
    "site",
    "sublayer_kind",
    "head_id",
    "probe_id",
    "statistic_name",
)
MATCHING_FIELDS = frozenset({"scene_group", "motion_bin", "brightness_bin", "duration_bin"})


@dataclass(frozen=True, slots=True)
class ContrastConfig:
    """Fixed, auditable choices for a contrast export."""

    bootstrap_replicates: int = 1000
    seed: int = 20270917
    phase: str = "explore"
    matching_fields: tuple[str, ...] = ("scene_group", "motion_bin")

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap_replicates, int) or self.bootstrap_replicates <= 0:
            raise ContrastError("bootstrap_replicates 必须是正整数")
        if not isinstance(self.seed, int):
            raise ContrastError("seed 必须是整数")
        if self.phase not in {"explore", "confirm", "select"}:
            raise ContrastError("phase 必须是 explore、confirm 或 select")
        if (
            not isinstance(self.matching_fields, tuple)
            or not self.matching_fields
            or len(set(self.matching_fields)) != len(self.matching_fields)
            or any(field not in MATCHING_FIELDS for field in self.matching_fields)
        ):
            raise ContrastError(
                f"matching_fields 必须是非空且不重复的白名单字段：{sorted(MATCHING_FIELDS)}"
            )


@dataclass(frozen=True, slots=True)
class ContrastAnalysis:
    """JSON/CSV-ready outputs of one bounded analysis run."""

    video_rows: tuple[dict[str, Any], ...]
    contrast_rows: tuple[dict[str, Any], ...]
    composition_rows: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    return str(value)


def _signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [name for name in SIGNATURE_FIELDS if name not in row]
    if missing:
        raise ContrastError(f"probe 行缺少签名字段：{missing}")
    for name in ("encoder_id", "site", "probe_id", "statistic_name"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise ContrastError(f"probe 行的 {name} 必须是非空字符串")
    return tuple(row[name] for name in SIGNATURE_FIELDS)


def signature_dict(signature: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(SIGNATURE_FIELDS, signature, strict=True))


def _available_value(row: Mapping[str, Any]) -> float | None:
    if row.get("status") != "available":
        return None
    value = row.get("statistic_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _label(value: Any) -> int | None:
    return value if type(value) is int and value in {0, 1} else None


def _candidate_signature(
    candidate: Mapping[str, Any], *, require_direction: bool
) -> dict[str, Any]:
    missing = [name for name in SIGNATURE_FIELDS if name not in candidate]
    if missing:
        raise ContrastError(f"冻结候选缺少精确签名字段：{missing}")
    identifier = candidate.get("candidate_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ContrastError("冻结候选必须含非空 candidate_id")
    direction = candidate.get("expected_direction")
    if require_direction and direction not in {"positive", "negative"}:
        raise ContrastError("confirm 冻结候选的 expected_direction 必须为 positive 或 negative")
    if direction is not None and direction not in {"positive", "negative"}:
        raise ContrastError("expected_direction 必须为 positive、negative 或省略")
    result = {name: candidate[name] for name in SIGNATURE_FIELDS}
    result["candidate_id"] = identifier
    result["expected_direction"] = direction
    return result


def validate_candidate_definition(
    value: Mapping[str, Any] | None, *, require_direction: bool
) -> tuple[dict[str, Any], ...]:
    """Validate an externally frozen candidate definition; never select one."""

    if value is None:
        if require_direction:
            raise ContrastError("confirm 阶段必须提供探索后冻结的 candidate definition")
        return ()
    candidates = value.get("candidates") if isinstance(value, Mapping) else None
    if not isinstance(candidates, list) or not candidates:
        raise ContrastError("candidate definition 必须含非空 candidates 数组")
    if len(candidates) > 2:
        raise ContrastError("一次最多定义两个候选")
    normalized = tuple(
        _candidate_signature(item, require_direction=require_direction)
        for item in candidates
        if isinstance(item, Mapping)
    )
    if len(normalized) != len(candidates):
        raise ContrastError("candidate definition 的每项都必须是对象")
    if len({item["candidate_id"] for item in normalized}) != len(normalized):
        raise ContrastError("candidate_id 不可重复")
    return normalized


def _matches_candidate(signature: tuple[Any, ...], candidate: Mapping[str, Any]) -> bool:
    return all(signature[index] == candidate[name] for index, name in enumerate(SIGNATURE_FIELDS))


def aggregate_probe_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Mean first over repeated rows in a clip, then over clips in a video.

    Labels and nuisance fields are only metadata here.  A video with 20 clips
    consequently has exactly the same inferential weight as one with two clips.
    Rows marked unavailable are retained only through their signature by
    :func:`analyze_contrasts`; they cannot contribute a fabricated value.
    """

    clip_values: dict[tuple[tuple[Any, ...], str, str], list[float]] = defaultdict(list)
    clip_meta: dict[tuple[tuple[Any, ...], str, str], dict[str, Any]] = {}
    video_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = _signature(row)
        video_id, clip_id = row.get("video_id"), row.get("clip_id")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ContrastError("probe 行必须含非空 video_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ContrastError("probe 行必须含非空 clip_id")
        meta = {
            "weak_label": _label(row.get("weak_label")),
            "partition": row.get("partition"),
            "scene_group": _text(row.get("scene_group")),
            "motion_bin": _text(row.get("motion_bin")),
            "brightness_bin": _text(row.get("brightness_bin")),
            "duration_bin": _text(row.get("duration_bin")),
            "category_for_analysis_only": _text(row.get("category_for_analysis_only")),
        }
        previous_video = video_meta.setdefault(video_id, meta)
        for field in (
            "weak_label",
            "partition",
            "scene_group",
            "motion_bin",
            "brightness_bin",
            "duration_bin",
            "category_for_analysis_only",
        ):
            if previous_video[field] != meta[field]:
                raise ContrastError(f"同一 video_id 的 {field} 不一致：{video_id}")
        key = (signature, video_id, clip_id)
        previous_clip = clip_meta.setdefault(key, meta)
        if previous_clip != meta:
            raise ContrastError(f"同一 clip 的分析元数据不一致：{clip_id}")
        value = _available_value(row)
        if value is not None:
            clip_values[key].append(value)

    by_video: dict[tuple[tuple[Any, ...], str], list[float]] = defaultdict(list)
    meta_by_video: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    for (signature, video_id, _clip_id), values in clip_values.items():
        if values:
            by_video[(signature, video_id)].append(float(np.mean(values)))
            meta_by_video[(signature, video_id)] = video_meta[video_id]
    result: list[dict[str, Any]] = []
    for (signature, video_id), values in sorted(
        by_video.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        result.append(
            {
                **signature_dict(signature),
                "video_id": video_id,
                "video_statistic_value": float(np.mean(values)),
                "clip_count": len(values),
                **meta_by_video[(signature, video_id)],
            }
        )
    return tuple(result)


def join_video_controls(
    probe_rows: Iterable[Mapping[str, Any]], controls: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Join derived video bins by exact ``(encoder_id, video_id)`` identity.

    A manually populated non-unknown bin cannot silently override the frozen
    raw-control result.  Missing control rows deliberately yield ``unknown``
    bins, making matched-control output unavailable instead of synthetic.
    """

    by_video: dict[tuple[str, str], Mapping[str, Any]] = {}
    for control in controls:
        encoder_id, video_id = control.get("encoder_id"), control.get("video_id")
        if (
            not isinstance(encoder_id, str)
            or not encoder_id
            or not isinstance(video_id, str)
            or not video_id
        ):
            raise ContrastError("video control 行必须含非空 encoder_id 和 video_id")
        key = (encoder_id, video_id)
        if key in by_video:
            raise ContrastError(f"video control 重复：{encoder_id}/{video_id}")
        by_video[key] = control
    result: list[dict[str, Any]] = []
    for source in probe_rows:
        row = dict(source)
        encoder_id, video_id = row.get("encoder_id"), row.get("video_id")
        control = by_video.get((encoder_id, video_id))
        if control is None:
            for field in ("motion_bin", "brightness_bin"):
                if _text(row.get(field)) == "unknown":
                    row[field] = "unknown"
            row["control_join_status"] = "missing_video_control"
            result.append(row)
            continue
        for field in ("motion_bin", "brightness_bin"):
            assigned = _text(control.get(field))
            existing = _text(row.get(field))
            if existing != "unknown" and existing != assigned:
                raise ContrastError(f"probe/control 的 {field} 冲突：{encoder_id}/{video_id}")
            row[field] = assigned
        row["control_join_status"] = "applied_frozen_video_control"
        row["control_input_sampling_id"] = control.get("input_sampling_id")
        result.append(row)
    return tuple(result)


def _ks_cdf_overlap(normal: np.ndarray, positive: np.ndarray) -> float:
    """Return ``1 - D_KS``: a bin-free empirical CDF-overlap descriptor."""

    values = np.sort(np.concatenate((normal, positive)))
    if not len(values):
        return float("nan")
    normal_cdf = np.searchsorted(np.sort(normal), values, side="right") / len(normal)
    positive_cdf = np.searchsorted(np.sort(positive), values, side="right") / len(positive)
    return float(1 - np.max(np.abs(normal_cdf - positive_cdf)))


def _common_language(positive: np.ndarray, normal: np.ndarray) -> float:
    comparisons = positive[:, None] - normal[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _hedges_g(positive: np.ndarray, normal: np.ndarray) -> float | None:
    n_positive, n_normal = len(positive), len(normal)
    if n_positive < 2 or n_normal < 2:
        return None
    numerator = (n_positive - 1) * np.var(positive, ddof=1) + (n_normal - 1) * np.var(
        normal, ddof=1
    )
    denominator = n_positive + n_normal - 2
    if denominator <= 0 or numerator <= 0:
        return None
    d = (float(np.mean(positive)) - float(np.mean(normal))) / sqrt(numerator / denominator)
    correction = 1 - 3 / (4 * (n_positive + n_normal) - 9)
    return float(correction * d)


def _raw_distribution(positive: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    """Keep raw video distributions visible before interpreting a standardized effect."""

    positive_mean = float(np.mean(positive)) if len(positive) else float("nan")
    normal_mean = float(np.mean(normal)) if len(normal) else float("nan")
    return {
        "positive_raw_mean": positive_mean,
        "normal_raw_mean": normal_mean,
        "positive_raw_std": float(np.std(positive, ddof=1)) if len(positive) > 1 else float("nan"),
        "normal_raw_std": float(np.std(normal, ddof=1)) if len(normal) > 1 else float("nan"),
        "raw_mean_delta_positive_minus_normal": positive_mean - normal_mean,
    }


def _normalization_caution(
    signature: tuple[Any, ...], positive: np.ndarray, normal: np.ndarray
) -> str:
    """Flag near-constant norm outputs before a tiny delta becomes a huge g."""

    site = str(signature[SIGNATURE_FIELDS.index("site")]).lower()
    sublayer = str(signature[SIGNATURE_FIELDS.index("sublayer_kind")]).lower()
    statistic = str(signature[SIGNATURE_FIELDS.index("statistic_name")]).lower()
    looks_normalized = any(
        token in site or token in sublayer
        for token in ("layernorm", ".norm", "_norm", ".ln", "_ln")
    )
    if not looks_normalized or "norm" not in statistic or len(positive) < 2 or len(normal) < 2:
        return "not_detected"
    raw = _raw_distribution(positive, normal)
    scale = max(abs(raw["positive_raw_mean"]), abs(raw["normal_raw_mean"]), 1.0)
    if max(raw["positive_raw_std"], raw["normal_raw_std"]) <= 1e-5 * scale:
        return "mechanically_near_constant_norm_output"
    return "not_detected"


def _ci(values: list[float]) -> tuple[float | None, float | None]:
    usable = np.asarray([value for value in values if isfinite(value)], dtype=float)
    if not len(usable):
        return None, None
    return float(np.quantile(usable, 0.025)), float(np.quantile(usable, 0.975))


def _bootstrap_hedges(
    positive: np.ndarray, normal: np.ndarray, *, replicates: int, rng: np.random.Generator
) -> tuple[float | None, float | None, int, int]:
    values: list[float] = []
    attempts = 0
    # A two-video group can draw a constant resample, for which standardized
    # effect is undefined.  Re-draw rather than silently treating it as zero;
    # retain the attempt count so this handling is auditable.
    while len(values) < replicates and attempts < replicates * 20:
        attempts += 1
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        sampled_normal = rng.choice(normal, size=len(normal), replace=True)
        effect = _hedges_g(sampled_positive, sampled_normal)
        if effect is not None:
            values.append(effect)
    low, high = _ci(values)
    return low, high, len(values), attempts


def _base_row(signature: tuple[Any, ...], **extra: Any) -> dict[str, Any]:
    return {**signature_dict(signature), **extra}


def _weak_video_contrast(
    signature: tuple[Any, ...],
    positive: np.ndarray,
    normal: np.ndarray,
    *,
    contrast_id: str,
    group_definition: str,
    stratum: str,
    config: ContrastConfig,
    rng: np.random.Generator,
    selection_source: str,
    confirmation_source: str | None,
) -> dict[str, Any]:
    row = _base_row(
        signature,
        contrast_id=contrast_id,
        group_definition=group_definition,
        stratum=stratum,
        resampling_unit="video",
        num_independent_videos=int(len(positive) + len(normal)),
        num_positive_videos=int(len(positive)),
        num_normal_videos=int(len(normal)),
        effect_name="hedges_g_positive_minus_normal",
        effect=None,
        mean_difference=None,
        positive_raw_mean=None,
        normal_raw_mean=None,
        positive_raw_std=None,
        normal_raw_std=None,
        raw_mean_delta_positive_minus_normal=None,
        normalization_caution="not_detected",
        ci_low=None,
        ci_high=None,
        bootstrap_valid_replicates=0,
        bootstrap_attempts=0,
        cdf_overlap=None,
        common_language_positive_gt_normal=None,
        selection_source=selection_source,
        confirmation_source=confirmation_source,
        status="unavailable",
        reason=None,
    )
    if len(positive) < 2 or len(normal) < 2:
        row.update(_raw_distribution(positive, normal))
        row["reason"] = "每组至少需要两个独立视频才能计算标准化效应与视频 bootstrap"
        return row
    row.update(_raw_distribution(positive, normal))
    row["normalization_caution"] = _normalization_caution(signature, positive, normal)
    effect = _hedges_g(positive, normal)
    if effect is None:
        row["reason"] = "组内方差为零或退化，Hedges g 不可定义"
        row["mean_difference"] = row["raw_mean_delta_positive_minus_normal"]
        return row
    low, high, valid, attempts = _bootstrap_hedges(
        positive, normal, replicates=config.bootstrap_replicates, rng=rng
    )
    if low is None or high is None:
        row["reason"] = "bootstrap 重抽样退化，无法形成有效置信区间"
        return row
    row.update(
        effect=effect,
        mean_difference=row["raw_mean_delta_positive_minus_normal"],
        ci_low=low,
        ci_high=high,
        bootstrap_valid_replicates=valid,
        bootstrap_attempts=attempts,
        cdf_overlap=_ks_cdf_overlap(normal, positive),
        common_language_positive_gt_normal=_common_language(positive, normal),
        status="available",
    )
    return row


def _matched_control_contrast(
    signature: tuple[Any, ...],
    records: list[dict[str, Any]],
    *,
    config: ContrastConfig,
    rng: np.random.Generator,
    selection_source: str,
    confirmation_source: str | None,
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    excluded_unknown = 0
    unknown_counts = {field: 0 for field in config.matching_fields}
    for record in records:
        values = tuple(record[field] for field in config.matching_fields)
        unknown = [
            field
            for field, value in zip(config.matching_fields, values, strict=True)
            if value == "unknown"
        ]
        if unknown:
            excluded_unknown += 1
            for field in unknown:
                unknown_counts[field] += 1
            continue
        groups[values][record["weak_label"]].append(record["video_statistic_value"])
    complete = [(key, values) for key, values in groups.items() if values.get(0) and values.get(1)]
    row = _base_row(
        signature,
        contrast_id="matched_control",
        group_definition=(
            f"exact {' × '.join(config.matching_fields)} matched control; "
            "equal weight per complete group"
        ),
        stratum="×".join(config.matching_fields),
        resampling_unit="matched_group",
        matching_fields=list(config.matching_fields),
        num_independent_videos=sum(len(values[0]) + len(values[1]) for _, values in complete),
        num_positive_videos=sum(len(values[1]) for _, values in complete),
        num_normal_videos=sum(len(values[0]) for _, values in complete),
        num_matched_groups=len(complete),
        excluded_unknown_nuisance_videos=excluded_unknown,
        effect_name="matched_group_mean_difference_positive_minus_normal",
        effect=None,
        mean_difference=None,
        ci_low=None,
        ci_high=None,
        bootstrap_valid_replicates=0,
        bootstrap_attempts=0,
        cdf_overlap=None,
        common_language_positive_gt_normal=None,
        selection_source=selection_source,
        confirmation_source=confirmation_source,
        status="unavailable",
        reason=None,
    )
    for field, count in unknown_counts.items():
        row[f"unknown_{field}_videos"] = count
    if len(complete) < 2:
        row["reason"] = "至少需要两个完整且非 unknown 的场景×运动匹配组"
        return row
    differences = np.asarray(
        [float(np.mean(values[1]) - np.mean(values[0])) for _, values in complete], dtype=float
    )
    boot = [
        float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
        for _ in range(config.bootstrap_replicates)
    ]
    low, high = _ci(boot)
    row.update(
        effect=float(np.mean(differences)),
        mean_difference=float(np.mean(differences)),
        ci_low=low,
        ci_high=high,
        bootstrap_valid_replicates=len(boot),
        bootstrap_attempts=len(boot),
        status="available",
    )
    return row


def _empty_unavailable(
    signature: tuple[Any, ...], *, selection_source: str, confirmation_source: str | None
) -> dict[str, Any]:
    return _base_row(
        signature,
        contrast_id="weak_video",
        group_definition="weak-video: V+ minus V−",
        stratum="all",
        resampling_unit="video",
        num_independent_videos=0,
        num_positive_videos=0,
        num_normal_videos=0,
        effect_name="hedges_g_positive_minus_normal",
        effect=None,
        mean_difference=None,
        positive_raw_mean=None,
        normal_raw_mean=None,
        positive_raw_std=None,
        normal_raw_std=None,
        raw_mean_delta_positive_minus_normal=None,
        normalization_caution="not_detected",
        ci_low=None,
        ci_high=None,
        bootstrap_valid_replicates=0,
        bootstrap_attempts=0,
        cdf_overlap=None,
        common_language_positive_gt_normal=None,
        selection_source=selection_source,
        confirmation_source=confirmation_source,
        status="unavailable",
        reason="没有同时具备 available 数值和有效弱视频标签的独立视频",
    )


def analyze_contrasts(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: ContrastConfig | None = None,
    candidate_definition: Mapping[str, Any] | None = None,
) -> ContrastAnalysis:
    """Compute C1, category strata, and C3 without selecting new candidates.

    ``phase='confirm'`` only evaluates exact probe signatures named in a
    frozen definition.  Direction is checked after the one pre-specified
    contrast is calculated; it is never used to invert or choose a statistic.
    """

    config = ContrastConfig() if config is None else config
    materialized = tuple(dict(row) for row in rows)
    candidates = validate_candidate_definition(
        candidate_definition, require_direction=config.phase in {"confirm", "select"}
    )
    all_signatures = {_signature(row) for row in materialized}
    selected_signatures = (
        {
            signature
            for signature in all_signatures
            if any(_matches_candidate(signature, item) for item in candidates)
        }
        if config.phase in {"confirm", "select"}
        else all_signatures
    )
    video_rows = aggregate_probe_rows(materialized)
    records_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in video_rows:
        signature = tuple(record[name] for name in SIGNATURE_FIELDS)
        if signature in selected_signatures and record["weak_label"] is not None:
            records_by_signature[signature].append(record)

    rng = np.random.default_rng(config.seed)
    contrasts: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for signature in sorted(selected_signatures, key=str):
        candidate = next((item for item in candidates if _matches_candidate(signature, item)), None)
        selection_source = (
            "frozen_candidate_definition" if candidate else "unselected_exploration_grid"
        )
        confirmation_source = (
            candidate["candidate_id"]
            if config.phase in {"confirm", "select"} and candidate
            else None
        )
        records = records_by_signature.get(signature, [])
        positive = np.asarray(
            [record["video_statistic_value"] for record in records if record["weak_label"] == 1],
            dtype=float,
        )
        normal = np.asarray(
            [record["video_statistic_value"] for record in records if record["weak_label"] == 0],
            dtype=float,
        )
        if not len(records):
            primary = _empty_unavailable(
                signature,
                selection_source=selection_source,
                confirmation_source=confirmation_source,
            )
        else:
            primary = _weak_video_contrast(
                signature,
                positive,
                normal,
                contrast_id="weak_video",
                group_definition="weak-video: V+ minus V−",
                stratum="all",
                config=config,
                rng=rng,
                selection_source=selection_source,
                confirmation_source=confirmation_source,
            )
        if candidate and candidate["expected_direction"]:
            direction = candidate["expected_direction"]
            primary["expected_direction"] = direction
            primary["direction_matches_frozen"] = (
                None
                if primary["effect"] is None
                else (primary["effect"] > 0 if direction == "positive" else primary["effect"] < 0)
            )
        contrasts.append(primary)

        categories = sorted(
            {
                record["category_for_analysis_only"]
                for record in records
                if record["weak_label"] == 1
            }
        )
        for category in categories:
            category_positive = np.asarray(
                [
                    record["video_statistic_value"]
                    for record in records
                    if record["weak_label"] == 1
                    and record["category_for_analysis_only"] == category
                ],
                dtype=float,
            )
            contrasts.append(
                _weak_video_contrast(
                    signature,
                    category_positive,
                    normal,
                    contrast_id="category_stratified_weak_video",
                    group_definition="weak-video category stratum: V+ category minus all V−",
                    stratum=category,
                    config=config,
                    rng=rng,
                    selection_source=selection_source,
                    confirmation_source=confirmation_source,
                )
            )
        contrasts.append(
            _matched_control_contrast(
                signature,
                records,
                config=config,
                rng=rng,
                selection_source=selection_source,
                confirmation_source=confirmation_source,
            )
        )
        for field in (
            "category_for_analysis_only",
            "scene_group",
            "motion_bin",
            "brightness_bin",
            "duration_bin",
        ):
            for label in (0, 1):
                counts: dict[str, int] = defaultdict(int)
                for record in records:
                    if record["weak_label"] == label:
                        counts[record[field]] += 1
                for value, count in sorted(counts.items()):
                    composition.append(
                        {
                            **signature_dict(signature),
                            "field": field,
                            "weak_label": label,
                            "stratum": value,
                            "num_independent_videos": count,
                        }
                    )

    unique_video_metadata: dict[str, dict[str, Any]] = {}
    for record in video_rows:
        unique_video_metadata.setdefault(record["video_id"], record)
    unknown_coverage = {
        field: {
            "unknown_videos": sum(
                item[field] == "unknown" for item in unique_video_metadata.values()
            ),
            "total_videos": len(unique_video_metadata),
        }
        for field in config.matching_fields
    }
    receipt = {
        "phase": config.phase,
        "bootstrap_replicates_requested": config.bootstrap_replicates,
        "seed": config.seed,
        "resampling": {
            "weak_video": "independent videos resampled separately by weak label",
            "matched_control": "complete non-unknown matching-field groups resampled",
        },
        "matching_fields": list(config.matching_fields),
        "matching_unknown_coverage": unknown_coverage,
        "effect_definition": "Hedges g for weak-video/category contrasts, V+ minus V−",
        "overlap_definition": "cdf_overlap = 1 - two-sample empirical KS distance; common_language is P(V+ > V−) with ties split",
        "clip_reduction": "mean available repeated rows within clip, then mean clips within each video",
        "input_rows": len(materialized),
        "available_video_rows": len(video_rows),
        "available_signatures": len(all_signatures),
        "analyzed_signatures": len(selected_signatures),
        "candidate_ids": [item["candidate_id"] for item in candidates],
        "research_conclusions": None,
    }
    return ContrastAnalysis(
        video_rows=tuple(video_rows),
        contrast_rows=tuple(contrasts),
        composition_rows=tuple(composition),
        receipt=receipt,
    )


def render_candidate_cards(
    analysis: ContrastAnalysis, candidates: Iterable[Mapping[str, Any]]
) -> str:
    """Render at most two fact-only drafts; it intentionally makes no choice."""

    normalized = validate_candidate_definition(
        {"candidates": list(candidates)}, require_direction=False
    )
    cards: list[str] = ["# Probe candidate cards (draft; not research conclusions)", ""]
    for candidate in normalized:
        signature = tuple(candidate[name] for name in SIGNATURE_FIELDS)
        related = [
            row
            for row in analysis.contrast_rows
            if tuple(row[name] for name in SIGNATURE_FIELDS) == signature
        ]
        primary = next((row for row in related if row["contrast_id"] == "weak_video"), None)
        matched = next((row for row in related if row["contrast_id"] == "matched_control"), None)
        category_rows = [
            row for row in related if row["contrast_id"] == "category_stratified_weak_video"
        ]
        cards.extend(
            [
                f"## {candidate['candidate_id']} — draft",
                "",
                f"- Probe: `{candidate['encoder_id']}` / layer `{candidate['layer_index']}` / `{candidate['site']}` / `{candidate['probe_id']}` / `{candidate['statistic_name']}`.",
                f"- Frozen expected direction: `{candidate.get('expected_direction') or 'not set'}`. The tool did not choose this candidate or its direction.",
                f"- Weak-video result: status `{primary['status'] if primary else 'unavailable'}`; effect `{primary['effect'] if primary else None}`; CI `[{primary['ci_low'] if primary else None}, {primary['ci_high'] if primary else None}]`; n(V+) `{primary['num_positive_videos'] if primary else 0}`, n(V−) `{primary['num_normal_videos'] if primary else 0}`.",
                f"- Matched scene×motion control: status `{matched['status'] if matched else 'unavailable'}`; effect `{matched['effect'] if matched else None}`; complete groups `{matched.get('num_matched_groups') if matched else 0}`.",
            ]
        )
        opposite = [
            row["stratum"]
            for row in category_rows
            if row["status"] == "available"
            and primary
            and primary["effect"] is not None
            and row["effect"] * primary["effect"] < 0
        ]
        cards.append(
            "- Counterexamples: "
            + (
                f"category strata with opposite standardized direction: {', '.join(opposite)}."
                if opposite
                else "none established by the available category strata; this is not evidence of absence."
            )
        )
        cards.append(
            "- Limits: weak positive videos do not label every clip/token; category and nuisance fields are analysis-only; association has not passed the equal-budget decision or deployment gates."
        )
        cards.append("")
    return "\n".join(cards)
