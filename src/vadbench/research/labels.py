"""Analysis-side label joins.  Collectors intentionally do not import this module."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .cohorts import CohortError, CohortIndex


def join_probe_rows(
    rows: Iterable[Mapping[str, Any]], cohort: CohortIndex
) -> tuple[dict[str, Any], ...]:
    """Join a probe export to its cohort sidecar by ``clip_id``.

    This function runs after collection.  It refuses rows without a stable clip
    identity rather than guessing from a path or a video filename.
    """

    labels = cohort.label_sidecar()
    joined: list[dict[str, Any]] = []
    for row in rows:
        clip_id = row.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            raise CohortError("probe 行必须含非空 clip_id，不能用文件名推断标签")
        if clip_id not in labels:
            raise CohortError(f"probe 行的 clip_id 不在 cohort：{clip_id}")
        if row.get("video_id") != labels[clip_id]["video_id"]:
            raise CohortError("probe video_id 与 clip sidecar 不一致")
        if row.get("partition") not in {None, labels[clip_id]["partition"]}:
            raise CohortError("probe partition 与 sidecar 不一致")
        result = dict(row)
        result.update(labels[clip_id])
        joined.append(result)
    return tuple(joined)
