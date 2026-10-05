"""可选 OpenCV 视频探测、稀疏解码与 :class:`ClipBatch` 构造。

OpenCV 不是核心依赖；即使环境没有 ``cv2``，导入本模块也不会失败。所有解码
路径按目标索引 seek 或有限近邻顺读，不会为了稀疏 clip 从头扫描长视频。同一次
batch 中重复的源帧（包括 padding）只解码一次。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..contracts import ClipBatch
from .manifest import VideoManifestRecord
from .sampling import ClipSample, FixedClipSample, sample_uniform_segment_clips

try:  # pragma: no cover - availability depends on the selected environment extra
    import cv2 as _cv2
except ImportError:  # pragma: no cover - tested by replacing the module-level backend
    _cv2 = None


class VideoIOError(RuntimeError):
    """视频元数据或解码结果不满足框架约束。"""


class OpenCVUnavailableError(VideoIOError):
    """调用了解码功能，但当前环境没有安装 OpenCV。"""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """由容器探测得到的最小视频元数据。"""

    path: Path
    num_frames: int
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        if (
            isinstance(self.num_frames, bool)
            or not isinstance(self.num_frames, Integral)
            or self.num_frames <= 0
        ):
            raise VideoIOError("num_frames 必须是正整数")
        object.__setattr__(self, "num_frames", int(self.num_frames))
        if isinstance(self.fps, bool):
            raise VideoIOError("fps 必须是有限正数")
        try:
            fps = float(self.fps)
        except (TypeError, ValueError) as exc:
            raise VideoIOError("fps 必须是有限正数") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise VideoIOError("fps 必须是有限正数")
        object.__setattr__(self, "fps", fps)
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, Integral)
            or not isinstance(self.height, Integral)
            or self.width <= 0
            or self.height <= 0
        ):
            raise VideoIOError("视频 width/height 必须大于 0")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps


def cv2_available() -> bool:
    """返回默认 OpenCV 后端是否可用。"""

    return _cv2 is not None


def _resolve_backend(backend: Any | None) -> Any:
    resolved = _cv2 if backend is None else backend
    if resolved is None:
        raise OpenCVUnavailableError("视频解码需要可选依赖 OpenCV；请安装项目的 video extra")
    required = (
        "VideoCapture",
        "CAP_PROP_FRAME_COUNT",
        "CAP_PROP_FPS",
        "CAP_PROP_FRAME_WIDTH",
        "CAP_PROP_FRAME_HEIGHT",
        "CAP_PROP_POS_FRAMES",
    )
    missing = [name for name in required if not hasattr(resolved, name)]
    if missing:
        raise VideoIOError(f"OpenCV backend 缺少属性：{missing}")
    return resolved


def _positive_capture_int(value: Any, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoIOError(f"OpenCV 返回的 {name} 不是数值") from exc
    if not math.isfinite(number):
        raise VideoIOError(f"OpenCV 返回的 {name} 不是有限数值")
    rounded = int(round(number))
    if rounded <= 0:
        raise VideoIOError(f"OpenCV 返回的 {name} 必须大于 0，实际为 {value!r}")
    return rounded


class OpenCVVideoReader:
    """单视频稀疏 reader；上下文退出时保证释放 capture。"""

    def __init__(self, path: str | Path, *, backend: Any | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.backend = _resolve_backend(backend)
        self._capture: Any | None = None
        self._info: VideoInfo | None = None

    def __enter__(self) -> OpenCVVideoReader:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            raise VideoIOError("reader 尚未打开")
        return self._info

    def open(self) -> None:
        if self.is_open:
            return
        if not self.path.is_file():
            raise VideoIOError(f"视频文件不存在：{self.path}")
        capture = self.backend.VideoCapture(str(self.path))
        if capture is None or not bool(capture.isOpened()):
            if capture is not None:
                capture.release()
            raise VideoIOError(f"OpenCV 无法打开视频：{self.path}")
        try:
            num_frames = _positive_capture_int(
                capture.get(self.backend.CAP_PROP_FRAME_COUNT), "frame_count"
            )
            fps_value = float(capture.get(self.backend.CAP_PROP_FPS))
            if not math.isfinite(fps_value) or fps_value <= 0:
                raise VideoIOError(f"OpenCV 返回非法 fps={fps_value!r}")
            width = _positive_capture_int(
                capture.get(self.backend.CAP_PROP_FRAME_WIDTH), "frame_width"
            )
            height = _positive_capture_int(
                capture.get(self.backend.CAP_PROP_FRAME_HEIGHT), "frame_height"
            )
            info = VideoInfo(
                path=self.path,
                num_frames=num_frames,
                fps=fps_value,
                width=width,
                height=height,
            )
        except Exception:
            capture.release()
            raise
        self._capture = capture
        self._info = info

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._info = None

    def _read_rgb_frame(self, capture: Any, index: int) -> np.ndarray:
        ok, bgr = capture.read()
        if not ok or bgr is None:
            raise VideoIOError(f"OpenCV 解码 frame={index} 失败：{self.path}")
        array = np.asarray(bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise VideoIOError(f"frame={index} 必须是 HWC 三通道，实际 shape={array.shape}")
        if array.shape[:2] != (self.info.height, self.info.width):
            raise VideoIOError(
                f"frame={index} 尺寸 {array.shape[:2]} 与 probe "
                f"{(self.info.height, self.info.width)} 不一致"
            )
        if array.dtype != np.uint8:
            if array.dtype.kind not in "biuf" or np.any(array < 0) or np.any(array > 255):
                raise VideoIOError(f"frame={index} 无法安全转换为 uint8")
            array = array.astype(np.uint8)
        # OpenCV 解码是 BGR；copy 同时消除反向 stride，便于后续堆叠/送入 torch。
        return np.ascontiguousarray(array[..., ::-1])

    def _seek_and_read(self, capture: Any, index: int) -> np.ndarray:
        seek_result = capture.set(self.backend.CAP_PROP_POS_FRAMES, float(index))
        if seek_result is not None and not bool(seek_result):
            raise VideoIOError(f"OpenCV 无法 seek 到 frame={index}")
        return self._read_rgb_frame(capture, index)

    def read_indices(self, frame_indices: Sequence[int]) -> np.ndarray:
        """稀疏读取并按请求顺序返回 RGB uint8 ``[T,H,W,3]``。

        索引会先去重后排序。每次调用的首目标和长跳跃均 seek；相邻目标最多前进两
        帧时顺读并丢弃中间帧。最终恢复调用方的原始顺序和重复 padding。
        """

        if not self.is_open:
            raise VideoIOError("reader 尚未打开")
        raw_requested = tuple(frame_indices)
        if not raw_requested:
            raise VideoIOError("frame_indices 不能为空")
        requested: list[int] = []
        for index in raw_requested:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise VideoIOError("frame_indices 必须是整数")
            index = int(index)
            if index < 0 or index >= self.info.num_frames:
                raise VideoIOError(f"帧索引 {index} 越界，有效范围为 [0, {self.info.num_frames})")
            requested.append(index)

        unique_indices = sorted(set(requested))
        decoded: dict[int, np.ndarray] = {}
        capture = self._capture
        assert capture is not None  # narrowed by is_open
        cursor: int | None = None
        for index in unique_indices:
            if cursor is None or index - cursor > 2:
                decoded[index] = self._seek_and_read(capture, index)
                cursor = index + 1
                continue
            try:
                while cursor < index:
                    self._read_rgb_frame(capture, cursor)
                    cursor += 1
                decoded[index] = self._read_rgb_frame(capture, index)
            except VideoIOError:
                # An unread intermediate frame must not become a new dependency.
                decoded[index] = self._seek_and_read(capture, index)
            cursor = index + 1

        return np.stack([decoded[index] for index in requested], axis=0)


def probe_video(path: str | Path, *, backend: Any | None = None) -> VideoInfo:
    """探测帧数、fps 和空间尺寸，不解码全视频。"""

    with OpenCVVideoReader(path, backend=backend) as reader:
        return reader.info


def decode_rgb_frames(
    path: str | Path,
    frame_indices: Sequence[int],
    *,
    backend: Any | None = None,
) -> np.ndarray:
    """按索引稀疏解码 RGB uint8 帧，同一调用内自动去重。"""

    with OpenCVVideoReader(path, backend=backend) as reader:
        return reader.read_indices(frame_indices)


def _sample_arrays(
    samples: Sequence[FixedClipSample | ClipSample],
) -> tuple[np.ndarray, np.ndarray]:
    items = tuple(samples)
    if not items:
        raise VideoIOError("samples 不能为空")
    clip_frames = items[0].clip_frames
    if any(item.clip_frames != clip_frames for item in items):
        raise VideoIOError("同一 ClipBatch 中所有 sample 的 clip_frames 必须一致")
    indices = np.asarray([item.frame_indices for item in items], dtype=np.int64)
    valid_mask = np.asarray([item.valid_mask for item in items], dtype=bool)
    return indices, valid_mask


def _batch_from_reader(
    reader: OpenCVVideoReader,
    video_id: str,
    samples: Sequence[FixedClipSample | ClipSample],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ClipBatch:
    if not video_id:
        raise VideoIOError("video_id 不能为空")
    indices, valid_mask = _sample_arrays(samples)
    flat_indices = tuple(int(index) for index in indices.reshape(-1))
    flat_frames = reader.read_indices(flat_indices)
    batch, steps = indices.shape
    frames = flat_frames.reshape(
        batch,
        steps,
        reader.info.height,
        reader.info.width,
        3,
    )
    timestamps_s = indices.astype(np.float64) / reader.info.fps
    batch_metadata = dict(metadata or {})
    batch_metadata.update(
        {
            "video_path": str(reader.info.path),
            "source_num_frames": reader.info.num_frames,
            "source_fps": reader.info.fps,
        }
    )
    return ClipBatch(
        frames=frames,
        timestamps_s=timestamps_s,
        video_ids=(video_id,) * batch,
        valid_mask=valid_mask,
        frame_indices=indices,
        metadata=batch_metadata,
    )


def build_clip_batch(
    path: str | Path,
    video_id: str,
    samples: Sequence[FixedClipSample | ClipSample],
    *,
    backend: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ClipBatch:
    """解码一组同视频 samples，构造 BTHWC RGB ``ClipBatch``。"""

    with OpenCVVideoReader(path, backend=backend) as reader:
        return _batch_from_reader(reader, video_id, samples, metadata=metadata)


def _validate_record_info(
    record: VideoManifestRecord,
    info: VideoInfo,
    *,
    strict_manifest_info: bool,
) -> None:
    if not strict_manifest_info:
        return
    if record.num_frames is not None and record.num_frames != info.num_frames:
        raise VideoIOError(
            f"{record.video_id}: manifest num_frames={record.num_frames}，probe={info.num_frames}"
        )
    if record.fps is not None and not math.isclose(record.fps, info.fps, rel_tol=1e-3):
        raise VideoIOError(f"{record.video_id}: manifest fps={record.fps}，probe={info.fps}")


def iter_fixed_segment_batches(
    records: Iterable[VideoManifestRecord],
    dataset_root: str | Path,
    *,
    num_segments: int = 32,
    clip_frames: int = 16,
    frame_stride: int = 2,
    position: Literal["start", "center", "random"] = "center",
    seed: int | None = None,
    strict_manifest_info: bool = True,
    backend: Any | None = None,
) -> Iterator[ClipBatch]:
    """逐视频生成固定段 batch；默认每个 batch 是一个 32-clip MIL bag。"""

    for record in records:
        if not isinstance(record, VideoManifestRecord):
            raise VideoIOError("records 必须由 VideoManifestRecord 组成")
        path = record.resolve_path(dataset_root)
        with OpenCVVideoReader(path, backend=backend) as reader:
            _validate_record_info(
                record,
                reader.info,
                strict_manifest_info=strict_manifest_info,
            )
            samples = sample_uniform_segment_clips(
                reader.info.num_frames,
                num_segments=num_segments,
                clip_frames=clip_frames,
                frame_stride=frame_stride,
                position=position,
                seed=seed,
            )
            metadata = {
                "clip_ids": [
                    f"{record.video_id}:segment-{sample.segment_index:02d}" for sample in samples
                ],
                "clip_indices": [sample.segment_index for sample in samples],
                "segment_start_frames": [sample.segment_start_frame for sample in samples],
                "segment_end_frames": [sample.segment_end_frame for sample in samples],
                "sampling_kind": "uniform_segments",
                "split": record.split.value,
                "category": record.category,
                "is_anomaly": record.is_anomaly,
            }
            yield _batch_from_reader(
                reader,
                record.video_id,
                samples,
                metadata=metadata,
            )


def _stream_stride(info: VideoInfo, *, frame_stride: int | None, sample_fps: float | None) -> int:
    if frame_stride is not None and (
        isinstance(frame_stride, bool)
        or not isinstance(frame_stride, Integral)
        or frame_stride <= 0
    ):
        raise VideoIOError("frame_stride 必须是正整数或 null")
    if sample_fps is not None:
        if isinstance(sample_fps, bool):
            raise VideoIOError("sample_fps 必须是有限正数或 null")
        try:
            requested_fps = float(sample_fps)
        except (TypeError, ValueError) as exc:
            raise VideoIOError("sample_fps 必须是有限正数或 null") from exc
        if not math.isfinite(requested_fps) or requested_fps <= 0:
            raise VideoIOError("sample_fps 必须是有限正数或 null")
        if frame_stride is not None:
            raise VideoIOError("frame_stride 与 sample_fps 只能指定一个")
        return max(1, int(round(info.fps / requested_fps)))
    return int(frame_stride) if frame_stride is not None else 1


def iter_streaming_chunk_batches(
    records: Iterable[VideoManifestRecord],
    dataset_root: str | Path,
    *,
    chunk_frames: int = 16,
    frame_stride: int | None = None,
    sample_fps: float | None = None,
    start_frame: int = 0,
    drop_last: bool = False,
    strict_manifest_info: bool = True,
    backend: Any | None = None,
) -> Iterator[ClipBatch]:
    """按时间顺序逐视频生成 B=1 的 streaming chunk batch。

    最后一个不足 ``chunk_frames`` 的 chunk 默认重复最后帧并用 ``valid_mask=False``
    标出 padding；``drop_last=True`` 可选择丢弃该 chunk。
    """

    if (
        isinstance(chunk_frames, bool)
        or not isinstance(chunk_frames, Integral)
        or chunk_frames <= 0
    ):
        raise VideoIOError("chunk_frames 必须是正整数")
    if isinstance(start_frame, bool) or not isinstance(start_frame, Integral) or start_frame < 0:
        raise VideoIOError("start_frame 必须是非负整数")
    chunk_frames = int(chunk_frames)
    start_frame = int(start_frame)

    for record in records:
        if not isinstance(record, VideoManifestRecord):
            raise VideoIOError("records 必须由 VideoManifestRecord 组成")
        path = record.resolve_path(dataset_root)
        with OpenCVVideoReader(path, backend=backend) as reader:
            _validate_record_info(
                record,
                reader.info,
                strict_manifest_info=strict_manifest_info,
            )
            if start_frame >= reader.info.num_frames:
                raise VideoIOError(f"{record.video_id}: start_frame={start_frame} 越过视频长度")
            stride = _stream_stride(
                reader.info,
                frame_stride=frame_stride,
                sample_fps=sample_fps,
            )
            chunk_index = 0
            chunk_start = start_frame
            while chunk_start < reader.info.num_frames:
                stop = min(reader.info.num_frames, chunk_start + chunk_frames * stride)
                valid_indices = tuple(range(chunk_start, stop, stride))
                if not valid_indices:
                    break
                if drop_last and len(valid_indices) < chunk_frames:
                    break
                padding = chunk_frames - len(valid_indices)
                indices = (*valid_indices, *((valid_indices[-1],) * padding))
                valid_mask = (*((True,) * len(valid_indices)), *((False,) * padding))
                sample = FixedClipSample(indices, valid_mask)
                metadata = {
                    "clip_ids": [f"{record.video_id}:chunk-{chunk_index:06d}"],
                    "clip_indices": [chunk_index],
                    "sampling_kind": "chronological_stream",
                    "sampling_stride": stride,
                    "chunk_source_start": valid_indices[0],
                    "chunk_source_end": valid_indices[-1] + 1,
                    "is_last_chunk": stop >= reader.info.num_frames,
                    "split": record.split.value,
                    "category": record.category,
                    "is_anomaly": record.is_anomaly,
                }
                yield _batch_from_reader(
                    reader,
                    record.video_id,
                    (sample,),
                    metadata=metadata,
                )
                chunk_index += 1
                chunk_start += chunk_frames * stride
