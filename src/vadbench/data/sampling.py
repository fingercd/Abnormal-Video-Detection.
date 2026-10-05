"""与编码器无关的确定性视频索引采样。

本模块只生成帧索引，不负责打开视频。32 段采样始终返回恰好 32 个逻辑段；当
视频短于段数或 clip 所需跨度时，采用末帧重复填充，保证输出张量形状稳定。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


class SamplingError(ValueError):
    """采样参数或帧区间非法。"""


@dataclass(frozen=True)
class Segment:
    """一个左闭右开的逻辑视频段。"""

    index: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise SamplingError("segment index 不能为负")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise SamplingError("segment 必须满足 0 <= start_frame < end_frame")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class FixedClipSample:
    """固定长度帧索引及有效位掩码。

    形状不足时 ``frame_indices`` 末尾重复最后一个真实帧，但对应
    ``valid_mask=False``。调用方必须只让有效位置参与损失、池化或统计。
    """

    frame_indices: tuple[int, ...]
    valid_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.frame_indices:
            raise SamplingError("frame_indices 不能为空")
        if len(self.frame_indices) != len(self.valid_mask):
            raise SamplingError("frame_indices 与 valid_mask 长度必须一致")
        if not self.valid_mask[0]:
            raise SamplingError("固定 clip 至少要有一个有效帧")

        padding_started = False
        previous_valid = -1
        last_valid = -1
        for index, is_valid in zip(self.frame_indices, self.valid_mask, strict=True):
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise SamplingError("frame_indices 必须是非负整数")
            if not isinstance(is_valid, bool):
                raise SamplingError("valid_mask 必须是 boolean")
            if is_valid:
                if padding_started:
                    raise SamplingError("valid_mask 的有效位置必须连续出现在前缀")
                if index <= previous_valid:
                    raise SamplingError("有效帧索引必须严格递增")
                previous_valid = index
                last_valid = index
            else:
                padding_started = True
                if index != last_valid:
                    raise SamplingError("padding 位置必须重复最后一个有效帧")

    @property
    def clip_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def valid_frames(self) -> int:
        return sum(self.valid_mask)

    @property
    def valid_frame_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, is_valid in zip(self.frame_indices, self.valid_mask, strict=True)
            if is_valid
        )


@dataclass(frozen=True)
class ClipSample:
    """某一逻辑段内的固定长度帧索引及 padding 掩码。"""

    segment_index: int
    segment_start_frame: int
    segment_end_frame: int
    frame_indices: tuple[int, ...]
    valid_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.segment_index < 0:
            raise SamplingError("segment_index 不能为负")
        if self.segment_start_frame < 0 or self.segment_end_frame <= self.segment_start_frame:
            raise SamplingError("segment 边界非法")
        if not self.frame_indices:
            raise SamplingError("frame_indices 不能为空")
        fixed = FixedClipSample(self.frame_indices, self.valid_mask)
        for index in fixed.frame_indices:
            if index < self.segment_start_frame or index >= self.segment_end_frame:
                raise SamplingError("clip 索引越过所属 segment")

    @property
    def clip_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def valid_frames(self) -> int:
        return sum(self.valid_mask)

    @property
    def valid_frame_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, is_valid in zip(self.frame_indices, self.valid_mask, strict=True)
            if is_valid
        )


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SamplingError(f"{name} 必须是正整数")
    return value


def uniform_segments(num_frames: int, num_segments: int = 32) -> tuple[Segment, ...]:
    """把视频均匀划分为固定数量的逻辑段。

    对足够长的视频，边界等价于整数 ``linspace(0, num_frames, N + 1)``。
    对短视频允许相邻段复用同一帧，但不会产生空段或越界索引。
    """

    num_frames = _positive_int(num_frames, "num_frames")
    num_segments = _positive_int(num_segments, "num_segments")
    segments: list[Segment] = []
    for index in range(num_segments):
        start = min((index * num_frames) // num_segments, num_frames - 1)
        raw_end = ((index + 1) * num_frames) // num_segments
        end = min(num_frames, max(start + 1, raw_end))
        segments.append(Segment(index=index, start_frame=start, end_frame=end))
    return tuple(segments)


def sample_fixed_clip(
    num_frames: int,
    *,
    clip_frames: int,
    frame_stride: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    position: Literal["start", "center", "random"] = "center",
    rng: random.Random | None = None,
) -> FixedClipSample:
    """在指定帧区间采固定长度 clip，并显式标出末帧 padding。"""

    num_frames = _positive_int(num_frames, "num_frames")
    clip_frames = _positive_int(clip_frames, "clip_frames")
    frame_stride = _positive_int(frame_stride, "frame_stride")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise SamplingError("start_frame 必须是整数")
    if end_frame is None:
        end_frame = num_frames
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise SamplingError("end_frame 必须是整数")
    if start_frame < 0 or end_frame <= start_frame or end_frame > num_frames:
        raise SamplingError("采样区间必须满足 0 <= start_frame < end_frame <= num_frames")
    if position not in {"start", "center", "random"}:
        raise SamplingError("position 必须是 start、center 或 random")

    required_span = (clip_frames - 1) * frame_stride + 1
    region_length = end_frame - start_frame
    if region_length >= required_span:
        latest_start = end_frame - required_span
        if position == "start":
            clip_start = start_frame
        elif position == "center":
            clip_start = start_frame + (latest_start - start_frame) // 2
        else:
            generator = rng if rng is not None else random.Random()
            clip_start = generator.randint(start_frame, latest_start)
    else:
        clip_start = start_frame

    indices: list[int] = []
    valid_mask: list[bool] = []
    last_valid = clip_start
    for offset in range(clip_frames):
        candidate = clip_start + offset * frame_stride
        is_valid = candidate < end_frame
        if is_valid:
            last_valid = candidate
        indices.append(last_valid)
        valid_mask.append(is_valid)
    return FixedClipSample(tuple(indices), tuple(valid_mask))


def sample_uniform_segment_clips(
    num_frames: int,
    *,
    num_segments: int = 32,
    clip_frames: int = 16,
    frame_stride: int = 2,
    position: Literal["start", "center", "random"] = "center",
    seed: int | None = None,
) -> tuple[ClipSample, ...]:
    """每个均匀段采一个固定 clip，典型弱监督设置为 32×16 帧。"""

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SamplingError("seed 必须是整数或 null")
    generator = random.Random(seed) if position == "random" else None
    samples: list[ClipSample] = []
    for segment in uniform_segments(num_frames, num_segments):
        fixed_clip = sample_fixed_clip(
            num_frames,
            clip_frames=clip_frames,
            frame_stride=frame_stride,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            position=position,
            rng=generator,
        )
        samples.append(
            ClipSample(
                segment_index=segment.index,
                segment_start_frame=segment.start_frame,
                segment_end_frame=segment.end_frame,
                frame_indices=fixed_clip.frame_indices,
                valid_mask=fixed_clip.valid_mask,
            )
        )
    return tuple(samples)


def sample_32_segments(
    num_frames: int,
    *,
    clip_frames: int = 16,
    frame_stride: int = 2,
    position: Literal["start", "center", "random"] = "center",
    seed: int | None = None,
) -> tuple[ClipSample, ...]:
    """UCF-Crime 32 段采样的便捷入口。"""

    return sample_uniform_segment_clips(
        num_frames,
        num_segments=32,
        clip_frames=clip_frames,
        frame_stride=frame_stride,
        position=position,
        seed=seed,
    )
