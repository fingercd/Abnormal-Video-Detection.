"""生成可再现的 64 帧监控风格冒烟视频（不作为精度样本）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_video(path: Path, *, frames: int = 64, fps: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV 无法创建 MP4 writer")
    try:
        for index in range(frames):
            image = np.full((height, width, 3), 32, dtype=np.uint8)
            cv2.line(image, (0, 180), (width, 180), (90, 90, 90), 2)
            cv2.rectangle(image, (20, 30), (135, 150), (55, 55, 55), -1)
            x = 5 + (index * 5) % 280
            cv2.rectangle(image, (x, 155), (x + 32, 176), (190, 190, 190), -1)
            person_x = 200 + int(18 * np.sin(index / 6.0))
            cv2.circle(image, (person_x, 130), 7, (150, 190, 210), -1)
            cv2.line(image, (person_x, 137), (person_x, 165), (150, 190, 210), 3)
            if 32 <= index < 40:
                # 短暂高变化区只用于检查时间链路和缓存，不代表真实异常标签。
                cv2.rectangle(image, (150, 75), (310, 118), (20, 20, 200), -1)
                cv2.putText(
                    image,
                    "SMOKE EVENT",
                    (160, 103),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            cv2.putText(
                image,
                f"frame {index:03d}",
                (10, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            writer.write(image)
    finally:
        writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()
    print(create_video(Path(args.output), frames=args.frames, fps=args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
