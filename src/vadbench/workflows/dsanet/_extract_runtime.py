"""Stream raw videos through the frozen CLIP visual tower and PairSelect.

One process owns one GPU and one output root. Video decoding and author-style
cropping are distributed over a bounded number of CPU workers; each worker
emits small image batches so long videos never become huge tensors in RAM.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ._bridge import bridge_symbols


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def transform_image() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def crop_image(rgb: np.ndarray, crop_id: int) -> np.ndarray:
    resized = cv2.resize(rgb, (340, 256))
    top = 16 if crop_id in (0, 5) else (0 if crop_id in (1, 2, 6, 7) else 32)
    left = 58 if crop_id in (0, 5) else (0 if crop_id in (1, 3, 6, 8) else 116)
    crop = resized[top : top + 224, left : left + 224]
    if crop_id >= 5:
        crop = cv2.flip(crop, 1)
    return np.ascontiguousarray(crop)


class RawImages(IterableDataset):
    def __init__(self, videos: list[dict], batch_size: int) -> None:
        self.videos = videos
        self.batch_size = batch_size

    def __iter__(self):
        torch.set_num_threads(1)
        cv2.setNumThreads(0)
        worker = get_worker_info()
        videos = self.videos if worker is None else self.videos[worker.id :: worker.num_workers]
        preprocess = transform_image()
        for video in videos:
            capture = cv2.VideoCapture(video["raw_path"])
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video['raw_path']}")
            pending_images = []
            pending_rows = []
            pending_crops = []
            pending_sources = []
            frame_group = []
            source_count = 0
            row_index = 0

            def append_representative(frame: np.ndarray, source_index: int):
                nonlocal row_index
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for crop_id in video["crop_ids"]:
                    image = Image.fromarray(crop_image(rgb, crop_id))
                    pending_images.append(preprocess(image))
                    pending_rows.append(row_index)
                    pending_crops.append(crop_id)
                    pending_sources.append(source_index)
                row_index += 1

            def flush():
                if not pending_images:
                    return None
                batch = {
                    "kind": "images",
                    "video_id": video["video_id"],
                    "images": torch.stack(pending_images),
                    "rows": pending_rows.copy(),
                    "crops": pending_crops.copy(),
                    "source_indices": pending_sources.copy(),
                }
                pending_images.clear()
                pending_rows.clear()
                pending_crops.clear()
                pending_sources.clear()
                return batch

            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_group.append(frame)
                    source_count += 1
                    if len(frame_group) == 16:
                        append_representative(frame_group[7], source_count - 9)
                        frame_group.clear()
                        if len(pending_images) + len(video["crop_ids"]) > self.batch_size:
                            batch = flush()
                            if batch is not None:
                                yield batch
                if frame_group:
                    offset = (len(frame_group) - 1) // 2
                    append_representative(frame_group[offset], source_count - len(frame_group) + offset)
                batch = flush()
                if batch is not None:
                    yield batch
                if source_count <= 0:
                    raise RuntimeError(f"zero decoded frames: {video['raw_path']}")
                yield {
                    "kind": "end",
                    "video_id": video["video_id"],
                    "raw_frames": source_count,
                    "rows": row_index,
                }
            finally:
                capture.release()


def write_video(root: Path, video: dict, chunks: list[tuple], end: dict, identity: dict) -> None:
    if len(chunks) != end["rows"] * len(video["crop_ids"]):
        raise RuntimeError(f"missing feature rows for {video['video_id']}")
    crops = defaultdict(list)
    for crop_id, row_index, source_index, feature in chunks:
        crops[crop_id].append((row_index, source_index, feature))
    output_files = {}
    for crop_id in video["crop_ids"]:
        entries = sorted(crops[crop_id], key=lambda item: item[0])
        if [item[0] for item in entries] != list(range(end["rows"])):
            raise RuntimeError(f"noncontiguous rows: {video['video_id']} crop {crop_id}")
        array = np.stack([item[2] for item in entries]).astype(np.float32)
        if array.shape != (end["rows"], 512) or not np.isfinite(array).all():
            raise RuntimeError(f"invalid feature matrix: {video['video_id']} crop {crop_id}")
        destination = root / "features" / f"{video['video_id']}__{crop_id}.npy"
        if destination.exists():
            raise FileExistsError(f"incomplete existing video attempt must be inspected: {destination}")
        temp = destination.with_suffix(".npy.part")
        with temp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temp, destination)
        output_files[str(crop_id)] = {"path": str(destination), "sha256": sha256(destination)}
    receipt = {
        "video_id": video["video_id"],
        "raw_path": video["raw_path"],
        "raw_frames": end["raw_frames"],
        "feature_rows": end["rows"],
        "source_frame_indices": [item[1] for item in sorted(crops[video["crop_ids"][0]])],
        "crop_files": output_files,
        "identity": identity,
    }
    destination = root / "receipts" / f"{video['video_id']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    temp = destination.with_suffix(".json.part")
    temp.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, destination)


def run(options) -> None:
    if options.batch_size < 10 or options.workers < 1:
        raise ValueError("batch-size must be >=10 and workers >=1")
    videos = [json.loads(line) for line in options.manifest.read_text(encoding="utf-8").splitlines()]
    roles = {video["role"] for video in videos}
    if len(roles) != 1:
        raise ValueError("one extraction run must have a single data role")
    if options.limit_train_videos:
        if roles != {"train"}:
            raise ValueError("limit-train-videos may only be used on the train manifest")
        videos = videos[: options.limit_train_videos]
    if not 0 <= options.shard_index < options.shards:
        raise ValueError("shard-index must be in [0, shards)")
    if options.shards != 1:
        if roles != {"train"}:
            raise ValueError("training sharding is not available for official test")
        videos = videos[options.shard_index :: options.shards]
    if len({video["video_id"] for video in videos}) != len(videos):
        raise ValueError("duplicate video ID in source manifest")
    output = options.output
    ClipVisionBridge, load_openai_clip, bridge_path = bridge_symbols(options.bridge)
    identity = {
        "manifest_sha256": sha256(options.manifest),
        "bridge_sha256": sha256(bridge_path),
        "weights_sha256": sha256(options.weights),
        "keep_ratio": options.keep_ratio,
        "batch_size": options.batch_size,
        "workers": options.workers,
        "shard_index": options.shard_index,
        "shards": options.shards,
        "source_group": "contiguous_16_stride_16_center7_tail_lower_median",
        "crop": "vadclip_resize_340x256_then_224_author_crop",
        "dtype": "float32",
    }
    if options.resume:
        existing = json.loads((output / "resolved.json").read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("resume identity differs from original extraction")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "features").mkdir()
        (output / "receipts").mkdir()
        (output / "resolved.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    target_count = len(videos)
    prior_completed = 0
    prior_images = 0
    if options.resume:
        pending = []
        for video in videos:
            receipt_path = output / "receipts" / f"{video['video_id']}.json"
            if not receipt_path.exists():
                pending.append(video)
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["identity"] != identity or receipt["raw_path"] != video["raw_path"]:
                raise ValueError(f"resume video identity differs: {video['video_id']}")
            for item in receipt["crop_files"].values():
                path = Path(item["path"])
                if not path.is_file() or sha256(path) != item["sha256"]:
                    raise ValueError(f"resume feature is absent or changed: {path}")
            prior_completed += 1
            prior_images += receipt["feature_rows"] * len(video["crop_ids"])
        videos = pending

    # Spawn decoder workers before initializing CUDA in the parent process.
    loader = DataLoader(
        RawImages(videos, options.batch_size),
        batch_size=None,
        num_workers=options.workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    iterator = iter(loader)
    torch.cuda.set_device(options.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda", options.gpu)
    model, _preprocess, _load_identity = load_openai_clip(
        options.clip_repo, options.weights, device=str(device)
    )
    model = model.float().eval()
    bridge = ClipVisionBridge(model)
    collected = defaultdict(list)
    by_id = {video["video_id"]: video for video in videos}
    started = time.perf_counter()
    completed = prior_completed
    image_count = prior_images
    with torch.no_grad():
        for item in iterator:
            video_id = item["video_id"]
            if item["kind"] == "end":
                write_video(output, by_id[video_id], collected.pop(video_id), item, identity)
                completed += 1
                print(f"completed={completed}/{target_count} video={video_id}", flush=True)
                continue
            images = item["images"].to(device, non_blocking=True)
            if options.keep_ratio is None:
                features = bridge.forward(images).pooled
            else:
                features = bridge.forward(images, capture_depth=6, keep_ratio=options.keep_ratio).pooled
            features_cpu = features.float().cpu().numpy()
            image_count += len(features_cpu)
            collected[video_id].extend(
                (crop_id, row_index, source_index, feature)
                for crop_id, row_index, source_index, feature in zip(
                    item["crops"], item["rows"], item["source_indices"], features_cpu
                )
            )
    summary = {
        "status": "completed" if completed == target_count else "incomplete",
        "videos": completed,
        "images": image_count,
        "elapsed_seconds": time.perf_counter() - started,
        "identity": identity,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if completed != target_count:
        raise RuntimeError(f"only {completed}/{target_count} videos completed")
    print(json.dumps(summary), flush=True)
