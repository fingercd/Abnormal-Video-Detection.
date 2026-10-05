"""Run a fit-only raw-video OpenAI CLIP visual-tower probe.

This is an engineering probe, not a detector benchmark.  It consumes only
official training videos and video-level labels, records native preprocessing,
checks a read-only intermediate hook, and performs one deterministic local
patch intervention at a verified CLIP block.  It never opens an official test
manifest and never consumes pre-extracted detector features.

Typical remote invocation (after a GPU lease is assigned)::

    PYTHONPATH=/users/fotile/VAD/src python scripts/icassp2027/probe_clip_train.py \
      --dataset ucf --manifest data/splits/ucf_crime/Anomaly_Train.txt \
      --dataset-root /users/fotile/datasets/UCF-Crime-official-verified \
      --clip-repo /data2/localdisk/fotile-icassp2027-kimi-20260919-a01/clip-assets/openai-clip/repo/source \
      --weights /data2/localdisk/fotile-icassp2027-kimi-20260919-a01/clip-assets/openai-clip/weights/ViT-B-16.pt \
      --max-videos 4 --windows-per-video 2 --frames-per-window 8 --frame-stride 2 \
      --output outputs/icassp2027/clip/probe-ucf-fit.json

For XD, omit ``--manifest`` and point ``--dataset-root`` to the verified raw
training tree.  The script enumerates only ``train_*`` directories and infers
the weak video label from the official ``_label_A`` filename convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load_bridge_module() -> Any:
    """Load the bridge directly so the Python 3.9 CLIP env need not import VADBench."""

    script_path = Path(__file__).resolve()
    candidates = [
        parent / "src/vadbench/token_reduction/bridges/clip.py"
        for parent in (script_path.parent, *script_path.parents)
    ]
    candidates.append(Path("/tmp/vadbench_clip_bridge_20260920.py"))
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError("找不到新增 CLIP bridge 源文件")
    spec = importlib.util.spec_from_file_location("vadbench_clip_bridge_runtime", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 CLIP bridge：{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BRIDGE = _load_bridge_module()
ClipBridgeError = _BRIDGE.ClipBridgeError
ClipVisionBridge = _BRIDGE.ClipVisionBridge
fixed_local_keep_indices = _BRIDGE.fixed_local_keep_indices
load_openai_clip = _BRIDGE.load_openai_clip


@dataclass(frozen=True)
class FitVideo:
    video_id: str
    path: Path
    is_anomaly: bool
    source: str


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(path: Path) -> str:
    return _sha256_file(path) if path.is_file() else "missing"


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative.replace("\\", "/"))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"视频路径逃逸 dataset root：{relative!r}") from exc
    return candidate


def _read_jsonl_sources(manifest: Path, dataset_root: Path) -> list[FitVideo]:
    rows: list[FitVideo] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_number}: 非法 JSONL") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{manifest}:{line_number}: manifest row 必须是 object")
            split = str(record.get("split", "train")).lower()
            if split not in {"train", "fit"}:
                raise ValueError(
                    f"CLIP fit probe 拒绝非训练 split={split!r}，禁止把 test/confirm 混入训练侧"
                )
            video_id = str(record.get("video_id", "")).strip()
            relative = str(record.get("path", "")).strip()
            if not video_id or not relative:
                raise ValueError(f"{manifest}:{line_number}: 缺少 video_id/path")
            path = _resolve_under(dataset_root, relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            weak_label = record.get("is_anomaly")
            if type(weak_label) is not bool:
                raise ValueError(f"{manifest}:{line_number}: fit row 缺少 bool video-level is_anomaly")
            rows.append(FitVideo(video_id, path, weak_label, "train-manifest"))
    return rows


def _read_split_sources(manifest: Path, dataset_root: Path) -> list[FitVideo]:
    """Read the official UCF-style train split when JSONL is unavailable."""

    rows: list[FitVideo] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().split()
            if not fields:
                continue
            relative = fields[0]
            path = _resolve_under(dataset_root, relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            video_id = path.stem
            category = path.parts[-2].lower() if len(path.parts) >= 2 else ""
            is_anomaly = category != "normal"
            rows.append(FitVideo(video_id, path, is_anomaly, "official-train-split"))
    if not rows:
        raise ValueError(f"训练 split 为空：{manifest}")
    return rows


def _read_fit_sources(manifest: Path, dataset_root: Path) -> list[FitVideo]:
    if manifest.suffix.lower() in {".jsonl", ".json"}:
        return _read_jsonl_sources(manifest, dataset_root)
    rows = _read_split_sources(manifest, dataset_root)
    # The official UCF ``Anomaly_Train.txt`` records only anomalous training
    # classes.  Normal training videos live in the separately named
    # ``Training_Normal_Videos_Anomaly`` directory and are still fit videos;
    # append them without reading the test directory.
    if not any(not row.is_anomaly for row in rows):
        normal_root = dataset_root / "Training_Normal_Videos_Anomaly"
        for path in sorted(normal_root.rglob("*.mp4")) if normal_root.is_dir() else ():
            rows.append(FitVideo(path.stem, path, False, "official-train-normal-tree"))
    return rows


def _xd_sources(dataset_root: Path) -> list[FitVideo]:
    root = dataset_root.resolve()
    train_files = [
        path
        for directory in sorted(root.glob("train_*"))
        if directory.is_dir()
        for path in sorted(directory.rglob("*.mp4"))
    ]
    rows: list[FitVideo] = []
    for path in train_files:
        # The official XD weak train convention uses ``_label_A`` for normal
        # clips.  Every other train member is the official weak abnormal side.
        is_anomaly = "_label_A" not in path.stem
        rows.append(FitVideo(path.stem, path, is_anomaly, "xd-official-train-tree"))
    if not rows:
        raise FileNotFoundError(f"dataset root 没有 train_*/*.mp4：{root}")
    return rows


def _balanced_fit_sources(rows: Iterable[FitVideo], max_videos: int) -> list[FitVideo]:
    if type(max_videos) is not int or max_videos <= 0:
        raise ValueError("max_videos 必须是正整数")
    ordered = list(rows)
    normal = [row for row in ordered if not row.is_anomaly]
    anomaly = [row for row in ordered if row.is_anomaly]
    if not normal or not anomaly:
        raise ValueError("fit-only CLIP probe 需要至少一个正常和一个异常训练视频")
    per_side = max(1, max_videos // 2)
    chosen = normal[:per_side] + anomaly[:per_side]
    if len(chosen) < max_videos:
        used = {row.video_id for row in chosen}
        chosen.extend(row for row in ordered if row.video_id not in used)
        chosen = chosen[:max_videos]
    if len(chosen) < 2:
        raise ValueError("fit-only CLIP probe 至少需要两个视频")
    return chosen


def _select_fit_sources(
    rows: Iterable[FitVideo], max_videos: int, ids_file: Path | None
) -> tuple[list[FitVideo], list[float] | None]:
    if ids_file is None:
        return _balanced_fit_sources(rows, max_videos), None
    payload = json.loads(ids_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("video_ids"), list):
        raise ValueError(f"ids file 缺少 video_ids 数组：{ids_file}")
    ids = [str(value) for value in payload["video_ids"]]
    by_id = {row.video_id: row for row in rows}
    missing = [video_id for video_id in ids if video_id not in by_id]
    if missing:
        raise ValueError(f"ids file 中视频不在 fit manifest：{missing}")
    centers = payload.get("window_center_fractions")
    if centers is not None:
        if not isinstance(centers, list) or not centers or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in centers
        ):
            raise ValueError("window_center_fractions 必须是 [0,1] 内有限数值数组")
        centers = [float(value) for value in centers]
    return [by_id[video_id] for video_id in ids], centers


def _windows(
    path: Path,
    *,
    windows_per_video: int,
    frames_per_window: int,
    frame_stride: int,
    center_fractions: list[float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if min(windows_per_video, frames_per_window, frame_stride) <= 0:
        raise ValueError("window 参数必须是正整数")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime-specific dependency
        raise RuntimeError("raw-video CLIP probe 需要 OpenCV") from exc
    capture = cv2.VideoCapture(str(path))
    if capture is None or not bool(capture.isOpened()):
        raise RuntimeError(f"OpenCV 无法打开视频：{path}")
    try:
        num_frames = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if num_frames <= 0 or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"视频元数据非法：frames={num_frames}, fps={fps}, size={width}x{height}")
        span = max(1, (frames_per_window - 1) * frame_stride + 1)
        max_start = max(0, num_frames - span)
        if center_fractions is None:
            centers = np.linspace(0, 1, windows_per_video).tolist()
        else:
            if len(center_fractions) != windows_per_video:
                raise ValueError("ids file 的 window_center_fractions 长度必须等于 windows_per_video")
            centers = center_fractions
        starts = np.rint(np.asarray(centers, dtype=np.float64) * max_start).astype(np.int64)
        frames: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for window_index, start in enumerate(starts.tolist()):
            requested = [min(num_frames - 1, int(start) + i * frame_stride) for i in range(frames_per_window)]
            valid = [int(start) + i * frame_stride < num_frames for i in range(frames_per_window)]
            decoded_by_index: dict[int, np.ndarray] = {}
            for index in sorted(set(requested)):
                if not bool(capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))):
                    raise RuntimeError(f"OpenCV 无法 seek 到 frame={index}: {path}")
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"OpenCV 解码 frame={index} 失败：{path}")
                decoded_by_index[index] = np.ascontiguousarray(np.asarray(bgr)[..., ::-1])
            frames.extend(decoded_by_index[index] for index in requested)
            metadata.append(
                {
                    "window_index": window_index,
                    "source_frame_indices": requested,
                    "valid_mask": valid,
                    "start_frame": int(requested[0]),
                    "end_frame_exclusive": int(requested[-1]) + 1,
                    "fps": fps,
                    "num_frames": num_frames,
                }
            )
    finally:
        capture.release()
    return frames, metadata, {"fps": fps, "num_frames": num_frames, "height": height, "width": width}


def _preprocess_frames(preprocess: Any, frames: list[np.ndarray], device: Any) -> Any:
    from PIL import Image
    import torch

    tensors = [preprocess(Image.fromarray(frame, mode="RGB")) for frame in frames]
    if not tensors or any(tuple(item.shape) != tuple(tensors[0].shape) for item in tensors):
        raise ValueError("CLIP native preprocess 输出 shape 不一致")
    return torch.stack(tensors, dim=0).to(device)


def _max_abs(left: Any, right: Any) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def _f06_attention_metrics(run: Any) -> dict[str, Any]:
    """Compute the approved fixed-depth attention diagnostic.

    The diagnostic excludes CLS and uses the real ``resblock[5].attn`` module
    output.  It is a property measurement, never a token selector.
    """

    import torch

    if run.block_input is None or run.attention_output is None:
        raise ClipBridgeError("F06 需要 block input 与 attention module output")
    tokens = run.block_input[1:]
    attention = run.attention_output[1:]
    if tokens.shape != attention.shape:
        raise ClipBridgeError("F06 X/U shape 不一致")
    flat_tokens = tokens.detach().float()
    flat_attention = attention.detach().float()
    token_norm = torch.linalg.vector_norm(flat_tokens)
    attention_norm = torch.linalg.vector_norm(flat_attention)
    token_rms = torch.sqrt(torch.mean(flat_tokens.square()))
    attention_rms = torch.sqrt(torch.mean(flat_attention.square()))
    return {
        "depth": int(run.depth),
        "cls_excluded": True,
        "token_shape_without_cls": list(tokens.shape),
        "attention_shape_without_cls": list(attention.shape),
        "x_frobenius": float(token_norm),
        "u_frobenius": float(attention_norm),
        "u_over_x_frobenius": float(attention_norm / token_norm) if float(token_norm) else None,
        "x_rms": float(token_rms),
        "u_rms": float(attention_rms),
        "attention_source": "visual.transformer.resblocks[5].attn output[0]",
        "external_gamma_or_layerscale": False,
        "used_as_selector": False,
    }


def _head_fit(features: Any, labels: Any, *, steps: int, seed: int) -> dict[str, Any] | None:
    if steps <= 0:
        return None
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = nn.Linear(int(features.shape[-1]), 1, device=features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    target = labels.to(features.device, dtype=features.dtype).reshape(-1, 1)
    losses: list[float] = []
    for _ in range(steps):
        logits = head(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        logits = head(features).reshape(-1)
        predictions = (torch.sigmoid(logits) >= 0.5).to(target.dtype)
        accuracy = float((predictions == target.reshape(-1)).float().mean().cpu())
    return {"steps": steps, "seed": seed, "final_loss": losses[-1], "train_accuracy": accuracy}


def run(options: argparse.Namespace) -> dict[str, Any]:
    import torch

    if options.dataset == "ucf" and options.manifest is None:
        raise ValueError("UCF fit probe 需要显式 --manifest data/manifests/ucf_crime/train.jsonl")
    dataset_root = Path(options.dataset_root).expanduser().resolve()
    manifest = None if options.manifest is None else Path(options.manifest).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if options.dataset == "ucf":
        candidates = _read_fit_sources(manifest, dataset_root)
    else:
        candidates = _read_fit_sources(manifest, dataset_root) if manifest else _xd_sources(dataset_root)
    ids_file = None if options.ids_file is None else Path(options.ids_file).expanduser().resolve()
    selected, center_fractions = _select_fit_sources(candidates, options.max_videos, ids_file)

    device_name = options.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    load_start = time.perf_counter()
    model, preprocess, load_identity = load_openai_clip(options.clip_repo, options.weights, device=str(device))
    if options.precision == "float32":
        # OpenAI's loader converts CUDA models to fp16 by default.  Repeated
        # half GEMM kernels on the V100 can differ by one or more ulps even
        # when the hook is read-only.  The identity gate is intended to detect
        # a changed computation graph, so the probe fixes the visual tower to
        # float32 instead of hiding this behind a wider tolerance.
        model.float()
    bridge = ClipVisionBridge(model)
    model.eval()
    load_seconds = time.perf_counter() - load_start
    receipt = bridge.receipt()
    depth = int(options.intervention_depth)
    if type(options.intervention_depth) is not int or not 0 <= depth < receipt.block_count:
        raise ValueError(f"--intervention-depth 必须在 [0,{receipt.block_count - 1}]")
    if type(options.f06_depth) is not int or not 0 <= options.f06_depth < receipt.block_count:
        raise ValueError(f"--f06-depth 必须在 [0,{receipt.block_count - 1}]")
    keep_indices = fixed_local_keep_indices(
        receipt.grid_h,
        receipt.grid_w,
        group_h=2,
        group_w=2,
        keep_per_group=2,
    )
    if depth < 0 or depth >= receipt.block_count:
        raise ValueError(f"depth_fraction 产生非法 block depth={depth}")

    rows: list[dict[str, Any]] = []
    dense_video_features: list[Any] = []
    reduced_video_features: list[Any] = []
    labels: list[bool] = []
    probe_start = time.perf_counter()
    for source in selected:
        raw_frames, windows, video_info = _windows(
            source.path,
            windows_per_video=options.windows_per_video,
            frames_per_window=options.frames_per_window,
            frame_stride=options.frame_stride,
            center_fractions=center_fractions,
        )
        images = _preprocess_frames(preprocess, raw_frames, device)
        with torch.no_grad():
            dense = bridge.forward(images)
            identity = bridge.forward(images, capture_depth=depth)
            reduced = bridge.forward(images, capture_depth=depth, keep_indices=keep_indices)
            f06 = bridge.forward(images, capture_depth=options.f06_depth, capture_attention=True)
        if dense.pooled.shape != identity.pooled.shape or dense.pooled.shape != reduced.pooled.shape:
            raise ClipBridgeError("CLIP native/reduced readout output shape changed")
        identity_diff = _max_abs(dense.pooled, identity.pooled)
        if identity_diff > options.identity_atol:
            raise ClipBridgeError(
                f"CLIP identity/hook parity failed: max_abs={identity_diff} > atol={options.identity_atol}"
            )
        if identity.block_input is None or identity.block_output is None:
            raise ClipBridgeError("CLIP block hook 没有读到真实中间层 tensor")
        if reduced.input_sequence_length != int(keep_indices.numel()):
            raise ClipBridgeError("CLIP reduced block input 没有真实缩短到 keep_indices 长度")
        f06_metrics = _f06_attention_metrics(f06)
        # Aggregate frames equally within each sampled video.  This is a
        # representation check only; no test score or event labels are read.
        dense_feature = dense.pooled.detach().float().mean(dim=0)
        reduced_feature = reduced.pooled.detach().float().mean(dim=0)
        dense_video_features.append(dense_feature)
        reduced_video_features.append(reduced_feature)
        labels.append(source.is_anomaly)
        rows.append(
            {
                "video_id": source.video_id,
                "path": str(source.path),
                "source": source.source,
                "split": "train",
                "weak_video_label": bool(source.is_anomaly),
                "video_info": video_info,
                "windows": windows,
                "frame_count_processed": len(raw_frames),
                "native_image_batch_shape": list(images.shape),
                "dense_readout_shape": list(dense.pooled.shape),
                "identity_readout_shape": list(identity.pooled.shape),
                "reduced_readout_shape": list(reduced.pooled.shape),
                "identity_hook_max_abs": identity_diff,
                "identity_hook_allclose": bool(identity_diff <= options.identity_atol),
                "hook_depth": depth,
                "hook_input_shape": list(identity.block_input.shape),
                "hook_output_shape": list(identity.block_output.shape),
                "reduced_input_shape": list(reduced.block_input.shape),
                "dense_tokens_at_hook": int(identity.input_sequence_length),
                "reduced_tokens_at_hook": int(reduced.input_sequence_length),
                "output_dim_preserved": int(reduced.pooled.shape[-1]) == receipt.projection_dim,
                "reduction_kind": "fixed_local_2x2_keep_2_neutral_control",
                "f06": f06_metrics,
            }
        )

    dense_features = torch.stack(dense_video_features, dim=0)
    reduced_features = torch.stack(reduced_video_features, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    result: dict[str, Any] = {
        "status": "completed",
        "probe": "clip_visual_fit_raw_v1",
        "representation_source": "raw_video",
        "preextracted_features_used": False,
        "dataset": options.dataset,
        "partition": "fit",
        "supervision": "video_level_only",
        "official_test_accessed": False,
        "official_test_used_for_selection": False,
        "selected_videos": len(rows),
        "selected_video_ids": [row["video_id"] for row in rows],
        "model": bridge.architecture(),
        "load_identity": load_identity,
        "clip_repo_head": _git_head(Path(options.clip_repo).expanduser().resolve()),
        "weights_sha256": _sha256_file(Path(options.weights).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "manifest": None if manifest is None else str(manifest),
        "manifest_sha256": None if manifest is None else _sha256_text(manifest),
        "ids_file": None if ids_file is None else str(ids_file),
        "ids_file_sha256": None if ids_file is None else _sha256_text(ids_file),
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "model_precision": options.precision,
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "sampling": {
            "windows_per_video": options.windows_per_video,
            "frames_per_window": options.frames_per_window,
            "frame_stride": options.frame_stride,
            "window_center_fractions": center_fractions,
            "shared_with_v2_probe": bool(options.shared_with_v2_probe),
        },
        "intervention": {
            "depth_selection": "explicit_block_index",
            "depth": depth,
            "keep_indices": keep_indices.tolist(),
            "keep_count": int(keep_indices.numel()),
            "dense_count": receipt.sequence_length,
            "grid": [receipt.grid_h, receipt.grid_w],
            "coordinates": bridge.token_coordinates().tolist(),
            "same_input_dense_comparison": True,
            "output_dim": receipt.projection_dim,
        },
        "f06_attention_probe": {
            "depth": options.f06_depth,
            "definition": "exclude CLS; Frobenius(U)/Frobenius(X) and both RMS",
            "attention_source": "visual.transformer.resblocks[5].attn output[0]",
            "external_gamma_or_layerscale": False,
            "used_as_selector": False,
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "fit_probe_forward": time.perf_counter() - probe_start,
        },
        "videos": rows,
        "head_fit": {
            "dense": _head_fit(dense_features, label_tensor, steps=options.train_head_steps, seed=options.seed),
            "reduced": _head_fit(reduced_features, label_tensor, steps=options.train_head_steps, seed=options.seed),
            "scope": "fit videos only; no quality claim",
        },
        "limitations": [
            "OpenAI center-crop preprocessing is not a reproduction of VadCLIP's official 10-crop feature cache.",
            "The fixed local intervention is a neutral engineering control; it is not a selector learned from labels.",
            "A reduced forward proves sequence shortening for this CLIP visual tower, not detector quality or end-to-end speed.",
        ],
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ucf", "xd"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clip-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "native"), default="float32")
    parser.add_argument("--max-videos", type=int, default=4)
    parser.add_argument("--windows-per-video", type=int, default=2)
    parser.add_argument("--frames-per-window", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--intervention-depth", type=int, default=2)
    parser.add_argument("--f06-depth", type=int, default=5)
    parser.add_argument("--identity-atol", type=float, default=1e-5)
    parser.add_argument("--train-head-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20270920)
    parser.add_argument("--shared-with-v2-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        result = run(options)
    except (ClipBridgeError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    output = options.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "videos": result["selected_video_ids"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
