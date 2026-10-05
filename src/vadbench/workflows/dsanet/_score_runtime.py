"""Run the fixed DSANet head on one sealed raw-CLIP feature tier.

This writes per-video scores only. It never opens official test annotations or
chooses a checkpoint, score branch, threshold, or compression budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


LABEL_MAPS = {
    "ucf": {
        "Normal": "Normal", "Abuse": "Abuse", "Arrest": "Arrest", "Arson": "Arson",
        "Assault": "Assault", "Burglary": "Burglary", "Explosion": "Explosion",
        "Fighting": "Fighting", "RoadAccidents": "RoadAccidents", "Robbery": "Robbery",
        "Shooting": "Shooting", "Shoplifting": "Shoplifting", "Stealing": "Stealing",
        "Vandalism": "Vandalism",
    },
    "xd": {
        "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
        "B5": "abuse", "B6": "car accident", "G": "explosion",
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run(options) -> None:
    sys.path.insert(0, str(options.upstream_src.resolve()))
    from model import DSANet
    from utils.tools import get_batch_mask, get_prompt_text, process_split

    author_options = __import__(f"{options.dataset}_option")
    config = author_options.parser.parse_args([])
    text = get_prompt_text(LABEL_MAPS[options.dataset])
    device = torch.device("cuda:0")
    model = DSANet(
        config.classes_num, config.embed_dim, config.visual_length,
        config.visual_width, config.visual_head, config.visual_layers,
        config.attn_window, config.prompt_prefix, config.prompt_postfix,
        config, str(device),
    ).to(device)
    model.load_state_dict(torch.load(options.checkpoint, map_location=device))
    model.eval()
    options.output.mkdir(parents=True, exist_ok=False)
    videos = [json.loads(line) for line in options.manifest.read_text(encoding="utf-8").splitlines()]
    predictions = options.output / "predictions.jsonl"
    started = time.perf_counter()
    completed = 0
    with predictions.open("w", encoding="utf-8") as writer, torch.inference_mode():
        for video in videos:
            video_id = video["video_id"]
            receipt = json.loads(
                (options.features / "receipts" / f"{video_id}.json").read_text(encoding="utf-8")
            )
            crop = str(video["crop_ids"][0])
            feature_path = Path(receipt["crop_files"][crop]["path"])
            feature = np.load(feature_path, allow_pickle=False)
            if feature.shape != (receipt["feature_rows"], 512):
                raise ValueError(f"feature shape changed: {video_id}")
            split, length = process_split(feature, config.visual_length)
            visual = torch.from_numpy(split).float().to(device)
            if length < config.visual_length:
                visual = visual.unsqueeze(0)
            lengths = torch.tensor(
                [min(config.visual_length, max(0, length - index * config.visual_length))
                 for index in range(visual.shape[0])], dtype=torch.int32,
            )
            mask = get_batch_mask(lengths, config.visual_length).to(device)
            output = model(visual, mask, text, lengths, config.DNP_use)
            logits1, logits2 = output[1], output[2]
            logits1 = logits1.reshape(-1, logits1.shape[-1])[:length]
            logits2 = logits2.reshape(-1, logits2.shape[-1])[:length]
            coarse = torch.sigmoid(logits1.squeeze(-1)).float()
            total_abnormal = torch.sigmoid(logits1.squeeze(-1) / config.temp).float()
            align_prob = torch.softmax(logits2.float() / config.temp, dim=1)[:, 1:]
            abnormal_distribution = align_prob / (align_prob.sum(dim=1, keepdim=True) + 1e-12)
            class_scores = torch.cat(
                ((1.0 - total_abnormal).unsqueeze(1),
                 total_abnormal.unsqueeze(1) * abnormal_distribution), dim=1,
            )
            refined = 1.0 - class_scores[:, 0]
            if not torch.isfinite(class_scores).all():
                raise RuntimeError(f"nonfinite DSANet score: {video_id}")
            record = {
                "video_id": video_id,
                "raw_path": video["raw_path"],
                "raw_frames": receipt["raw_frames"],
                "feature_rows": length,
                "source_frame_indices": receipt["source_frame_indices"],
                "crop_id": int(crop),
                "coarse_scores": coarse.cpu().tolist(),
                "refined_scores": refined.cpu().tolist(),
                "class_scores": class_scores.cpu().tolist(),
            }
            writer.write(json.dumps(record, separators=(",", ":")) + "\n")
            completed += 1
            if completed % 25 == 0:
                print(f"predicted={completed}/{len(videos)}", flush=True)
    summary = {
        "status": "completed", "dataset": options.dataset, "videos": completed,
        "checkpoint_sha256": digest(options.checkpoint),
        "manifest_sha256": digest(options.manifest),
        "feature_root": str(options.features),
        "score_rule": "coarse_sigmoid_logit1_and_refined_1_minus_normal",
        "test_truth_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if completed != len(videos):
        raise RuntimeError("prediction coverage incomplete")
    print(json.dumps(summary), flush=True)

