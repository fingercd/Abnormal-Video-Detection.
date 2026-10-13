"""Frozen deployment of the 2026-10-13 single-site relation ablation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from .bridges.post_attention import PostAttentionIntervention
from .contracts import TokenLayout
from .relation_selection import METHODS, retained_count, select_relational_tokens


@dataclass(frozen=True, init=False, eq=False)
class RelationDeployment:
    """Patch-only native VideoMAEv2; no dependency on the retired L2 policy."""

    def __init__(self, bridge, dense_layout: TokenLayout, *, depth: int, method: str, keep_ratio: float):
        if bridge.receipt().encoder_id != "videomaev2" or bridge.receipt().has_cls is not False:
            raise ValueError("relation deployment currently supports verified patch-only VideoMAEv2")
        if dense_layout.batch_size != 1 or bool(dense_layout.special_token_mask.any()):
            raise ValueError("setup must be a verified B=1 patch-only layout")
        if not bool(dense_layout.valid_mask.all()) or not bool(dense_layout.mass.eq(1).all()):
            raise ValueError("setup needs complete, unpadded, unit-mass native tokens")
        if method not in METHODS or keep_ratio not in (0.3, 0.5, 1.0):
            raise ValueError("relation experiment supports its three methods at keep 0.3/0.5/1.0")
        if type(depth) is not int or not 0 <= depth < bridge.receipt().block_count:
            raise ValueError("invalid native block index")
        if dense_layout.provenance.get("coordinate_source") != "verified-conv3d-flatten-t-h-w":
            raise ValueError("native Conv3d flatten geometry must be verified before deployment")
        for name, value in {
            "bridge": bridge, "depth": depth, "method": method, "keep_ratio": keep_ratio,
            "source_tokens": dense_layout.token_capacity,
            "output_tokens": retained_count(dense_layout.token_capacity, keep_ratio),
            "clip_frames": dense_layout.provenance["grid"][0] * dense_layout.provenance["stride"][0],
        }.items():
            object.__setattr__(self, name, value)
        root = Path(__file__).parent
        files = [Path(__file__), root / "relation_selection.py", root / "bridges/post_attention.py"]
        identity = {
            "name": method, "variant": "key_attention_single_site_v1", "encoder_id": "videomaev2",
            "depth": depth, "block_index": depth, "site": "post_attention_residual_pre_norm2_mlp",
            "patch_keep_ratio": keep_ratio, "budget_rounding": "nearest_half_up",
            "source_tokens": self.source_tokens, "output_tokens": self.output_tokens,
            "signal": "none" if method == "uniform" else "mean_received_native_attention",
            "key_signal": "mean_head_native_keys_cosine" if method == "visionzip_internal" else None,
            "grouping": "none", "special_tokens": "unsupported_patch_only_bridge",
            "dominant_contextual_weights": [4, 1] if method == "visionzip_internal" else None,
            "merge_value": "target_plus_assigned_mean_fp32_accumulation_cast_back" if method == "visionzip_internal" else None,
            "readout": "native_unchanged_no_mass_reweighting", "calibration": "none",
            "position_strategy": "native_position_algorithm_unchanged",
            "position_semantics": "source_group_first_order" if method == "visionzip_internal" else "retained_native_order",
            "implementation_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        }
        object.__setattr__(self, "_identity", identity)

    @property
    def reducer_identity(self):
        return deepcopy(self._identity)

    def __call__(self, clean_batch):
        if clean_batch.frames.shape[1] != self.clip_frames or not bool(clean_batch.valid_mask.all()):
            raise ValueError("deployment requires complete native clips")
        return _RelationContext(self)


class _RelationContext:
    def __init__(self, deployment):
        self.deployment = deployment
        self.selection = None
        active = deployment.keep_ratio < 1
        self.intervention = PostAttentionIntervention(
            deployment.bridge, deployment.depth, self.reduce,
            need_keys=active and deployment.method == "visionzip_internal",
            need_attention=active and deployment.method != "uniform",
        )

    def reduce(self, hidden, keys, scores):
        if hidden.shape[1] != self.deployment.source_tokens:
            raise ValueError("actual token count differs from verified setup geometry")
        self.selection = select_relational_tokens(
            hidden, method=self.deployment.method, keep_ratio=self.deployment.keep_ratio,
            keys=keys, attention_scores=scores,
        )
        return self.selection.tokens

    def __enter__(self):
        self.intervention.__enter__()
        return self

    def __exit__(self, *args):
        return self.intervention.__exit__(*args)

    def validate_execution(self):
        receipt = self.intervention.validate_execution()
        if receipt["gathered_tokens"] != self.deployment.output_tokens:
            raise ValueError("actual compressed count differs from frozen budget")
        return {
            **receipt,
            "requested_patch_keep_ratio": self.deployment.keep_ratio,
            "actual_patch_keep_ratio": receipt["gathered_tokens"] / self.deployment.source_tokens,
            "source_membership_recorded": True,
            "source_coverage": "all_sources_merged" if self.deployment.method == "visionzip_internal" else "retained_only",
            "mass_used_to_reweight_readout": False,
        }
