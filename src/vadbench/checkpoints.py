"""显式、可校验的模型权重注册与下载。"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vadbench.hashing import sha256_file as sha256_file


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    adapter: str
    source: str
    repo_id: str
    revision: str
    license: str
    allow_patterns: tuple[str, ...]
    sha256: Mapping[str, str]
    notes: str = ""
    local_path: str | None = None
    status: str = "verified"
    file_sizes: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, checkpoint_id: str, data: Mapping[str, Any]) -> CheckpointSpec:
        required = ("adapter", "source", "repo_id", "revision", "license")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise CheckpointError(f"checkpoint {checkpoint_id!r} 缺少字段：{missing}")
        return cls(
            id=checkpoint_id,
            adapter=str(data["adapter"]),
            source=str(data["source"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            license=str(data["license"]),
            allow_patterns=tuple(str(item) for item in data.get("allow_patterns", ())),
            sha256={str(k): str(v).lower() for k, v in dict(data.get("sha256", {})).items()},
            notes=str(data.get("notes", "")),
            local_path=str(data["local_path"]) if data.get("local_path") else None,
            status=str(data.get("status", "verified")),
            file_sizes={
                str(item["path"]): int(item["size_bytes"])
                for item in data.get("files", ())
                if item.get("size_bytes") is not None
            },
        )


def load_checkpoint_registry(path: str | Path) -> dict[str, CheckpointSpec]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entries = data.get("checkpoints", data)
    if not isinstance(entries, Mapping):
        raise CheckpointError("checkpoint registry 顶层必须包含 checkpoints 对象")
    return {
        str(checkpoint_id): CheckpointSpec.from_mapping(str(checkpoint_id), value)
        for checkpoint_id, value in entries.items()
    }


def verify_checkpoint(spec: CheckpointSpec, root: str | Path) -> dict[str, str]:
    checkpoint_root = Path(root)
    if spec.status == "planned" or not spec.sha256:
        raise CheckpointError(f"checkpoint {spec.id!r} 没有可验证的冻结资产")
    if not checkpoint_root.exists():
        raise CheckpointError(f"权重路径不存在：{checkpoint_root}")
    if checkpoint_root.is_file() and len(spec.sha256) != 1:
        raise CheckpointError("多文件 checkpoint 必须提供目录")
    actual: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in spec.sha256.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CheckpointError(f"非法 checkpoint 文件路径：{relative}")
        file_path = checkpoint_root if checkpoint_root.is_file() else checkpoint_root / relative
        if not file_path.is_file():
            errors.append(f"缺少 {relative}")
            continue
        actual_digest = sha256_file(file_path)
        actual[relative] = actual_digest
        if actual_digest.lower() != expected.lower():
            errors.append(f"{relative}: expected={expected}, actual={actual_digest}")
        expected_size = spec.file_sizes.get(relative)
        if expected_size is not None and file_path.stat().st_size != expected_size:
            errors.append(f"{relative}: size != {expected_size}")
    if errors:
        raise CheckpointError("权重校验失败：\n- " + "\n- ".join(errors))
    return actual


def fetch_checkpoint(
    spec: CheckpointSpec,
    destination: str | Path,
    *,
    accepted_license: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """下载冻结 revision；许可证必须被调用方显式确认。"""

    if accepted_license != spec.license:
        raise CheckpointError(
            f"下载 {spec.id!r} 前必须显式确认许可证：accepted_license={spec.license!r}"
        )
    if spec.source != "huggingface":
        raise CheckpointError(f"暂不支持 checkpoint source={spec.source!r}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CheckpointError("缺少 huggingface_hub；请安装 vadbench[videomaev2]") from exc

    destination_path = Path(destination).resolve()
    if destination_path.exists():
        verify_checkpoint(spec, destination_path)
        return destination_path
    if spec.status == "planned" or not spec.sha256:
        raise CheckpointError(f"checkpoint {spec.id!r} 没有可验证的冻结资产")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination_path.parent))
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=str(staging),
            allow_patterns=list(spec.allow_patterns) or None,
            local_files_only=local_files_only,
        )
        actual = verify_checkpoint(spec, staging)
        (staging / "vadbench-checkpoint.json").write_text(
            json.dumps(
                {
                    "schema": "vadbench.checkpoint/v1",
                    "spec": asdict(spec),
                    "verified_sha256": actual,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # rename refuses an existing nonempty destination; never replace a verified asset.
        staging.rename(destination_path)
    finally:
        if staging.exists():
            assert staging.resolve().parent == destination_path.parent
            shutil.rmtree(staging)
    return destination_path
