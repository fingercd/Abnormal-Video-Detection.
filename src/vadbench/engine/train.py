"""Minimal, auditable PyTorch training step and checkpoint artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vadbench.hashing import sha256_file as _sha256

try:  # The framework's extraction/metric path does not require PyTorch.
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal environments.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training and checkpoints. Install the "
            "project's 'torch' extra or a compatible PyTorch build."
        )


def move_to_device(value: Any, device: Any) -> Any:
    """Recursively move tensors while preserving batch container structure."""

    _require_torch()
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        moved = {key: move_to_device(item, device) for key, item in value.items()}
        try:
            return type(value)(moved)
        except TypeError:
            return moved
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            descriptor.name: move_to_device(getattr(value, descriptor.name), device)
            for descriptor in fields(value)
        }
        return replace(value, **updates)
    return value


def _to_log_value(value: Any) -> Any:
    if TORCH_AVAILABLE and torch.is_tensor(value):
        detached = value.detach().cpu()
        return float(detached.item()) if detached.numel() == 1 else detached.numpy().tolist()
    return value


def _extract_loss(output: Any) -> Any:
    if TORCH_AVAILABLE and torch.is_tensor(output) and output.ndim == 0:
        return output
    if isinstance(output, Mapping) and "loss" in output:
        return output["loss"]
    loss = getattr(output, "loss", None)
    if loss is not None:
        return loss
    raise TypeError(
        "training output must be a scalar tensor or expose a scalar 'loss'; "
        "provide loss_fn for a plain model output"
    )


@dataclass(frozen=True)
class TrainStepResult(Mapping[str, Any]):
    """JSON-friendly summary of exactly one optimizer update."""

    loss: float
    step: int
    metrics: dict[str, Any]
    grad_norm: float | None = None

    def __getitem__(self, key: str) -> Any:
        if key in {"loss", "step", "metrics", "grad_norm"}:
            return getattr(self, key)
        if key in self.metrics:
            return self.metrics[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("loss", "step", "metrics", "grad_norm"))

    def __len__(self) -> int:
        return 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_one_step(
    model: Any,
    batch: Any,
    optimizer: Any,
    *,
    step: int = 0,
    device: Any | None = None,
    loss_fn: Any | None = None,
    max_grad_norm: float | None = None,
    scaler: Any | None = None,
) -> TrainStepResult:
    """Run one forward/backward/update step for a task or a plain model.

    A task should expose ``training_step(batch)`` and return an object with a
    scalar ``loss`` plus optional ``metrics``.  For a plain model, pass a
    ``loss_fn(output, batch)`` callable.
    """

    _require_torch()
    if step < 0:
        raise ValueError("step must be non-negative")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be greater than zero")
    if device is not None:
        model.to(device)
        batch = move_to_device(batch, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_step = getattr(model, "training_step", None)
    if callable(training_step):
        output = training_step(batch)
    else:
        predictions = model(batch)
        if loss_fn is None:
            output = predictions
        else:
            output = {"loss": loss_fn(predictions, batch), "predictions": predictions}
    loss = _extract_loss(output)
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise TypeError("loss must be a scalar torch.Tensor")
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError(f"non-finite training loss: {loss.detach().cpu().item()}")

    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

    parameters_with_grad = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    grad_norm: float | None = None
    try:
        if max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters_with_grad, max_grad_norm, error_if_nonfinite=True
            )
            grad_norm = float(norm.detach().cpu()) if torch.is_tensor(norm) else float(norm)
        else:
            for parameter in parameters_with_grad:
                gradient = parameter.grad
                values = gradient._values() if gradient.is_sparse else gradient
                if not bool(torch.isfinite(values).all().detach().cpu()):
                    raise FloatingPointError("non-finite gradient detected")
    except (FloatingPointError, RuntimeError):
        optimizer.zero_grad(set_to_none=True)
        raise

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()

    raw_metrics = getattr(output, "metrics", None)
    if raw_metrics is None and isinstance(output, Mapping):
        raw_metrics = output.get("metrics", {})
    metrics = {key: _to_log_value(value) for key, value in dict(raw_metrics or {}).items()}
    return TrainStepResult(
        loss=float(loss.detach().cpu()),
        step=int(step) + 1,
        metrics=metrics,
        grad_norm=grad_norm,
    )


@dataclass(frozen=True)
class CheckpointArtifact:
    """Traceable description stored beside a ``.pt`` checkpoint."""

    path: str
    manifest_path: str
    sha256: str
    size_bytes: int
    step: int
    epoch: int
    created_at: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if TORCH_AVAILABLE and torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    metadata: Mapping[str, Any] | None = None,
    write_manifest: bool = True,
) -> CheckpointArtifact:
    """Atomically save model/training state and a checksum manifest."""

    _require_torch()
    if step < 0 or epoch < 0:
        raise ValueError("step and epoch must be non-negative")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    metadata_payload = dict(metadata or {})
    if write_manifest:
        # Fail before committing the .pt artifact when metadata violates the
        # JSON sidecar contract.
        json.dumps(metadata_payload, ensure_ascii=False, allow_nan=False, default=_json_default)
    state: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "step": int(step),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metadata": metadata_payload,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest_path = destination.with_suffix(destination.suffix + ".json")
    artifact = CheckpointArtifact(
        path=str(destination),
        manifest_path=str(manifest_path),
        sha256=_sha256(destination),
        size_bytes=destination.stat().st_size,
        step=int(step),
        epoch=int(epoch),
        created_at=created_at,
    )
    if write_manifest:
        payload = artifact.to_dict()
        payload["metadata"] = metadata_payload
        encoded = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
        )
        descriptor, manifest_temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        os.close(descriptor)
        manifest_temporary_path = Path(manifest_temporary_name)
        try:
            manifest_temporary_path.write_text(encoded + "\n", encoding="utf-8")
            os.replace(manifest_temporary_path, manifest_path)
        finally:
            manifest_temporary_path.unlink(missing_ok=True)
    return artifact


def load_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: Any = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return non-tensor run metadata."""

    _require_torch()
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if verify:
        manifest_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"checkpoint checksum manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected_size = int(manifest.get("size_bytes", -1))
        expected_sha256 = str(manifest.get("sha256", ""))
        actual_size = checkpoint_path.stat().st_size
        actual_sha256 = _sha256(checkpoint_path)
        if expected_size != actual_size or expected_sha256 != actual_sha256:
            raise ValueError(
                "checkpoint checksum verification failed: "
                f"expected size/sha256={expected_size}/{expected_sha256}, "
                f"actual={actual_size}/{actual_sha256}"
            )
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        state = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("invalid checkpoint: missing model_state_dict")
    incompatible = model.load_state_dict(state["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    if restore_rng and "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    return {
        "schema_version": int(state.get("schema_version", 0)),
        "created_at": state.get("created_at"),
        "step": int(state.get("step", 0)),
        "epoch": int(state.get("epoch", 0)),
        "metadata": dict(state.get("metadata", {})),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


__all__ = [
    "CheckpointArtifact",
    "TORCH_AVAILABLE",
    "TrainStepResult",
    "load_checkpoint",
    "move_to_device",
    "save_checkpoint",
    "train_one_step",
]
