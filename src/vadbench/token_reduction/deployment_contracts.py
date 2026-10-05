"""Contract for a prepared reduction around one native encoder forward.

This differs from ``TokenReducer.reduce``: a deployment binds frozen identity
and a clean clip batch to a context that checks the completed native forward.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from vadbench.contracts import ClipBatch


@runtime_checkable
class ReductionExecutionContext(Protocol):
    """Active native-forward intervention with a checked execution receipt."""

    def __enter__(self) -> ReductionExecutionContext: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any: ...

    def validate_execution(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ReductionDeployment(Protocol):
    """Frozen reducer identity and per-batch native-forward context factory."""

    @property
    def reducer_identity(self) -> Mapping[str, Any]: ...

    def __call__(self, clean_batch: ClipBatch) -> AbstractContextManager[ReductionExecutionContext]: ...


__all__ = ["ReductionDeployment", "ReductionExecutionContext"]
