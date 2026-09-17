"""Token-reduction contracts and currently supported no-op baseline."""

from .contracts import (
    ReductionContext,
    ReductionResult,
    TokenLayout,
    TokenReducer,
    TokenReductionContractError,
)
from .identity import IdentityReducer

__all__ = [
    "IdentityReducer",
    "ReductionContext",
    "ReductionResult",
    "TokenLayout",
    "TokenReducer",
    "TokenReductionContractError",
]
