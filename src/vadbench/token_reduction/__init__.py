"""Token-reduction contracts and currently supported no-op baseline."""

from .contracts import (
    ReductionContext,
    ReductionResult,
    TokenLayout,
    TokenReducer,
    TokenReductionContractError,
)
from .identity import IdentityReducer
from .pair_merge import (
    PairLinearGate,
    PairMergeError,
    PairMergePlanCache,
    PairMergeSpec,
    PairWeightedMerge,
    horizontal_pair_merge_spec,
)

__all__ = [
    "IdentityReducer",
    "PairLinearGate",
    "PairMergeError",
    "PairMergePlanCache",
    "PairMergeSpec",
    "PairWeightedMerge",
    "ReductionContext",
    "ReductionResult",
    "TokenLayout",
    "TokenReducer",
    "TokenReductionContractError",
    "horizontal_pair_merge_spec",
]
