"""Read-only observation tools for the ICASSP 2027 research workflow.

This package deliberately has no dependency on :mod:`vadbench.token_reduction`.
Labels live in cohort sidecars and are joined only after encoder observations
have been collected.
"""

from .cohorts import CohortError, CohortIndex, CohortRecord, LabelPolicy
from .collectors import (
    ProbeCollector,
    ProbeLimits,
    ProbeObservation,
    ProbeSiteMetadata,
    ProbeTokenMetadata,
)
from .labels import join_probe_rows

__all__ = [
    "CohortError",
    "CohortIndex",
    "CohortRecord",
    "LabelPolicy",
    "ProbeCollector",
    "ProbeLimits",
    "ProbeObservation",
    "ProbeSiteMetadata",
    "ProbeTokenMetadata",
    "join_probe_rows",
]
