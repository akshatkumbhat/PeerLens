"""The query-plan contract and shared agent types.

The model produces a ``QueryPlan`` (loose — institution and metric are free text);
deterministic resolution against the warehouse turns a valid plan into a
``ResolvedPlan`` or an ``Abstention``. Keeping ``metric``/``institution`` as
strings here (rather than enums) lets the resolver produce a *precise* abstention
message ("unknown metric: graduation_rate; I can answer …") instead of an opaque
parse error — grounding lives in the resolver, not the type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SINGLE = "single"    # one institution, one metric
    COMPARE = "compare"  # institution vs a comparison set
    # TREND deferred until multi-year data (Phase 4)


class ComparisonKind(str, Enum):
    NONE = "none"
    PEERS = "peers"
    ASPIRANTS = "aspirants"
    EXPLICIT = "explicit"


class ComparisonSet(BaseModel):
    kind: ComparisonKind = ComparisonKind.NONE
    institutions: list[str] = Field(default_factory=list)  # only for EXPLICIT
    k: int = Field(default=10, ge=1, le=25)


class QueryPlan(BaseModel):
    """What the model emits. Intentionally permissive; grounded by the resolver."""

    intent: Intent
    institution: str
    metric: str
    comparison: ComparisonSet = Field(default_factory=ComparisonSet)
    years: list[int] = Field(default_factory=lambda: [2020])


class ResolvedPlan(BaseModel):
    """A plan that passed catalog grounding — every reference is a real id/metric."""

    intent: Intent
    target_unitid: int
    target_name: str
    metric: str
    comparison: ComparisonSet
    comparison_unitids: list[int] = Field(default_factory=list)
    years: list[int]


class AbstainReason(str, Enum):
    UNKNOWN_INSTITUTION = "unknown_institution"
    AMBIGUOUS_INSTITUTION = "ambiguous_institution"
    UNKNOWN_METRIC = "unknown_metric"
    OUT_OF_SCOPE = "out_of_scope"
    NO_DATA = "no_data"
    LOW_AGREEMENT = "low_agreement"
    NO_VALID_PLAN = "no_valid_plan"


@dataclass
class Abstention:
    """A decision NOT to answer — with a reason and, where useful, options."""

    reason: AbstainReason
    message: str
    options: list[str] = field(default_factory=list)
