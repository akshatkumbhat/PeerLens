"""Self-consistency — the core confidence signal.

Sample N plans at nonzero temperature, resolve + execute each, and group by a
unified key: an execution's canonical result signature, OR the abstention reason
when a sample resolves to "don't answer", OR "invalid" when no plan parses. The
fraction of samples in the largest group is the agreement score. Grouping
abstentions alongside answers means a question the model reliably reads as
"unknown institution" abstains *with high agreement*, exactly as it should.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from peerlens.agent.catalog import link
from peerlens.agent.execute import Execution, execute_resolved, result_signature
from peerlens.agent.model import PlanModel
from peerlens.agent.plan import Abstention, QueryPlan, ResolvedPlan
from peerlens.agent.resolve import resolve_plan


@dataclass
class Sample:
    plan: QueryPlan | None
    resolved: ResolvedPlan | None
    execution: Execution | None
    abstention: Abstention | None
    key: tuple


@dataclass
class ConsistencyResult:
    samples: list[Sample]
    top_key: tuple
    agreement: float
    top_samples: list[Sample]


def _one_sample(
    con: duckdb.DuckDBPyConnection, model: PlanModel, question: str, temperature: float
) -> Sample:
    linked = link(question)
    plan = model.propose(question, linked, temperature=temperature)
    if plan is None:
        return Sample(None, None, None, None, ("invalid",))
    outcome = resolve_plan(con, plan)
    if isinstance(outcome, Abstention):
        return Sample(plan, None, None, outcome, ("abstain", outcome.reason.value))
    execution = execute_resolved(con, outcome)
    return Sample(plan, outcome, execution, None, ("answer", result_signature(execution)))


def run_consistency(
    con: duckdb.DuckDBPyConnection,
    model: PlanModel,
    question: str,
    *,
    n: int,
    temperature: float,
) -> ConsistencyResult:
    """Draw N samples and group them; return the top group and agreement."""
    samples = [_one_sample(con, model, question, temperature) for _ in range(n)]

    groups: dict[tuple, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.key, []).append(s)
    top_key = max(groups, key=lambda k: len(groups[k]))
    top_samples = groups[top_key]
    agreement = len(top_samples) / len(samples) if samples else 0.0
    return ConsistencyResult(samples, top_key, agreement, top_samples)
