"""The PeerLens agent pipeline — one clean grounded path.

schema-link -> sample N plans -> resolve + template SQL + execute each ->
self-consistency grouping -> correct-or-silent decision -> programmatic number
injection. Every answer surfaces its work (plan, SQL, rows, agreement) — making
the grounding visible is the whole trust argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from peerlens.agent.answer import render_answer
from peerlens.agent.consistency import run_consistency
from peerlens.agent.decide import decide
from peerlens.agent.model import PlanModel
from peerlens.agent.plan import Abstention, ResolvedPlan
from peerlens.config import Settings, get_settings


@dataclass
class AgentResponse:
    question: str
    answered: bool
    agreement: float
    n_samples: int
    answer: str | None = None
    abstention: Abstention | None = None
    resolved_plan: ResolvedPlan | None = None
    sql: str | None = None
    params: list[object] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)


def run_agent(
    con: duckdb.DuckDBPyConnection,
    model: PlanModel,
    question: str,
    settings: Settings | None = None,
) -> AgentResponse:
    """Answer ``question`` or abstain, surfacing the work either way."""
    s = settings or get_settings()
    cr = run_consistency(
        con, model, question, n=s.agent_samples, temperature=s.agent_temperature
    )
    decision = decide(cr, tau=s.agent_tau)

    if not decision.answered:
        return AgentResponse(
            question=question,
            answered=False,
            agreement=cr.agreement,
            n_samples=len(cr.samples),
            abstention=decision.abstention,
        )

    plan = decision.answer_plan
    execution = decision.answer_execution
    assert plan is not None and execution is not None
    return AgentResponse(
        question=question,
        answered=True,
        agreement=cr.agreement,
        n_samples=len(cr.samples),
        answer=render_answer(plan, execution),
        resolved_plan=plan,
        sql=execution.sql,
        params=execution.params,
        rows=execution.rows,
    )
