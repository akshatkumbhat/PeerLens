"""The correct-or-silent decision.

Maps a ConsistencyResult to either an answer or an abstention, implementing the
five abstain/clarify conditions:

1. unknown institution/metric          -> resolver abstention is the majority vote
2. ambiguous institution               -> resolver abstention is the majority vote
5. out of scope (year/metric/unparsed) -> resolver/invalid is the majority vote
4. empty / suppressed result set       -> NO_DATA on the winning answer
3. agreement below tau                 -> LOW_AGREEMENT (the samples disagree)

Conditions 1/2/5 ride the unified grouping: when the model reliably reads a
question as "don't answer", that abstention is the top group. Below tau we never
answer — we report uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

from peerlens.agent.consistency import ConsistencyResult
from peerlens.agent.execute import Execution
from peerlens.agent.plan import Abstention, AbstainReason, ResolvedPlan


@dataclass
class Decision:
    answer_plan: ResolvedPlan | None = None
    answer_execution: Execution | None = None
    abstention: Abstention | None = None

    @property
    def answered(self) -> bool:
        return self.abstention is None


def decide(cr: ConsistencyResult, *, tau: float) -> Decision:
    kind = cr.top_key[0]

    # Below tau, the samples don't agree enough to answer — abstain as uncertain.
    if cr.agreement < tau:
        return Decision(
            abstention=Abstention(
                AbstainReason.LOW_AGREEMENT,
                f"I'm not confident enough to answer — the model's samples disagreed "
                f"(agreement {cr.agreement:.0%}, below the {tau:.0%} threshold). "
                "Try narrowing the question.",
            )
        )

    if kind == "invalid":
        return Decision(
            abstention=Abstention(
                AbstainReason.NO_VALID_PLAN,
                "I couldn't form a grounded query for that question. "
                "I answer questions about admit rate, yield, retention, applicants, "
                "and enrollment for four-year institutions (2020).",
            )
        )

    if kind == "abstain":
        # Every sample in this group carries the same reason; surface it.
        return Decision(abstention=cr.top_samples[0].abstention)

    # kind == "answer"
    rep = cr.top_samples[0]
    execution = rep.execution
    assert execution is not None and rep.resolved is not None
    has_value = any(r.get("metric_value") is not None for r in execution.rows)
    if not execution.rows or not has_value:
        return Decision(
            abstention=Abstention(
                AbstainReason.NO_DATA,
                f"No data for {rep.resolved.target_name} on that metric in "
                f"{rep.resolved.years[0]}.",
            )
        )
    return Decision(answer_plan=rep.resolved, answer_execution=execution)
