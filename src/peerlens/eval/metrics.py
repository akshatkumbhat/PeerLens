"""Evaluation metrics and the risk-coverage sweep.

All metrics are functions of the records and a threshold τ. A question is
*answered* when the consistent sample is an answer AND agreement ≥ τ; otherwise
the system abstains. Sweeping τ over the records traces the risk-coverage curve
and picks the operating point — no model calls involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from peerlens.eval.harness import EvalRecord


def _answered(r: EvalRecord, tau: float) -> bool:
    return r.top_kind == "answer" and r.agreement >= tau


def _is_wrong(r: EvalRecord, tau: float) -> bool:
    """A confidently-wrong answer: answered, but the answer is incorrect.

    Answerable: answered with the wrong institution/metric.
    Unanswerable: answered at all (it should have abstained).
    """
    if not _answered(r, tau):
        return False
    if r.kind == "answerable":
        return r.answer_correct is not True
    return True  # answered an unanswerable question


@dataclass
class Metrics:
    tau: float
    n: int
    coverage: float            # fraction of all questions answered
    selective_risk: float      # error rate among answered
    execution_accuracy: float  # of answered answerable, fraction correct
    confident_wrong_rate: float  # of ALL questions, fraction answered-but-wrong
    abstention_recall: float   # of unanswerable, fraction abstained
    over_abstention: float     # of answerable, fraction wrongly refused


def metrics_at(records: list[EvalRecord], tau: float) -> Metrics:
    answerable = [r for r in records if r.kind == "answerable"]
    unanswerable = [r for r in records if r.kind == "unanswerable"]
    n = len(records)

    answered = [r for r in records if _answered(r, tau)]
    wrong = [r for r in records if _is_wrong(r, tau)]

    ans_answered = [r for r in answerable if _answered(r, tau)]
    ex = (
        sum(1 for r in ans_answered if r.answer_correct) / len(ans_answered)
        if ans_answered else 1.0
    )
    return Metrics(
        tau=tau,
        n=n,
        coverage=len(answered) / n if n else 0.0,
        selective_risk=len(wrong) / len(answered) if answered else 0.0,
        execution_accuracy=ex,
        confident_wrong_rate=len(wrong) / n if n else 0.0,
        abstention_recall=(
            sum(1 for r in unanswerable if not _answered(r, tau)) / len(unanswerable)
            if unanswerable else 1.0
        ),
        over_abstention=(
            sum(1 for r in answerable if not _answered(r, tau)) / len(answerable)
            if answerable else 0.0
        ),
    )


def risk_coverage(records: list[EvalRecord], taus: list[float] | None = None) -> list[Metrics]:
    """Trace the risk-coverage curve across thresholds."""
    taus = taus if taus is not None else [i / 100 for i in range(0, 101, 5)]
    return [metrics_at(records, t) for t in taus]


def operating_point(records: list[EvalRecord], target_risk: float = 0.02) -> Metrics:
    """Pick the lowest τ (max coverage) whose selective risk ≤ target.

    Falls back to the τ with the minimum selective risk if none meets the target.
    """
    curve = risk_coverage(records, [i / 100 for i in range(0, 101, 1)])
    feasible = [m for m in curve if m.selective_risk <= target_risk]
    if feasible:
        return max(feasible, key=lambda m: m.coverage)
    return min(curve, key=lambda m: (m.selective_risk, -m.coverage))
