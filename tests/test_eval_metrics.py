"""Deterministic tests for the eval metric math and risk-coverage sweep."""

from __future__ import annotations

from peerlens.eval.harness import EvalRecord
from peerlens.eval.metrics import metrics_at, operating_point, risk_coverage


def _records() -> list[EvalRecord]:
    return [
        EvalRecord("R1", "answerable", "q", 1.0, "answer", None, True, None),   # correct, confident
        EvalRecord("R2", "answerable", "q", 1.0, "answer", None, False, None),  # confident WRONG
        EvalRecord("R3", "answerable", "q", 1.0, "abstain", "ambiguous_institution", None, None),  # over-abstain
        EvalRecord("R4", "answerable", "q", 0.4, "answer", None, True, None),   # correct but low agreement
        EvalRecord("R5", "unanswerable", "q", 1.0, "abstain", "unknown_institution", None, "unknown_institution"),
        EvalRecord("R6", "unanswerable", "q", 1.0, "answer", None, None, "out_of_scope"),  # answered -> wrong
    ]


def test_metrics_at_tau_0_6() -> None:
    m = metrics_at(_records(), 0.6)
    assert m.coverage == 0.5                       # R1, R2, R6 answered of 6
    assert abs(m.selective_risk - 2 / 3) < 1e-9    # R2, R6 wrong of 3 answered
    assert m.execution_accuracy == 0.5             # of {R1,R2}, R1 correct
    assert abs(m.confident_wrong_rate - 2 / 6) < 1e-9
    assert m.abstention_recall == 0.5              # R5 abstained, R6 did not
    assert m.over_abstention == 0.5                # R3, R4 of 4 answerable refused


def test_higher_tau_reduces_coverage() -> None:
    curve = risk_coverage(_records(), [0.0, 0.5, 1.0])
    covs = [m.coverage for m in curve]
    assert covs[0] >= covs[1] >= covs[2]           # coverage is non-increasing in tau


def test_operating_point_prefers_low_risk_then_coverage() -> None:
    # clean set: confident answers are all correct; the only error is at low agreement
    recs = [
        EvalRecord("c1", "answerable", "q", 1.0, "answer", None, True, None),
        EvalRecord("c2", "answerable", "q", 1.0, "answer", None, True, None),
        EvalRecord("c3", "answerable", "q", 0.4, "answer", None, False, None),  # wrong, low agreement
        EvalRecord("u1", "unanswerable", "q", 1.0, "abstain", "out_of_scope", None, "out_of_scope"),
    ]
    op = operating_point(recs, target_risk=0.02)
    # the wrong answer only enters at tau <= 0.4, so the chosen tau excludes it
    assert op.tau > 0.4
    assert op.selective_risk <= 0.02
    assert op.coverage > 0.0
