"""Harness test with a scripted FakePlanModel against the synthetic warehouse."""

from __future__ import annotations

from peerlens.agent.model import FakePlanModel, ModelError
from peerlens.agent.plan import Intent, QueryPlan
from peerlens.config import Settings
from peerlens.eval.harness import run_eval


class _ScriptedModel:
    """Returns a plan keyed by a substring of the question (None if unmatched)."""

    def __init__(self, routes: dict[str, QueryPlan]):
        self._routes = routes

    def propose(self, question, linked, *, temperature):
        for needle, plan in self._routes.items():
            if needle.lower() in question.lower():
                return plan
        return None


def test_run_eval_scores_correct_and_abstain(agent_warehouse) -> None:
    gold = {
        "answerable": [
            {"id": "a1", "question": "retention at Alpha University", "institution": "Alpha University", "metric": "retention_rate"},
        ],
        "unanswerable": [
            {"id": "u1", "question": "retention at Nowhere Tech", "expected_reason": "unknown_institution"},
        ],
    }
    routes = {
        "Alpha University": QueryPlan(intent=Intent.SINGLE, institution="Alpha University", metric="retention_rate"),
        "Nowhere Tech": QueryPlan(intent=Intent.SINGLE, institution="Nowhere Tech", metric="retention_rate"),
    }
    recs = run_eval(agent_warehouse, _ScriptedModel(routes), Settings(agent_samples=3, agent_temperature=0.0), gold)

    by_id = {r.id: r for r in recs}
    assert by_id["a1"].top_kind == "answer"
    assert by_id["a1"].answer_correct is True
    assert by_id["u1"].top_kind == "abstain"
    assert by_id["u1"].predicted_reason == "unknown_institution"


def test_run_eval_flags_wrong_metric_as_incorrect(agent_warehouse) -> None:
    gold = {
        "answerable": [
            {"id": "a1", "question": "admit rate at Alpha University", "institution": "Alpha University", "metric": "admit_rate"},
        ],
        "unanswerable": [],
    }
    # model answers the right institution but the WRONG metric -> not correct
    wrong = QueryPlan(intent=Intent.SINGLE, institution="Alpha University", metric="retention_rate")
    recs = run_eval(agent_warehouse, FakePlanModel([wrong]), Settings(agent_samples=3, agent_temperature=0.0), gold)
    assert recs[0].top_kind == "answer"
    assert recs[0].answer_correct is False


def test_run_eval_preserves_partial_on_api_error(agent_warehouse) -> None:
    """A mid-run ModelError returns the records collected so far, not nothing."""

    class _FlakyModel:
        def __init__(self):
            self.calls = 0

        def propose(self, question, linked, *, temperature):
            self.calls += 1
            if self.calls > 3:  # first question's 3 samples ok, then fail
                raise ModelError("Gemini API error 429: quota exhausted")
            return QueryPlan(intent=Intent.SINGLE, institution="Alpha University", metric="retention_rate")

    gold = {
        "answerable": [
            {"id": "a1", "question": "retention at Alpha University", "institution": "Alpha University", "metric": "retention_rate"},
            {"id": "a2", "question": "retention at Beta University", "institution": "Beta University", "metric": "retention_rate"},
        ],
        "unanswerable": [],
    }
    recs = run_eval(agent_warehouse, _FlakyModel(), Settings(agent_samples=3, agent_temperature=0.0), gold)
    assert len(recs) == 1  # first question survived; run stopped cleanly on the second
    assert recs[0].id == "a1"
