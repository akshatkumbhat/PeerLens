"""End-to-end agent tests with a deterministic FakePlanModel (no network)."""

from __future__ import annotations

from peerlens.agent.model import FakePlanModel
from peerlens.agent.pipeline import run_agent
from peerlens.agent.plan import (
    AbstainReason,
    ComparisonKind,
    ComparisonSet,
    Intent,
    QueryPlan,
)
from peerlens.config import Settings


def _settings(**kw) -> Settings:
    base = dict(agent_samples=5, agent_tau=0.6, agent_temperature=0.0)
    base.update(kw)
    return Settings(**base)


def _plan(**kw) -> QueryPlan:
    base = dict(intent=Intent.SINGLE, institution="Alpha University", metric="retention_rate")
    base.update(kw)
    return QueryPlan(**base)


def test_confident_single_answer(agent_warehouse) -> None:
    model = FakePlanModel([_plan()])  # repeats -> 5 identical samples
    resp = run_agent(agent_warehouse, model, "What was Alpha's retention?", _settings())
    assert resp.answered
    assert resp.agreement == 1.0
    assert "97.0%" in resp.answer
    assert resp.sql and "fact_retention" in resp.sql


def test_confident_comparison_answer(agent_warehouse) -> None:
    plan = _plan(
        intent=Intent.COMPARE,
        institution="Delta College",
        metric="admit_rate",
        comparison=ComparisonSet(kind=ComparisonKind.PEERS, k=3),
    )
    resp = run_agent(agent_warehouse, FakePlanModel([plan]), "Delta vs peers admit rate?", _settings())
    assert resp.answered
    assert "Delta College" in resp.answer
    assert "ranks" in resp.answer.lower()
    assert len(resp.rows) >= 2


def test_unknown_institution_abstains(agent_warehouse) -> None:
    model = FakePlanModel([_plan(institution="Nowhere Tech")])
    resp = run_agent(agent_warehouse, model, "Nowhere Tech retention?", _settings())
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.UNKNOWN_INSTITUTION


def test_ambiguous_institution_clarifies(agent_warehouse) -> None:
    model = FakePlanModel([_plan(institution="University")])
    resp = run_agent(agent_warehouse, model, "University retention?", _settings())
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.AMBIGUOUS_INSTITUTION
    assert len(resp.abstention.options) > 1


def test_out_of_scope_year_abstains(agent_warehouse) -> None:
    model = FakePlanModel([_plan(years=[2019])])
    resp = run_agent(agent_warehouse, model, "Alpha retention in 2019?", _settings())
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.OUT_OF_SCOPE


def test_low_agreement_abstains(agent_warehouse) -> None:
    # five different institutions -> five distinct answers -> no group reaches tau
    plans = [_plan(institution=n) for n in [
        "Alpha University", "Beta University", "Gamma College", "Delta College", "Epsilon University",
    ]]
    resp = run_agent(agent_warehouse, FakePlanModel(plans), "ambiguous-ish", _settings())
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.LOW_AGREEMENT


def test_no_valid_plan_abstains(agent_warehouse) -> None:
    resp = run_agent(agent_warehouse, FakePlanModel([None]), "gibberish", _settings())
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.NO_VALID_PLAN


def test_split_between_answer_and_abstain_is_low_agreement(agent_warehouse) -> None:
    # 3 valid (same answer) + 2 unknown -> top group 3/5 = 0.6 >= tau -> answers;
    # tighten tau to 0.8 so neither group wins -> low agreement
    plans = [_plan(), _plan(), _plan(), _plan(institution="Nowhere"), _plan(institution="Nowhere")]
    resp = run_agent(agent_warehouse, FakePlanModel(plans), "q", _settings(agent_tau=0.8))
    assert not resp.answered
    assert resp.abstention.reason == AbstainReason.LOW_AGREEMENT
