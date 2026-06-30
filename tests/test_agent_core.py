"""Unit tests for the agent's deterministic pieces: resolve, execute, answer."""

from __future__ import annotations

from peerlens.agent import resolve
from peerlens.agent.execute import build_sql, execute_resolved, result_signature
from peerlens.agent.answer import render_answer
from peerlens.agent.plan import (
    Abstention,
    AbstainReason,
    ComparisonKind,
    ComparisonSet,
    Intent,
    QueryPlan,
    ResolvedPlan,
)


def _plan(**kw) -> QueryPlan:
    base = dict(intent=Intent.SINGLE, institution="Alpha University", metric="retention_rate")
    base.update(kw)
    return QueryPlan(**base)


def test_resolve_known_institution(agent_warehouse) -> None:
    out = resolve.resolve_plan(agent_warehouse, _plan())
    assert isinstance(out, ResolvedPlan)
    assert out.target_name == "Alpha University"


def test_resolve_unknown_institution_suggests(agent_warehouse) -> None:
    out = resolve.resolve_plan(agent_warehouse, _plan(institution="Nowhere Tech"))
    assert isinstance(out, Abstention)
    assert out.reason == AbstainReason.UNKNOWN_INSTITUTION


def test_resolve_ambiguous_institution(agent_warehouse) -> None:
    # "University" is a substring of several names -> ambiguous
    out = resolve.resolve_plan(agent_warehouse, _plan(institution="University"))
    assert isinstance(out, Abstention)
    assert out.reason == AbstainReason.AMBIGUOUS_INSTITUTION
    assert len(out.options) > 1


def test_resolve_unknown_metric(agent_warehouse) -> None:
    out = resolve.resolve_plan(agent_warehouse, _plan(metric="graduation_rate"))
    assert isinstance(out, Abstention)
    assert out.reason == AbstainReason.UNKNOWN_METRIC


def test_resolve_unspecified_metric_asks_which(agent_warehouse) -> None:
    # question named no measure -> clarify (ask which), not a flat unknown_metric
    out = resolve.resolve_plan(agent_warehouse, _plan(metric="unspecified"))
    assert isinstance(out, Abstention)
    assert out.reason == AbstainReason.UNSPECIFIED_METRIC
    assert out.options  # offers the metric menu to pick from


def test_resolve_out_of_scope_year(agent_warehouse) -> None:
    out = resolve.resolve_plan(agent_warehouse, _plan(years=[2019]))
    assert isinstance(out, Abstention)
    assert out.reason == AbstainReason.OUT_OF_SCOPE


def test_resolve_peers(agent_warehouse) -> None:
    plan = _plan(
        intent=Intent.COMPARE,
        institution="Delta College",
        metric="admit_rate",
        comparison=ComparisonSet(kind=ComparisonKind.PEERS, k=3),
    )
    out = resolve.resolve_plan(agent_warehouse, plan)
    assert isinstance(out, ResolvedPlan)
    assert len(out.comparison_unitids) >= 1
    assert out.target_unitid not in out.comparison_unitids


def test_build_sql_single_is_parameterized(agent_warehouse) -> None:
    rp = resolve.resolve_plan(agent_warehouse, _plan())
    assert isinstance(rp, ResolvedPlan)
    sql, params = build_sql(rp)
    assert "JOIN fact_retention" in sql
    assert sql.count("?") == len(params)


def test_execute_and_signature_stable(agent_warehouse) -> None:
    rp = resolve.resolve_plan(agent_warehouse, _plan())
    ex = execute_resolved(agent_warehouse, rp)
    assert ex.rows and ex.rows[0]["metric_value"] is not None
    assert result_signature(ex) == result_signature(ex)


def test_render_answer_injects_real_number(agent_warehouse) -> None:
    rp = resolve.resolve_plan(agent_warehouse, _plan())
    ex = execute_resolved(agent_warehouse, rp)
    text = render_answer(rp, ex)
    assert "Alpha University" in text
    assert "97.0%" in text  # the seeded retention rate, injected from the row
