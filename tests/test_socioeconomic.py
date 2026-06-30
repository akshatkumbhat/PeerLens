"""Socio-economic augmentation: new metrics resolve + answer, Pell joins peer features."""

from __future__ import annotations

from peerlens.agent import resolve
from peerlens.agent.answer import render_answer
from peerlens.agent.execute import execute_resolved
from peerlens.agent.plan import Intent, QueryPlan
from peerlens.peers.features import load_features


def _plan(metric: str, institution: str = "Alpha University") -> QueryPlan:
    return QueryPlan(intent=Intent.SINGLE, institution=institution, metric=metric)


def test_net_price_resolves_and_formats_as_dollars(socio_warehouse) -> None:
    rp = resolve.resolve_plan(socio_warehouse, _plan("net_price"))
    text = render_answer(rp, execute_resolved(socio_warehouse, rp))
    assert "Alpha University" in text
    assert "$12,000" in text  # seeded net price for unitid 1


def test_pell_rate_formats_as_percent(socio_warehouse) -> None:
    rp = resolve.resolve_plan(socio_warehouse, _plan("pell_rate"))
    assert "30.0%" in render_answer(rp, execute_resolved(socio_warehouse, rp))


def test_median_earnings_formats_as_dollars(socio_warehouse) -> None:
    rp = resolve.resolve_plan(socio_warehouse, _plan("median_earnings"))
    assert "$70,000" in render_answer(rp, execute_resolved(socio_warehouse, rp))


def test_pell_enters_peer_features_when_present(socio_warehouse) -> None:
    fm = load_features(socio_warehouse)
    assert "pell_rate" in fm.feature_names
    assert fm.X.shape[1] == len(fm.feature_names)


def test_pell_absent_from_features_without_scorecard(agent_warehouse) -> None:
    fm = load_features(agent_warehouse)
    assert "pell_rate" not in fm.feature_names  # base features only
