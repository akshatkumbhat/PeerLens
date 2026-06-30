"""Offline tests for the College Scorecard client (mocked transport)."""

from __future__ import annotations

import httpx

from peerlens.ingest.scorecard import ScorecardClient


def _rec(uid, pub=None, priv=None, pell=0.2, earn=50000):
    return {
        "id": uid,
        "latest.cost.avg_net_price.public": pub,
        "latest.cost.avg_net_price.private": priv,
        "latest.aid.pell_grant_rate": pell,
        "latest.earnings.10_yrs_after_entry.median": earn,
    }


def _client(pages: list[list[dict]], total: int) -> ScorecardClient:
    def handler(req: httpx.Request) -> httpx.Response:
        page = int(req.url.params.get("page", 0))
        results = pages[page] if page < len(pages) else []
        return httpx.Response(200, json={"metadata": {"total": total}, "results": results})

    return ScorecardClient(api_key="dummy", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_fetch_all_parses_and_coalesces_net_price() -> None:
    c = _client([[_rec(1, pub=12000, pell=0.30, earn=60000),
                  _rec(2, priv=25000, pell=0.18, earn=70000)]], total=2)
    rows = {r["unitid"]: r for r in c.fetch_all().to_dicts()}
    assert rows[1]["net_price"] == 12000.0          # public used
    assert rows[2]["net_price"] == 25000.0          # private coalesced in
    assert rows[1]["pell_rate"] == 0.30
    assert rows[2]["median_earnings"] == 70000.0
    # only the four output columns survive (split net-price cols dropped)
    assert set(rows[1]) == {"unitid", "net_price", "pell_rate", "median_earnings"}


def test_pagination_follows_total() -> None:
    page0 = [_rec(i, pub=10000) for i in range(1, 101)]   # 100 rows
    page1 = [_rec(i, pub=10000) for i in range(101, 151)]  # 50 rows
    df = _client([page0, page1], total=150).fetch_all()
    assert df.height == 150  # both pages fetched and concatenated


def test_empty_results_returns_typed_empty_frame() -> None:
    df = _client([[]], total=0).fetch_all()
    assert df.height == 0
    assert set(df.columns) == {"unitid", "net_price", "pell_rate", "median_earnings"}
