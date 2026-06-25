"""Offline tests for the Urban IPEDS client — no network, via MockTransport."""

from __future__ import annotations

import httpx
import polars as pl

from peerlens.ingest import urban


def _make_client(pages: list[dict]) -> httpx.Client:
    """Build an httpx.Client whose transport replays ``pages`` in order."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        page = pages[state["i"]]
        state["i"] += 1
        return httpx.Response(200, json=page)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_iter_records_follows_pagination() -> None:
    pages = [
        {
            "count": 3,
            "next": "https://example.test/page2",
            "previous": None,
            "results": [{"unitid": 1}, {"unitid": 2}],
        },
        {"count": 3, "next": None, "previous": None, "results": [{"unitid": 3}]},
    ]
    with urban.UrbanIPEDSClient(client=_make_client(pages)) as c:
        recs = list(c.iter_records("directory", 2020))
    assert [r["unitid"] for r in recs] == [1, 2, 3]


def test_fetch_all_returns_dataframe() -> None:
    pages = [
        {
            "count": 2,
            "next": None,
            "previous": None,
            "results": [
                {"unitid": 234030, "number_applied": 18242, "sex": 99},
                {"unitid": 234030, "number_applied": 6723, "sex": 1},
            ],
        }
    ]
    with urban.UrbanIPEDSClient(client=_make_client(pages)) as c:
        df = c.fetch_all("admissions-enrollment", 2020, sex=99)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert set(df.columns) >= {"unitid", "number_applied", "sex"}


def test_pull_to_parquet_caches_and_skips(tmp_path) -> None:
    pages = [{"count": 1, "next": None, "previous": None, "results": [{"unitid": 1}]}]
    out = urban.pull_to_parquet("directory", 2020, tmp_path, client=_make_client(pages))
    assert out.exists()
    assert out == urban.cache_path(tmp_path, "directory", 2020)
    # second call with a transport that would raise if hit — cache short-circuits it
    def boom(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("network should not be called when cache exists")

    again = urban.pull_to_parquet(
        "directory", 2020, tmp_path, client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    assert again == out
