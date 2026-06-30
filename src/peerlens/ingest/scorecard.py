"""Client for the U.S. Dept. of Education College Scorecard API (api.data.gov).

Needs a free key (``SCORECARD_API_KEY`` from https://api.data.gov/signup/). Pulls
the socio-economic fields PeerLens augments the warehouse with — average net
price, Pell-grant share, and median earnings — keyed by **IPEDS unitid** (the
Scorecard ``id`` field), and caches the parsed result to parquet so every
downstream step is reproducible offline.

The HTTP layer is injectable (``client`` arg) so tests drive it with an
``httpx.MockTransport`` and never touch the network — same pattern as the Urban
IPEDS client.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import polars as pl

BASE_URL = "https://api.data.gov/ed/collegescorecard/v1/schools"
PER_PAGE = 100  # API maximum

# Scorecard dotted field path -> our column name. Requesting flat `fields=` makes
# the API return these exact dotted keys on each result.
_FIELDS: dict[str, str] = {
    "id": "unitid",
    "latest.cost.avg_net_price.public": "_net_price_public",
    "latest.cost.avg_net_price.private": "_net_price_private",
    "latest.aid.pell_grant_rate": "pell_rate",
    "latest.earnings.10_yrs_after_entry.median": "median_earnings",
}
_FLOATS = ("_net_price_public", "_net_price_private", "pell_rate", "median_earnings")


class ScorecardClient:
    """Paginating reader for the socio-economic Scorecard fields."""

    def __init__(self, api_key: str, client: httpx.Client | None = None, timeout: float = 60.0):
        if not api_key:
            raise ValueError("SCORECARD_API_KEY is required for the College Scorecard client")
        self._api_key = api_key
        self._own_client = client is None
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    def __enter__(self) -> ScorecardClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def iter_records(self) -> Iterator[dict]:
        """Yield every school record, following the API's page metadata."""
        fields = ",".join(_FIELDS)
        page = 0
        while True:
            resp = self._client.get(
                BASE_URL,
                params={"api_key": self._api_key, "fields": fields,
                        "per_page": PER_PAGE, "page": page},
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results") or []
            yield from results
            total = (payload.get("metadata") or {}).get("total") or 0
            page += 1
            if not results or page * PER_PAGE >= total:
                break

    def fetch_all(self) -> pl.DataFrame:
        """Return one row per institution: unitid, net_price, pell_rate, median_earnings."""
        rows = [{our: rec.get(src) for src, our in _FIELDS.items()} for rec in self.iter_records()]
        df = pl.DataFrame(rows)
        if df.is_empty():
            return pl.DataFrame(schema={"unitid": pl.Int64, "net_price": pl.Float64,
                                        "pell_rate": pl.Float64, "median_earnings": pl.Float64})
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in _FLOATS])
        return df.with_columns(
            pl.coalesce(["_net_price_public", "_net_price_private"]).alias("net_price"),
            pl.col("unitid").cast(pl.Int64, strict=False),
        ).select("unitid", "net_price", "pell_rate", "median_earnings").drop_nulls("unitid")


def cache_path(raw_dir: Path) -> Path:
    return raw_dir / "scorecard_socioeconomic.parquet"


def pull_to_parquet(
    raw_dir: Path, api_key: str, *, client: httpx.Client | None = None, overwrite: bool = False
) -> Path:
    """Pull the socio-economic fields and cache to parquet; return the path."""
    out = cache_path(raw_dir)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ScorecardClient(api_key, client=client) as c:
        c.fetch_all().write_parquet(out)
    return out
