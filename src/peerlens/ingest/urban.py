"""Client for the Urban Institute Education Data API (IPEDS).

No API key needed. The client paginates a topic fully (following ``next``) and
caches the raw pull to parquet so every downstream step is reproducible offline.

Design: the HTTP layer is injectable (``client`` arg) so tests can drive it with
an ``httpx.MockTransport`` and never touch the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import polars as pl

BASE_URL = "https://educationdata.urban.org/api/v1/college-university/ipeds"
# Urban API caps page size at 10000.
PAGE_LIMIT = 10000


class UrbanIPEDSClient:
    """Paginating reader for one IPEDS topic/year."""

    def __init__(self, client: httpx.Client | None = None, timeout: float = 60.0) -> None:
        self._own_client = client is None
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    def __enter__(self) -> UrbanIPEDSClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def iter_records(self, topic: str, year: int, **filters: object) -> Iterator[dict]:
        """Yield every record for ``topic``/``year``, following pagination.

        ``filters`` become query params (e.g. ``sex=99``, ``sector=1``).
        """
        url: str | None = f"{BASE_URL}/{topic}/{year}/"
        params: dict[str, object] | None = {"limit": PAGE_LIMIT, **filters}
        while url:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            yield from payload.get("results") or []
            # `next` is an absolute URL already carrying its own query string.
            url = payload.get("next")
            params = None

    def fetch_all(self, topic: str, year: int, **filters: object) -> pl.DataFrame:
        """Return all records for ``topic``/``year`` as a Polars DataFrame."""
        records = list(self.iter_records(topic, year, **filters))
        return pl.DataFrame(records)


def cache_path(raw_dir: Path, topic: str, year: int) -> Path:
    """Deterministic parquet location for a topic/year raw pull."""
    return raw_dir / f"ipeds_{topic}_{year}.parquet"


def pull_to_parquet(
    topic: str,
    year: int,
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    overwrite: bool = False,
    **filters: object,
) -> Path:
    """Pull a topic/year and cache it to parquet; return the path.

    Skips the network when a cache already exists and ``overwrite`` is False.
    """
    out = cache_path(raw_dir, topic, year)
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with UrbanIPEDSClient(client=client) as c:
        frame = c.fetch_all(topic, year, **filters)
    frame.write_parquet(out)
    return out
