"""Shared test fixtures: a small synthetic warehouse with peer sets."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from peerlens.ingest import urban
from peerlens.peers.build import build_peer_sets
from peerlens.warehouse.build import build_warehouse


def _seed_six(raw_dir, year=2020) -> None:
    n = 6
    uids = list(range(1, n + 1))
    names = [
        "Alpha University",
        "Beta University",
        "Gamma College",
        "Delta College",
        "Epsilon University",
        "Zeta College",
    ]
    pl.DataFrame(
        {
            "unitid": uids,
            "inst_name": names,
            "state_abbr": ["VA"] * n,
            "region": [5] * n,
            "sector": [1, 1, 1, 2, 2, 2],
            "institution_level": [4] * n,
            "inst_control": [1] * n,
            "inst_size": [3] * n,
            "hbcu": [2] * n,
            "cc_basic_2021": [15] * n,
            "longitude": [-77.0] * n,
            "latitude": [37.0] * n,
        }
    ).write_parquet(urban.cache_path(raw_dir, "directory", year))

    admitted = [2000, 5400, 8000, 9800, 9600, 9000]
    pl.DataFrame(
        {
            "unitid": uids,
            "sex": [99] * n,
            "number_applied": [20000, 18000, 16000, 14000, 12000, 10000],
            "number_admitted": admitted,
            "number_enrolled_ft": [a // 3 for a in admitted],
            "number_enrolled_pt": [10] * n,
            "number_enrolled_total": [a // 3 + 10 for a in admitted],
        }
    ).write_parquet(urban.cache_path(raw_dir, "admissions-enrollment", year))

    pl.DataFrame(
        {
            "unitid": uids,
            "ftpt": [1] * n,
            "retention_rate": [0.97, 0.94, 0.90, 0.85, 0.80, 0.75],
            "returning_students": [900] * n,
            "prev_cohort_adj": [1000] * n,
        }
    ).write_parquet(urban.cache_path(raw_dir, "fall-retention", year))


@pytest.fixture
def agent_warehouse(tmp_path):
    """Return a read-only connection to a built 6-institution warehouse + peers."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_six(raw)
    db = tmp_path / "wh.duckdb"
    build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)
    build_peer_sets(db_path=db, k=3, n_bands=3)
    con = duckdb.connect(str(db), read_only=True)
    yield con
    con.close()


def _seed_scorecard(raw_dir, year=2020) -> None:
    from peerlens.ingest import scorecard
    pl.DataFrame(
        {
            "unitid": [1, 2, 3, 4, 5, 6],
            "net_price": [12000.0, 15000.0, 18000.0, 21000.0, 24000.0, 27000.0],
            "pell_rate": [0.30, 0.26, 0.22, 0.18, 0.14, 0.10],
            "median_earnings": [70000.0, 66000.0, 62000.0, 58000.0, 54000.0, 50000.0],
        }
    ).write_parquet(scorecard.cache_path(raw_dir))


@pytest.fixture
def socio_warehouse(tmp_path):
    """Like agent_warehouse, but with College Scorecard socio-economic data."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_six(raw)
    _seed_scorecard(raw)
    db = tmp_path / "wh.duckdb"
    build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)
    build_peer_sets(db_path=db, k=3, n_bands=3)
    con = duckdb.connect(str(db), read_only=True)
    yield con
    con.close()
