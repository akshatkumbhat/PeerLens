"""Hermetic tests for the warehouse build and query template.

The build test fabricates tiny parquet inputs (no network); the query test
builds a small in-memory-style warehouse and exercises the template.
"""

from __future__ import annotations

import duckdb
import polars as pl

from peerlens.ingest import urban
from peerlens.warehouse import queries
from peerlens.warehouse.build import build_warehouse


def _seed_raw(raw_dir, year=2020) -> None:
    """Write minimal directory/admissions/retention parquet for two four-year insts."""
    directory = pl.DataFrame(
        {
            "unitid": [1, 2, 3],
            "inst_name": ["Alpha U", "Beta College", "Gamma CC"],
            "state_abbr": ["VA", "CA", "TX"],
            "region": [5, 8, 6],
            "sector": [1, 2, 4],  # 1,2 four-year; 4 = two-year (excluded)
            "institution_level": [4, 4, 2],
            "inst_control": [1, 2, 1],
            "inst_size": [4, 2, 3],
            "hbcu": [2, 2, 2],
            "cc_basic_2021": [15, 21, 2],
            "longitude": [-77.0, -118.0, -97.0],
            "latitude": [37.0, 34.0, 30.0],
        }
    )
    admissions = pl.DataFrame(
        {
            "unitid": [1, 1, 2, 2, 3],
            "sex": [99, 1, 99, 1, 99],
            "number_applied": [10000, 4000, 5000, 2000, 0],
            "number_admitted": [3000, 1200, 4000, 1500, 0],
            "number_enrolled_ft": [900, 400, 1200, 500, 0],
            "number_enrolled_pt": [100, 50, 100, 40, 0],
            "number_enrolled_total": [1000, 450, 1300, 540, 0],
        }
    )
    retention = pl.DataFrame(
        {
            "unitid": [1, 1, 2, 3],
            "ftpt": [1, 2, 1, 1],
            "retention_rate": [0.92, 0.70, 0.85, 0.60],
            "returning_students": [920, 70, 850, 600],
            "prev_cohort_adj": [1000, 100, 1000, 1000],
        }
    )
    directory.write_parquet(urban.cache_path(raw_dir, "directory", year))
    admissions.write_parquet(urban.cache_path(raw_dir, "admissions-enrollment", year))
    retention.write_parquet(urban.cache_path(raw_dir, "fall-retention", year))


def test_build_warehouse_curates_four_year_cohort(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_raw(raw)
    db = tmp_path / "wh.duckdb"
    counts = build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)

    # Gamma (two-year, zero funnel) excluded; Alpha + Beta kept.
    assert counts["dim_institution"] == 2
    assert counts["fact_admissions_funnel"] == 2
    assert counts["fact_retention"] == 2

    con = duckdb.connect(str(db), read_only=True)
    try:
        # admit_rate derived correctly for Alpha: 3000/10000 = 0.30
        admit = con.execute(
            "SELECT admit_rate FROM fact_admissions_funnel WHERE unitid = 1"
        ).fetchone()[0]
        assert abs(admit - 0.30) < 1e-9
        # retention is the full-time (ftpt=1) row, not part-time
        ret = con.execute("SELECT retention_rate FROM fact_retention WHERE unitid = 1").fetchone()[0]
        assert abs(ret - 0.92) < 1e-9
        names = {r[0] for r in con.execute("SELECT inst_name FROM dim_institution").fetchall()}
        assert names == {"Alpha U", "Beta College"}
    finally:
        con.close()


def test_compare_to_peers_returns_rows_and_sql(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_raw(raw)
    db = tmp_path / "wh.duckdb"
    build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)

    con = duckdb.connect(str(db), read_only=True)
    try:
        res = queries.compare_to_peers(con, target_unitid=1, peer_unitids=[2], metric="admit_rate")
    finally:
        con.close()

    assert res.metric == "admit_rate"
    assert "fact_admissions_funnel" in res.sql
    assert res.rows.height == 2
    target_row = res.rows.filter(pl.col("is_target"))
    assert target_row.height == 1
    assert target_row["unitid"][0] == 1


def test_compare_to_peers_unknown_metric_raises(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_raw(raw)
    db = tmp_path / "wh.duckdb"
    build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)
    con = duckdb.connect(str(db), read_only=True)
    try:
        import pytest

        with pytest.raises(KeyError):
            queries.compare_to_peers(con, 1, [2], metric="graduation_rate")
    finally:
        con.close()
