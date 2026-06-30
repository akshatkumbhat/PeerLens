"""Tests for the data-quality gate."""

from __future__ import annotations

import duckdb
import pytest

from peerlens.quality.checks import CheckResult, DataQualityError, assert_quality, run_checks


def _clean_warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dim_institution AS SELECT * FROM (VALUES (1,'A',1),(2,'B',2)) t(unitid,inst_name,sector)")
    con.execute("CREATE TABLE fact_admissions_funnel AS SELECT * FROM (VALUES (1,2020,0.3,0.4),(2,2020,0.5,0.6)) t(unitid,year,admit_rate,yield_rate)")
    con.execute("CREATE TABLE fact_retention AS SELECT * FROM (VALUES (1,2020,0.92),(2,2020,0.85)) t(unitid,year,retention_rate)")
    con.execute("CREATE TABLE fact_socioeconomic AS SELECT * FROM (VALUES (1,2020,15000.0,0.20,60000.0),(2,2020,22000.0,0.15,72000.0)) t(unitid,year,net_price,pell_rate,median_earnings)")
    return con


def test_clean_warehouse_passes() -> None:
    con = _clean_warehouse()
    results = assert_quality(con)
    assert all(r.passed for r in results)
    assert {r.name for r in results} >= {"admit_rate_bounded", "cohort_completeness"}


def test_out_of_range_admit_rate_fails() -> None:
    con = _clean_warehouse()
    con.execute("UPDATE fact_admissions_funnel SET admit_rate = 1.5 WHERE unitid = 1")
    with pytest.raises(DataQualityError, match="admit_rate_bounded"):
        assert_quality(con)


def test_referential_integrity_fails() -> None:
    con = _clean_warehouse()
    con.execute("INSERT INTO fact_retention VALUES (999, 2020, 0.8)")  # orphan unitid
    with pytest.raises(DataQualityError, match="retention_referential_integrity"):
        assert_quality(con)


def test_cohort_completeness_fails_when_retention_missing() -> None:
    con = _clean_warehouse()
    con.execute("INSERT INTO dim_institution VALUES (3, 'C', 1)")  # no funnel/retention rows
    with pytest.raises(DataQualityError, match="cohort_completeness"):
        assert_quality(con)


def test_run_checks_does_not_raise() -> None:
    con = _clean_warehouse()
    con.execute("UPDATE fact_retention SET retention_rate = 2.0 WHERE unitid = 1")
    results = run_checks(con)
    assert isinstance(results[0], CheckResult)
    bad = [r for r in results if not r.passed]
    assert any(r.name == "retention_rate_bounded" for r in bad)
