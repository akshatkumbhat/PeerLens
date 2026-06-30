"""Data-quality layer — fails the build on violations.

Each check is a SQL probe that counts violating rows; a non-zero count fails it.
``assert_quality`` raises ``DataQualityError`` if any error-severity check fails,
so a bad warehouse never reaches the agent or the page. Warn-severity checks are
reported but do not fail the build.

Checks (Phase 2):
- bounded rates: admit / yield / retention in [0, 1]
- referential integrity: every fact unitid exists in dim_institution
- not-null keys: dim_institution identity columns are populated
- cohort completeness: every institution has both a funnel and a retention row
- year-over-year sanity: |Δ retention| ≤ 0.5 between adjacent years (active only
  when the warehouse holds more than one year; a no-op for the Phase 1/2 slice)
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


class DataQualityError(Exception):
    """Raised when one or more error-severity checks fail."""


@dataclass(frozen=True)
class Check:
    name: str
    sql: str  # must SELECT a single integer: the count of violating rows
    severity: str = "error"  # "error" fails the build; "warn" is reported only
    detail: str = ""


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: str
    n_violations: int
    detail: str

    @property
    def passed(self) -> bool:
        return self.n_violations == 0


CHECKS: tuple[Check, ...] = (
    Check(
        "admit_rate_bounded",
        "SELECT COUNT(*) FROM fact_admissions_funnel "
        "WHERE admit_rate IS NOT NULL AND (admit_rate < 0 OR admit_rate > 1)",
        detail="admit_rate must be in [0, 1]",
    ),
    Check(
        "yield_rate_bounded",
        "SELECT COUNT(*) FROM fact_admissions_funnel "
        "WHERE yield_rate IS NOT NULL AND (yield_rate < 0 OR yield_rate > 1)",
        detail="yield_rate must be in [0, 1]",
    ),
    Check(
        "retention_rate_bounded",
        "SELECT COUNT(*) FROM fact_retention "
        "WHERE retention_rate IS NOT NULL AND (retention_rate < 0 OR retention_rate > 1)",
        detail="retention_rate must be in [0, 1]",
    ),
    Check(
        "admissions_referential_integrity",
        "SELECT COUNT(*) FROM fact_admissions_funnel f "
        "LEFT JOIN dim_institution d USING (unitid) WHERE d.unitid IS NULL",
        detail="every fact_admissions_funnel.unitid must exist in dim_institution",
    ),
    Check(
        "retention_referential_integrity",
        "SELECT COUNT(*) FROM fact_retention f "
        "LEFT JOIN dim_institution d USING (unitid) WHERE d.unitid IS NULL",
        detail="every fact_retention.unitid must exist in dim_institution",
    ),
    Check(
        "dim_keys_not_null",
        "SELECT COUNT(*) FROM dim_institution "
        "WHERE unitid IS NULL OR inst_name IS NULL OR sector IS NULL",
        detail="dim_institution identity columns must be populated",
    ),
    Check(
        "cohort_completeness",
        "SELECT COUNT(*) FROM dim_institution d "
        "WHERE NOT EXISTS (SELECT 1 FROM fact_admissions_funnel f WHERE f.unitid = d.unitid) "
        "   OR NOT EXISTS (SELECT 1 FROM fact_retention r WHERE r.unitid = d.unitid)",
        detail="every cohort institution must have a funnel and a retention row",
    ),
    Check(
        "yoy_retention_sanity",
        # Adjacent-year retention should not swing more than 0.5; no-op with one year.
        "SELECT COUNT(*) FROM ("
        "  SELECT unitid, retention_rate,"
        "         LAG(retention_rate) OVER (PARTITION BY unitid ORDER BY year) AS prev"
        "  FROM fact_retention"
        ") t WHERE prev IS NOT NULL AND ABS(retention_rate - prev) > 0.5",
        detail="year-over-year retention change should be within 0.5",
    ),
    # Socio-economic augmentation (College Scorecard). Empty table -> 0 violations,
    # so these are no-ops until the Scorecard pull has populated it.
    Check(
        "pell_rate_bounded",
        "SELECT COUNT(*) FROM fact_socioeconomic "
        "WHERE pell_rate IS NOT NULL AND (pell_rate < 0 OR pell_rate > 1)",
        detail="pell_rate must be in [0, 1]",
    ),
    Check(
        "net_price_sane",
        "SELECT COUNT(*) FROM fact_socioeconomic "
        "WHERE net_price IS NOT NULL AND (net_price < 0 OR net_price > 200000)",
        detail="net_price must be a non-negative, plausible dollar amount",
    ),
    Check(
        "median_earnings_nonneg",
        "SELECT COUNT(*) FROM fact_socioeconomic "
        "WHERE median_earnings IS NOT NULL AND median_earnings < 0",
        detail="median_earnings must be non-negative",
    ),
    Check(
        "socioeconomic_referential_integrity",
        "SELECT COUNT(*) FROM fact_socioeconomic f "
        "LEFT JOIN dim_institution d USING (unitid) WHERE d.unitid IS NULL",
        detail="every fact_socioeconomic.unitid must exist in dim_institution",
    ),
)


def run_checks(
    con: duckdb.DuckDBPyConnection, checks: tuple[Check, ...] = CHECKS
) -> list[CheckResult]:
    """Run every check and return its result (does not raise)."""
    results: list[CheckResult] = []
    for c in checks:
        (n,) = con.execute(c.sql).fetchone()
        results.append(CheckResult(c.name, c.severity, int(n), c.detail))
    return results


def assert_quality(
    con: duckdb.DuckDBPyConnection, checks: tuple[Check, ...] = CHECKS
) -> list[CheckResult]:
    """Run checks; raise ``DataQualityError`` on any error-severity failure."""
    results = run_checks(con, checks)
    failures = [r for r in results if not r.passed and r.severity == "error"]
    if failures:
        lines = [f"  - {r.name}: {r.n_violations} violation(s) — {r.detail}" for r in failures]
        raise DataQualityError("data-quality gate failed:\n" + "\n".join(lines))
    return results
