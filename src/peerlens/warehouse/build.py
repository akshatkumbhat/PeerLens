"""Build the PeerLens DuckDB star schema from cached IPEDS parquet.

Idempotent: every table is CREATE OR REPLACE, so re-running rebuilds the
warehouse deterministically from the raw caches. No network here — operates only
on ``data/raw/*.parquet``.

Star schema (Phase 1):
    dim_year
    dim_institution          (four-year cohort)
    fact_admissions_funnel   (sex=99 totals + derived admit/yield rates)
    fact_retention           (ftpt=1 first-time full-time retention)

Cohort: four-year (institution_level=4), public + private-nonprofit
(sector in {1,2}), with a non-zero admissions funnel and present retention,
ranked by applicants — top ``cohort_size``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from peerlens import config
from peerlens.ingest import scorecard, urban

COHORT_SIZE = 200
FOUR_YEAR_LEVEL = 4
FOUR_YEAR_SECTORS = (1, 2)

# IPEDS BEA region codes -> labels.
_REGION_CASE = """
    CASE region
        WHEN 0 THEN 'US Service Schools'
        WHEN 1 THEN 'New England'
        WHEN 2 THEN 'Mid East'
        WHEN 3 THEN 'Great Lakes'
        WHEN 4 THEN 'Plains'
        WHEN 5 THEN 'Southeast'
        WHEN 6 THEN 'Southwest'
        WHEN 7 THEN 'Rocky Mountains'
        WHEN 8 THEN 'Far West'
        WHEN 9 THEN 'Outlying Areas'
        ELSE 'Unknown'
    END
"""

_SECTOR_CASE = """
    CASE sector
        WHEN 1 THEN 'Public, 4-year'
        WHEN 2 THEN 'Private nonprofit, 4-year'
        ELSE 'Other'
    END
"""


def build_warehouse(
    year: int | None = None,
    raw_dir: Path | None = None,
    db_path: Path | None = None,
    *,
    cohort_size: int = COHORT_SIZE,
) -> dict[str, int]:
    """Build the warehouse for ``year``; return ``{table: row_count}``."""
    year = year if year is not None else config.get_settings().ipeds_year
    raw_dir = raw_dir or config.RAW_DIR
    db_path = db_path or config.WAREHOUSE_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)

    dir_pq = str(urban.cache_path(raw_dir, "directory", year))
    adm_pq = str(urban.cache_path(raw_dir, "admissions-enrollment", year))
    ret_pq = str(urban.cache_path(raw_dir, "fall-retention", year))
    for p in (dir_pq, adm_pq, ret_pq):
        if not Path(p).exists():
            raise FileNotFoundError(f"missing raw cache: {p} — run `peerlens ingest` first")

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE _stg_directory AS SELECT * FROM read_parquet(?)", [dir_pq])
        con.execute("CREATE OR REPLACE TABLE _stg_admissions AS SELECT * FROM read_parquet(?)", [adm_pq])
        con.execute("CREATE OR REPLACE TABLE _stg_retention AS SELECT * FROM read_parquet(?)", [ret_pq])

        # dim_year (single year for Phase 1)
        con.execute(
            "CREATE OR REPLACE TABLE dim_year AS SELECT ? AS year, ? AS academic_year",
            [year, f"{year}-{(year + 1) % 100:02d}"],
        )

        # Candidate four-year institutions.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE _cand_institution AS
            SELECT
                unitid,
                inst_name,
                state_abbr,
                region,
                {_REGION_CASE} AS region_name,
                sector,
                {_SECTOR_CASE} AS sector_name,
                inst_control,
                inst_size,
                CAST(hbcu AS INTEGER) AS hbcu,
                cc_basic_2021,
                longitude,
                latitude
            FROM _stg_directory
            WHERE institution_level = {FOUR_YEAR_LEVEL}
              AND sector IN {FOUR_YEAR_SECTORS}
            """
        )

        # Totals (sex=99) and full-time retention (ftpt=1).
        con.execute(
            """
            CREATE OR REPLACE TABLE _adm_total AS
            SELECT unitid, number_applied, number_admitted,
                   number_enrolled_ft, number_enrolled_pt, number_enrolled_total
            FROM _stg_admissions WHERE sex = 99
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE _ret_ft AS
            SELECT unitid, retention_rate, returning_students, prev_cohort_adj
            FROM _stg_retention WHERE ftpt = 1
            """
        )

        # Cohort: complete data + non-zero funnel, top N by applicants.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE _cohort AS
            SELECT c.unitid
            FROM _cand_institution c
            JOIN _adm_total a USING (unitid)
            JOIN _ret_ft r USING (unitid)
            WHERE a.number_applied > 0 AND r.retention_rate IS NOT NULL
            ORDER BY a.number_applied DESC
            LIMIT {int(cohort_size)}
            """
        )

        # Final dimension + facts, restricted to the cohort.
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_institution AS
            SELECT c.* FROM _cand_institution c
            JOIN _cohort co USING (unitid)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE fact_admissions_funnel AS
            SELECT
                a.unitid,
                {year} AS year,
                a.number_applied,
                a.number_admitted,
                a.number_enrolled_total,
                a.number_enrolled_ft,
                a.number_enrolled_pt,
                a.number_admitted::DOUBLE / NULLIF(a.number_applied, 0) AS admit_rate,
                a.number_enrolled_total::DOUBLE / NULLIF(a.number_admitted, 0) AS yield_rate
            FROM _adm_total a
            JOIN _cohort co USING (unitid)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE fact_retention AS
            SELECT
                r.unitid,
                {year} AS year,
                r.retention_rate,
                r.returning_students,
                r.prev_cohort_adj
            FROM _ret_ft r
            JOIN _cohort co USING (unitid)
            """
        )

        # Socio-economic augmentation (College Scorecard). Built from the cached
        # pull when present; otherwise an empty table so the schema and metrics
        # always exist (querying them then abstains with no_data rather than failing).
        sc_pq = scorecard.cache_path(raw_dir)
        if Path(sc_pq).exists():
            con.execute("CREATE OR REPLACE TABLE _stg_scorecard AS SELECT * FROM read_parquet(?)", [str(sc_pq)])
            con.execute(
                f"""
                CREATE OR REPLACE TABLE fact_socioeconomic AS
                SELECT s.unitid, {year} AS year, s.net_price, s.pell_rate, s.median_earnings
                FROM _stg_scorecard s JOIN _cohort co USING (unitid)
                """
            )
            con.execute("DROP TABLE IF EXISTS _stg_scorecard")
        else:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE fact_socioeconomic AS
                SELECT unitid, {year} AS year,
                       CAST(NULL AS DOUBLE) AS net_price,
                       CAST(NULL AS DOUBLE) AS pell_rate,
                       CAST(NULL AS DOUBLE) AS median_earnings
                FROM _cohort WHERE FALSE
                """
            )

        # Drop staging/intermediate tables.
        for t in ("_stg_directory", "_stg_admissions", "_stg_retention",
                  "_cand_institution", "_adm_total", "_ret_ft", "_cohort"):
            con.execute(f"DROP TABLE IF EXISTS {t}")

        # Data-quality gate — raises DataQualityError on any error-severity violation,
        # so a malformed warehouse never reaches the agent or the page.
        from peerlens.quality.checks import assert_quality

        assert_quality(con)

        tables = ["dim_year", "dim_institution", "fact_admissions_funnel",
                  "fact_retention", "fact_socioeconomic"]
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        return counts
    finally:
        con.close()
