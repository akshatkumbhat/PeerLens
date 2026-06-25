"""Template-first SQL — the Phase 1 seed of PeerLens's parameterized query layer.

One template: compare a target institution to a fixed peer list on a single
metric. The model (Phase 3) will fill the slots — target, peers, metric — but
never free-write SQL on this path. Every call returns the rows *and* the exact
SQL + params, because surfacing the query is the whole trust argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import polars as pl

# metric key -> (fact table, column, human label, higher_is_better)
METRICS: dict[str, tuple[str, str, str, bool]] = {
    "admit_rate": ("fact_admissions_funnel", "admit_rate", "Admit rate", False),
    "yield_rate": ("fact_admissions_funnel", "yield_rate", "Yield rate", True),
    "retention_rate": ("fact_retention", "retention_rate", "Retention rate (first-time, full-time)", True),
    "applied": ("fact_admissions_funnel", "number_applied", "Applicants", True),
    "enrolled": ("fact_admissions_funnel", "number_enrolled_total", "Enrolled (total)", True),
}


@dataclass
class ComparisonResult:
    """A rendered template execution: the data plus the query behind it."""

    metric: str
    label: str
    sql: str
    params: list[object]
    rows: pl.DataFrame


def compare_to_peers(
    con: duckdb.DuckDBPyConnection,
    target_unitid: int,
    peer_unitids: list[int],
    metric: str,
) -> ComparisonResult:
    """Compare ``target_unitid`` to ``peer_unitids`` on ``metric``.

    Raises ``KeyError`` for an unknown metric (the deterministic catalog check
    that, in Phase 3, becomes an abstention rather than a guess).
    """
    if metric not in METRICS:
        raise KeyError(f"unknown metric: {metric!r}; known: {sorted(METRICS)}")
    fact_table, column, label, _higher = METRICS[metric]

    unitids = [target_unitid, *peer_unitids]
    placeholders = ", ".join("?" for _ in unitids)
    sql = f"""
        SELECT
            i.unitid,
            i.inst_name,
            i.state_abbr,
            i.sector_name,
            m.{column} AS metric_value,
            (i.unitid = ?) AS is_target
        FROM dim_institution i
        JOIN {fact_table} m USING (unitid)
        WHERE i.unitid IN ({placeholders})
        ORDER BY metric_value DESC NULLS LAST
    """.strip()
    params: list[object] = [target_unitid, *unitids]
    rows = con.execute(sql, params).pl()
    return ComparisonResult(metric=metric, label=label, sql=sql, params=params, rows=rows)


def list_institutions(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """All cohort institutions (for UI pickers), ordered by name."""
    return con.execute(
        "SELECT unitid, inst_name, state_abbr, sector_name FROM dim_institution ORDER BY inst_name"
    ).pl()
