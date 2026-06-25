"""Template-first SQL for a ResolvedPlan, plus a canonical result signature.

The model fills slots; it does not free-write SQL on this path (that kills most
hallucination and injection risk). Free-form read-only SQL is a constrained
fallback handled in ``freeform.py`` with execution-guided correction. Every
execution carries its SQL + params so the answer can show its work.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from peerlens.agent import catalog
from peerlens.agent.plan import Intent, ResolvedPlan


@dataclass
class Execution:
    sql: str
    params: list[object]
    rows: list[dict]


def build_sql(plan: ResolvedPlan) -> tuple[str, list[object]]:
    """Render the template for ``plan`` into SQL + params."""
    spec = catalog.METRICS[plan.metric]
    year = plan.years[0]

    if plan.intent == Intent.SINGLE or not plan.comparison_unitids:
        sql = (
            f"SELECT i.unitid, i.inst_name, i.state_abbr, m.{spec.column} AS metric_value, "
            f"TRUE AS is_target "
            f"FROM dim_institution i JOIN {spec.table} m USING (unitid) "
            f"WHERE i.unitid = ? AND m.year = ?"
        )
        return sql, [plan.target_unitid, year]

    unitids = [plan.target_unitid, *plan.comparison_unitids]
    placeholders = ", ".join("?" for _ in unitids)
    sql = (
        f"SELECT i.unitid, i.inst_name, i.state_abbr, m.{spec.column} AS metric_value, "
        f"(i.unitid = ?) AS is_target "
        f"FROM dim_institution i JOIN {spec.table} m USING (unitid) "
        f"WHERE m.year = ? AND i.unitid IN ({placeholders}) "
        f"ORDER BY metric_value DESC NULLS LAST"
    )
    return sql, [plan.target_unitid, year, *unitids]


def execute_resolved(con: duckdb.DuckDBPyConnection, plan: ResolvedPlan) -> Execution:
    """Run the template for ``plan`` and return rows + the SQL behind them."""
    sql, params = build_sql(plan)
    rows = con.execute(sql, params).pl().to_dicts()
    return Execution(sql=sql, params=params, rows=rows)


def result_signature(execution: Execution, *, ndigits: int = 3) -> tuple:
    """Canonical, order-stable signature of a result — the self-consistency key.

    Rates rounded to ``ndigits``; counts kept exact. Two executions with the same
    signature are 'the same answer' when grouping samples.
    """
    items = []
    for r in execution.rows:
        v = r.get("metric_value")
        v = round(float(v), ndigits) if isinstance(v, (int, float)) else v
        items.append((int(r["unitid"]), v))
    return tuple(sorted(items))
