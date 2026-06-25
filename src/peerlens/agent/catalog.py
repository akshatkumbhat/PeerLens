"""Data catalog and schema linking.

The catalog is the deterministic source of truth the model is checked against.
Schema linking retrieves only the tables/metrics relevant to a question rather
than dumping the whole schema (the failure mode of off-the-shelf tools). Our
schema is tiny so linking is keyword-based; the README explains how the same
interface scales to a large schema (swap the linker for embedding retrieval —
the rest of the pipeline is unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from peerlens.warehouse import queries


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    table: str
    column: str
    is_rate: bool


# Built from the warehouse query layer so the catalog can't drift from the SQL.
METRICS: dict[str, MetricSpec] = {
    key: MetricSpec(key, label, table, column, key in {"admit_rate", "yield_rate", "retention_rate"})
    for key, (table, column, label, _higher) in queries.METRICS.items()
}

# Keyword → metric key. Lowercased substring match; first hit wins by order.
_SYNONYMS: list[tuple[tuple[str, ...], str]] = [
    (("retention", "retain", "retained", "came back", "returning"), "retention_rate"),
    (("admit", "admission", "acceptance", "accept rate", "selectiv"), "admit_rate"),
    (("yield", "matriculat"), "yield_rate"),
    (("applicant", "application", "applied", "number of app"), "applied"),
    (("enroll", "enrolled", "class size", "freshman class"), "enrolled"),
]


@dataclass
class LinkedCatalog:
    """The relevant slice of the catalog for one question."""

    metrics: list[MetricSpec]
    tables: list[str]


def link(question: str) -> LinkedCatalog:
    """Return the metrics/tables relevant to ``question`` (schema linking)."""
    q = question.lower()
    hits: list[MetricSpec] = []
    seen: set[str] = set()
    for needles, key in _SYNONYMS:
        if any(n in q for n in needles) and key not in seen:
            hits.append(METRICS[key])
            seen.add(key)
    # Fall back to the whole (small) metric catalog when nothing matched, so the
    # model still gets a grounded menu rather than guessing blind.
    metrics = hits or list(METRICS.values())
    tables = ["dim_institution", *sorted({m.table for m in metrics})]
    return LinkedCatalog(metrics=metrics, tables=tables)


def metric_menu() -> str:
    """Human-readable list of answerable metrics (for prompts and abstentions)."""
    return ", ".join(f"{m.key} ({m.label})" for m in METRICS.values())


def available_years(con: duckdb.DuckDBPyConnection) -> list[int]:
    return [r[0] for r in con.execute("SELECT year FROM dim_year ORDER BY year").fetchall()]
