"""Deterministic grounding: QueryPlan -> ResolvedPlan or Abstention.

No model in this module. Institution names resolve by normalized exact match,
then deterministic containment (not fuzzy ML) — which honestly surfaces
ambiguity ("Virginia" matches many) and unknowns (with closest suggestions by
word overlap). Metric and year checks come straight from the catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from peerlens.agent import catalog
from peerlens.agent.plan import (
    Abstention,
    AbstainReason,
    ComparisonKind,
    QueryPlan,
    ResolvedPlan,
)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


# Acronyms / short names people actually type, mapped to the canonical institution
# name in the IPEDS cohort. Each was validated to resolve 1:1 against the warehouse.
# Keyed and valued through _normalize at import, so a typed acronym becomes an
# exact-name hit in _match. Irregular ones (UVA, UCLA, MIT, NYU…) can't be derived
# by initials, so a curated map is the reliable approach for a small fixed cohort.
_ACRONYMS_RAW = {
    "uva": "University of Virginia-Main Campus",
    "vcu": "Virginia Commonwealth University",
    "virginia tech": "Virginia Polytechnic Institute and State University",
    "vt": "Virginia Polytechnic Institute and State University",
    "wvu": "West Virginia University",
    "mit": "Massachusetts Institute of Technology",
    "nyu": "New York University",
    "usc": "University of Southern California",
    "unc": "University of North Carolina at Chapel Hill",
    "umich": "University of Michigan-Ann Arbor",
    "penn": "University of Pennsylvania",
    "upenn": "University of Pennsylvania",
    "penn state": "The Pennsylvania State University",
    "umd": "University of Maryland-College Park",
    "uga": "University of Georgia",
    "georgia tech": "Georgia Institute of Technology-Main Campus",
    "gt": "Georgia Institute of Technology-Main Campus",
    "gwu": "George Washington University",
    "wustl": "Washington University in St Louis",
    "uw madison": "University of Wisconsin-Madison",
    "umn": "University of Minnesota-Twin Cities",
    "ucla": "University of California-Los Angeles",
    "ucb": "University of California-Berkeley",
    "uc berkeley": "University of California-Berkeley",
    "cal": "University of California-Berkeley",
    "uf": "University of Florida",
    "ut austin": "The University of Texas at Austin",
    "osu": "Ohio State University-Main Campus",
    "msu": "Michigan State University",
}
_ALIASES = {_normalize(k): _normalize(v) for k, v in _ACRONYMS_RAW.items()}


@dataclass
class _Inst:
    unitid: int
    name: str
    norm: str
    tokens: frozenset[str]


def _load_index(con: duckdb.DuckDBPyConnection) -> list[_Inst]:
    rows = con.execute("SELECT unitid, inst_name FROM dim_institution").fetchall()
    out: list[_Inst] = []
    for unitid, name in rows:
        norm = _normalize(name)
        out.append(_Inst(int(unitid), name, norm, frozenset(norm.split())))
    return out


def _match(index: list[_Inst], query: str) -> list[_Inst]:
    nq = _normalize(query)
    if not nq:
        return []
    nq = _ALIASES.get(nq, nq)  # expand a known acronym (UVA, UCLA, MIT, …) to its name
    exact = [i for i in index if i.norm == nq]
    if exact:
        return exact
    contains = [i for i in index if nq in i.norm]  # query is a substring of the name
    if contains:
        return contains
    return [i for i in index if i.norm in nq]  # name is a substring of the query


def _suggestions(index: list[_Inst], query: str, k: int = 5) -> list[str]:
    qtok = frozenset(_normalize(query).split())
    if not qtok:
        return []
    scored = sorted(
        index,
        key=lambda i: len(qtok & i.tokens) / max(1, len(qtok | i.tokens)),
        reverse=True,
    )
    return [i.name for i in scored[:k] if qtok & i.tokens]


def resolve_institution(con: duckdb.DuckDBPyConnection, name: str) -> list[int]:
    """Return the unitid(s) a name resolves to (0 = unknown, >1 = ambiguous).

    Public helper reused by the eval harness to compute gold institution ids.
    """
    return [m.unitid for m in _match(_load_index(con), name)]


def resolve_plan(con: duckdb.DuckDBPyConnection, plan: QueryPlan) -> ResolvedPlan | Abstention:
    """Ground a plan against the warehouse, or abstain with a precise reason."""
    index = _load_index(con)

    # 0) did the question name a measure at all? If not, clarify rather than guess
    # (distinct from an unknown metric, which we can name and decline precisely).
    if plan.metric.strip().lower() in {"", "unspecified", "unspecified_metric", "none"}:
        return Abstention(
            AbstainReason.UNSPECIFIED_METRIC,
            "Which measure would you like — admit rate, yield rate, retention rate, "
            "applicants, or enrollment?",
            options=list(catalog.METRICS),
        )

    # 1) metric in catalog?
    if plan.metric not in catalog.METRICS:
        return Abstention(
            AbstainReason.UNKNOWN_METRIC,
            f"I can't compute '{plan.metric}'. I can answer: {catalog.metric_menu()}.",
            options=list(catalog.METRICS),
        )

    # 2) years in scope?
    years_available = set(catalog.available_years(con))
    if not set(plan.years) <= years_available:
        return Abstention(
            AbstainReason.OUT_OF_SCOPE,
            f"I only have data for year(s) {sorted(years_available)}; "
            f"'{plan.years}' is outside that.",
        )

    # 3) target institution
    matches = _match(index, plan.institution)
    if not matches:
        return Abstention(
            AbstainReason.UNKNOWN_INSTITUTION,
            f"I don't have '{plan.institution}' in the dataset.",
            options=_suggestions(index, plan.institution),
        )
    if len(matches) > 1:
        return Abstention(
            AbstainReason.AMBIGUOUS_INSTITUTION,
            f"'{plan.institution}' matches several institutions — which did you mean?",
            options=[m.name for m in sorted(matches, key=lambda i: i.name)][:8],
        )
    target = matches[0]

    # 4) comparison set
    comparison_unitids: list[int] = []
    if plan.comparison.kind == ComparisonKind.EXPLICIT:
        for name in plan.comparison.institutions:
            m = _match(index, name)
            if not m:
                return Abstention(
                    AbstainReason.UNKNOWN_INSTITUTION,
                    f"I don't have comparison institution '{name}'.",
                    options=_suggestions(index, name),
                )
            if len(m) > 1:
                return Abstention(
                    AbstainReason.AMBIGUOUS_INSTITUTION,
                    f"Comparison institution '{name}' is ambiguous.",
                    options=[x.name for x in m][:8],
                )
            comparison_unitids.append(m[0].unitid)
    elif plan.comparison.kind in (ComparisonKind.PEERS, ComparisonKind.ASPIRANTS):
        set_type = "peer" if plan.comparison.kind == ComparisonKind.PEERS else "aspirant"
        rows = con.execute(
            "SELECT peer_unitid FROM bridge_peer_set "
            "WHERE target_unitid = ? AND set_type = ? ORDER BY rank LIMIT ?",
            [target.unitid, set_type, plan.comparison.k],
        ).fetchall()
        comparison_unitids = [int(r[0]) for r in rows]
        if not comparison_unitids:
            return Abstention(
                AbstainReason.NO_DATA,
                f"No {set_type} set is available for {target.name} "
                f"(it may be in the most-selective band, which has no aspirants).",
            )

    return ResolvedPlan(
        intent=plan.intent,
        target_unitid=target.unitid,
        target_name=target.name,
        metric=plan.metric,
        comparison=plan.comparison,
        comparison_unitids=comparison_unitids,
        years=plan.years,
    )
