"""PeerLens — Phase 1 minimal page.

Pick an institution and a peer list, choose a metric, and see the comparison
*and the SQL behind it*. This is deliberately spare: it proves the thin slice
end to end and establishes the "show your work" pattern the agent will inherit.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from peerlens.peers.build import peers_for
from peerlens.warehouse import db, queries

st.set_page_config(page_title="PeerLens", layout="wide")


@st.cache_resource
def _connect():
    return db.connect(read_only=True)


@st.cache_data
def _institutions() -> pl.DataFrame:
    return queries.list_institutions(_connect())


def _has_bridge() -> bool:
    rows = _connect().execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'bridge_peer_set'"
    ).fetchall()
    return bool(rows)


def _neighbor_ids(target_unitid: int, set_type: str, limit: int = 8) -> list[int]:
    df = peers_for(_connect(), target_unitid, set_type, limit=limit)
    return df["unitid"].to_list() if df.height else []


def _agent_panel() -> None:
    """Natural-language agent — correct or silent. Inert without a Gemini key."""
    from peerlens.config import get_settings

    st.subheader("Ask PeerLens")
    q = st.text_input(
        "Ask about admit rate, yield, retention, applicants, or enrollment (2020)",
        placeholder="e.g. How does UVA's retention compare to its peers?",
    )
    if not q:
        return

    s = get_settings()
    if not s.gemini_api_key:
        st.info(
            "Set `GEMINI_API_KEY` in `.env` to enable the agent. "
            "Until then, use the comparison tool below."
        )
        return

    from peerlens.agent.model import ModelError, get_plan_model
    from peerlens.agent.pipeline import run_agent

    try:
        resp = run_agent(_connect(), get_plan_model(s), q, s)
    except ModelError as e:
        st.error(str(e))
        return

    if resp.answered:
        st.success(resp.answer)
        st.caption(f"confidence: {resp.agreement:.0%} agreement over {resp.n_samples} samples")
        with st.expander("the query behind this answer"):
            st.code(resp.sql, language="sql")
            st.dataframe(resp.rows, hide_index=True)
    else:
        ab = resp.abstention
        st.warning(f"**Abstained ({ab.reason.value}).** {ab.message}")
        if ab.options:
            st.caption("Options: " + ", ".join(ab.options[:6]))
        st.caption(f"agreement: {resp.agreement:.0%} over {resp.n_samples} samples")


def main() -> None:
    st.title("PeerLens")
    st.caption(
        "Grounded higher-ed comparisons — every number is computed by SQL and the "
        "query is always shown. (Phase 1 thin slice: IPEDS 2020, 200 four-year institutions.)"
    )

    _agent_panel()

    try:
        insts = _institutions()
    except FileNotFoundError:
        st.error("Warehouse not found. Run `uv run peerlens ingest && uv run peerlens build` first.")
        return

    labels = {
        f"{r['inst_name']} ({r['state_abbr']})": r["unitid"]
        for r in insts.iter_rows(named=True)
    }
    names = list(labels)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        target_label = st.selectbox("Institution", names, index=0)
        target_unitid = labels[target_label]

        metric_key = st.selectbox(
            "Metric",
            list(queries.METRICS),
            format_func=lambda k: queries.METRICS[k][2],
        )

        id_to_label = {v: k for k, v in labels.items()}
        options = ["Manual"]
        if _has_bridge():
            options = ["Mahalanobis peers", "Aspirants", "Manual"]
        comparison_set = st.radio("Comparison set", options, index=0)

        if comparison_set == "Mahalanobis peers":
            seed_ids = _neighbor_ids(target_unitid, "peer")
        elif comparison_set == "Aspirants":
            seed_ids = _neighbor_ids(target_unitid, "aspirant")
        else:
            seed_ids = [labels[n] for n in names if n != target_label][:5]

        default_peers = [id_to_label[i] for i in seed_ids if i in id_to_label]
        if comparison_set == "Aspirants" and not default_peers:
            st.caption("No aspirants — this institution is already in the most-selective band.")

        peer_labels = st.multiselect(
            "Peer institutions",
            [n for n in names if n != target_label],
            default=default_peers,
        )
        peer_unitids = [labels[n] for n in peer_labels]

    if not peer_unitids:
        st.info("Pick at least one peer institution to compare against.")
        return

    con = _connect()
    res = queries.compare_to_peers(con, target_unitid, peer_unitids, metric_key)

    with col_r:
        st.subheader(res.label)
        df = res.rows.with_columns(
            pl.when(pl.col("is_target")).then(pl.lit("★ ")).otherwise(pl.lit("")).alias("mark")
        )
        # target rank among the compared set
        ranked = df.sort("metric_value", descending=True, nulls_last=True)
        order = [r["unitid"] for r in ranked.iter_rows(named=True)]
        rank = order.index(target_unitid) + 1
        st.metric(
            f"{target_label} — {res.label}",
            value=_fmt(res.rows.filter(pl.col("is_target"))["metric_value"][0], metric_key),
            help=f"Rank {rank} of {len(order)} in the compared set",
        )

        chart = df.select(
            label=pl.col("mark") + pl.col("inst_name"),
            value="metric_value",
        ).to_pandas().set_index("label")
        st.bar_chart(chart, horizontal=True)

        st.dataframe(
            df.select("inst_name", "state_abbr", "sector_name", "metric_value", "is_target"),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("The query behind this answer", expanded=False):
        st.code(res.sql, language="sql")
        st.caption(f"params: {res.params}")


def _fmt(value: float | None, metric_key: str) -> str:
    if value is None:
        return "—"
    if metric_key in {"admit_rate", "yield_rate", "retention_rate"}:
        return f"{value:.1%}"
    return f"{value:,.0f}"


main()
