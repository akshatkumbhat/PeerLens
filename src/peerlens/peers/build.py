"""Build the ``bridge_peer_set`` table from Mahalanobis neighbor sets.

Reads the cohort features from the warehouse, computes peer and aspirant sets,
and writes them to ``bridge_peer_set`` (target_unitid, peer_unitid, set_type,
rank, distance). Idempotent: CREATE OR REPLACE.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from peerlens.peers.features import load_features
from peerlens.peers.mahalanobis import build_neighbor_sets
from peerlens.warehouse import db


def build_peer_sets(
    db_path: Path | None = None,
    *,
    k: int = 10,
    n_bands: int = 5,
) -> dict[str, object]:
    """Compute and persist ``bridge_peer_set``. Returns a small summary dict."""
    con = db.connect(db_path, read_only=False)
    try:
        fm = load_features(con)
        sets, used_diag = build_neighbor_sets(
            fm.unitids, fm.X, fm.admit_rate, k=k, n_bands=n_bands
        )

        records: list[dict] = []
        for s in sets:
            for rank, (uid, dist) in enumerate(s.peers, start=1):
                records.append(
                    {"target_unitid": s.target_unitid, "peer_unitid": uid,
                     "set_type": "peer", "rank": rank, "distance": dist}
                )
            for rank, (uid, dist) in enumerate(s.aspirants, start=1):
                records.append(
                    {"target_unitid": s.target_unitid, "peer_unitid": uid,
                     "set_type": "aspirant", "rank": rank, "distance": dist}
                )

        bridge = pl.DataFrame(
            records,
            schema={
                "target_unitid": pl.Int64,
                "peer_unitid": pl.Int64,
                "set_type": pl.Utf8,
                "rank": pl.Int64,
                "distance": pl.Float64,
            },
        )
        con.register("_bridge_df", bridge)
        con.execute("CREATE OR REPLACE TABLE bridge_peer_set AS SELECT * FROM _bridge_df")
        con.unregister("_bridge_df")

        # Referential integrity for the bridge (built after the main DQ gate).
        (orphans,) = con.execute(
            "SELECT COUNT(*) FROM bridge_peer_set b "
            "LEFT JOIN dim_institution d ON b.peer_unitid = d.unitid "
            "WHERE d.unitid IS NULL"
        ).fetchone()
        if orphans:
            raise ValueError(f"bridge_peer_set has {orphans} peer_unitid(s) not in dim_institution")

        n_peer = bridge.filter(pl.col("set_type") == "peer").height
        n_asp = bridge.filter(pl.col("set_type") == "aspirant").height
        return {
            "rows": bridge.height,
            "peer_rows": n_peer,
            "aspirant_rows": n_asp,
            "n_targets": len(sets),
            "used_diagonal_fallback": used_diag,
            "k": k,
            "n_bands": n_bands,
        }
    finally:
        con.close()


def peers_for(
    con,
    target_unitid: int,
    set_type: str = "peer",
    limit: int | None = None,
) -> pl.DataFrame:
    """Return a target's peer (or aspirant) institutions, nearest first."""
    sql = (
        "SELECT b.peer_unitid AS unitid, d.inst_name, d.state_abbr, d.sector_name, "
        "       b.rank, b.distance "
        "FROM bridge_peer_set b JOIN dim_institution d ON b.peer_unitid = d.unitid "
        "WHERE b.target_unitid = ? AND b.set_type = ? ORDER BY b.rank"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql, [target_unitid, set_type]).pl()
