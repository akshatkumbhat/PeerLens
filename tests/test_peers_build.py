"""End-to-end test: build bridge_peer_set from a small synthetic warehouse."""

from __future__ import annotations

import duckdb
import polars as pl

from peerlens.ingest import urban
from peerlens.peers.build import build_peer_sets, peers_for
from peerlens.warehouse.build import build_warehouse


def _seed_six(raw_dir, year=2020) -> None:
    n = 6
    uids = list(range(1, n + 1))
    directory = pl.DataFrame(
        {
            "unitid": uids,
            "inst_name": [f"Inst {i}" for i in uids],
            "state_abbr": ["VA"] * n,
            "region": [5] * n,
            "sector": [1, 1, 1, 2, 2, 2],
            "institution_level": [4] * n,
            "inst_control": [1] * n,
            "inst_size": [3] * n,
            "hbcu": [2] * n,
            "cc_basic_2021": [15] * n,
            "longitude": [-77.0] * n,
            "latitude": [37.0] * n,
        }
    )
    # admit_rate spreads selectivity across bands; sizes/retention vary
    applied = [20000, 18000, 16000, 14000, 12000, 10000]
    admitted = [2000, 5400, 8000, 9800, 9600, 9000]  # admit rates .10..90
    admissions = pl.DataFrame(
        {
            "unitid": uids,
            "sex": [99] * n,
            "number_applied": applied,
            "number_admitted": admitted,
            "number_enrolled_ft": [a // 3 for a in admitted],
            "number_enrolled_pt": [10] * n,
            "number_enrolled_total": [a // 3 + 10 for a in admitted],
        }
    )
    retention = pl.DataFrame(
        {
            "unitid": uids,
            "ftpt": [1] * n,
            "retention_rate": [0.97, 0.94, 0.90, 0.85, 0.80, 0.75],
            "returning_students": [900] * n,
            "prev_cohort_adj": [1000] * n,
        }
    )
    directory.write_parquet(urban.cache_path(raw_dir, "directory", year))
    admissions.write_parquet(urban.cache_path(raw_dir, "admissions-enrollment", year))
    retention.write_parquet(urban.cache_path(raw_dir, "fall-retention", year))


def test_build_peer_sets_populates_bridge(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _seed_six(raw)
    db = tmp_path / "wh.duckdb"
    build_warehouse(year=2020, raw_dir=raw, db_path=db, cohort_size=200)

    summary = build_peer_sets(db_path=db, k=3, n_bands=3)
    assert summary["n_targets"] == 6
    assert summary["peer_rows"] > 0

    con = duckdb.connect(str(db), read_only=True)
    try:
        # bridge exists and every peer_unitid is a real institution
        (orphans,) = con.execute(
            "SELECT COUNT(*) FROM bridge_peer_set b "
            "LEFT JOIN dim_institution d ON b.peer_unitid = d.unitid WHERE d.unitid IS NULL"
        ).fetchone()
        assert orphans == 0

        # a target's peers exclude itself and are ranked
        peers = peers_for(con, target_unitid=1, set_type="peer")
        assert peers.height > 0
        assert 1 not in peers["unitid"].to_list()
        assert peers["rank"].to_list() == sorted(peers["rank"].to_list())
    finally:
        con.close()
