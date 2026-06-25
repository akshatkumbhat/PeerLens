"""DuckDB connection helper."""

from __future__ import annotations

from pathlib import Path

import duckdb

from peerlens import config


def connect(db_path: Path | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse. Read-only by default (query path is never a writer)."""
    db_path = db_path or config.WAREHOUSE_DB
    if read_only and not Path(db_path).exists():
        raise FileNotFoundError(f"warehouse not found: {db_path} — run `peerlens build` first")
    return duckdb.connect(str(db_path), read_only=read_only)
