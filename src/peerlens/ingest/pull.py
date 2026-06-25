"""Phase 1 ingest orchestration.

Pulls the three IPEDS topics PeerLens needs for the thin slice and caches each
full year pull to parquet under ``data/raw/``. Full (unfiltered) pulls keep the
raw layer faithful and reproducible; scope/cohort selection happens later in the
warehouse layer, not here.
"""

from __future__ import annotations

from pathlib import Path

from peerlens import config
from peerlens.ingest import urban

# Topics required for the Phase 1 thin slice.
PHASE1_TOPICS: tuple[str, ...] = (
    "directory",
    "admissions-enrollment",
    "fall-retention",
)


def pull_phase1(
    year: int | None = None,
    raw_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Cache the Phase 1 topics for ``year`` to parquet.

    Returns a ``{topic: parquet_path}`` map. Existing caches are reused unless
    ``overwrite`` is set.
    """
    year = year if year is not None else config.get_settings().ipeds_year
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for topic in PHASE1_TOPICS:
        out = urban.pull_to_parquet(topic, year, raw_dir, overwrite=overwrite)
        paths[topic] = out
    return paths
