"""Central configuration and filesystem paths for PeerLens.

Settings load from environment / .env (never commit real keys). Paths are
derived from the repo root so ingest, warehouse, and app agree on locations.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = three parents up from this file: src/peerlens/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
WAREHOUSE_DB = WAREHOUSE_DIR / "peerlens.duckdb"


class Settings(BaseSettings):
    """Runtime settings, populated from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # College Scorecard (Phase 4); IPEDS needs no key.
    scorecard_api_key: str = ""

    # Model provider (Phase 3): gemini | ollama | claude
    peerlens_model_provider: str = "gemini"
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    # Thin-slice scope (Phase 1). 2020 is the most recent IPEDS year for which
    # the Urban API has retention data (2021+ return zero rows), so all three
    # facts — admissions funnel, enrollment, retention — align on one year.
    ipeds_year: int = 2020


def get_settings() -> Settings:
    """Return a fresh Settings instance (cheap; reads .env each call)."""
    return Settings()


def ensure_dirs() -> None:
    """Create the data directories if missing (idempotent)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
