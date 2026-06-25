"""Smoke test: the Streamlit script runs end to end without raising.

Uses Streamlit's AppTest to actually execute the script. Skips when the
warehouse hasn't been built (it's gitignored / network-derived), so CI without
data stays green; locally, after `peerlens build`, it exercises the real page.
"""

from __future__ import annotations

import pytest

from peerlens import config


@pytest.mark.skipif(
    not config.WAREHOUSE_DB.exists(),
    reason="warehouse not built; run `peerlens ingest && peerlens build`",
)
def test_streamlit_app_runs_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = config.REPO_ROOT / "src" / "peerlens" / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not at.exception, f"app raised: {at.exception}"
    # the title renders and at least one selectbox (institution picker) exists
    assert any("PeerLens" in t.value for t in at.title)
    assert len(at.selectbox) >= 2
