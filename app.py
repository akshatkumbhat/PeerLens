"""Hugging Face Spaces entrypoint.

Streamlit re-executes this file top-to-bottom on every rerun, so we put the
``src/`` layout on the import path and run the real app module *fresh* each time
with ``runpy`` — a plain ``import`` would render only once, because the module
would be cached in ``sys.modules`` after the first run and never re-execute.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
runpy.run_module("peerlens.app.streamlit_app", run_name="__main__")
