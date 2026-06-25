.PHONY: setup ingest build pipeline test app clean

setup:        ## install deps into the venv
	uv sync --extra dev

ingest:       ## pull Phase 1 IPEDS topics to parquet
	uv run peerlens ingest

build:        ## build the DuckDB warehouse from cached parquet
	uv run peerlens build

pipeline:     ## ingest -> build (full Phase 1 data pipeline)
	uv run peerlens ingest && uv run peerlens build

test:         ## run the test suite
	uv run pytest -q

app:          ## launch the Streamlit page
	uv run peerlens app

clean:        ## remove the built warehouse (keeps raw parquet cache)
	rm -f data/warehouse/peerlens.duckdb
