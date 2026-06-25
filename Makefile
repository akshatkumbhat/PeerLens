.PHONY: setup ingest build peers validate pipeline test app clean

setup:        ## install deps into the venv
	uv sync --extra dev

ingest:       ## pull Phase 1 IPEDS topics to parquet
	uv run peerlens ingest

build:        ## build the DuckDB warehouse from cached parquet
	uv run peerlens build

peers:        ## build Mahalanobis peer/aspirant sets (bridge_peer_set)
	uv run peerlens peers

validate:     ## run data-quality checks on the warehouse
	uv run peerlens validate

pipeline:     ## ingest -> build -> peers (full data pipeline through Phase 2)
	uv run peerlens ingest && uv run peerlens build && uv run peerlens peers

test:         ## run the test suite
	uv run pytest -q

app:          ## launch the Streamlit page
	uv run peerlens app

clean:        ## remove the built warehouse (keeps raw parquet cache)
	rm -f data/warehouse/peerlens.duckdb
