# PeerLens — The Grounded Insights Agent

A natural-language insights agent over public U.S. higher-education data that is
**correct or silent, never confidently wrong.** Every number in an answer is
computed by SQL over a clean warehouse and injected programmatically; the
language model never emits a figure of its own. When the system is not confident
the answer is right, it **abstains or asks a clarifying question instead of
guessing.**

The headline metric is not accuracy alone — it is the rate of *confident-and-wrong*
answers, driven toward zero.

> Status: **Phase 1 (scaffold)** — see [Build order](#build-order).

## Architecture (target)

```
Urban Institute IPEDS API ─┐
College Scorecard API ──────┼─▶ ingest (httpx + parquet cache)
                            │
                            ▼
                 DuckDB star schema  ──▶  data-quality gate (fails build on violations)
                 dim_institution / dim_year
                 fact_admissions_funnel / fact_retention
                 bridge_peer_set (Mahalanobis k-NN peers + aspirants)
                            │
                            ▼
        Agent pipeline (correct-or-silent)
        plan contract (Pydantic) → schema linking → template-first SQL
        → execution-guided correction → N-sample self-consistency
        → abstention decision → programmatic number injection
                            │
                            ▼
                 Streamlit: answer + confidence + the query behind it
```

## Design notes (research-grounded)

Borrowed: schema linking (not whole-schema dumping), constrained generation into a
validated plan before SQL, template-first SQL, execution-guided self-correction,
self-consistency as a confidence signal (CSC-SQL style), programmatic number injection.

Deliberately dropped (restraint is the point): no RL fine-tuning, no multi-agent
swarm, no vector DB for schema linking (the schema is small).

## Results

_Populated in Phase 4: execution accuracy, confident-wrong rate, abstention recall,
over-abstention, and the risk-coverage curve._

## Build order

- **Phase 1** — Ingest one IPEDS year → DuckDB dims + one fact table → one templated
  comparison query → minimal page. Runnable end to end. _(current)_
- **Phase 2** — Mahalanobis peer/aspirant sets, retention cohorts, data-quality layer.
- **Phase 3** — The agent (plan contract, schema linking, templates, self-consistency,
  abstention, number injection).
- **Phase 4** — Evaluation harness + CI, Streamlit polish, README with risk-coverage
  plot and the MARKETview-stack mapping.

## Setup

```sh
uv sync                 # create venv + install deps (Python 3.11+)
cp .env.example .env    # fill in keys as needed (IPEDS needs none)
```

## Tech stack

Python 3.11+, httpx, polars, DuckDB, scikit-learn, Pydantic v2, FastAPI,
LangGraph + LangChain, Streamlit, pytest, GitHub Actions. Provider-swappable model
layer (Gemini, local Ollama, Claude).
