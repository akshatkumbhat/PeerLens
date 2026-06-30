# Evaluation (Phase 4)

The eval harness is the credibility multiplier: it measures not raw accuracy but
the **rate of confident-and-wrong answers**, and it sets the abstention threshold
τ from data rather than by guess.

## Gold set (`src/peerlens/eval/gold.json`)

- **Answerable** — natural-language questions paired with a gold institution +
  metric resolved directly from the warehouse. The system is *correct* on one of
  these when it answers AND its resolved plan targets the right institution and
  metric (execution is deterministic, so the value follows).
- **Unanswerable / ambiguous** — questions whose correct behavior is abstention
  or clarification: unknown institution, ambiguous reference, out-of-scope metric
  or year, or non-questions. The system is correct here when it abstains.

Institution phrasings resolve under the resolver, which includes a curated acronym
map (e.g. "UVA", "UCLA", "Penn State") on top of exact/containment matching, so a
common abbreviation resolves rather than showing up as honest *over-abstention*.

## How it runs (model called once per question)

`run_eval` runs the agent's self-consistency over each question **once** and
stores τ-independent primitives: the agreement score, what the consistent sample
decided (answer / abstain / invalid), and — for answerable questions — whether
that decision matches gold. The τ-sweep then happens entirely in `metrics.py`
over these records, so **no model call is repeated per threshold**. Records are
cached to `docs/eval/records.json`; `peerlens eval --from-cache` recomputes every
metric and the plot without touching the model (this is what CI runs).

## Metrics (`metrics.py`)

A question is *answered* when the consistent sample is an answer **and** agreement
≥ τ; otherwise the system abstains.

- **Execution accuracy (EX)** — of answered answerable questions, the fraction correct.
- **Confident-wrong rate** (the headline) — of *all* questions, the fraction
  answered confidently but wrong (wrong answer, or answered something unanswerable).
- **Abstention recall** — of truly unanswerable questions, the fraction abstained.
- **Over-abstention** — of answerable questions, the fraction wrongly refused (the cost of caution).
- **Risk-coverage curve** — sweep τ; plot selective risk (error among answered)
  against coverage (fraction answered). The **operating point** is the lowest τ
  (max coverage) whose selective risk ≤ 2%.

## Run it

```sh
peerlens eval --update-readme          # live: calls the model once per question, writes report + plot + README
peerlens eval --from-cache             # recompute metrics/plot from cached records (no model) — CI uses this
peerlens eval --samples 3              # fewer self-consistency samples (free-tier friendly)
```

Outputs land in `docs/eval/`: `records.json`, `report.md`, `risk_coverage.png`.
The README "Results" section is regenerated between its `EVAL` markers.

## CI

`.github/workflows/ci.yml` runs ruff + the full hermetic test suite (no network,
no key) on every push, then recomputes the eval metrics from the committed
records so the README numbers are reproducible in CI.
