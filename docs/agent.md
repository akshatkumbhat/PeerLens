# The agent (Phase 3)

A natural-language question becomes a grounded answer — or an honest abstention —
through one clean pipeline. Numbers are computed by SQL and injected by code; the
model never emits a figure.

## Pipeline

```
question
  → schema linking            (catalog.link: relevant tables/metrics only)
  → sample N plans            (model.propose × N at temperature; provider-swappable)
  → resolve each plan         (resolve.resolve_plan: deterministic catalog grounding)
  → template-first SQL + run  (execute.build_sql / execute_resolved; read-only)
  → self-consistency          (consistency.run_consistency: group by result, agreement)
  → correct-or-silent         (decide.decide: τ + the five conditions)
  → number injection          (answer.render_answer: figures from rows, by code)
→ AgentResponse (answer or abstention, always with plan + SQL + rows + agreement)
```

Each stage is a small, separately tested module (`src/peerlens/agent/`). The
orchestrator (`pipeline.run_agent`) is plain Python; its stages map 1:1 onto a
LangGraph graph, the documented next step.

## The query-plan contract

The model emits a permissive `QueryPlan` (institution + metric as free text);
**grounding happens in the resolver**, which produces either a `ResolvedPlan`
(every reference a real id/metric) or an `Abstention` with a precise reason. This
is where an unknown institution or metric is caught — deterministically, before
any SQL exists.

- `intent`: `single` | `compare` (trend deferred until multi-year data)
- `institution`: resolved by a curated acronym map (UVA, UCLA, MIT, …) → normalized
  exact match → deterministic containment — which honestly surfaces ambiguity and
  unknowns (with suggestions)
- `metric`: one of the catalog keys — admissions (admit_rate, yield_rate, applied,
  enrolled), retention (retention_rate), and socio-economic from College Scorecard
  (net_price, pell_rate, median_earnings). A question naming no measure is set to the
  literal `unspecified` so the resolver can ask which one.
- `comparison`: `none` | `peers` | `aspirants` (from `bridge_peer_set`) | `explicit`
- `years`: validated ⊆ available (currently {2020})

## Self-consistency (the confidence signal)

Sample **N = 5** plans at **temperature 0.7**, execute each, and group by a
**canonical result signature** (rates rounded to 3 dp, counts exact). Abstentions
are votes too: a question the model reliably reads as "unknown institution" lands
as the top group and abstains *with high agreement*. **Agreement** = fraction of
samples in the largest group.

## Correct-or-silent (the centerpiece)

| # | Condition | Behavior |
|---|---|---|
| 1 | unknown institution / metric | abstain, list closest matches / the metric menu |
| 2 | ambiguous institution, or no metric named | clarify — offer the matches, or ask which measure |
| 3 | agreement < τ (**0.6**) | abstain: uncertain, offer to narrow |
| 4 | empty / suppressed result | "no data," never invent |
| 5 | out of scope (year/metric/unparseable) | decline, state what's answerable |

τ and N are config defaults (`agent_tau`, `agent_samples`); **τ is *set* by the
Phase 4 risk-coverage sweep**, not guessed. Below τ the agent never answers.

## Number injection

`answer.render_answer` composes prose over the returned rows with every figure
substituted from the result set by code — hallucinated numbers are structurally
impossible. An LLM-narration variant could replace the templating, but it would
still receive numbers only as code-filled slots; the guarantee is the injection.

## Providers

`model.get_plan_model` builds the configured provider. **Gemini** is the default,
called over its REST API with httpx (no SDK, no extra deps); set `GEMINI_API_KEY`.
Adding Ollama or Claude is one new class implementing the `PlanModel` protocol —
the pipeline is provider-agnostic. `FakePlanModel` makes the whole agent testable
offline (61 tests, no network, no key).

## Schema linking at scale

Our schema is tiny, so `link()` is keyword-based. The interface is the scaling
seam: for MARKETview's large schema, swap the linker for embedding/retrieval over
table+column descriptions and return the same `LinkedCatalog` — every downstream
stage is unchanged. We deliberately avoid a vector DB here (the schema is small;
it would be theater).

## Run it

```sh
peerlens ask "How does UVA's retention compare to its peers?"   # CLI
peerlens app                                                    # Ask panel in the page
```
