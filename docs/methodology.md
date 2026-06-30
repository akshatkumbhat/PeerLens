# Methodology

## Peer & aspirant sets (Mahalanobis nearest neighbors)

For target institution *i* and candidate *j* with standardized feature vectors
x_i, x_j and feature covariance S:

```
d(i, j) = sqrt( (x_i - x_j)^T  S^-1  (x_i - x_j) )
```

Mahalanobis rather than Euclidean because the features are correlated and on
different scales; S^-1 whitens them so no single feature dominates. If S is
ill-conditioned on the small cohort (condition number > 1e6, or singular), we
fall back to a diagonal S — z-score-standardized Euclidean. This mirrors how
NCES finds similar institutions in IPEDS (a nearest-neighbor procedure over key
statistics).

**Features** (standardized; `peers/features.py`): admit rate (selectivity),
log enrollment (size), retention rate, yield rate, sector (public vs private
nonprofit), and **Pell share** (College Scorecard) — which joins the distance
automatically when present and is median-imputed for the few institutions missing
it. Kept compact so S is well-conditioned on ~200 institutions. **Extension
points**: region and Carnegie class.

- **Peer set**: the *k* nearest institutions overall (excluding self).
- **Aspirant set**: the *k* nearest among institutions **one selectivity band
  above** (more selective). Bands are admit-rate quantiles; band 0 = most
  selective. The most-selective band has no aspirants by construction.

Sanity check (real IPEDS 2020): the University of Virginia's nearest peers are
Georgia Tech, UNC-Chapel Hill, NC State, Florida, and Maryland — the public
flagship peer group, with no hand-tuning.

`bridge_peer_set` columns: `target_unitid, peer_unitid, set_type ∈ {peer,
aspirant}, rank, distance`.

## Retention cohorts

`fact_retention` models the first-time, full-time retention cohort directly from
IPEDS: `prev_cohort_adj` is the adjusted prior-fall cohort, `returning_students`
those retained, and `retention_rate` their ratio. We use the full-time (`ftpt=1`)
series, the canonical IPEDS retention metric.

## Data-quality gate (`quality/checks.py`)

The build **fails** on any error-severity violation, so a malformed warehouse
never reaches the agent or the page:

- bounded rates: admit / yield / retention in [0, 1]
- referential integrity: every fact `unitid` exists in `dim_institution`
- not-null identity keys on `dim_institution`
- cohort completeness: every institution has both a funnel and a retention row
- year-over-year retention sanity (|Δ| ≤ 0.5; inactive with a single year)

Run ad hoc with `peerlens validate`.
