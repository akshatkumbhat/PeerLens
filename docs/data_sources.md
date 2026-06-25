# Data sources & dictionary

## Urban Institute Education Data API (IPEDS)

No API key required. Base:
`https://educationdata.urban.org/api/v1/college-university/ipeds/{topic}/{year}/`

Responses are paginated JSON: `{count, next, previous, results}`. Follow `next`
until null. Server-side filters are passed as query params (e.g. `sector=1`,
`fips=51`, `unitid=234030`).

### Year alignment (verified by probe)

| Topic | Years with data |
|---|---|
| `directory` | through 2022+ |
| `admissions-enrollment` | through 2022+ |
| `fall-retention` | **2018–2020 only** (2021+ return `count=0`) |

→ **Phase 1 uses year 2020**, the most recent year where retention is present,
so all facts align on one consistent year.

### `directory/{year}/` — institution characteristics → `dim_institution`

Key fields: `unitid` (PK), `inst_name`, `state_abbr`, `region`, `sector`,
`institution_level`, `inst_control`, `inst_size`, `hbcu`, and Carnegie classes
(`cc_basic_2021`, `cc_basic_2018`, …), `longitude`, `latitude`.

Code values (verified):
- `institution_level`: **4 = four-year (and above)**, 2 = two-year, 1 = < two-year.
- `sector`: 1 = Public 4yr, 2 = Private nonprofit 4yr, 3 = Private for-profit 4yr,
  4 = Public 2yr, 5 = Private nonprofit 2yr, 6 = Private for-profit 2yr, 9 = other.
- `inst_size`: ordinal enrollment-size category (1 small … 5 large; -1/-2 = N/A).

### `admissions-enrollment/{year}/` — funnel → `fact_admissions_funnel`

Fields: `unitid`, `year`, `fips`, `number_applied`, `number_admitted`,
`number_enrolled_ft`, `number_enrolled_pt`, `number_enrolled_total`, `sex`.

- `sex`: 1 = men, 2 = women, 9 = unknown, **99 = total** ← use 99.
- Derived metrics: admit_rate = admitted / applied; yield_rate = enrolled_total / admitted.

### `fall-retention/{year}/` — → `fact_retention`

Fields: `unitid`, `year`, `fips`, `ftpt`, `retention_rate`, `returning_students`,
`prev_cohort`, `prev_exclusions`, `prev_cohort_adj`.

- `ftpt`: **1 = full-time** (the canonical first-time full-time retention rate),
  2 = part-time, 99 = total.
- `retention_rate` is a fraction in [0, 1].

## Phase 1 cohort selection

Raw parquet caches the **full** year pull (reproducible). The warehouse curates
the thin-slice cohort: four-year (`institution_level=4`), public + private-nonprofit
(`sector ∈ {1,2}`), with a non-zero admissions funnel and present retention,
ranked by applicants — top ~200. Documented & deterministic.

## College Scorecard API (Phase 4)

Free key from https://api.data.gov/signup/ → `.env` `SCORECARD_API_KEY`. Adds
socio-economic context (Pell share, net price, earnings). Not used in Phase 1.
