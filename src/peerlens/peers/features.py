"""Institution feature matrix for peer-set construction.

Pulls the features the peer distance is computed over, standardizes them
(z-score), and returns them with the raw admit_rate kept aside for selectivity
banding. The feature set is deliberately compact so the covariance is
well-conditioned on a ~200-institution cohort; region / Carnegie / Pell share are
documented extension points (Pell arrives with College Scorecard in Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardized features used in the Mahalanobis distance.
FEATURE_NAMES: tuple[str, ...] = (
    "admit_rate",      # selectivity
    "log_enrolled",    # size (log to tame the heavy tail)
    "retention_rate",  # outcome
    "yield_rate",      # demand realized
    "is_private",      # sector (private nonprofit vs public)
)


@dataclass
class FeatureMatrix:
    unitids: np.ndarray          # shape (n,)
    X: np.ndarray                # shape (n, d), standardized
    feature_names: tuple[str, ...]
    admit_rate: np.ndarray       # shape (n,), raw — for selectivity banding
    scaler: StandardScaler


def load_features(con: duckdb.DuckDBPyConnection) -> FeatureMatrix:
    """Build the standardized feature matrix from the warehouse."""
    df = con.execute(
        """
        SELECT
            i.unitid,
            f.admit_rate,
            f.yield_rate,
            f.number_enrolled_total,
            r.retention_rate,
            CASE WHEN i.sector = 2 THEN 1 ELSE 0 END AS is_private
        FROM dim_institution i
        JOIN fact_admissions_funnel f USING (unitid)
        JOIN fact_retention r USING (unitid)
        WHERE f.admit_rate IS NOT NULL
          AND f.yield_rate IS NOT NULL
          AND f.number_enrolled_total IS NOT NULL
          AND r.retention_rate IS NOT NULL
        ORDER BY i.unitid
        """
    ).pl()

    unitids = df["unitid"].to_numpy()
    admit_rate = df["admit_rate"].to_numpy().astype(float)
    raw = np.column_stack(
        [
            admit_rate,
            np.log1p(df["number_enrolled_total"].to_numpy().astype(float)),
            df["retention_rate"].to_numpy().astype(float),
            df["yield_rate"].to_numpy().astype(float),
            df["is_private"].to_numpy().astype(float),
        ]
    )
    scaler = StandardScaler().fit(raw)
    X = scaler.transform(raw)
    return FeatureMatrix(unitids, X, FEATURE_NAMES, admit_rate, scaler)
