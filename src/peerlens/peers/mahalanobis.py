"""Mahalanobis nearest-neighbor peer and aspirant sets.

For target i and candidate j with feature vectors x_i, x_j and feature covariance
S:  d(i, j) = sqrt( (x_i - x_j)^T S^-1 (x_i - x_j) ).

Mahalanobis (rather than Euclidean) because the features are correlated and on
different scales; S^-1 whitens them so no single feature dominates. When S is
ill-conditioned on a small sample we fall back to a diagonal S — i.e. z-score
Euclidean — exactly as the brief prescribes. This mirrors how NCES finds similar
institutions in IPEDS (a nearest-neighbor procedure over key statistics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this condition number, treat S as ill-conditioned and use diagonal S.
CONDITION_THRESHOLD = 1e6


def inverse_covariance(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (S^-1, used_diagonal_fallback) for the feature matrix ``X``.

    Falls back to a diagonal S (z-score Euclidean) when the full covariance is
    singular or ill-conditioned.
    """
    if X.shape[0] < 2:
        return np.eye(X.shape[1]), True
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    var = np.diag(S).copy()
    var[var <= 0] = 1.0
    diag_inv = np.diag(1.0 / var)

    finite = np.all(np.isfinite(S))
    cond = np.linalg.cond(S) if finite else np.inf
    if not finite or cond > CONDITION_THRESHOLD:
        return diag_inv, True
    try:
        return np.linalg.inv(S), False
    except np.linalg.LinAlgError:
        return diag_inv, True


def distance_matrix(X: np.ndarray, S_inv: np.ndarray) -> np.ndarray:
    """Pairwise Mahalanobis distances for all rows of ``X`` (shape (n, n))."""
    # G = X S^-1 X^T ; D2[i,j] = G[i,i] + G[j,j] - 2 G[i,j]
    G = X @ S_inv @ X.T
    g = np.diag(G)
    d2 = g[:, None] + g[None, :] - 2.0 * G
    np.maximum(d2, 0.0, out=d2)  # clamp tiny negatives from float error
    return np.sqrt(d2)


def selectivity_bands(admit_rate: np.ndarray, n_bands: int = 5) -> np.ndarray:
    """Assign each institution a selectivity band; 0 = most selective.

    Bands are quantiles of admit_rate (lower admit_rate ⇒ more selective ⇒ lower
    band index). Robust to ties and small samples.
    """
    n = admit_rate.shape[0]
    n_bands = max(1, min(n_bands, n))
    ranks = admit_rate.argsort().argsort()  # 0 = lowest admit_rate = most selective
    return np.floor(ranks * n_bands / n).astype(int)


@dataclass
class NeighborSet:
    target_unitid: int
    peers: list[tuple[int, float]]      # (unitid, distance), nearest first
    aspirants: list[tuple[int, float]]  # (unitid, distance), nearest first


def build_neighbor_sets(
    unitids: np.ndarray,
    X: np.ndarray,
    admit_rate: np.ndarray,
    *,
    k: int = 10,
    n_bands: int = 5,
) -> tuple[list[NeighborSet], bool]:
    """Compute peer and aspirant sets for every institution.

    Peer set = k nearest overall (excluding self). Aspirant set = k nearest among
    institutions one selectivity band above (more selective). Returns the sets and
    whether the diagonal-S fallback was used.
    """
    S_inv, used_diag = inverse_covariance(X)
    D = distance_matrix(X, S_inv)
    bands = selectivity_bands(admit_rate, n_bands)
    n = len(unitids)

    sets: list[NeighborSet] = []
    for i in range(n):
        order = np.argsort(D[i], kind="stable")
        peers: list[tuple[int, float]] = []
        aspirants: list[tuple[int, float]] = []
        target_band = bands[i]
        for j in order:
            if j == i:
                continue
            if len(peers) < k:
                peers.append((int(unitids[j]), float(D[i, j])))
            if bands[j] == target_band - 1 and len(aspirants) < k:
                aspirants.append((int(unitids[j]), float(D[i, j])))
            if len(peers) >= k and len(aspirants) >= k:
                break
        sets.append(NeighborSet(int(unitids[i]), peers, aspirants))
    return sets, used_diag
