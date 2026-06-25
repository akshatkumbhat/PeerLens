"""Tests for the Mahalanobis peer/aspirant math."""

from __future__ import annotations

import numpy as np

from peerlens.peers import mahalanobis as mh


def test_distance_matrix_euclidean_when_identity() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    D = mh.distance_matrix(X, np.eye(2))
    assert np.allclose(np.diag(D), 0.0)            # self-distance zero
    assert np.allclose(D, D.T)                      # symmetric
    assert abs(D[0, 1] - 5.0) < 1e-9                # 3-4-5 triangle


def test_inverse_covariance_full_when_well_conditioned() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    S_inv, used_diag = mh.inverse_covariance(X)
    assert not used_diag
    assert S_inv.shape == (3, 3)


def test_inverse_covariance_falls_back_when_singular() -> None:
    # second column is a copy of the first -> singular covariance
    a = np.linspace(0, 1, 50)
    X = np.column_stack([a, a, np.zeros_like(a)])
    S_inv, used_diag = mh.inverse_covariance(X)
    assert used_diag
    assert np.all(np.isfinite(S_inv))


def test_selectivity_bands_lower_admit_is_more_selective() -> None:
    admit = np.array([0.05, 0.20, 0.40, 0.60, 0.80])
    bands = mh.selectivity_bands(admit, n_bands=5)
    assert bands[0] < bands[-1]                      # most selective = lowest band
    assert list(bands) == sorted(bands)              # monotonic with admit_rate


def test_build_neighbor_sets_peers_exclude_self_and_sort() -> None:
    unitids = np.array([10, 20, 30, 40])
    # 1-D feature; distances are just gaps along the line
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    admit = np.array([0.1, 0.2, 0.3, 0.4])
    sets, _ = mh.build_neighbor_sets(unitids, X, admit, k=2, n_bands=4)

    by_id = {s.target_unitid: s for s in sets}
    s10 = by_id[10]
    assert 10 not in [p for p, _ in s10.peers]       # excludes self
    assert [p for p, _ in s10.peers] == [20, 30]     # nearest first
    assert s10.peers[0][1] <= s10.peers[1][1]         # distances ascending


def test_aspirants_are_one_band_more_selective() -> None:
    unitids = np.array([1, 2, 3, 4])
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    admit = np.array([0.8, 0.6, 0.4, 0.2])            # inst 4 most selective
    sets, _ = mh.build_neighbor_sets(unitids, X, admit, k=3, n_bands=4)
    by_id = {s.target_unitid: s for s in sets}
    # inst 1 (least selective, band 3) -> aspirants drawn from band 2 (inst 2)
    aspirant_ids = [a for a, _ in by_id[1].aspirants]
    assert 2 in aspirant_ids
    # most-selective inst 4 (band 0) has no band above -> no aspirants
    assert by_id[4].aspirants == []
