"""Tests for the Part 2 table selection and the Part 3 relative-contrast / binning helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from annbench import tables
from annbench.stages import part3_experiment as part3


def _rows(kind, build, points):
    """points: (search value, tune recall, report recall, report qps)."""
    rows = []
    for value, r_tune, r_report, qps in points:
        common = dict(experiment="part2_benchmark", dataset="sift", index_type=kind, build_params=build,
                      search_params={"ef": value}, build_time_s=1.0, index_size_bytes=1000, bytes_per_vector=10.0)
        rows.append(dict(common, split="tune", recall=r_tune, qps=np.nan))
        rows.append(dict(common, split="report", recall=r_report, qps=qps))
    return rows


def test_operating_point_is_chosen_on_tune_recall_and_reported_on_report_half():
    df = pd.DataFrame(_rows("hnsw", {"M": 8}, [(10, 0.80, 0.81, 900.0), (20, 0.91, 0.89, 500.0), (40, 0.96, 0.94, 300.0)])
                      + _rows("hnsw", {"M": 16}, [(10, 0.92, 0.93, 700.0), (20, 0.97, 0.97, 400.0)]))
    ops = tables.operating_points(df, "sift")
    at90 = ops[ops["target"] == 0.90].iloc[0]
    assert at90["build_params"] == {"M": 16} and at90["qps"] == 700.0      # fastest with tune recall >= 0.90
    at95 = ops[ops["target"] == 0.95].iloc[0]
    assert at95["build_params"] == {"M": 16} and at95["recall"] == 0.97


def test_index_that_never_reaches_a_target_gets_its_best_setting():
    df = pd.DataFrame(_rows("ivf_pq", {"m": 8}, [(1, 0.40, 0.41, 900.0), (2, 0.60, 0.58, 500.0)]))
    ops = tables.operating_points(df, "sift")
    assert not ops["reached"].any()
    assert set(ops["recall"]) == {0.58}


def test_relative_contrast_matches_brute_force():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1200, 5), dtype=np.float32)
    queries = rng.standard_normal((7, 5), dtype=np.float32)
    dist = np.sqrt(((queries[:, None, :].astype(np.float64) - base[None, :, :]) ** 2).sum(-1))
    expected = dist.mean(1) / dist.min(1)
    got = part3.relative_contrast(base, queries, dist.min(1) ** 2)
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_difficulty_bins_put_the_lowest_contrast_in_bin_zero():
    rc = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, np.inf])
    bins = part3.difficulty_bins(rc, n_bins=4)
    assert bins[0] == 0 and bins[-1] == 3
    assert np.bincount(bins).tolist() == [2, 2, 2, 2]
    assert (np.diff(bins) >= 0).all()                      # sorted RC -> non-decreasing bins
