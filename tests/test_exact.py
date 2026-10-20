"""Tests for exact.py (exact search and ground-truth plumbing) on tiny synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from annbench import data, exact


def _toy(n: int = 200, d: int = 8, nq: int = 12, seed: int = 0, stored=None) -> data.Dataset:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, size=(n, d)).astype(np.float32)       # small integers: many exact ties, like SIFT
    queries = rng.integers(0, 4, size=(nq, d)).astype(np.float32)
    return data.Dataset(name="toy", base=base, queries=queries, stored_neighbors=stored, quick=False)


def test_rescore_sorts_by_distance_then_id():
    base = np.array([[0.0], [2.0], [1.0], [-1.0], [5.0]], dtype=np.float32)   # squared distances to 0: 0, 4, 1, 1, 25
    queries = np.zeros((1, 1), dtype=np.float32)
    ids, dist = exact.rescore(base, queries, np.array([[4, 3, 2, 1, 0]]))
    assert ids.tolist() == [[0, 2, 3, 1, 4]]
    assert dist.tolist() == [[0.0, 1.0, 1.0, 4.0, 25.0]]


def test_rescore_rejects_padding_ids():
    with pytest.raises(ValueError):
        exact.rescore(np.zeros((3, 2), np.float32), np.zeros((1, 2), np.float32), np.array([[0, -1]]))


def test_ground_truth_distances_match_brute_force():
    ds = _toy()
    gt = exact.ground_truth(ds, k=20, cache=False)
    brute = np.sort(((ds.queries[:, None, :] - ds.base[None, :, :]) ** 2).sum(-1), axis=1)[:, :20]
    np.testing.assert_array_equal(gt.dist_sq, brute)                           # integer data: exact equality
    recomputed = ((ds.base[gt.ids] - ds.queries[:, None, :]) ** 2).sum(-1)
    np.testing.assert_array_equal(recomputed, gt.dist_sq)
    tied = gt.dist_sq[:, 1:] == gt.dist_sq[:, :-1]
    assert (gt.ids[:, 1:][tied] > gt.ids[:, :-1][tied]).all()                  # ties ordered by id
    np.testing.assert_array_equal(gt.kth_dist_sq(10), gt.dist_sq[:, 9])


def test_ground_truth_is_cached_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNBENCH_DATA_DIR", str(tmp_path))
    meta = tmp_path / "processed" / "toy" / "meta.json"          # a fingerprint makes the toy data cacheable
    meta.parent.mkdir(parents=True)
    meta.write_text('{"source_sha256": "toy"}')
    ds = _toy()
    first = exact.ground_truth(ds, k=10)
    assert (tmp_path / "ground_truth" / "toy_n200_q12_k10.npz").exists()
    second = exact.ground_truth(ds, k=10)
    np.testing.assert_array_equal(first.ids, second.ids)
    np.testing.assert_array_equal(first.dist_sq, second.dist_sq)


def test_data_without_a_fingerprint_is_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNBENCH_DATA_DIR", str(tmp_path))
    exact.ground_truth(_toy(), k=10)
    assert not (tmp_path / "ground_truth").exists()


def test_agreement_with_file_counts_identical_id_sets():
    ds = _toy()
    gt = exact.ground_truth(ds, k=10, cache=False)
    stored = gt.ids[:, ::-1].copy()                                            # same sets, reversed order
    stored[0, 0] = next(i for i in range(ds.n_base) if i not in set(gt.ids[0].tolist()))
    stats = exact.agreement_with_file(_toy(stored=stored), gt)
    assert stats["id_sets_identical"] == ds.n_queries - 1
    assert stats["first_mismatches"] == [0]


def test_agreement_with_file_needs_the_full_base():
    ds = _toy()
    with pytest.raises(ValueError):
        exact.agreement_with_file(ds, exact.ground_truth(ds, k=10, cache=False))


def test_build_flat_index_holds_every_vector():
    ds = _toy()
    index, seconds = exact.build_flat_index(ds.base)
    assert index.ntotal == ds.n_base and seconds >= 0
