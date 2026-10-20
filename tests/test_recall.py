"""Unit tests for metrics.recall_at_k / recall_at_10 (assignment Part 1: "add a few unit tests").

Written by Claude at the student's request; they test the student's recall_at_k. Base vectors sit on a line
(squared distance from a query at 0 is id**2) or on a circle of integer points, so every expected value can be
checked by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from annbench.metrics import recall_at_10, recall_at_k

LINE = np.arange(20, dtype=np.float32)[:, None]        # id i sits at x = i: squared distance from 0 is i**2
ORIGIN = np.zeros((1, 1), dtype=np.float32)
KTH_LINE = np.array([81.0], dtype=np.float32)          # the 10th neighbour of 0 is id 9, at squared distance 81


def _recall(ids, base=LINE, queries=ORIGIN, kth=KTH_LINE, **kwargs):
    return recall_at_10(np.asarray(ids, dtype=np.int64), queries, base, kth, **kwargs)


def test_perfect_answer_scores_one():
    assert _recall([list(range(10))]) == pytest.approx(1.0)


def test_order_within_a_row_does_not_matter():
    assert _recall([list(range(9, -1, -1))]) == pytest.approx(1.0)


def test_disjoint_answer_scores_zero():
    assert _recall([list(range(10, 20))]) == pytest.approx(0.0)


def test_partial_overlap_scores_the_fraction_found():
    assert _recall([[0, 1, 2, 3, 4, 5, 6, 15, 16, 17]]) == pytest.approx(0.7)


def test_tie_at_the_kth_neighbour_is_not_a_mistake():
    # 12 integer points at squared distance exactly 25 from the origin, then two far-away points. Exact search
    # lists any 10 of the 12; returning the "other" ones is still a perfect answer.
    circle = [(5, 0), (-5, 0), (0, 5), (0, -5), (3, 4), (-3, 4), (3, -4), (-3, -4), (4, 3), (-4, 3), (4, -3), (-4, -3)]
    base = np.array(circle + [(20, 20), (30, 30)], dtype=np.float32)
    origin = np.zeros((1, 2), dtype=np.float32)
    returned = [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]       # skips ids 0 and 1, which exact search would list first
    assert _recall(returned, base=base, queries=origin, kth=np.array([25.0], np.float32)) == pytest.approx(1.0)


def test_just_beyond_the_threshold_is_a_miss():
    base = np.vstack([LINE, [[9.01]]]).astype(np.float32)   # id 20 at 9.01: squared distance 81.18 > 81 * (1 + 1e-5)
    assert _recall([[0, 1, 2, 3, 4, 5, 6, 7, 8, 20]], base=base) == pytest.approx(0.9)


def test_padding_ids_are_misses():
    assert _recall([[0, 1, 2, 3, 4, 5, 6, 7, 8, -1]]) == pytest.approx(0.9)
    assert _recall([[-1] * 10]) == pytest.approx(0.0)


def test_duplicate_ids_count_once():
    assert _recall([[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]]) == pytest.approx(0.8)


def test_only_the_first_k_columns_count():
    assert _recall([[10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]) == pytest.approx(0.8)   # ids 8, 9 sit in columns 11, 12


def test_per_query_returns_one_value_per_query():
    queries = np.zeros((2, 1), dtype=np.float32)
    kth = np.array([81.0, 81.0], dtype=np.float32)
    ids = [list(range(10)), [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]]
    per_query = _recall(ids, queries=queries, kth=kth, per_query=True)
    assert per_query.shape == (2,)
    np.testing.assert_allclose(per_query, [1.0, 0.5])
    assert _recall(ids, queries=queries, kth=kth) == pytest.approx(float(np.mean(per_query)))


def test_recall_at_10_is_recall_at_k_with_k_10():
    ids = np.array([[0, 1, 2, 3, 4, 5, 6, 15, 16, 17]], dtype=np.int64)
    assert recall_at_10(ids, ORIGIN, LINE, KTH_LINE) == recall_at_k(ids, ORIGIN, LINE, KTH_LINE, k=10)


def test_works_on_a_memory_mapped_base(tmp_path):
    path = tmp_path / "base.npy"
    np.save(path, LINE)
    assert _recall([list(range(10))], base=np.load(path, mmap_mode="r")) == pytest.approx(1.0)
