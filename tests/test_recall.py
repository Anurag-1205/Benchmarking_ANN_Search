"""Unit tests for metrics.recall_at_k / recall_at_10 -- YOURS TO WRITE (assignment Part 1: "add a few unit tests").

Each stub names one behaviour from the recall_at_k docstring. Replace `pytest.skip(TODO)` with a tiny hand-made
case and its assertion. Small 1-D or 2-D bases keep the expected answers easy to work out by hand.

    cd code && ../.venv/bin/python -m pytest tests/test_recall.py -v
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (for your test data)
import pytest

from annbench.metrics import recall_at_10, recall_at_k  # noqa: F401

TODO = "TODO(you): write this test"


def test_perfect_answer_scores_one():
    """Returning exactly the true top-k ids gives recall 1.0."""
    pytest.skip(TODO)


def test_order_within_a_row_does_not_matter():
    """The same ids in a different order still score 1.0."""
    pytest.skip(TODO)


def test_disjoint_answer_scores_zero():
    """Ids that are all farther than the k-th neighbour score 0.0."""
    pytest.skip(TODO)


def test_partial_overlap_scores_the_fraction_found():
    """Finding 7 of the true 10 scores 0.7."""
    pytest.skip(TODO)


def test_tie_at_the_kth_neighbour_is_not_a_mistake():
    """Several vectors at exactly the k-th distance: returning one that exact search did not list still counts."""
    pytest.skip(TODO)


def test_just_beyond_the_threshold_is_a_miss():
    """A vector farther than the k-th distance by more than rel_eps is not a hit."""
    pytest.skip(TODO)


def test_padding_ids_are_misses():
    """-1 entries (FAISS padding) count as misses and never index `base`."""
    pytest.skip(TODO)


def test_duplicate_ids_count_once():
    """The same id returned twice in one row is one hit, not two."""
    pytest.skip(TODO)


def test_only_the_first_k_columns_count():
    """Hits in column k+1 and beyond are ignored."""
    pytest.skip(TODO)


def test_per_query_returns_one_value_per_query():
    """per_query=True returns an array of shape (n_queries,) whose mean equals the default return value."""
    pytest.skip(TODO)


def test_recall_at_10_is_recall_at_k_with_k_10():
    """recall_at_10 agrees with recall_at_k(..., k=10) on the same inputs."""
    pytest.skip(TODO)
