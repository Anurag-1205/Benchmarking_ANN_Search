"""Tests for the Part 4 generator: seeded, stored in 128 dimensions, distances follow the intrinsic dimension."""
from __future__ import annotations

import numpy as np

from annbench import config, synthetic


def test_rotation_is_orthogonal_and_fixed():
    r = synthetic.rotation()
    np.testing.assert_allclose(r @ r.T, np.eye(config.PART4_STORED_DIM), atol=1e-5)
    np.testing.assert_array_equal(r, synthetic.rotation())


def test_same_seed_same_data_and_stored_dimension_is_fixed():
    a = synthetic.intrinsic_dataset(8, 500, 20)
    b = synthetic.intrinsic_dataset(8, 500, 20)
    np.testing.assert_array_equal(a.base, b.base)
    assert a.base.shape == (500, config.PART4_STORED_DIM) and a.queries.shape == (20, config.PART4_STORED_DIM)
    assert a.name == "synthetic_m8"


def test_data_spans_exactly_m_dimensions():
    ds = synthetic.intrinsic_dataset(16, 2000, 10)
    singular = np.linalg.svd(ds.base, compute_uv=False)
    assert (singular[:16] > 1.0).all() and (singular[16:] < 1e-2).all()
