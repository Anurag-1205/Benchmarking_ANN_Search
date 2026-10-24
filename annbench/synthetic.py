"""Seeded synthetic data for Part 4: Gaussian vectors of intrinsic dimension m stored in a fixed 128-d space.

A dataset with intrinsic dimension m is z @ B, where z ~ N(0, I_m) and B holds the first m rows of one fixed random
128 x 128 rotation. The rows of B are orthonormal, so distances between the stored vectors equal distances between
the z's: the geometry is that of an m-dimensional Gaussian, while the memory and the cost of one distance are those
of 128-d vectors in every condition. The rotation and every draw are seeded, so each run produces the same data.
"""
from __future__ import annotations

import numpy as np

from . import config, memory
from .data import Dataset


def rotation(dim: int = config.PART4_STORED_DIM, seed: int = config.SEED) -> np.ndarray:
    """A fixed random orthogonal dim x dim matrix (QR of a seeded Gaussian, signs fixed so it is unique)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def intrinsic_dataset(m: int, n_base: int, n_queries: int) -> Dataset:
    """Base and query vectors of intrinsic dimension m, stored in config.PART4_STORED_DIM dimensions."""
    dim = config.PART4_STORED_DIM
    if not 1 <= m <= dim:
        raise ValueError(f"intrinsic dimension {m} outside 1..{dim}")
    memory.require_available((n_base + n_queries) * dim * 4 * 2, f"synthetic data m={m}")
    basis = rotation()[:m]                                             # m orthonormal rows in R^dim
    rng = np.random.default_rng([config.SEED, m])
    base = np.ascontiguousarray(rng.standard_normal((n_base, m), dtype=np.float32) @ basis)
    queries = np.ascontiguousarray(rng.standard_normal((n_queries, m), dtype=np.float32) @ basis)
    return Dataset(name=f"synthetic_m{m}", base=base, queries=queries, stored_neighbors=None, quick=config.is_quick())
