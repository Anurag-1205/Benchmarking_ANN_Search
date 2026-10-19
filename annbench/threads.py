"""Thread control. Query timing is single-threaded (assignment Part 2); index builds may use more.

FAISS keeps one *global* OpenMP thread count, so it is set explicitly before every phase.
hnswlib keeps a per-index default (`Index.set_num_threads`); `knn_query` also accepts `num_threads=`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import faiss

from . import config


def set_faiss_threads(n: int) -> int:
    """Set FAISS's global OpenMP thread count and return the value FAISS reports back."""
    faiss.omp_set_num_threads(int(n))
    return faiss.omp_get_max_threads()


@contextmanager
def faiss_threads(n: int) -> Iterator[int]:
    """Temporarily run FAISS with `n` threads (e.g. for a build), restoring the previous value afterwards."""
    previous = faiss.omp_get_max_threads()
    try:
        yield set_faiss_threads(n)
    finally:
        faiss.omp_set_num_threads(previous)


def force_single_thread_search(hnsw_index=None) -> int:
    """Enforce the benchmark rule: one thread for every search. Call it before timing anything.

    Sets FAISS to one thread and, when an hnswlib index is given, calls its `set_num_threads(1)`.
    Returns the FAISS thread count now in effect (must be 1).
    """
    n = set_faiss_threads(config.SEARCH_THREADS)
    if hnsw_index is not None:
        hnsw_index.set_num_threads(config.SEARCH_THREADS)
    return n
