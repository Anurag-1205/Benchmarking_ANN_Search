"""Exact k-NN search with FAISS IndexFlatL2: the reference every ANN index is measured against (Part 1).

Ground truth for one (dataset, n_base, n_queries) combination is computed once and cached in
<data>/ground_truth/<name>_n<n_base>_q<n_queries>_k<k>.npz:

    ids      (n_queries, k) int64    exact neighbours, nearest first, ties broken by the smaller id
    dist_sq  (n_queries, k) float32  their squared L2 distances, recomputed by direct differences

Why the re-scoring: for a large batch FAISS uses the BLAS identity |x|^2 + |y|^2 - 2<x,y>, whose float32 rounding
error grows with |x|^2 + |y|^2 (squared norms are ~2.6e5 on SIFT). Ties at the 10th neighbour matter in this
assignment (SIFT coordinates are integers, so exact ties are common), so the k candidates FAISS returns are
re-scored with direct differences -- exact for SIFT's integers -- and re-sorted. A vector outside FAISS's top
k = 100 could only belong in the true top 10 if the rounding error exceeded the gap between the 10th and the
100th neighbour, which is orders of magnitude larger on both datasets.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from . import config, memory, threads
from .data import Dataset, source_fingerprint

GT_K = 100            # neighbours kept per query (the HDF5 files also store 100)
RESCORE_CHUNK = 256   # queries per re-scoring block: 256 x 100 x 128 float32 = 13 MB


@dataclass(frozen=True)
class GroundTruth:
    ids: np.ndarray        # (n_queries, k) int64
    dist_sq: np.ndarray    # (n_queries, k) float32

    def kth_dist_sq(self, k: int = config.K) -> np.ndarray:
        """Squared distance from every query to its true k-th neighbour (the threshold recall_at_k uses)."""
        return self.dist_sq[:, k - 1]


def build_flat_index(base: np.ndarray) -> tuple[faiss.IndexFlatL2, float]:
    """IndexFlatL2 holding a copy of `base`. Returns (index, seconds spent in add).

    One add of the whole array means one allocation; adding in chunks would let FAISS's buffer grow
    geometrically and briefly hold two copies.
    """
    n, d = base.shape
    memory.require_available(n * d * 4, f"IndexFlatL2 over {n:,} x {d} float32")
    index = faiss.IndexFlatL2(d)
    start = time.perf_counter()
    index.add(base)
    return index, time.perf_counter() - start


def rescore(base: np.ndarray, queries: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact squared L2 distances of the candidate `ids` (direct differences), each row sorted by (distance, id)."""
    if (ids < 0).any():
        raise ValueError("candidate list contains -1 padding (fewer base vectors than k?)")
    dist = np.empty(ids.shape, dtype=np.float32)
    for sl in memory.chunk_slices(len(ids), RESCORE_CHUNK):
        diff = base[ids[sl]] - queries[sl, None, :]                    # (chunk, k, dim)
        dist[sl] = np.einsum("qkd,qkd->qk", diff, diff, dtype=np.float64)
        del diff
    order = np.lexsort((ids, dist), axis=-1)                          # by distance, then by id
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(dist, order, axis=1)


def _cache_path(ds: Dataset, k: int) -> Path:
    return config.data_dir() / "ground_truth" / f"{ds.name}_n{ds.n_base}_q{ds.n_queries}_k{k}.npz"


def ground_truth(ds: Dataset, index: faiss.IndexFlatL2 | None = None, k: int = GT_K,
                 cache: bool = True) -> GroundTruth:
    """Exact top-k of every query in `ds` against `ds.base`, cached on disk.

    Pass the IndexFlatL2 you already built to avoid a second copy of the base. This is not timed, so FAISS may
    use every core here. Data without a source fingerprint (e.g. synthetic data) is never cached, because two
    different datasets could share a name and a size.
    """
    path = _cache_path(ds, k)
    fingerprint = source_fingerprint(ds.name)
    cache = cache and fingerprint != "unknown"
    if cache and path.exists():
        with np.load(path) as cached:
            if str(cached["fingerprint"]) == fingerprint:
                return GroundTruth(ids=cached["ids"], dist_sq=cached["dist_sq"])

    own_index = index is None
    if own_index:
        index, _ = build_flat_index(ds.base)
    with threads.faiss_threads(os.cpu_count() or 1):
        _, candidates = index.search(ds.queries, k)
    if own_index:
        del index
        gc.collect()

    ids, dist_sq = rescore(ds.base, ds.queries, candidates)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=ids, dist_sq=dist_sq, fingerprint=np.array(fingerprint))
    return GroundTruth(ids=ids, dist_sq=dist_sq)


def agreement_with_file(ds: Dataset, gt: GroundTruth, k: int = config.K) -> dict:
    """How often our exact top-k id set equals the one stored in the HDF5 file (full base only).

    Differences are expected where several vectors tie at the k-th distance. The tie-aware comparison is done
    by the Part 1 stage with metrics.recall_at_10, scoring the stored ids as if an index had returned them.
    """
    if ds.stored_neighbors is None:
        raise ValueError("the stored neighbours describe the full base; load the dataset without a subset")
    ours, theirs = gt.ids[:, :k].tolist(), ds.stored_neighbors[:, :k].tolist()
    same = np.fromiter((set(a) == set(b) for a, b in zip(ours, theirs)), dtype=bool, count=len(ours))
    return {
        "n_queries": int(len(same)),
        "id_sets_identical": int(same.sum()),
        "id_set_match_rate": float(same.mean()),
        "first_mismatches": np.flatnonzero(~same)[:10].tolist(),
    }
