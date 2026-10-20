"""Evaluation metrics.

YOURS TO WRITE (assignment Part 1: "write your own recall@10 function ... and add a few unit tests").
The signatures below are what the benchmark runner (Part 1 onwards) calls; if you change one, tell me and I
will adapt the runner. Your unit tests belong in tests/test_recall.py (stubs are waiting there).
"""
from __future__ import annotations

import numpy as np


def recall_at_k(
    returned_ids: np.ndarray,
    queries: np.ndarray,
    base: np.ndarray,
    kth_true_dist_sq: np.ndarray,
    k: int = 10,
    rel_eps: float = 1e-5,
    per_query: bool = False,
) -> float | np.ndarray:
    """Tie-aware recall@k of an index's answers against exact search.

    Parameters
    ----------
    returned_ids : (n_queries, >= k) int array
        Ids returned by the index under test, best first. FAISS pads a row with -1 when it finds fewer than k.
    queries : (n_queries, dim) float32
        The queries that produced `returned_ids`, in the same order.
    base : (n_base, dim) float32
        The indexed vectors (may be a np.memmap). Used to recompute the TRUE distance of every returned id,
        because an index such as IVF-PQ reports approximate distances.
    kth_true_dist_sq : (n_queries,) float
        Squared L2 distance from each query to its TRUE k-th nearest neighbour: `exact.GroundTruth.kth_dist_sq(k)`
        (FAISS IndexFlatL2 candidates re-scored exactly; note that FAISS reports *squared* L2).
    k : int
        Number of neighbours that count (10 in this assignment).
    rel_eps : float
        Relative slack on the threshold to absorb float32 rounding (normalised GloVe carries ~1e-7 noise).
    per_query : bool
        False -> mean recall over the queries (a float).
        True  -> array of shape (n_queries,) with the recall of every query; the Part 3 "query difficulty"
        experiment needs per-query values, not just the average.

    Returns
    -------
    float, or (n_queries,) float array when per_query=True.

    Definition to implement
    -----------------------
    A returned id is a hit when its true squared distance to the query is <= kth_true_dist_sq * (1 + rel_eps).
    Comparing distances instead of ids is what makes ties at the k-th neighbour harmless: SIFT coordinates
    are integers, so several vectors can sit at exactly the same distance, and returning any of them is correct.
        recall_i = (number of distinct hits among the first k returned ids of query i) / k

    Cases your implementation must handle (each deserves a unit test)
    ------------------------------------------------------------------
    * padding ids (-1) are never hits and must not be used to index `base`;
    * a duplicate id inside one row counts once;
    * only the first k columns count, and the order inside the row is irrelevant;
    * a tie at the k-th neighbour (more than k vectors within the threshold) is not penalised;
    * memory: `base[returned_ids]` is (n_queries, k, dim) floats -- process the queries in chunks
      (see memory.chunk_slices) rather than in one shot.
    """
    n_queries = queries.shape[0]
    recalls = np.zeros(n_queries, dtype=np.float32)

    # Chunk queries to keep memory footprint under 2 GiB
    # (chunk_size * k * dim * 4 bytes is trivial at 1000 queries)
    chunk_size = 1000

    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx +chunk_size, n_queries)
        q_chunk = queries[start_idx:end_idx]
        ids_chunk = returned_ids[start_idx:end_idx, :k]
        thresh_chunk = kth_true_dist_sq[start_idx:end_idx] * (1.0 + rel_eps)

        chunk_len = end_idx - start_idx
        chunk_hits = np.zeros(chunk_len, dtype=np.float32)

        for i in range(chunk_len):
            row_ids = ids_chunk[i]
            q_vec = q_chunk[i]
            threshold = thresh_chunk[i]

            #1. Filter out invalid/padding entries (-1)
            valid_ids = row_ids[row_ids >= 0]

            #2 Keep unique IDs (duplicates within the top-k count only once)
            unique_ids = np.unique(valid_ids)

            if len(unique_ids) == 0:
                continue

            #3. Retrieve actual base vectors and compute true squared L2 distance
            retrieved_vectors = base[unique_ids]
            diff = retrieved_vectors - q_vec
            dists_sq = np.sum(diff * diff, axis=1)

            # 4. Count valid hits within threshold (tie-aware condition)
            hits = np.sum(dists_sq <= threshold)
            chunk_hits[i] = hits / float(k)

        recalls[start_idx:end_idx] = chunk_hits

    return recalls if per_query else float(np.mean(recalls))

def recall_at_10(
    returned_ids: np.ndarray,
    queries: np.ndarray,
    base: np.ndarray,
    kth_true_dist_sq: np.ndarray,
    rel_eps: float = 1e-5,
    per_query: bool = False,
) -> float | np.ndarray:
    """The assignment's recall@10: `recall_at_k` with k = 10 (same contract, see its docstring)."""
    return recall_at_k(returned_ids, queries, base, kth_true_dist_sq, k=10, rel_eps=rel_eps, per_query=per_query)
