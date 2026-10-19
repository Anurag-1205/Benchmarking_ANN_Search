"""Timing code.

YOURS TO WRITE (assignment ground rules: "write the recall function, timing code and plots yourself").
The functions below are what the benchmark runner will call. Return a TimingResult; the runner passes
qps_median / qps_runs / n_repeats on to results_log.log_run. If you change a signature, tell me and I will
adapt the runner.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimingResult:
    qps_median: float          # headline number: median over the repeats of n_queries / elapsed seconds
    qps_runs: list[float]      # QPS of every repeat, in order (logged so the spread stays visible)
    seconds_runs: list[float]  # wall-clock seconds of every repeat
    n_queries: int             # queries per pass
    n_repeats: int


def time_single_queries(
    search_fn: Callable[[np.ndarray], Any],
    queries: np.ndarray,
    n_repeats: int = 5,
    n_warmup: int = 100,
) -> TimingResult:
    """Queries per second when queries arrive ONE AT A TIME on ONE thread (assignment Part 2).

    Contract the benchmark runner relies on
    ---------------------------------------
    * `search_fn(q)` is called with one query of shape (1, dim), float32, C-contiguous, and performs one
      complete k-NN search, e.g. `lambda q: index.search(q, 10)` for FAISS or
      `lambda q: hnsw.knn_query(q, k=10)` for hnswlib. Its return value is ignored: no recall work and no
      result handling inside the timed region.
    * This function does not touch thread settings. The caller has already run
      threads.force_single_thread_search(...).
    * Warm up first (`n_warmup` untimed queries: page the index in, fill caches), then time `n_repeats`
      full passes over `queries`. The MEDIAN of the per-pass QPS is the headline number ("repeat each timing
      a few times and take the median").

    Things to decide
    ----------------
    * clock: time.perf_counter(); keep per-query slicing/array preparation outside the timed region;
    * garbage collector: consider gc.disable() during the timed passes (re-enable in `finally`);
    * every pass uses the same queries in the same order.
    """
    raise NotImplementedError("TODO(you): the one-query-at-a-time timing loop")


def time_batch(
    search_fn: Callable[[np.ndarray], Any],
    queries: np.ndarray,
    n_repeats: int = 3,
) -> TimingResult:
    """Queries per second when ALL queries are sent in a single call (Part 1: exact-search batch timing).

    `search_fn(queries)` receives the whole (n_queries, dim) float32 array at once. The thread count is
    whatever the caller configured -- state it in the report, because it is part of why batch and
    one-at-a-time differ so much. QPS = n_queries / elapsed seconds; repeat and take the median as above.
    """
    raise NotImplementedError("TODO(you): the all-queries-in-one-call timing")
