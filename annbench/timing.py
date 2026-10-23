"""Timing code.

YOURS TO WRITE (assignment ground rules: "write the recall function, timing code and plots yourself").
The functions below are what the benchmark runner will call. Return a TimingResult; the runner passes
qps_median / qps_runs / n_repeats on to results_log.log_run. If you change a signature, tell me and I will
adapt the runner.
"""
from __future__ import annotations

import gc
import time
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

    @property
    def mean_latency_ms(self) -> float:
        """Average milliseconds per query implied by the median QPS (for a batch: amortised over the batch)."""
        return 1000.0 / self.qps_median


def time_single_queries(
    search_fn: Callable[[np.ndarray], Any],
    queries: np.ndarray,
    n_repeats: int = 5,
    n_warmup: int = 100,
) -> TimingResult:
    """Queries per second when queries arrive ONE AT A TIME on ONE thread (assignment Part 2)."""
    n_queries = queries.shape[0]

    # Pre-slice queries to shape (1, dim) outside the timed loop to avoid slicing overhead
    q_list = [queries[i:i+1] for i in range(n_queries)]

    # Warmup phase (untimed)
    warmup_iters = min(n_warmup, n_queries)
    for i in range(warmup_iters):
        search_fn(q_list[i])

    seconds_runs = []
    qps_runs = []

    # Disable garbage collector to prevent random stalls during timing
    gc.disable()
    try:
        for _ in range(n_repeats):
            start = time.perf_counter()
            for q in q_list:
                search_fn(q)
            elapsed = time.perf_counter() - start
            
            seconds_runs.append(elapsed)
            qps_runs.append(n_queries / elapsed)
    finally:
        gc.enable()

    qps_median = float(np.median(qps_runs))

    return TimingResult(
        qps_median=qps_median,
        qps_runs=qps_runs,
        seconds_runs=seconds_runs,
        n_queries=n_queries,
        n_repeats=n_repeats
    )


def time_batch(
    search_fn: Callable[[np.ndarray], Any],
    queries: np.ndarray,
    n_repeats: int = 3,
) -> TimingResult:
    """Queries per second when ALL queries are sent in a single call (Part 1: exact-search batch timing)."""
    n_queries = queries.shape[0]
    seconds_runs = []
    qps_runs = []

    # Disable garbage collector to prevent random stalls during timing
    gc.disable()
    try:
        for _ in range(n_repeats):
            start = time.perf_counter()
            search_fn(queries)
            elapsed = time.perf_counter() - start
            
            seconds_runs.append(elapsed)
            qps_runs.append(n_queries / elapsed)
    finally:
        gc.enable()

    qps_median = float(np.median(qps_runs))

    return TimingResult(
        qps_median=qps_median,
        qps_runs=qps_runs,
        seconds_runs=seconds_runs,
        n_queries=n_queries,
        n_repeats=n_repeats
    )