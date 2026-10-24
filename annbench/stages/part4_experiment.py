"""Part 4 - hard geometry: intrinsic dimension. Hypothesis: code/hypothesis.md, committed (b7ea947) before this ran.

For each intrinsic dimension m in config.PART4_DIMS (m = 8 is the unchanged control), on seeded synthetic data stored
in 128 dimensions (synthetic.py; 200,000 base vectors and 2,000 queries, 50,000 / 1,000 in --quick):
  1. exact ground truth (not cached: synthetic data has no fingerprint) and the mean relative contrast of the report
     queries (part3_experiment.relative_contrast);
  2. exact search timed one query at a time on one thread (YOUR timing loop) as the baseline;
  3. HNSW (M=16, efConstruction=200) and IVF-Flat (nlist = sqrt(n) -> 512) with FIXED build settings, each swept over
     efSearch / nprobe exactly as in Part 2 (part2_benchmark.sweep: tune + report rows, single-thread QPS).
Only m changes between conditions. Every row carries extra = {m, role, rc_mean, rc_median, stored_dim}.
Rows are logged after each dimension; a re-run replaces the experiment's rows.

    python -m annbench.stages.part4_experiment
"""
from __future__ import annotations

import gc
import sys

import numpy as np

from .. import config, exact, indexes, memory, metrics, results_log, synthetic, threads, timing
from .part2_benchmark import sweep
from .part3_experiment import relative_contrast

EXPERIMENT = "part4_intrinsic_dim"


def _say(message: str) -> None:
    print(f"[part4] {message}", flush=True)


def run_dimension(m: int) -> list[dict]:
    n_base, n_queries = config.part4_sizes()
    ds = synthetic.intrinsic_dataset(m, n_base, n_queries)
    k = config.K
    flat, add_seconds = exact.build_flat_index(ds.base)
    gt = exact.ground_truth(ds, index=flat, cache=False)
    report_q = ds.report_queries()
    rc = relative_contrast(ds.base, report_q, gt.dist_sq[ds.report, 0])
    finite = rc[np.isfinite(rc)]
    tags = {"m": m, "role": "control" if m == config.PART4_CONTROL_DIM else "setting",
            "stored_dim": config.PART4_STORED_DIM, "rc_mean": float(finite.mean()), "rc_median": float(np.median(finite))}

    _say(f"m={m}: exact search baseline")
    kth_report = gt.kth_dist_sq(k)[ds.report]
    with threads.faiss_threads(config.build_threads()):
        _, found = flat.search(report_q, k)
    exact_recall = metrics.recall_at_10(found, report_q, ds.base, kth_report)
    threads.force_single_thread_search()
    timed = timing.time_single_queries(lambda q: flat.search(q, k), report_q[:config.PART4_EXACT_TIMED_QUERIES], n_repeats=3)
    rows = [dict(experiment=EXPERIMENT, dataset=ds.name, index_type="exact", n_base=ds.n_base, split="report",
                 n_queries=len(report_q), recall=exact_recall, build_params={"index": "IndexFlatL2"},
                 search_params={"mode": "single"}, qps=timed.qps_median, qps_runs=timed.qps_runs,
                 n_repeats=timed.n_repeats, threads_search=1, threads_build=1, build_time_s=add_seconds,
                 index_size_bytes=flat.ntotal * flat.code_size, bytes_per_vector=flat.code_size,
                 peak_rss_mb=memory.peak_rss_mb(), extra={**tags, "n_timed_queries": timed.n_queries})]
    del flat, found
    gc.collect()

    for kind, params in config.part4_builds(ds.n_base).items():
        _say(f"m={m}: {kind} {params}: building and sweeping")
        index, stats = indexes.get_or_build(ds, kind, params, cache=False)
        for row in sweep(ds, gt, index, stats, config.part2_sweep(kind, params), experiment=EXPERIMENT):
            row["extra"] = {**row["extra"], **tags}
            rows.append(row)
        del index
        gc.collect()
    del ds, gt
    gc.collect()
    return rows


def main(argv: list[str] | None = None) -> None:
    del argv
    results_log.drop_rows(EXPERIMENT)
    for m in config.PART4_DIMS:
        try:
            rows = run_dimension(m)
        except NotImplementedError as exc:
            sys.exit(f"[part4] m={m}: stopped at a function you still have to write -> {exc}")
        for row in rows:
            results_log.log_run(**row)
        _say(f"m={m}: {len(rows)} rows written to {config.results_csv()}")


if __name__ == "__main__":
    main()
