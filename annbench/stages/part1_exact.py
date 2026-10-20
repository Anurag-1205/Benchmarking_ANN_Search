"""Part 1 - exact baseline.

Assignment: "Build exact search with FAISS and use it as your reference. Write your own recall@10 function ...
Check that your exact results agree with the neighbours stored in the data file. Finally, time exact search
both one query at a time and with all queries in one batch."

Per dataset:
  1. load (quick mode: first 100k vectors and 2,000 queries) and build IndexFlatL2;
  2. exact ground truth for every query, cached on disk (exact.ground_truth);
  3. full base only: agreement with the neighbours stored in the HDF5 file -- the id-set match rate, plus
     tie-aware agreement through YOUR metrics.recall_at_10 (the stored ids scored as if an index returned them);
  4. timing on the report half of the queries through YOUR timing.py:
       single  one query at a time on 1 thread, over the first SINGLE_QUERY_SAMPLE report queries
               (brute force costs the same for every query, so a sample gives the same QPS);
       batch   all report queries in one call, on 1 thread and on every core (the two thread counts
               separate the effect of batching from the effect of parallelism);
  5. every number goes to results.csv through results_log (experiment "part1_exact"); a run that fails part-way
     leaves the previous rows untouched.

    python -m annbench.stages.part1_exact [--datasets sift glove]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

from .. import config, data, exact, memory, metrics, results_log, threads, timing

EXPERIMENT = "part1_exact"
SINGLE_QUERY_SAMPLE = 300          # report queries timed one at a time (tens of ms each on the full base)
REPEATS_SINGLE = 3
REPEATS_BATCH = 3
BATCH_THREADS = (1, os.cpu_count() or 1)


def _say(message: str) -> None:
    print(f"[part1] {message}", flush=True)       # flush: keep progress in order with errors on stderr


def run_dataset(name: str) -> int:
    """Run Part 1 on one dataset; returns the number of rows written to results.csv."""
    k = config.K
    ds = data.load(name)
    _say(f"{name}: building IndexFlatL2 over {ds.n_base:,} vectors")
    index, add_seconds = exact.build_flat_index(ds.base)
    _say(f"{name}: exact ground truth for {ds.n_queries:,} queries (cached after the first run)")
    gt = exact.ground_truth(ds, index=index)
    queries = ds.report_queries()
    kth_report = gt.kth_dist_sq(k)[ds.report]

    common = dict(experiment=EXPERIMENT, dataset=name, index_type="exact", n_base=ds.n_base,
                  build_params={"index": "IndexFlatL2"}, build_time_s=add_seconds,
                  threads_build=1,                                 # IndexFlat.add is a plain copy
                  index_size_bytes=index.ntotal * index.code_size, bytes_per_vector=index.code_size)
    rows: list[dict] = []

    if ds.stored_neighbors is not None:
        agreement = exact.agreement_with_file(ds, gt, k)
        tie_aware = metrics.recall_at_10(ds.stored_neighbors[:, :k], ds.queries, ds.base, gt.kth_dist_sq(k))
        rows.append(dict(common, split="all", n_queries=ds.n_queries, recall=tie_aware,
                         search_params={"check": "hdf5_neighbors_vs_exact"}, extra=agreement,
                         notes="recall = tie-aware agreement of the file's stored top-10 with our exact distances"))

    _say(f"{name}: recall of exact search through metrics.recall_at_10")
    with threads.faiss_threads(os.cpu_count() or 1):
        _, found = index.search(queries, k)
    exact_recall = metrics.recall_at_10(found, queries, ds.base, kth_report)

    _say(f"{name}: timing one query at a time (1 thread)")
    threads.force_single_thread_search()
    single = timing.time_single_queries(lambda q: index.search(q, k), queries[:SINGLE_QUERY_SAMPLE],
                                        n_repeats=REPEATS_SINGLE)
    rows.append(dict(common, split="report", n_queries=single.n_queries, recall=exact_recall,
                     search_params={"mode": "single"}, threads_search=1,
                     qps=single.qps_median, qps_runs=single.qps_runs, n_repeats=single.n_repeats,
                     peak_rss_mb=memory.peak_rss_mb(),
                     notes=f"timed on the first {single.n_queries} report queries; recall over all report queries"))

    for n_threads in BATCH_THREADS:
        _say(f"{name}: timing all report queries in one batch ({n_threads} thread(s))")
        threads.set_faiss_threads(n_threads)
        batch = timing.time_batch(lambda qs: index.search(qs, k), queries, n_repeats=REPEATS_BATCH)
        rows.append(dict(common, split="report", n_queries=batch.n_queries, recall=exact_recall,
                         search_params={"mode": "batch"}, threads_search=n_threads,
                         qps=batch.qps_median, qps_runs=batch.qps_runs, n_repeats=batch.n_repeats,
                         peak_rss_mb=memory.peak_rss_mb()))
    threads.force_single_thread_search()                          # leave FAISS in the benchmark state

    results_log.drop_rows(EXPERIMENT, dataset=name)
    for row in rows:
        results_log.log_run(**row)
    del index, gt, found, queries, ds
    gc.collect()
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m annbench.stages.part1_exact",
                                     description="Part 1 - exact baseline.")
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    args = parser.parse_args(argv)
    for name in args.datasets:
        try:
            n_rows = run_dataset(name)
        except NotImplementedError as exc:
            sys.exit(f"[part1] {name}: stopped at a function you still have to write -> {exc}")
        _say(f"{name}: {n_rows} rows written to {config.results_csv()}")


if __name__ == "__main__":
    main()
