"""Part 2 - benchmark HNSW, IVF-Flat and IVF-PQ on both datasets.

For every build setting in config (one index each), the search parameter (ef for HNSW, nprobe for IVF) is swept
from low to high. At every sweep point:
  * recall@10 on the tune half and on the report half of the queries through YOUR metrics.recall_at_10, using
    answers from an untimed multi-threaded batch search (the answers do not depend on the thread count);
  * single-query QPS through YOUR timing.time_single_queries on the first config.part2_timed_queries() report
    queries, on one thread (index.force_single_thread(): FAISS omp_set_num_threads(1), hnswlib num_threads=1).
The sweep stops once the tune recall reaches config.SWEEP_STOP_RECALL, or when it has stopped improving (two
consecutive gains below config.SWEEP_PLATEAU, e.g. IVF-PQ's quantisation ceiling).

Each sweep point gives two rows in results.csv: split "tune" (recall only, used to choose settings) and split
"report" (recall + QPS: the numbers the report shows). Both carry the build statistics: build time, serialized
index size, bytes per vector, build threads. Rows are logged as each index finishes, so an interrupted run keeps
the finished indexes; re-running an index family replaces its rows. Built indexes are cached (indexes.py).

    python -m annbench.stages.part2_benchmark [--datasets sift glove] [--indexes hnsw ivf_flat ivf_pq] [--rebuild]
"""
from __future__ import annotations

import argparse
import gc
import sys

from .. import config, data, exact, indexes, memory, metrics, results_log, timing

EXPERIMENT = "part2_benchmark"


def _say(message: str) -> None:
    print(f"[part2] {message}", flush=True)


def _stalled(history: list[float]) -> bool:
    """True once each of the last two steps improved the tune recall by less than config.SWEEP_PLATEAU."""
    return len(history) >= 3 and all(b - a < config.SWEEP_PLATEAU for a, b in zip(history[-3:-1], history[-2:]))


def sweep(ds: data.Dataset, gt: exact.GroundTruth, index: indexes.AnnIndex, stats: indexes.BuildStats,
          values: list[int]) -> list[dict]:
    """Rows for one built index: two per sweep value (tune recall; report recall + single-query QPS)."""
    k = config.K
    kth = gt.kth_dist_sq(k)
    tune_q, report_q = ds.tune_queries(), ds.report_queries()
    kth_tune, kth_report = kth[ds.tune], kth[ds.report]
    timed_q = report_q[: config.part2_timed_queries()]
    common = dict(experiment=EXPERIMENT, dataset=ds.name, index_type=stats.kind, n_base=ds.n_base,
                  build_params=stats.params, build_time_s=stats.build_time_s, index_size_bytes=stats.index_bytes,
                  bytes_per_vector=stats.bytes_per_vector, threads_build=stats.threads_build)
    build_extra = {"train_time_s": stats.train_time_s, "index_from_cache": stats.from_cache}

    rows: list[dict] = []
    history: list[float] = []
    for value in values:
        index.set_search(value)
        search_params = {index.search_param: value}
        recall_tune = metrics.recall_at_10(index.search(tune_q, k), tune_q, ds.base, kth_tune)
        recall_report = metrics.recall_at_10(index.search(report_q, k), report_q, ds.base, kth_report)
        search_threads = index.force_single_thread()
        timed = timing.time_single_queries(index.single_query(k), timed_q, n_repeats=config.PART2_REPEATS)
        rows.append(dict(common, split="tune", n_queries=len(tune_q), recall=recall_tune,
                         search_params=search_params, extra=build_extra,
                         notes="recall on the tune queries (used to choose settings); not timed"))
        rows.append(dict(common, split="report", n_queries=len(report_q), recall=recall_report,
                         search_params=search_params, qps=timed.qps_median, qps_runs=timed.qps_runs,
                         n_repeats=timed.n_repeats, threads_search=search_threads, peak_rss_mb=memory.peak_rss_mb(),
                         extra=build_extra | {"n_timed_queries": timed.n_queries},
                         notes="recall over all report queries; QPS over the first n_timed_queries of them"))
        history.append(recall_tune)
        if recall_tune >= config.SWEEP_STOP_RECALL or _stalled(history):
            break
    return rows


def run_dataset(name: str, kinds: list[str], rebuild: bool) -> int:
    """Benchmark every requested index family on one dataset; returns the number of rows written."""
    ds = data.load(name)
    gt = exact.ground_truth(ds)
    written = 0
    for kind in kinds:
        results_log.drop_rows(EXPERIMENT, dataset=name, index_type=kind)
        for params in config.part2_builds(kind, name, ds.n_base):
            _say(f"{name}: {kind} {params}: building (or loading from the cache)")
            index, stats = indexes.get_or_build(ds, kind, params, rebuild=rebuild)
            _say(f"{name}: {kind} {params}: {'loaded from the cache' if stats.from_cache else 'built'}; "
                 f"sweeping {index.search_param}")
            rows = sweep(ds, gt, index, stats, config.part2_sweep(kind, params))
            for row in rows:
                results_log.log_run(**row)
            written += len(rows)
            del index
            gc.collect()
    del ds, gt
    gc.collect()
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m annbench.stages.part2_benchmark",
                                     description="Part 2 - benchmark HNSW, IVF-Flat and IVF-PQ.")
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--indexes", nargs="+", choices=list(indexes.KINDS), default=list(indexes.KINDS))
    parser.add_argument("--rebuild", action="store_true", help="ignore cached indexes and build them again")
    args = parser.parse_args(argv)
    for name in args.datasets:
        try:
            n_rows = run_dataset(name, args.indexes, args.rebuild)
        except NotImplementedError as exc:
            sys.exit(f"[part2] {name}: stopped at a function you still have to write -> {exc}")
        _say(f"{name}: {n_rows} rows written to {config.results_csv()}")


if __name__ == "__main__":
    main()
