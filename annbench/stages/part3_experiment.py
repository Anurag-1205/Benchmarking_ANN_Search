"""Part 3 - query difficulty on GloVe-100. Hypothesis: code/hypothesis.md, committed (b7ea947) before this ran.

  1. relative contrast of every report query: RC = mean distance to every base vector / distance to its nearest
     neighbour (exact, chunked over queries and base vectors; never an n_queries x n_base matrix);
  2. four equal bins by RC quartile, bin 1 = lowest RC = hardest;
  3. operating points chosen on the tune half from the Part 2 rows (tables.operating_points): for HNSW and IVF-Flat
     the fastest setting reaching tune recall 0.90 and 0.95; IVF-PQ never reaches them, so its highest-recall
     setting is used;
  4. each operating point reuses the cached Part 2 index; recall@10 per report query through YOUR
     metrics.recall_at_10(per_query=True), averaged per bin. Nothing is timed.

One results.csv row per (operating point, bin). Needs the Part 2 GloVe rows of the same mode (full or --quick).

    python -m annbench.stages.part3_experiment
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np

from .. import config, data, exact, indexes, memory, metrics, results_log, tables

EXPERIMENT = "part3_query_difficulty"
DATASET = "glove"
N_BINS = 4
QUERY_CHUNK = 500
BASE_CHUNK = 50_000          # 500 x 50,000 float32 = 100 MB per block


def _say(message: str) -> None:
    print(f"[part3] {message}", flush=True)


def relative_contrast(base: np.ndarray, queries: np.ndarray, nn_dist_sq: np.ndarray) -> np.ndarray:
    """Mean Euclidean distance from each query to all of `base`, divided by its nearest-neighbour distance.

    A query whose nearest neighbour is at distance 0 gets RC = inf (the easiest possible query)."""
    q_norm = (queries.astype(np.float64) ** 2).sum(1).astype(np.float32)
    total = np.zeros(len(queries), dtype=np.float64)
    for qs in memory.chunk_slices(len(queries), QUERY_CHUNK):
        q = np.ascontiguousarray(queries[qs])
        for bs in memory.chunk_slices(len(base), BASE_CHUNK):
            block = np.asarray(base[bs])
            dist = q @ block.T                                   # in place from here: no further big temporaries
            dist *= -2.0
            dist += q_norm[qs, None]
            dist += (block * block).sum(1)[None, :]
            np.maximum(dist, 0.0, out=dist)
            np.sqrt(dist, out=dist)
            total[qs] += dist.sum(1, dtype=np.float64)
            del dist, block
    gc.collect()
    nn = np.sqrt(np.maximum(nn_dist_sq.astype(np.float64), 0.0))
    with np.errstate(divide="ignore"):
        return np.where(nn > 0, (total / len(base)) / nn, np.inf)


def difficulty_bins(rc: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin index 0..n_bins-1 by RC rank, so every bin holds the same number of queries (+-1); 0 holds the lowest
    RC (hardest queries). Ranking instead of quantile edges keeps the bins equal even with ties or RC = inf."""
    order = np.argsort(rc, kind="stable")
    bins = np.empty(len(rc), dtype=np.int64)
    bins[order] = np.arange(len(rc)) * n_bins // len(rc)
    return bins


def _unique_points(ops) -> list[dict]:
    """Operating points with the targets each one serves (IVF-PQ's best setting serves both)."""
    merged: dict[str, dict] = {}
    for _, op in ops.iterrows():
        key = json.dumps([op["index_type"], op["build_params"], op["search_params"]], sort_keys=True)
        point = merged.setdefault(key, {**op.to_dict(), "targets": []})
        point["targets"].append(float(op["target"]) if op["reached"] else f"best (target {op['target']} not reached)")
    return list(merged.values())


def main(argv: list[str] | None = None) -> None:
    del argv
    if not config.results_csv().exists():
        sys.exit("[part3] results.csv does not exist yet - run Part 2 first")
    ops = tables.operating_points(results_log.load_results(), DATASET)
    if ops.empty:
        sys.exit(f"[part3] no Part 2 rows for {DATASET} in {config.results_csv()} - run Part 2 first")

    ds = data.load(DATASET)
    gt = exact.ground_truth(ds)
    queries = ds.report_queries()
    kth = gt.kth_dist_sq(config.K)[ds.report]
    _say(f"relative contrast of {len(queries):,} report queries against {ds.n_base:,} base vectors")
    rc = relative_contrast(ds.base, queries, gt.dist_sq[ds.report, 0])
    bins = difficulty_bins(rc)

    rows: list[dict] = []
    try:
        for point in _unique_points(ops):
            kind, build, search = point["index_type"], point["build_params"], point["search_params"]
            _say(f"{kind} {build} {search}: per-query recall (targets {point['targets']})")
            index, stats = indexes.get_or_build(ds, kind, build)
            index.set_search(next(iter(search.values())))
            per_query = metrics.recall_at_10(index.search(queries, config.K), queries, ds.base, kth, per_query=True)
            del index
            gc.collect()
            for b in range(N_BINS):
                mask = bins == b
                in_bin = rc[mask]
                finite = in_bin[np.isfinite(in_bin)]
                rows.append(dict(
                    experiment=EXPERIMENT, dataset=DATASET, index_type=kind, n_base=ds.n_base, split="report",
                    n_queries=int(mask.sum()), recall=float(np.mean(per_query[mask])),
                    build_params=build, search_params=search, build_time_s=stats.build_time_s,
                    index_size_bytes=stats.index_bytes, bytes_per_vector=stats.bytes_per_vector,
                    threads_build=stats.threads_build,
                    extra={"bin": b + 1, "n_bins": N_BINS, "rc_min": float(finite.min()), "rc_max": float(finite.max()),
                           "rc_median": float(np.median(finite)), "targets": point["targets"],
                           "recall_all_report": float(np.mean(per_query)), "recall_min_in_bin": float(per_query[mask].min())},
                    notes="bin 1 = lowest relative contrast (hardest queries); recall averaged over the bin"))
    except NotImplementedError as exc:
        sys.exit(f"[part3] stopped at a function you still have to write -> {exc}")

    results_log.drop_rows(EXPERIMENT, dataset=DATASET)
    for row in rows:
        results_log.log_run(**row)
    _say(f"{len(rows)} rows written to {config.results_csv()}")


if __name__ == "__main__":
    main()
