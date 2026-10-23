"""Report figures (tables follow) from results.csv.

This module only selects rows and passes them to YOUR plotting functions in annbench/plotting.py.

    python -m annbench.stages.report_assets
"""
from __future__ import annotations

import sys

from .. import config, plotting, results_log


def _say(message: str) -> None:
    print(f"[report] {message}", flush=True)


def exact_single_query_qps(df, dataset: str) -> float | None:
    """Single-thread, one-query-at-a-time QPS of exact search (Part 1), or None if Part 1 has not been run."""
    rows = df[(df["experiment"] == "part1_exact") & (df["dataset"] == dataset) & (df["split"] == "report")]
    rows = rows[rows["search_params"].map(lambda p: isinstance(p, dict) and p.get("mode") == "single")]
    return float(rows["qps"].iloc[-1]) if len(rows) else None


def part2_report_rows(df, dataset: str):
    """Part 2 rows of one dataset measured on the report queries: what the recall-vs-QPS figure shows."""
    return df[(df["experiment"] == "part2_benchmark") & (df["dataset"] == dataset) & (df["split"] == "report")]


def main() -> None:
    if not config.results_csv().exists():
        sys.exit(f"[report] {config.results_csv()} does not exist yet - run the stages first")
    df = results_log.load_results()
    figures = config.figures_dir()
    figures.mkdir(parents=True, exist_ok=True)
    for name in config.DATASETS:
        rows = part2_report_rows(df, name)
        if rows.empty:
            _say(f"{name}: no Part 2 rows yet; skipping its figure")
            continue
        try:
            path = plotting.plot_recall_vs_qps(rows, name, out_path=figures / f"recall_vs_qps_{name}.png",
                                               exact_qps=exact_single_query_qps(df, name))
        except NotImplementedError as exc:
            sys.exit(f"[report] stopped at a function you still have to write -> {exc}")
        _say(f"{name}: wrote {path}")


if __name__ == "__main__":
    main()
