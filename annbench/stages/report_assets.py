"""Report tables and figures from results.csv.

Tables (annbench/tables.py) are written first, so they do not depend on the plotting code. Figures are drawn by
YOUR plotting functions in annbench/plotting.py; this module only selects rows and passes them in.

    python -m annbench.stages.report_assets
"""
from __future__ import annotations

import sys

from .. import config, plotting, results_log, tables


def _say(message: str) -> None:
    print(f"[report] {message}", flush=True)


def part2_report_rows(df, dataset: str):
    """Part 2 rows of one dataset measured on the report queries: what the recall-vs-QPS figure shows."""
    return df[(df["experiment"] == "part2_benchmark") & (df["dataset"] == dataset) & (df["split"] == "report")]


def main() -> None:
    if not config.results_csv().exists():
        sys.exit(f"[report] {config.results_csv()} does not exist yet - run the stages first")
    df = results_log.load_results()

    for path in tables.write_part2_tables(df, config.results_dir() / "tables"):
        _say(f"wrote {path}")

    figures = config.figures_dir()
    figures.mkdir(parents=True, exist_ok=True)
    for name in config.DATASETS:
        rows = part2_report_rows(df, name)
        if rows.empty:
            _say(f"{name}: no Part 2 rows yet; skipping its figure")
            continue
        exact = tables.exact_baseline(df, name)
        try:
            path = plotting.plot_recall_vs_qps(rows, name, out_path=figures / f"recall_vs_qps_{name}.png",
                                               exact_qps=exact["qps"] if exact else None)
        except NotImplementedError as exc:
            sys.exit(f"[report] stopped at a function you still have to write -> {exc}")
        _say(f"{name}: wrote {path}")


if __name__ == "__main__":
    main()
