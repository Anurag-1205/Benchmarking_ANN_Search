"""Report tables and figures from results.csv.

Tables (annbench/tables.py) are written first, so they never depend on the plotting code. Figures are drawn by
YOUR plotting functions in annbench/plotting.py; this module only selects rows and passes them in. A figure whose
function is not written yet is reported and skipped; the stage then exits non-zero so run_all.sh notices.

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

    table_dir = config.results_dir() / "tables"
    for path in tables.write_part2_tables(df, table_dir) + tables.write_part3_part4_tables(df, table_dir):
        _say(f"wrote {path}")

    figures = config.figures_dir()
    figures.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in config.DATASETS:
        rows = part2_report_rows(df, name)
        if not rows.empty:
            exact = tables.exact_baseline(df, name)
            jobs.append((f"recall vs QPS ({name})", plotting.plot_recall_vs_qps,
                         dict(results=rows, dataset=name, out_path=figures / f"recall_vs_qps_{name}.png",
                              exact_qps=exact["qps"] if exact else None)))
    part3 = tables.part3_summary(df)
    if not part3.empty:
        jobs.append(("Part 3 recall by difficulty", plotting.plot_recall_by_difficulty,
                     dict(summary=part3, out_path=figures / "part3_recall_by_difficulty.png")))
    part4 = tables.part4_summary(df)
    if not part4.empty:
        jobs.append(("Part 4 intrinsic dimension", plotting.plot_intrinsic_dimension,
                     dict(summary=part4, out_path=figures / "part4_intrinsic_dimension.png")))

    missing = []
    for label, draw, kwargs in jobs:
        try:
            _say(f"{label}: wrote {draw(**kwargs)}")
        except NotImplementedError as exc:
            _say(f"{label}: skipped -> {exc}")
            missing.append(label)
    if missing:
        sys.exit(f"[report] figures still to write: {', '.join(missing)}")


if __name__ == "__main__":
    main()
