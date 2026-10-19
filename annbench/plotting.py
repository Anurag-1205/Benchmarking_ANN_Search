"""Plotting code.

YOURS TO WRITE (assignment ground rules: plots are written by you). One function per report figure; each
saves into config.figures_dir() and returns the path it wrote. Part 3 and Part 4 figures get their own
functions here later.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def plot_recall_vs_qps(
    results: pd.DataFrame,
    dataset: str,
    out_path: Path | None = None,
    exact_qps: float | None = None,
    title: str | None = None,
) -> Path:
    """Part 2 main figure: recall@10 (x) against queries per second (y) for one dataset.

    Parameters
    ----------
    results : DataFrame from results_log.load_results(). Use the rows of `dataset` measured on the
        "report" half of the queries (column `split`), one row per (index, build setting, search setting).
    dataset : "sift" or "glove".
    out_path : where to save; default is config.figures_dir() / f"recall_vs_qps_{dataset}.png".
    exact_qps : single-thread, one-query-at-a-time QPS of exact search; drawn as one point at recall 1.0.
    title : optional figure title.

    What the figure must show (assignment Part 2)
    ---------------------------------------------
    * the best trade-off curve of each index: for every recall level the highest QPS over all its build
      settings (the upper-right envelope of its points), not every raw curve;
    * QPS on a log axis, recall on a linear axis, both labelled with units;
    * the exact baseline, a legend, and a size that stays readable when two datasets share a report page
      (the whole report is limited to 5 pages).
    """
    raise NotImplementedError("TODO(you): the recall-vs-QPS figure")
