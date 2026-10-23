"""Plotting code.

YOURS TO WRITE (assignment ground rules: plots are written by you). One function per report figure; each saves into
config.figures_dir() and returns the path it wrote. stages/report_assets.py selects the rows from results.csv and
calls these functions. Part 3 and Part 4 figures get their own functions here later.

run_all.sh runs without a display, so select a non-interactive backend before importing pyplot
(`import matplotlib; matplotlib.use("Agg")`), and close every figure after saving it (`plt.close(fig)`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def best_tradeoff_curve(points: pd.DataFrame) -> pd.DataFrame:
    """The best trade-off curve of ONE index on ONE dataset: the Pareto frontier of its (recall, qps) points.

    Parameters
    ----------
    points : the report-split rows of one index_type on one dataset -- every build setting and every sweep value.
        Useful columns: `recall` (float), `qps` (float, single-query QPS), `build_params` (dict, e.g.
        {"M": 16, "efConstruction": 200}), `search_params` (dict, e.g. {"ef": 64}).

    Returns
    -------
    The rows on the frontier, sorted by recall (ascending). A row is on the frontier when no other row has
    recall >= its recall and qps >= its qps with at least one of the two strictly greater. Because the frontier
    can mix build settings, keep the build_params / search_params columns: they make good point labels.

    This is what "the best trade-off curve for each index" means in Part 2: plot it, one line per index, instead
    of every raw sweep curve.
    """
    raise NotImplementedError("TODO(you): the best trade-off curve (Pareto frontier) of one index")


def plot_recall_vs_qps(
    results: pd.DataFrame,
    dataset: str,
    out_path: Path | None = None,
    exact_qps: float | None = None,
    title: str | None = None,
) -> Path:
    """Part 2 main figure: recall@10 (x) against single-query QPS (y) for one dataset.

    Parameters
    ----------
    results : the Part 2 rows of `dataset` measured on the report half of the queries (experiment
        "part2_benchmark", split "report"; stages/report_assets.py does the filtering). One row per
        (index_type, build_params, search_params). index_type is one of "hnsw", "ivf_flat", "ivf_pq".
    dataset : "sift" or "glove" (for the title and the default file name).
    out_path : where to save; default config.figures_dir() / f"recall_vs_qps_{dataset}.png". Create the parent
        folder if it does not exist.
    exact_qps : single-thread, one-query-at-a-time QPS of exact search from Part 1, or None if Part 1 has not
        run. Draw it as a single marker at recall 1.0.
    title : optional figure title.

    Returns
    -------
    The path of the saved figure.

    What the figure must show (assignment Part 2)
    ---------------------------------------------
    * one line per index: best_tradeoff_curve(rows of that index_type);
    * QPS on a log axis (it spans orders of magnitude), recall on a linear axis, both labelled with units;
    * the exact baseline marker and a legend;
    * a size that stays readable at half a page, since two datasets must share the 5-page report
      (e.g. figsize around (5, 3.5), dpi 200).
    """
    raise NotImplementedError("TODO(you): the recall-vs-QPS figure")
