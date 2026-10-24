"""Plotting code.

YOURS TO WRITE (assignment ground rules: plots are written by you). One function per report figure; each saves into
config.figures_dir() and returns the path it wrote. stages/report_assets.py selects the rows from results.csv and
calls these functions. Part 3 and Part 4 figures get their own functions here later.

run_all.sh runs without a display, so select a non-interactive backend before importing pyplot
(`import matplotlib; matplotlib.use("Agg")`), and close every figure after saving it (`plt.close(fig)`).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def best_tradeoff_curve(points: pd.DataFrame) -> pd.DataFrame:
    """The best trade-off curve of ONE index on ONE dataset: the Pareto frontier of its (recall, qps) points."""
    if points.empty:
        return points

    # Sort by recall descending. If tied, sort by QPS descending.
    # This allows us to sweep from highest recall downwards, keeping only points that offer strictly better QPS.
    sorted_pts = points.sort_values(by=["recall", "qps"], ascending=[False, False])
    
    frontier_indices = []
    max_qps_so_far = -1.0
    
    for idx, row in sorted_pts.iterrows():
        if row["qps"] > max_qps_so_far:
            frontier_indices.append(idx)
            max_qps_so_far = row["qps"]
            
    # The assignment requires the final curve to be sorted by recall ascending for plotting
    frontier = sorted_pts.loc[frontier_indices].sort_values(by="recall", ascending=True)
    return frontier


def plot_recall_vs_qps(
    results: pd.DataFrame,
    dataset: str,
    out_path: Path | None = None,
    exact_qps: float | None = None,
    title: str | None = None,
) -> Path:
    """Part 2 main figure: recall@10 (x) against single-query QPS (y) for one dataset."""
    if out_path is None:
        out_path = Path("results/figures") / f"recall_vs_qps_{dataset}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)

    # Standardize display names and colors for the report
    display_names = {
        "hnsw": "HNSW",
        "ivf_flat": "IVF-Flat",
        "ivf_pq": "IVF-PQ"
    }
    colors = {
        "hnsw": "tab:blue",
        "ivf_flat": "tab:orange",
        "ivf_pq": "tab:green"
    }

    for index_type in results["index_type"].unique():
        index_data = results[results["index_type"] == index_type]
        frontier = best_tradeoff_curve(index_data)
        
        label = display_names.get(index_type, index_type)
        color = colors.get(index_type, "tab:gray")
        
        ax.plot(
            frontier["recall"], 
            frontier["qps"], 
            marker="o", 
            markersize=4, 
            label=label,
            color=color,
            linewidth=1.5
        )

    if exact_qps is not None:
        ax.plot(
            1.0, 
            exact_qps, 
            marker="*", 
            markersize=10, 
            color="tab:red", 
            linestyle="None",
            label="Exact Baseline"
        )

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (QPS)")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Recall vs. QPS ({dataset.upper()})")
        
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    
    return out_path