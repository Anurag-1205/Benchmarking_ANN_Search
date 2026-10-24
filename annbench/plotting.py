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

def plot_recall_by_difficulty(summary: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Part 3 figure: Recall by query difficulty bin."""
    if out_path is None:
        out_path = Path("results/figures/part3_recall_by_difficulty.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    summary = summary.assign(label=summary["index"].astype(str) + " (target: " + summary["target"].astype(str) + ")")
    
    bins = sorted(summary["bin"].unique())
    x_labels = [f"{b}\n{summary[summary['bin'] == b].iloc[0]['rc_min']:.2f}–{summary[summary['bin'] == b].iloc[0]['rc_max']:.2f}" for b in bins]

    color_map = {"hnsw": "tab:blue", "ivf_flat": "tab:orange", "ivf_pq": "tab:green"}
    
    for label, group in summary.groupby("label"):
        group = group.sort_values("bin")
        idx_type = group["index_type"].iloc[0]
        target = str(group["target"].iloc[0])
        
        color = color_map.get(idx_type, "tab:gray")
        style = "--" if "0.90" in target else ("-" if "0.95" in target else ":")

        ax.plot(group["bin"], group["recall"], marker="o", linestyle=style, linewidth=2, color=color, label=label)

    ax.set_xticks(bins)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bin (relative contrast)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall by Query Difficulty (GloVe-100)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_intrinsic_dimension(summary: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Part 4 figure: QPS degradation against intrinsic dimension."""
    if out_path is None:
        out_path = Path("results/figures/part4_intrinsic_dimension.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    color_map = {"hnsw": "tab:blue", "ivf_flat": "tab:orange", "exact": "tab:red"}

    for index_name, group in summary.groupby("index"):
        group = group.sort_values("m")
        idx_type = group["index_type"].iloc[0]
        
        if index_name.lower().startswith("exact"):
            style, marker, color = "--", "*", color_map["exact"]
        else:
            style, marker, color = "-", "o", color_map.get(idx_type, "tab:gray")
            
        reached = group[group["reached"] == True]
        not_reached = group[group["reached"] == False]
        
        ax.plot(group["m"], group["qps"], linestyle=style, linewidth=2, color=color, label=index_name, zorder=2)
        if not reached.empty:
            ax.scatter(reached["m"], reached["qps"], marker=marker, color=color, s=40, zorder=3)
        if not not_reached.empty:
            ax.scatter(not_reached["m"], not_reached["qps"], marker=marker, facecolors="none", edgecolors=color, s=40, linewidths=1.5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    m_values = sorted(summary["m"].unique())
    ax.set_xticks(m_values)
    ax.set_xticklabels(m_values)
    ax.set_xlabel("Intrinsic Dimension (m)")
    ax.set_ylabel("Queries per second (QPS)")
    ax.set_title("Performance vs. Intrinsic Dimension")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path