"""Part 2 tables from results.csv. Settings are chosen on the tune half of the queries; the numbers shown come from
the report half (assignment "Data": tune on the first 5,000 queries, report on the last 5,000).

    operating point   for each index and target recall: the fastest (build, search) setting whose TUNE recall reaches
                      the target. An index that never reaches it gets its highest-recall setting, marked not reached.
    table 1           single-query QPS of each index at recall 0.90 and 0.95, next to exact search
    table 2           build time, index size and bytes per vector of each index at its 0.95 operating point
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

TARGETS = (0.90, 0.95)
INDEX_ORDER = ("hnsw", "ivf_flat", "ivf_pq")
NAMES = {"hnsw": "HNSW", "ivf_flat": "IVF-Flat", "ivf_pq": "IVF-PQ", "exact": "Exact (IndexFlatL2)"}


def _key(params) -> str:
    return json.dumps(params, sort_keys=True)


def _setting(params: dict) -> str:
    return ", ".join(f"{name}={value}" for name, value in params.items())


def sweep_points(df: pd.DataFrame, dataset: str, experiment: str = "part2_benchmark") -> pd.DataFrame:
    """One row per (index, build setting, search setting): report-half recall/QPS/build stats plus the tune recall."""
    rows = df[(df["experiment"] == experiment) & (df["dataset"] == dataset)].copy()
    rows["bkey"] = rows["build_params"].map(_key)
    rows["skey"] = rows["search_params"].map(_key)
    on = ["index_type", "bkey", "skey"]
    tune = rows[rows["split"] == "tune"][on + ["recall"]].rename(columns={"recall": "recall_tune"})
    return rows[rows["split"] == "report"].merge(tune, on=on, how="inner").reset_index(drop=True)


def operating_points(df: pd.DataFrame, dataset: str, targets=TARGETS, experiment: str = "part2_benchmark") -> pd.DataFrame:
    """For every index and target: the fastest setting whose tune recall reaches the target (see module docstring)."""
    points = sweep_points(df, dataset, experiment)
    out = []
    for kind in [k for k in INDEX_ORDER if k in set(points["index_type"])]:
        mine = points[points["index_type"] == kind]
        for target in targets:
            ok = mine[mine["recall_tune"] >= target]
            row = ok.loc[ok["qps"].idxmax()] if len(ok) else mine.loc[mine["recall_tune"].idxmax()]
            out.append({"dataset": dataset, "index_type": kind, "target": target, "reached": bool(len(ok)),
                        "build_params": row["build_params"], "search_params": row["search_params"],
                        "recall_tune": float(row["recall_tune"]), "recall": float(row["recall"]),
                        "qps": float(row["qps"]), "build_time_s": float(row["build_time_s"]),
                        "index_size_bytes": int(row["index_size_bytes"]),
                        "bytes_per_vector": float(row["bytes_per_vector"])})
    return pd.DataFrame(out)


def exact_baseline(df: pd.DataFrame, dataset: str) -> dict | None:
    """Part 1's single-thread, one-query-at-a-time exact search row, or None if Part 1 has not run."""
    rows = df[(df["experiment"] == "part1_exact") & (df["dataset"] == dataset) & (df["split"] == "report")]
    rows = rows[rows["search_params"].map(lambda p: isinstance(p, dict) and p.get("mode") == "single")]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {"qps": float(row["qps"]), "build_time_s": float(row["build_time_s"]),
            "index_size_bytes": int(row["index_size_bytes"]), "bytes_per_vector": float(row["bytes_per_vector"])}


def write_part2_tables(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write both tables as Markdown (for the report) plus the operating points as CSV. Returns the files written."""
    datasets = [d for d in config.DATASETS
                if ((df["experiment"] == "part2_benchmark") & (df["dataset"] == d)).any()]
    if not datasets:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = pd.concat([operating_points(df, d) for d in datasets], ignore_index=True)

    qps_lines = ["| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 |", "|---|---|---|---|"]
    cost_lines = ["| Dataset | Index | Setting at recall 0.95 | Build time (s) | Index size (MB) | Bytes / vector |",
                  "|---|---|---|---|---|---|"]
    for dataset in datasets:
        exact = exact_baseline(df, dataset)
        if exact:
            qps_lines.append(f"| {dataset} | {NAMES['exact']} | {exact['qps']:,.0f} (recall 1.000) "
                             f"| {exact['qps']:,.0f} (recall 1.000) |")
            cost_lines.append(f"| {dataset} | {NAMES['exact']} | - | {exact['build_time_s']:.1f} "
                              f"| {exact['index_size_bytes'] / 1e6:,.0f} | {exact['bytes_per_vector']:.1f} |")
        for kind in INDEX_ORDER:
            mine = ops[(ops["dataset"] == dataset) & (ops["index_type"] == kind)]
            if mine.empty:
                continue
            cells = []
            for target in TARGETS:
                row = mine[mine["target"] == target].iloc[0]
                setting = _setting({**row["build_params"], **row["search_params"]})
                cells.append(f"{row['qps']:,.0f} (recall {row['recall']:.3f}; {setting})" if row["reached"]
                             else f"not reached (best recall {row['recall']:.3f})")
            qps_lines.append(f"| {dataset} | {NAMES[kind]} | {cells[0]} | {cells[1]} |")
            row = mine[mine["target"] == 0.95].iloc[0]
            setting = _setting({**row["build_params"], **row["search_params"]})
            if not row["reached"]:
                setting += f" (never reaches 0.95; best recall {row['recall']:.3f})"
            cost_lines.append(f"| {dataset} | {NAMES[kind]} | {setting} | {row['build_time_s']:.1f} "
                              f"| {row['index_size_bytes'] / 1e6:,.0f} | {row['bytes_per_vector']:.1f} |")

    note = ("Settings are chosen on the tune half (fastest setting whose tune recall reaches the target); the recall "
            "and single-thread, one-query-at-a-time QPS shown are measured on the report half.")
    markdown = out_dir / "part2_tables.md"
    markdown.write_text("\n".join(["## QPS at recall 0.90 and 0.95", "", *qps_lines, "", note, "",
                                   "## Cost at recall 0.95", "", *cost_lines, ""]))
    csv = out_dir / "part2_operating_points.csv"
    ops.assign(build_params=ops["build_params"].map(_key), search_params=ops["search_params"].map(_key)).to_csv(csv, index=False)
    return [markdown, csv]


# --- Part 3 and Part 4 ------------------------------------------------------------------------------
def part3_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (operating point, difficulty bin) of the Part 3 experiment, ready to plot or tabulate."""
    rows = df[df["experiment"] == "part3_query_difficulty"]
    out = []
    for _, r in rows.iterrows():
        targets = r["extra"]["targets"]
        target = f"{targets[0]:.2f}" if isinstance(targets[0], float) else "best"
        out.append({"index_type": r["index_type"], "index": NAMES[r["index_type"]], "target": target,
                    "setting": _setting({**r["build_params"], **r["search_params"]}), "bin": r["extra"]["bin"],
                    "rc_min": r["extra"]["rc_min"], "rc_max": r["extra"]["rc_max"], "recall": float(r["recall"]),
                    "recall_all": r["extra"]["recall_all_report"], "n_queries": int(r["n_queries"])})
    return pd.DataFrame(out).sort_values(["index_type", "target", "bin"]).reset_index(drop=True) if out else pd.DataFrame()


def part4_summary(df: pd.DataFrame, target: float = 0.95) -> pd.DataFrame:
    """Per intrinsic dimension m and index: the fastest setting reaching `target` (chosen on the tune half), its
    report recall and QPS, next to the exact-search QPS and the dataset's mean relative contrast."""
    rows = df[df["experiment"] == "part4_intrinsic_dim"]
    out = []
    for dataset in sorted(set(rows["dataset"]), key=lambda d: int(d.split("_m")[-1])):
        mine = rows[rows["dataset"] == dataset]
        tags = mine["extra"].iloc[0]
        exact = mine[mine["index_type"] == "exact"]
        exact_qps = float(exact["qps"].iloc[0]) if len(exact) else float("nan")
        out.append({"m": tags["m"], "role": tags["role"], "rc_mean": tags["rc_mean"], "index_type": "exact",
                    "index": NAMES["exact"], "reached": True, "setting": "-", "recall": 1.0, "qps": exact_qps,
                    "speedup_vs_exact": 1.0})
        for _, op in operating_points(df, dataset, targets=(target,), experiment="part4_intrinsic_dim").iterrows():
            out.append({"m": tags["m"], "role": tags["role"], "rc_mean": tags["rc_mean"], "index_type": op["index_type"],
                        "index": NAMES[op["index_type"]], "reached": bool(op["reached"]),
                        "setting": _setting({**op["build_params"], **op["search_params"]}), "recall": op["recall"],
                        "qps": op["qps"], "speedup_vs_exact": op["qps"] / exact_qps})
    return pd.DataFrame(out)


def write_part3_part4_tables(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Markdown tables for Parts 3 and 4 (whichever has rows). Returns the files written."""
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    p3 = part3_summary(df)
    if not p3.empty:
        lines = ["| Index | Target | Setting | Overall | " + " | ".join(
                     f"Bin {b} (RC {g['rc_min'].iloc[0]:.2f}-{g['rc_max'].iloc[0]:.2f})" for b, g in p3.groupby("bin")) + " |",
                 "|---|---|---|---|" + "---|" * p3["bin"].nunique()]
        for (_, target, setting), g in p3.groupby(["index", "target", "setting"], sort=False):
            g = g.sort_values("bin")
            lines.append(f"| {g['index'].iloc[0]} | {target} | {setting} | {g['recall_all'].iloc[0]:.3f} | "
                         + " | ".join(f"{x:.3f}" for x in g["recall"]) + " |")
        path = out_dir / "part3_recall_by_difficulty.md"
        path.write_text("## Part 3: recall@10 per difficulty bin (bin 1 = lowest relative contrast = hardest)\n\n"
                        + "\n".join(lines) + "\n")
        written.append(path)
    p4 = part4_summary(df)
    if not p4.empty:
        lines = ["| m | Role | Mean RC | Index | Setting at recall 0.95 | Recall | QPS | Speed-up vs exact |",
                 "|---|---|---|---|---|---|---|---|"]
        for _, r in p4.iterrows():
            qps = f"{r['qps']:,.0f}" if r["reached"] else "not reached"
            lines.append(f"| {r['m']} | {r['role']} | {r['rc_mean']:.2f} | {r['index']} | {r['setting']} | "
                         f"{r['recall']:.3f} | {qps} | {r['speedup_vs_exact']:.1f}x |")
        path = out_dir / "part4_intrinsic_dimension.md"
        path.write_text("## Part 4: cost of recall 0.95 against intrinsic dimension (stored dimension 128)\n\n"
                        + "\n".join(lines) + "\n")
        written.append(path)
    return written
