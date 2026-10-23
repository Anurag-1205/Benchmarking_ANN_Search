"""The one place that writes benchmark numbers: append-only results.csv.

Every benchmark run (one index configuration x one search setting x one query set) is one row, and nothing
else in the code base prints or stores metrics -- experiments call `log_run(...)`.
Dict/list values go into JSON columns, so Part 3 / Part 4 can record their own parameters (put them in
`build_params`, `search_params` or `extra`) without a schema change.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config

COLUMNS: tuple[str, ...] = (
    # identity (filled automatically)
    "run_id", "timestamp_utc", "git_commit", "mode",
    # what was run
    "experiment", "dataset", "index_type", "n_base", "n_queries", "split",
    "build_params", "search_params",
    # outcome
    "k", "recall", "qps", "qps_runs", "n_repeats",
    # environment
    "threads_search", "threads_build",
    # cost
    "build_time_s", "index_size_bytes", "bytes_per_vector", "peak_rss_mb",
    # free-form
    "extra", "notes",
)
JSON_COLUMNS = frozenset({"build_params", "search_params", "qps_runs", "extra"})
REQUIRED = ("experiment", "dataset", "index_type", "n_base", "split")
_AUTO = ("run_id", "timestamp_utc", "git_commit", "mode")


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """Short hash of the code that produced the rows ('+dirty' if the tree has uncommitted changes)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(config.CODE_DIR), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"      # no git, or no commit yet


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):          # numpy scalars and arrays
        return obj.tolist()
    return str(obj)                     # Paths and anything else printable


def _encode(column: str, value: Any) -> Any:
    if value is None:
        return ""
    if column in JSON_COLUMNS:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    if hasattr(value, "item"):          # numpy scalar -> plain Python scalar
        try:
            value = value.item()
        except ValueError:
            raise TypeError(f"column {column!r} takes a scalar, got an array") from None
    if isinstance(value, (list, tuple, dict, set)):
        raise TypeError(f"column {column!r} takes a scalar; put structured values in a JSON column")
    return value


def log_run(*, path: Path | None = None, **fields: Any) -> dict[str, Any]:
    """Append one benchmark run to results.csv and return the row as written.

    Required: experiment, dataset, index_type, n_base, split.
    Filled in automatically: run_id, timestamp_utc, git_commit, mode ("quick"/"full"); k defaults to config.K.
    Everything else is optional. Unknown column names raise KeyError, so a typo can never silently drop a number.
    `path` is for tests; normally rows go to config.results_csv().
    """
    unknown = sorted(set(fields) - set(COLUMNS))
    if unknown:
        raise KeyError(f"unknown results column(s) {unknown}; allowed: {list(COLUMNS)}")
    reserved = sorted(set(fields) & set(_AUTO))
    if reserved:
        raise KeyError(f"column(s) {reserved} are filled in automatically")
    missing = [c for c in REQUIRED if fields.get(c) is None]
    if missing:
        raise KeyError(f"missing required results column(s): {missing}")

    row: dict[str, Any] = {c: "" for c in COLUMNS}
    row.update(
        run_id=uuid.uuid4().hex[:12],
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        mode="quick" if config.is_quick() else "full",
        k=config.K,
    )
    for column, value in fields.items():
        row[column] = _encode(column, value)

    target = Path(path) if path is not None else config.results_csv()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)           # safe even if two processes log at once
        try:
            f.seek(0)
            header = f.readline()
            if not header.strip():
                csv.writer(f, lineterminator="\n").writerow(COLUMNS)
            elif tuple(next(csv.reader([header]))) != COLUMNS:
                raise RuntimeError(f"{target} has a different header than results_log.COLUMNS; "
                                   "move the old file away or migrate it before logging")
            f.seek(0, os.SEEK_END)
            csv.writer(f, lineterminator="\n").writerow([row[c] for c in COLUMNS])
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return row


def drop_rows(experiment: str, *, dataset: str | None = None, index_type: str | None = None,
              path: Path | None = None) -> int:
    """Delete earlier rows of `experiment` (optionally only one dataset and/or index type) for the current mode, so
    a re-run replaces its rows instead of piling up duplicates. Returns the number of rows removed."""
    target = Path(path) if path is not None else config.results_csv()
    if not target.exists():
        return 0
    mode = "quick" if config.is_quick() else "full"
    col = {name: i for i, name in enumerate(COLUMNS)}
    with open(target, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            rows = list(csv.reader(f))
            if not rows:
                return 0
            if tuple(rows[0]) != COLUMNS:
                raise RuntimeError(f"{target} has a different header than results_log.COLUMNS")
            keep = [r for r in rows[1:]
                    if not (r[col["experiment"]] == experiment and r[col["mode"]] == mode
                            and (dataset is None or r[col["dataset"]] == dataset)
                            and (index_type is None or r[col["index_type"]] == index_type))]
            removed = len(rows) - 1 - len(keep)
            if removed:
                f.seek(0)
                f.truncate()
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(COLUMNS)
                writer.writerows(keep)
                f.flush()
                os.fsync(f.fileno())
            return removed
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_results(path: Path | None = None):
    """Read results.csv into a DataFrame; JSON columns are parsed back into Python objects."""
    import pandas as pd

    df = pd.read_csv(path if path is not None else config.results_csv())
    for column in JSON_COLUMNS & set(df.columns):
        df[column] = df[column].map(lambda s: json.loads(s) if isinstance(s, str) and s else None)
    return df
