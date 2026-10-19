"""Snapshot of the machine and the library versions (the report must state both).

    python -m annbench.sysinfo [--out path.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import config, memory

_DISTRIBUTIONS = ("numpy", "h5py", "faiss-cpu", "hnswlib", "pandas", "matplotlib", "pytest")


def _cpu_model() -> str:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    """Number of distinct (socket, core) pairs listed in /proc/cpuinfo."""
    pairs, socket = set(), None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("physical id"):
                socket = line.split(":")[1].strip()
            elif line.startswith("core id"):
                pairs.add((socket, line.split(":")[1].strip()))
    return len(pairs) or None


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect() -> dict:
    import faiss

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": _physical_cores(),
        "ram_total_gib": round(memory.total_bytes() / 2**30, 2),
        "libraries": {d: _version(d) for d in _DISTRIBUTIONS},
        "faiss_compile_options": faiss.get_compile_options(),
        "faiss_default_omp_threads": faiss.omp_get_max_threads(),
        "search_threads": config.SEARCH_THREADS,
        "build_threads": config.build_threads(),
        "mode": "quick" if config.is_quick() else "full",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="also write the JSON to this file")
    args = parser.parse_args()
    text = json.dumps(collect(), indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
