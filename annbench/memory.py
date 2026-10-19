"""RAM guards and chunking helpers. The dev laptop has ~2 GiB free, so nothing may allocate blindly.

Rules used across the code base:
  * all-pairs / big-matrix work goes through `chunk_slices` (never materialise n_queries x n_base);
  * large temporaries are freed with `del` + `gc.collect()` as soon as they are done;
  * anything that could allocate hundreds of MB calls `require_available` first.
"""
from __future__ import annotations

import os
import resource
from collections.abc import Iterator

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def _meminfo_kib(field: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def total_bytes() -> int:
    return _meminfo_kib("MemTotal") * 1024


def available_bytes() -> int:
    """RAM the kernel says can be allocated without swapping (MemAvailable)."""
    return _meminfo_kib("MemAvailable") * 1024


def rss_bytes() -> int:
    """Resident set size of this process right now."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE_SIZE


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MiB. It never decreases within a process."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # ru_maxrss is KiB on Linux


def require_available(n_bytes: int, what: str, headroom: float = 0.25) -> None:
    """Raise MemoryError *before* allocating if `n_bytes` (+ headroom) would not fit in available RAM."""
    need = int(n_bytes * (1 + headroom))
    have = available_bytes()
    if need > have:
        raise MemoryError(f"{what}: needs ~{need / 2**20:,.0f} MiB but only {have / 2**20:,.0f} MiB is available")


def chunk_slices(n: int, chunk: int) -> Iterator[slice]:
    """Yield slices covering range(n) in blocks of at most `chunk` rows."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))
