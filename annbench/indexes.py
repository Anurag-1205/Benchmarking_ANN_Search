"""The three ANN indexes of Part 2 behind one small interface, plus an on-disk cache of built indexes.

    kind       library                      build parameters        search parameter
    hnsw       hnswlib.Index(space="l2")    M, efConstruction       ef (efSearch)
    ivf_flat   faiss.IndexIVFFlat           nlist                   nprobe
    ivf_pq     faiss.IndexIVFPQ, 8 bits     nlist, m (code bytes)   nprobe

Builds use config.build_threads() threads (logged). Timed searches run on one thread: force_single_thread(), then
single_query(k). The untimed batch search() used for recall may use every build thread -- the answers do not
depend on the thread count.

A built index is saved under <data>/indexes/<dataset>_n<n_base>/ next to a JSON file with its build statistics.
The serialized file size is the "index size" of the report, and later runs (or Part 3) reload the file instead of
rebuilding; get_or_build(..., rebuild=True) forces a fresh build. hnswlib inserts in parallel, so two builds of the
same HNSW setting are not bit-identical; the cache keeps the one that was measured.

Memory: one index lives in RAM at a time. The base stays memory-mapped and is added in ADD_CHUNK slices, and IVF is
trained on a seeded random sample of at most TRAIN_PER_LIST * nlist vectors.
"""
from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import faiss
import hnswlib
import numpy as np

from . import config, memory, threads
from .data import Dataset, source_fingerprint

KINDS = ("hnsw", "ivf_flat", "ivf_pq")
ADD_CHUNK = 100_000          # vectors per add call (memory-mapped slices, no copies)
TRAIN_PER_LIST = 64          # IVF training points per list (FAISS warns below 39)
TRAIN_MIN = 20_000           # ... and never fewer than this (PQ codebooks want >= 39 * 256 points)


@dataclass(frozen=True)
class BuildStats:
    kind: str
    params: dict
    n_base: int
    build_time_s: float      # wall clock of the whole build (training + adding vectors)
    train_time_s: float      # the IVF / PQ training part of it (0 for HNSW)
    threads_build: int
    index_bytes: int         # size of the serialized index file
    from_cache: bool = False

    @property
    def bytes_per_vector(self) -> float:
        return self.index_bytes / self.n_base


class HnswIndex:
    kind = "hnsw"
    search_param = "ef"

    def __init__(self, index: hnswlib.Index):
        self.index = index

    def set_search(self, value: int) -> None:
        self.index.set_ef(int(value))

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Untimed batch search on every build thread; int64 ids (hnswlib itself returns uint64 labels)."""
        labels, _ = self.index.knn_query(queries, k=k, num_threads=config.build_threads())
        return labels.astype(np.int64)

    def single_query(self, k: int) -> Callable[[np.ndarray], Any]:
        """One complete search for one (1, dim) query on one thread: what the timing loop calls."""
        knn_query = self.index.knn_query
        return lambda q: knn_query(q, k=k, num_threads=1)

    def force_single_thread(self) -> int:
        return threads.force_single_thread_search(self.index)

    def save(self, path: Path) -> None:
        self.index.save_index(str(path))


class FaissIvfIndex:
    search_param = "nprobe"

    def __init__(self, kind: str, index: faiss.Index):
        self.kind, self.index = kind, index

    def set_search(self, value: int) -> None:
        self.index.nprobe = int(value)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Untimed batch search on every build thread; int64 ids."""
        with threads.faiss_threads(config.build_threads()):
            _, ids = self.index.search(queries, k)
        return ids

    def single_query(self, k: int) -> Callable[[np.ndarray], Any]:
        """One complete search for one (1, dim) query; single-threaded after force_single_thread()."""
        search = self.index.search
        return lambda q: search(q, k)

    def force_single_thread(self) -> int:
        return threads.force_single_thread_search()

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))


AnnIndex = HnswIndex | FaissIvfIndex


def build_hnsw(base: np.ndarray, M: int, efConstruction: int) -> tuple[HnswIndex, float]:
    """HNSW over `base` with ids 0..n-1. Returns (index, build seconds)."""
    n, d = base.shape
    memory.require_available(int(n * (4 * d + 8 * M + 12) * 1.1), f"HNSW M={M} over {n:,} vectors")
    index = hnswlib.Index(space="l2", dim=d)
    index.init_index(max_elements=n, M=M, ef_construction=efConstruction, random_seed=config.SEED)
    start = time.perf_counter()
    for sl in memory.chunk_slices(n, ADD_CHUNK):
        index.add_items(base[sl], np.arange(sl.start, sl.stop), num_threads=config.build_threads())
    return HnswIndex(index), time.perf_counter() - start


def _training_sample(base: np.ndarray, n_train: int) -> np.ndarray:
    """A seeded random sample of rows (a copy), or the whole memory-mapped base when it is no bigger."""
    if n_train >= len(base):
        return base
    rows = np.sort(np.random.default_rng(config.SEED).choice(len(base), size=n_train, replace=False))
    return np.ascontiguousarray(base[rows])


def build_ivf(base: np.ndarray, nlist: int, m: int | None = None) -> tuple[FaissIvfIndex, float, float]:
    """IVF-Flat (m=None) or IVF-PQ with m bytes per code. Returns (index, build seconds, of which training)."""
    n, d = base.shape
    if m is not None and d % m:
        raise ValueError(f"PQ needs the dimension ({d}) to be divisible by m ({m})")
    n_train = min(n, max(TRAIN_MIN, TRAIN_PER_LIST * nlist))
    codes = 4 * d if m is None else m
    tables = 0 if m is None else nlist * m * 256 * 4                 # IVF-PQ precomputed distance tables
    memory.require_available(n * (codes + 8) + n_train * d * 4 + tables, f"IVF nlist={nlist} m={m} over {n:,} vectors")
    quantizer = faiss.IndexFlatL2(d)
    if m is None:
        index, kind = faiss.IndexIVFFlat(quantizer, d, nlist), "ivf_flat"
    else:
        index, kind = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8), "ivf_pq"
    sample = _training_sample(base, n_train)
    with threads.faiss_threads(config.build_threads()):
        start = time.perf_counter()
        index.train(sample)
        train_s = time.perf_counter() - start
        del sample
        gc.collect()
        for sl in memory.chunk_slices(n, ADD_CHUNK):
            index.add(base[sl])
        build_s = time.perf_counter() - start
    return FaissIvfIndex(kind, index), build_s, train_s


def _build(kind: str, base: np.ndarray, params: dict) -> tuple[AnnIndex, float, float]:
    if kind == "hnsw":
        built, seconds = build_hnsw(base, params["M"], params["efConstruction"])
        return built, seconds, 0.0
    if kind == "ivf_flat":
        return build_ivf(base, params["nlist"])
    if kind == "ivf_pq":
        return build_ivf(base, params["nlist"], params["m"])
    raise ValueError(f"unknown index kind {kind!r}; expected one of {KINDS}")


def _load(kind: str, path: Path, dim: int, n_base: int) -> AnnIndex:
    if kind == "hnsw":
        index = hnswlib.Index(space="l2", dim=dim)
        index.load_index(str(path), max_elements=n_base)
        return HnswIndex(index)
    return FaissIvfIndex(kind, faiss.read_index(str(path)))


def _cache_paths(ds: Dataset, kind: str, params: dict) -> tuple[Path, Path]:
    folder = config.data_dir() / "indexes" / f"{ds.name}_n{ds.n_base}"
    key = kind + "".join(f"_{name}{value}" for name, value in sorted(params.items()))
    return folder / f"{key}.{'hnsw' if kind == 'hnsw' else 'faiss'}", folder / f"{key}.json"


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def get_or_build(ds: Dataset, kind: str, params: dict, rebuild: bool = False,
                 cache: bool = True) -> tuple[AnnIndex, BuildStats]:
    """The `kind` index with build `params` over `ds.base`: reloaded when a valid cached copy exists, otherwise built
    and saved (its file size is the index size) and kept for next time unless cache=False. Data without a source
    fingerprint (e.g. synthetic data) is never cached."""
    index_path, meta_path = _cache_paths(ds, kind, params)
    fingerprint = source_fingerprint(ds.name)
    cache = cache and fingerprint != "unknown"
    if cache and not rebuild and index_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint and meta.get("n_base") == ds.n_base and meta.get("params") == params:
            fields = {f: meta[f] for f in ("kind", "params", "n_base", "build_time_s", "train_time_s",
                                           "threads_build", "index_bytes")}
            return _load(kind, index_path, ds.dim, ds.n_base), BuildStats(**fields, from_cache=True)

    built, build_s, train_s = _build(kind, ds.base, params)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    built.save(index_path)
    stats = BuildStats(kind=kind, params=dict(params), n_base=ds.n_base, build_time_s=build_s, train_time_s=train_s,
                       threads_build=config.build_threads(), index_bytes=index_path.stat().st_size)
    if cache:
        meta = asdict(stats)
        meta.pop("from_cache")
        meta.update(fingerprint=fingerprint, dim=ds.dim, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    faiss=_version("faiss-cpu"), hnswlib=_version("hnswlib"))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    else:
        index_path.unlink()
    return built, stats
