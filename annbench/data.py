"""Dataset preparation and loading.

    ann-benchmarks HDF5  --prepare-->  data/processed/<name>/{train,test,neighbors,distances}.npy + meta.json
    load(name)                         Dataset: memory-mapped base, query split, optional subset

Notes
* GloVe is L2-normalised once, here, so every later step (index builds, exact search, generators) sees unit
  vectors and plain Euclidean distance ranks neighbours exactly like cosine (assignment "Data").
* Everything is streamed in row chunks of 100k rows. Measured peak RSS for SIFT is ~600 MB: one 51 MB chunk plus
  the ~490 MB output file mapped into memory (file-backed pages the kernel can drop, not heap).
* The HDF5 `neighbors` are the ground truth for the FULL base. They are only returned when the full base is
  loaded; on any subset (--quick, scaling experiment) recompute the truth with your own exact search.
* The HDF5 `distances` are in the file's own metric (angular for GloVe): never compare them with the L2
  distances you get after normalisation. Take thresholds from your own exact search instead.

CLI
    python -m annbench.data prepare [sift glove] [--overwrite]     (no names = every dataset)
    python -m annbench.data info    [sift glove]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from . import config
from .memory import chunk_slices

CHUNK_ROWS = 100_000          # 100k x 128 float32 = 51 MB per chunk


# --- paths -------------------------------------------------------------------------------------------
def raw_hdf5_path(name: str) -> Path:
    return config.data_dir() / "raw" / name / config.DATASETS[name].hdf5_name


def processed_dir(name: str) -> Path:
    return config.data_dir() / "processed" / name


# --- HDF5 -> .npy ------------------------------------------------------------------------------------
def _sha256(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value):
    """h5py attribute -> JSON-friendly Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _check(spec: config.DatasetSpec, f: h5py.File, attrs: dict) -> None:
    """Fail early (and clearly) if the file is not the dataset we think it is."""
    n, d = f["train"].shape
    lo, hi = spec.n_train_bounds
    problems = []
    if "distance" in attrs and attrs["distance"] != spec.file_metric:
        problems.append(f"metric {attrs['distance']!r}, expected {spec.file_metric!r}")
    if d != spec.dim:
        problems.append(f"dimension {d}, expected {spec.dim}")
    if not lo <= n <= hi:
        problems.append(f"{n:,} base vectors, expected {lo:,}..{hi:,}")
    if f["test"].shape != (config.N_QUERIES, spec.dim):
        problems.append(f"test shape {f['test'].shape}, expected {(config.N_QUERIES, spec.dim)}")
    if f["neighbors"].shape[0] != config.N_QUERIES:
        problems.append(f"neighbors shape {f['neighbors'].shape}")
    if problems:
        raise ValueError(f"{spec.name}: unexpected file contents: " + "; ".join(problems))


def _convert_matrix(src: h5py.Dataset, dst: Path, normalize: bool) -> tuple[int, int]:
    """Stream `src` into a float32 .npy at `dst` in row chunks, optionally L2-normalising every row."""
    n, d = src.shape
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=(n, d))
    for sl in chunk_slices(n, CHUNK_ROWS):
        block = np.asarray(src[sl], dtype=np.float32)        # this chunk only (<= 51 MB)
        if normalize:
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)               # an all-zero row stays zero instead of NaN
            block /= norms
        out[sl] = block
        del block
    out.flush()
    del out
    gc.collect()
    return n, d


def prepare(name: str, overwrite: bool = False) -> Path:
    """Convert one dataset from HDF5 to memory-mappable .npy files (idempotent). Returns the output folder."""
    spec = config.DATASETS[name]
    raw, out = raw_hdf5_path(name), processed_dir(name)
    if (out / "meta.json").exists() and not overwrite:
        return out
    if not raw.exists():
        raise FileNotFoundError(f"{raw} not found - run `bash scripts/download_data.sh {name}` first")
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw, "r") as f:
        attrs = {k: _plain(v) for k, v in f.attrs.items()}
        _check(spec, f, attrs)
        n_train, dim = _convert_matrix(f["train"], out / "train.npy", spec.normalize)
        n_test, _ = _convert_matrix(f["test"], out / "test.npy", spec.normalize)
        neighbors = np.asarray(f["neighbors"][:], dtype=np.int32)
        np.save(out / "neighbors.npy", neighbors)
        np.save(out / "distances.npy", np.asarray(f["distances"][:], dtype=np.float32))
        neighbors_shape = list(neighbors.shape)
    del neighbors
    gc.collect()

    meta = {
        "name": name,
        "source_file": raw.name,
        "source_bytes": raw.stat().st_size,
        "source_sha256": _sha256(raw),
        "hdf5_attrs": attrs,
        "n_train": n_train,
        "n_test": n_test,
        "dim": dim,
        "neighbors_shape": neighbors_shape,
        "normalized": spec.normalize,
        "dtype": "float32",
        "prepared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")   # written last = completion marker
    return out


# --- loading -----------------------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    base: np.ndarray                       # (n_base, dim) float32, C-contiguous, memory-mapped (no RAM copy)
    queries: np.ndarray                    # (n_queries, dim) float32
    stored_neighbors: np.ndarray | None    # (n_queries, 100) int32 file ground truth; None unless the FULL base is loaded
    quick: bool

    @property
    def n_base(self) -> int:
        return self.base.shape[0]

    @property
    def dim(self) -> int:
        return self.base.shape[1]

    @property
    def n_queries(self) -> int:
        return self.queries.shape[0]

    @property
    def tune(self) -> slice:
        """First half of the queries: tune everything here."""
        return slice(0, int(self.n_queries * config.TUNE_FRACTION))

    @property
    def report(self) -> slice:
        """Second half of the queries: final numbers are reported here."""
        return slice(int(self.n_queries * config.TUNE_FRACTION), self.n_queries)

    def tune_queries(self) -> np.ndarray:
        return self.queries[self.tune]

    def report_queries(self) -> np.ndarray:
        return self.queries[self.report]


def load(name: str, n_base: int | None = None, n_queries: int | None = None, quick: bool | None = None) -> Dataset:
    """Load a prepared dataset.

    n_base     use only the first n_base vectors (scaling experiment); default: all, or 100,000 in quick mode
    n_queries  use only the first n_queries queries; default: all 10,000, or 2,000 in quick mode
    quick      default: config.is_quick()

    The base is memory-mapped copy-on-write (mode "c"): no RAM copy, writable as far as FAISS's SWIG layer is
    concerned, and the file on disk is never modified.
    """
    quick = config.is_quick() if quick is None else quick
    folder = processed_dir(name)
    if not (folder / "meta.json").exists():
        raise FileNotFoundError(f"{folder} is not prepared - run `python -m annbench.data prepare {name}`")

    base_full = np.load(folder / "train.npy", mmap_mode="c")
    if n_base is None:
        n_base = config.QUICK_N_BASE if quick else base_full.shape[0]
    if not 0 < n_base <= base_full.shape[0]:
        raise ValueError(f"n_base={n_base} outside 1..{base_full.shape[0]}")

    queries = np.load(folder / "test.npy")
    if n_queries is None:
        n_queries = config.QUICK_N_QUERIES if quick else queries.shape[0]
    if not 0 < n_queries <= queries.shape[0]:
        raise ValueError(f"n_queries={n_queries} outside 1..{queries.shape[0]}")
    queries = np.ascontiguousarray(queries[:n_queries])

    stored = None
    if n_base == base_full.shape[0]:      # the file's ground truth only describes the full base
        stored = np.array(np.load(folder / "neighbors.npy", mmap_mode="r")[:n_queries])
    return Dataset(name=name, base=base_full[:n_base], queries=queries, stored_neighbors=stored, quick=quick)


def describe(name: str) -> str:
    """Summary printed by `info`; doubles as a sanity check of the conversion (GloVe norms must be ~1)."""
    ds = load(name, quick=False)
    base_norm = np.linalg.norm(ds.base[:10_000], axis=1)
    query_norm = np.linalg.norm(ds.queries[:1_000], axis=1)
    truth = "none" if ds.stored_neighbors is None else str(tuple(ds.stored_neighbors.shape))
    return (
        f"{name}: {ds.n_base:,} base x {ds.dim} dims, {ds.n_queries:,} queries "
        f"(tune {ds.tune.stop:,} / report {ds.n_queries - ds.report.start:,}), stored neighbours {truth}\n"
        f"  row norm, base[:10k]   mean {base_norm.mean():.4f}  min {base_norm.min():.4f}  max {base_norm.max():.4f}\n"
        f"  row norm, queries[:1k] mean {query_norm.mean():.4f}  min {query_norm.min():.4f}  max {query_norm.max():.4f}"
    )


# --- CLI ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m annbench.data", description="Prepare / inspect the datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prepare = sub.add_parser("prepare", help="convert HDF5 -> memory-mappable .npy (idempotent)")
    p_prepare.add_argument("--overwrite", action="store_true")
    p_info = sub.add_parser("info", help="print shapes and row norms of the prepared data")
    for p in (p_prepare, p_info):
        p.add_argument("names", nargs="*", help=f"any of {list(config.DATASETS)}; default: all")
    args = parser.parse_args(argv)

    names = args.names or list(config.DATASETS)
    unknown = [n for n in names if n not in config.DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {list(config.DATASETS)}")
    for name in names:
        if args.cmd == "prepare":
            print(f"[data] {name}: ready in {prepare(name, overwrite=args.overwrite)}")
        else:
            print(describe(name))


if __name__ == "__main__":
    main()
