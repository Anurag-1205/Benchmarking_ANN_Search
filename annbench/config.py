"""Central configuration: paths, dataset specs, constants, the quick-mode switch and the Part 2 grids.

Everything that decides *where* files live or *how big* a run is goes through this module, so that
`bash run_all.sh --quick` (which exports ANNBENCH_QUICK=1) and the full run differ in exactly one place.

Layout (<root> is the folder that contains code/):

    <root>/code/           git repo: this package, run_all.sh, README.md, hypothesis.md   -> submitted
    <root>/results/        results.csv + figures/ of the full run                         -> submitted
    <root>/results_quick/  the same for `run_all.sh --quick`                              -> NOT submitted
    <root>/data/           raw HDF5, converted .npy, ground truth, cached indexes         -> NOT submitted
    <root>/.venv/          virtual environment                                            -> NOT submitted

Override the locations with ANNBENCH_DATA_DIR / ANNBENCH_RESULTS_DIR.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

# --- layout --------------------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = CODE_DIR.parent

# --- experiment constants ------------------------------------------------------------------------
K = 10                       # neighbours per query: recall@10 everywhere
SEED = 20261025              # one seed for every random choice (subsampling, k-means, generators)
N_QUERIES = 10_000           # queries per dataset in the ann-benchmarks files
TUNE_FRACTION = 0.5          # first half of the queries tunes, second half reports (assignment "Data")
SEARCH_THREADS = 1           # query timing is single-threaded (assignment Part 2)
DEFAULT_BUILD_THREADS = 4    # physical cores of the dev laptop; builds may be parallel but must log it

QUICK_N_BASE = 100_000       # --quick: first 100,000 base vectors
QUICK_N_QUERIES = 2_000      # --quick: first 2,000 queries (1,000 tune / 1,000 report)

# --- Part 2 grids --------------------------------------------------------------------------------
# One index per build setting; for each, the search parameter is swept upwards until the recall on the tune
# queries reaches SWEEP_STOP_RECALL or stops improving (stages/part2_benchmark.py).
HNSW_BUILDS = (                              # M = graph degree, efConstruction = candidate list while building
    {"M": 8, "efConstruction": 200},
    {"M": 16, "efConstruction": 200},
    {"M": 32, "efConstruction": 200},
    {"M": 16, "efConstruction": 100},
)
HNSW_EF_SEARCH = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
IVF_NLIST_FACTORS = (0.25, 1.0, 4.0)         # nlist = factor * sqrt(n_base), rounded to a power of two
IVF_NPROBE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PQ_NLIST_FACTOR = 1.0
PQ_CODE_BYTES = {"sift": (16, 32, 64), "glove": (10, 25, 50)}   # m sub-quantizers x 8 bits; m must divide dim
SWEEP_STOP_RECALL = 0.995
SWEEP_PLATEAU = 0.002                        # "stopped improving": two consecutive gains below this
PART2_REPEATS = 3                            # timing repeats per sweep point (the median is reported)
PART2_TIMED_QUERIES = 1_000                  # report queries timed one at a time per sweep point
QUICK_PART2_TIMED_QUERIES = 300


# --- run-time switches (read at call time so tests and run_all.sh can change them) ------------------
def is_quick() -> bool:
    return os.environ.get("ANNBENCH_QUICK", "0") == "1"


def data_dir() -> Path:
    return Path(os.environ.get("ANNBENCH_DATA_DIR", ROOT_DIR / "data"))


def results_dir() -> Path:
    override = os.environ.get("ANNBENCH_RESULTS_DIR")
    if override:
        return Path(override)
    return ROOT_DIR / ("results_quick" if is_quick() else "results")


def figures_dir() -> Path:
    return results_dir() / "figures"


def results_csv() -> Path:
    return results_dir() / "results.csv"


def build_threads() -> int:
    return int(os.environ.get("ANNBENCH_BUILD_THREADS", DEFAULT_BUILD_THREADS))


# --- Part 2 helpers --------------------------------------------------------------------------------
def nlist_for(n_base: int, factor: float) -> int:
    """factor * sqrt(n_base) rounded to the nearest power of two (at least 16): 256 / 1024 / 4096 at 1M vectors."""
    return max(16, 2 ** round(math.log2(factor * math.sqrt(n_base))))


def part2_builds(kind: str, dataset: str, n_base: int) -> list[dict]:
    """Build settings of one index kind on one dataset; one index is built per entry."""
    if kind == "hnsw":
        return [dict(build) for build in HNSW_BUILDS]
    if kind == "ivf_flat":
        return [{"nlist": nlist_for(n_base, factor)} for factor in IVF_NLIST_FACTORS]
    if kind == "ivf_pq":
        nlist = nlist_for(n_base, PQ_NLIST_FACTOR)
        return [{"nlist": nlist, "m": m} for m in PQ_CODE_BYTES[dataset]]
    raise ValueError(f"unknown index kind {kind!r}")


def part2_sweep(kind: str, build: dict) -> list[int]:
    """Search-parameter values tried on one built index, low to high (nprobe never exceeds nlist)."""
    if kind == "hnsw":
        return list(HNSW_EF_SEARCH)
    return [nprobe for nprobe in IVF_NPROBE if nprobe <= build["nlist"]]


def part2_timed_queries() -> int:
    return QUICK_PART2_TIMED_QUERIES if is_quick() else PART2_TIMED_QUERIES


# --- datasets --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str                          # short key used everywhere
    hdf5_name: str                     # file name on ann-benchmarks.com (URLs: scripts/download_data.sh)
    file_metric: str                   # metric the file was built for (HDF5 attribute "distance")
    normalize: bool                    # L2-normalise train + test while converting
    dim: int
    n_train_bounds: tuple[int, int]    # sanity bounds on the number of base vectors


DATASETS: dict[str, DatasetSpec] = {
    "sift": DatasetSpec("sift", "sift-128-euclidean.hdf5", "euclidean", False, 128, (1_000_000, 1_000_000)),
    "glove": DatasetSpec("glove", "glove-100-angular.hdf5", "angular", True, 100, (1_100_000, 1_250_000)),
}
