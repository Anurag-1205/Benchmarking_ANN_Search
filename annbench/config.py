"""Central configuration: paths, dataset specs, constants and the quick-mode switch.

Everything that decides *where* files live or *how big* a run is goes through this module, so that
`bash run_all.sh --quick` (which exports ANNBENCH_QUICK=1) and the full run differ in exactly one place.

Layout (<root> is the folder that contains code/):

    <root>/code/           git repo: this package, run_all.sh, README.md, hypothesis.md   -> submitted
    <root>/results/        results.csv + figures/ of the full run                         -> submitted
    <root>/results_quick/  the same for `run_all.sh --quick`                              -> NOT submitted
    <root>/data/           raw HDF5, converted .npy, temporary index files                -> NOT submitted
    <root>/.venv/          virtual environment                                            -> NOT submitted

Override the locations with ANNBENCH_DATA_DIR / ANNBENCH_RESULTS_DIR.
"""
from __future__ import annotations

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
