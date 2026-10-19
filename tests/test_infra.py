"""Tests for the plumbing: results logger, memory helpers, quick-mode switch, thread control.

The recall tests are yours (Part 1) and belong in tests/test_recall.py.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from annbench import config, memory, results_log


def _row(**extra):
    row = dict(experiment="unit", dataset="sift", index_type="exact", n_base=1000, split="report")
    row.update(extra)
    return row


def test_chunk_slices_cover_the_range_exactly_once():
    slices = list(memory.chunk_slices(10, 4))
    assert [(s.start, s.stop) for s in slices] == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError):
        list(memory.chunk_slices(10, 0))


def test_log_run_writes_one_header_and_round_trips_json(tmp_path):
    path = tmp_path / "results.csv"
    results_log.log_run(path=path, **_row(build_params={"M": 16}, search_params={"efSearch": 64},
                                          recall=0.93, qps=1234.5, qps_runs=[1200.0, 1234.5, 1250.0]))
    results_log.log_run(path=path, **_row(index_type="hnsw"))
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert tuple(rows[0]) == results_log.COLUMNS
    assert len(rows) == 3                                   # header + two runs
    df = results_log.load_results(path)
    assert df.loc[0, "build_params"] == {"M": 16}
    assert df.loc[0, "qps_runs"] == [1200.0, 1234.5, 1250.0]
    assert df.loc[0, "recall"] == pytest.approx(0.93)


def test_log_run_accepts_numpy_scalars(tmp_path):
    path = tmp_path / "results.csv"
    results_log.log_run(path=path, **_row(n_base=np.int64(5), recall=np.float32(0.5)))
    df = results_log.load_results(path)
    assert df.loc[0, "n_base"] == 5
    assert df.loc[0, "recall"] == pytest.approx(0.5)


def test_log_run_rejects_unknown_and_missing_columns(tmp_path):
    path = tmp_path / "results.csv"
    with pytest.raises(KeyError, match="unknown"):
        results_log.log_run(path=path, **_row(recal=0.9))   # a typo must not silently drop a number
    with pytest.raises(KeyError, match="missing"):
        results_log.log_run(path=path, experiment="unit", dataset="sift")
    assert not path.exists()                                # nothing is written on error


def test_log_run_refuses_a_file_with_a_different_header(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("some,other,header\n1,2,3\n")
    with pytest.raises(RuntimeError, match="header"):
        results_log.log_run(path=path, **_row())


def test_quick_switch_moves_the_results_dir(monkeypatch):
    monkeypatch.delenv("ANNBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("ANNBENCH_QUICK", "1")
    assert config.is_quick() and config.results_dir().name == "results_quick"
    monkeypatch.setenv("ANNBENCH_QUICK", "0")
    assert not config.is_quick() and config.results_dir().name == "results"


def test_require_available_refuses_an_absurd_allocation():
    with pytest.raises(MemoryError):
        memory.require_available(memory.total_bytes() * 4, "unit test")
    memory.require_available(1024, "tiny request")          # must not raise


def test_force_single_thread_search_pins_faiss_and_hnswlib():
    import faiss
    import hnswlib

    from annbench import threads

    index = hnswlib.Index(space="l2", dim=8)
    index.init_index(max_elements=10, M=4, ef_construction=10)
    assert threads.force_single_thread_search(index) == 1
    assert faiss.omp_get_max_threads() == 1
    with threads.faiss_threads(2) as n:
        assert n == 2
    assert faiss.omp_get_max_threads() == 1                 # the previous value is restored
