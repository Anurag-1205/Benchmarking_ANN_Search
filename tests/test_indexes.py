"""Tests for indexes.py on a small synthetic dataset (each build takes well under a second)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from annbench import config, data, indexes


@pytest.fixture
def toy(tmp_path, monkeypatch) -> data.Dataset:
    monkeypatch.setenv("ANNBENCH_DATA_DIR", str(tmp_path))
    meta = tmp_path / "processed" / "toy" / "meta.json"          # a fingerprint makes the toy data cacheable
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"source_sha256": "toy"}))
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3000, 16), dtype=np.float32)
    queries = rng.standard_normal((50, 16), dtype=np.float32)
    return data.Dataset(name="toy", base=base, queries=queries, stored_neighbors=None, quick=False)


def _overlap_at_10(found: np.ndarray, ds: data.Dataset) -> float:
    dist = ((ds.queries[:, None, :] - ds.base[None, :, :]) ** 2).sum(-1)
    truth = np.argsort(dist, axis=1)[:, :10]
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found.tolist(), truth.tolist())]))


CASES = [
    ("hnsw", {"M": 8, "efConstruction": 64}, 128, 0.95),
    ("ivf_flat", {"nlist": 16}, 16, 0.99),           # nprobe = nlist scans every list: exact
    ("ivf_pq", {"nlist": 16, "m": 8}, 16, 0.25),      # 8-byte codes are lossy
]


@pytest.mark.parametrize("kind,params,search_value,floor", CASES)
def test_build_search_and_cache_round_trip(toy, kind, params, search_value, floor):
    built, stats = indexes.get_or_build(toy, kind, params)
    assert not stats.from_cache and stats.index_bytes > 0
    assert stats.bytes_per_vector == pytest.approx(stats.index_bytes / toy.n_base)
    built.set_search(search_value)
    found = built.search(toy.queries, 10)
    assert found.shape == (toy.n_queries, 10) and found.dtype == np.int64
    assert _overlap_at_10(found, toy) >= floor
    assert built.force_single_thread() == 1
    built.single_query(10)(np.ascontiguousarray(toy.queries[:1]))   # one query on one thread must work

    loaded, cached = indexes.get_or_build(toy, kind, params)
    assert cached.from_cache and cached.build_time_s == stats.build_time_s
    loaded.set_search(search_value)
    assert np.mean(loaded.search(toy.queries, 10) == found) >= 0.99  # same index file, same answers


def test_rebuild_ignores_the_cache(toy):
    indexes.get_or_build(toy, "ivf_flat", {"nlist": 16})
    _, stats = indexes.get_or_build(toy, "ivf_flat", {"nlist": 16}, rebuild=True)
    assert not stats.from_cache


def test_pq_rejects_a_code_size_that_does_not_divide_the_dimension(toy):
    with pytest.raises(ValueError):
        indexes.get_or_build(toy, "ivf_pq", {"nlist": 16, "m": 5})     # 16 % 5 != 0


def test_part2_grids():
    assert config.nlist_for(1_000_000, 1.0) == 1024
    assert [b["nlist"] for b in config.part2_builds("ivf_flat", "sift", 1_000_000)] == [256, 1024, 4096]
    for name, spec in config.DATASETS.items():
        assert all(spec.dim % b["m"] == 0 for b in config.part2_builds("ivf_pq", name, 100_000))
    assert max(config.part2_sweep("ivf_flat", {"nlist": 64})) == 64
