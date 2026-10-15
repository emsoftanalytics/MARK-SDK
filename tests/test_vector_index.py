# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import pytest

from mark.index import DimensionMismatchError, VectorIndex, VectorSearchResult

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False

BACKENDS = ["python"] + (["numpy"] if HAS_NUMPY else [])


def _rand(rng: random.Random, dim: int) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    return request.param


def test_basic_add_get_contains_len_ids(backend: str) -> None:
    index = VectorIndex(backend=backend)
    assert index.backend == backend
    assert len(index) == 0 and index.dim is None
    assert bool(index) is True  # empty index is still truthy (no `or` swaps)
    index.add("a", [3.0, 4.0], "A", agent_id="x")
    index.add("b", [1.0, 0.0], agent_id="y")
    assert index.dim == 2
    assert "a" in index and "zzz" not in index
    assert len(index) == 2
    assert index.get("a") == pytest.approx([0.6, 0.8])
    assert sorted(index.ids()) == ["a", "b"]
    assert index.ids("x") == ["a"]
    index.remove("a")
    index.remove("missing")
    assert "a" not in index and index.get("a") is None


def test_dimension_mismatch(backend: str) -> None:
    index = VectorIndex(backend=backend)
    assert index.search([1.0, 0.0]) == []  # empty index: no dim yet
    index.add("a", [1.0, 0.0, 0.0])
    with pytest.raises(DimensionMismatchError):
        index.add("b", [1.0, 0.0])
    with pytest.raises(DimensionMismatchError):
        index.search([1.0, 0.0])
    assert issubclass(DimensionMismatchError, ValueError)
    index.clear()
    assert index.dim is None
    index.add("c", [1.0, 0.0])  # dim resets once empty
    assert index.dim == 2


def test_search_ranking_include_ids_and_agent_scope(backend: str) -> None:
    index = VectorIndex(backend=backend)
    index.add("x1", [1.0, 0.0], "one", agent_id="x")
    index.add("x2", [0.7, 0.7], agent_id="x")
    index.add("y1", [1.0, 0.01], agent_id="y")
    hits = index.search([1.0, 0.0], top_k=3)
    assert [h.fragment_id for h in hits] == ["x1", "y1", "x2"]
    assert isinstance(hits[0], VectorSearchResult) and hits[0].content == "one"
    assert [h.fragment_id for h in index.search([1.0, 0.0], agent_id="x")] == ["x1", "x2"]
    assert [h.fragment_id for h in index.search([1.0, 0.0], agent_id="nobody")] == []
    assert [h.fragment_id for h in index.search([1.0, 0.0], include_ids={"x2", "y1"})] == ["y1", "x2"]
    assert [h.fragment_id for h in index.search([1.0, 0.0], include_ids={"x2", "y1"}, agent_id="x")] == ["x2"]
    assert index.search([1.0, 0.0], include_ids=set()) == []
    assert index.search([1.0, 0.0], top_k=0) == []
    assert [h.fragment_id for h in index.search([1.0, 0.0], min_score=0.9)] == ["x1", "y1"]


def test_zero_vectors_score_zero(backend: str) -> None:
    index = VectorIndex(backend=backend)
    index.add("zero", [0.0, 0.0])
    index.add("one", [1.0, 0.0])
    hits = {h.fragment_id: h.score for h in index.search([0.0, 0.0])}
    assert hits == {"zero": 0.0, "one": 0.0}
    assert {h.fragment_id: h.score for h in index.search([1.0, 0.0])}["zero"] == 0.0


def test_readd_moves_between_agents_and_clear_by_agent(backend: str) -> None:
    index = VectorIndex(backend=backend)
    index.add("f", [1.0, 0.0], agent_id="x")
    index.add("f", [0.0, 1.0], agent_id="y")
    assert index.ids("x") == [] and index.ids("y") == ["f"]
    assert index.get("f") == pytest.approx([0.0, 1.0])
    index.add("g", [1.0, 1.0], agent_id="x")
    index.clear("x")
    assert index.ids() == ["f"]
    assert [h.fragment_id for h in index.search([0.0, 1.0])] == ["f"]


def test_incremental_adds_after_search_are_visible(backend: str) -> None:
    rng = random.Random(1)
    index = VectorIndex(backend=backend)
    for i in range(50):
        index.add(f"f{i}", _rand(rng, 8), agent_id="a")
        q = index.get(f"f{i}")
        assert index.search(q, top_k=1, agent_id="a")[0].fragment_id == f"f{i}"
    for i in range(0, 50, 2):
        index.remove(f"f{i}")
    assert len(index.search(_rand(rng, 8), top_k=100, min_score=-1.0)) == 25


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_numpy_and_python_backends_return_identical_results() -> None:
    rng = random.Random(42)
    dim = 32
    py, np_ = VectorIndex(backend="python"), VectorIndex(backend="numpy")
    for i in range(600):
        vec = _rand(rng, dim)
        agent = f"agent-{i % 3}"
        py.add(f"f{i}", vec, agent_id=agent)
        np_.add(f"f{i}", vec, agent_id=agent)
    # Duplicate vectors create exact ties: order must still agree.
    tie = _rand(rng, dim)
    for idx in (py, np_):
        idx.add("tie-a", tie, agent_id="agent-0")
        idx.add("tie-b", tie, agent_id="agent-0")
    include = {f"f{i}" for i in range(0, 600, 7)} | {"tie-a", "tie-b"}
    small = {f"f{i}" for i in range(0, 600, 97)}  # exercises the row-subset path
    for _ in range(20):
        query = _rand(rng, dim)
        for kwargs in (
            {"top_k": 10},
            {"top_k": 25, "agent_id": "agent-1"},
            {"top_k": 5, "include_ids": include},
            {"top_k": 3, "include_ids": small},
            {"top_k": 1000, "min_score": 0.1},
        ):
            a, b = py.search(query, **kwargs), np_.search(query, **kwargs)
            assert [h.fragment_id for h in a] == [h.fragment_id for h in b]
            assert [h.score for h in a] == pytest.approx([h.score for h in b], abs=1e-5)
    ta = py.search(tie, top_k=2, agent_id="agent-0")
    tb = np_.search(tie, top_k=2, agent_id="agent-0")
    assert [h.fragment_id for h in ta] == [h.fragment_id for h in tb] == ["tie-a", "tie-b"]


def test_save_load_roundtrip_atomic_and_agent_ids(tmp_path: Path, backend: str) -> None:
    index = VectorIndex(backend=backend)
    index.add("a", [1.0, 0.0], "A", agent_id="x")
    index.add("b", [0.0, 1.0], "B")
    path = tmp_path / "sub" / "index.json"
    index.save(path)
    assert [p.name for p in path.parent.iterdir()] == ["index.json"]  # no temp leftovers
    loaded = VectorIndex.load(path, backend=backend)
    assert loaded.ids("x") == ["a"]
    assert loaded.get("b") == pytest.approx([0.0, 1.0])
    assert loaded.search([1.0, 0.0], top_k=1)[0].content == "A"


def test_load_legacy_v1_json(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"vectors": {"a": [0.6, 0.8]}, "content": {"a": "legacy"}}), encoding="utf-8")
    index = VectorIndex.load(path)
    assert index.get("a") == pytest.approx([0.6, 0.8])
    assert index.ids() == ["a"]
    assert index.search([0.6, 0.8])[0].content == "legacy"


def test_concurrent_add_remove_search(backend: str) -> None:
    index = VectorIndex(backend=backend)
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        rng = random.Random(n)
        try:
            for i in range(200):
                index.add(f"{n}-{i}", _rand(rng, 16), agent_id=f"a{n % 2}")
                if i % 3 == 0:
                    index.remove(f"{n}-{i - 1}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        rng = random.Random(99)
        try:
            for _ in range(200):
                for hit in index.search(_rand(rng, 16), top_k=5, agent_id="a0"):
                    assert -1.0001 <= hit.score <= 1.0001
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    index.add("seed", [1.0] * 16, agent_id="a0")
    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for fid in index.ids():
        assert index.get(fid) is not None


def test_invalid_backend_and_empty_embedding() -> None:
    with pytest.raises(ValueError):
        VectorIndex(backend="faiss")
    with pytest.raises(ValueError):
        VectorIndex().add("a", [])
