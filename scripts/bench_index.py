"""Benchmark VectorIndex search latency and HashEmbeddingProvider throughput.

Usage:
    uv run python scripts/bench_index.py [--sizes 10000 100000] [--dim 384] [--queries 20]

The pure-Python backend is skipped above --python-max vectors (default 100000)
because building it is slow; pass a larger value to force it.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

from mark.embeddings import HashEmbeddingProvider
from mark.index import VectorIndex


def _bench_search(backend: str, size: int, dim: int, queries: int, rng: random.Random) -> None:
    index = VectorIndex(backend=backend)
    start = time.perf_counter()
    for i in range(size):
        index.add(f"f{i}", [rng.gauss(0.0, 1.0) for _ in range(dim)], agent_id="agent")
    build_s = time.perf_counter() - start
    probe = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    first = time.perf_counter()
    index.search(probe, top_k=10, agent_id="agent")  # includes lazy matrix build
    first_ms = (time.perf_counter() - first) * 1000
    timings = []
    for _ in range(queries):
        query = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        t0 = time.perf_counter()
        index.search(query, top_k=10, agent_id="agent")
        timings.append((time.perf_counter() - t0) * 1000)
    subset = {f"f{i}" for i in range(0, size, 10)}
    t0 = time.perf_counter()
    index.search(probe, top_k=10, include_ids=subset)
    subset_ms = (time.perf_counter() - t0) * 1000
    print(
        f"{backend:>6} n={size:>7} dim={dim}: add {build_s:6.2f}s | first search {first_ms:8.1f} ms | "
        f"search p50 {statistics.median(timings):8.2f} ms p95 {sorted(timings)[int(len(timings) * 0.95) - 1]:8.2f} ms | "
        f"10% include_ids {subset_ms:8.2f} ms"
    )


def _bench_hash() -> None:
    provider = HashEmbeddingProvider()
    text = "Elena carries a brass key into the warehouse at midnight."
    provider.embed(text)
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        provider.embed(text)
    warm = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for i in range(n):
        provider.embed(f"{text} variant {i} token{i * 7}")
    cold = (time.perf_counter() - t0) / n * 1000
    print(f"HashEmbeddingProvider(dim=256): repeated sentence {warm:.4f} ms, novel sentences {cold:.4f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--python-max", type=int, default=100_000)
    args = parser.parse_args()
    rng = random.Random(0)
    _bench_hash()
    for size in args.sizes:
        for backend in ("numpy", "python"):
            if backend == "python" and size > args.python_max:
                continue
            try:
                _bench_search(backend, size, args.dim, args.queries, rng)
            except ImportError as exc:
                print(f"{backend}: skipped ({exc})")


if __name__ == "__main__":
    main()
