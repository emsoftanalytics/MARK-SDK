"""Reproducible local benchmark: store/retrieve latency and retrieval quality.

Usage::

    uv run python benchmarks/bench_runtime.py            # 2k and 10k memories
    uv run python benchmarks/bench_runtime.py --sizes 2000 50000 --queries 200

Runs fully offline with the default zero-dependency configuration (hash
embedder, SQLite on disk). Pass ``--json`` for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mark.memory.runtime import MarkRuntime  # noqa: E402
from test_retrieval_hybrid import FACTS, QUERIES  # noqa: E402

_WORDS = (
    "build deploy cache queue worker schema migration token session index shard "
    "replica latency budget review incident rollback feature flag metric alert "
    "contract invoice vendor roadmap sprint ticket parser compiler gateway proxy"
).split()


def _filler(rng: random.Random, i: int) -> str:
    words = rng.sample(_WORDS, 6)
    return f"Note {i}: the {words[0]} {words[1]} affects the {words[2]} {words[3]} during {words[4]} {words[5]}."


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run(size: int, queries: int, seed: int = 7) -> dict:
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        runtime = MarkRuntime.local(store_path=Path(tmp) / "bench.db")
        mem = runtime.memory("bench")
        fact_ids = [mem.store_sync(fact, importance=0.5) for fact in FACTS]

        t0 = time.perf_counter()
        for i in range(size):
            mem.store_sync(_filler(rng, i), importance=round(rng.uniform(0.3, 0.9), 2))
        store_ms = (time.perf_counter() - t0) * 1000 / size

        # Warm-up builds the index once (what a long-lived process pays at start).
        t0 = time.perf_counter()
        mem.retrieve_sync("warm up")
        warmup_ms = (time.perf_counter() - t0) * 1000

        latencies: list[float] = []
        for i in range(queries):
            query = QUERIES[i % len(QUERIES)][0] if i % 2 else " ".join(rng.sample(_WORDS, 3))
            t0 = time.perf_counter()
            mem.retrieve_sync(query)
            latencies.append((time.perf_counter() - t0) * 1000)

        hit1 = hit5 = 0
        for query, gold in QUERIES:
            result = mem.retrieve_sync(query)
            rank = next((n for n, f in enumerate(result.fragments, 1) if f.id == fact_ids[gold]), None)
            hit1 += rank == 1
            hit5 += bool(rank and rank <= 5)
        runtime.shutdown()

    return {
        "memories": size + len(FACTS),
        "store_ms_per_op": round(store_ms, 3),
        "index_warmup_ms": round(warmup_ms, 1),
        "retrieve_p50_ms": round(statistics.median(latencies), 2),
        "retrieve_p95_ms": round(_pct(latencies, 0.95), 2),
        "hit_at_1": f"{hit1}/{len(QUERIES)}",
        "hit_at_5": f"{hit5}/{len(QUERIES)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [run(size, args.queries) for size in args.sizes]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print("  ".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
