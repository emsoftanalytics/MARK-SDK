"""Compare embedders on the paraphrase / semantic retrieval set.

Usage::

    python benchmarks/embeddings/eval_embedders.py                       # hash + auto
    python benchmarks/embeddings/eval_embedders.py --embedders hash model2vec fastembed
    python benchmarks/embeddings/eval_embedders.py --allow-network \\
        --embedders hash "ollama:nomic-embed-text-v2-moe:latest"
    python benchmarks/embeddings/eval_embedders.py --check               # validate dataset

Each embedder is given as a spec string (see ``mark.embeddings.auto``). Two
numbers are reported per embedder and query category:

- ``vector``: the embedder alone — cosine ranking of all 60 facts;
- ``hybrid``: MARK's real pipeline (``retrieve_sync``: FTS5 BM25 + vector,
  reciprocal-rank fusion, scoring), which shows how much lexical retrieval
  already compensates for a weak embedder.

Latency is per query: ``embed`` is one query embedding, ``retrieve`` is one
full hybrid ``retrieve_sync`` call. Network specs (``ollama:``/``openai:``)
are refused unless ``--allow-network`` is passed, so the script never talks
to a server by accident.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from dataset import CATEGORIES, FACTS, QUERIES  # noqa: E402

from mark.embeddings import EmbeddingProvider  # noqa: E402
from mark.embeddings.auto import embedder_from_spec, is_network_spec  # noqa: E402
from mark.embeddings.base import _POSSESSIVE_RE, _STOPWORDS, _WORD_RE  # noqa: E402
from mark.memory.runtime import MarkRuntime  # noqa: E402


def _content_words(text: str) -> set[str]:
    words = _WORD_RE.findall(_POSSESSIVE_RE.sub(r"\1", text.lower()))
    return {word for word in words if word not in _STOPWORDS}


def check_dataset() -> list[str]:
    """Problems with the set: paraphrases sharing words, bad indices."""
    problems = []
    for query, gold, category in QUERIES:
        if not 0 <= gold < len(FACTS) or category not in CATEGORIES:
            problems.append(f"bad entry: {query!r}")
            continue
        shared = _content_words(query) & _content_words(FACTS[gold])
        if category == "paraphrase" and shared:
            problems.append(f"paraphrase shares {sorted(shared)}: {query!r}")
    return problems


def _rank(order: list[int], gold: int) -> int | None:
    return order.index(gold) + 1 if gold in order else None


def _ms(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {"p50": round(statistics.median(ordered) * 1000, 2), "p95": round(p95 * 1000, 2)}


def evaluate(spec: str, provider: EmbeddingProvider) -> dict:
    """Vector-only and hybrid ranks for every query, plus latencies."""
    start = time.perf_counter()
    fact_vectors = provider.embed_batch(FACTS)
    index_seconds = time.perf_counter() - start

    vector_ranks, embed_times = [], []
    for query, gold, _ in QUERIES:
        t0 = time.perf_counter()
        q = provider.embed(query)
        embed_times.append(time.perf_counter() - t0)
        scores = [sum(a * b for a, b in zip(q, v)) for v in fact_vectors]
        order = sorted(range(len(FACTS)), key=lambda i: -scores[i])
        vector_ranks.append(_rank(order, gold))

    hybrid_ranks, retrieve_times = [], []
    with tempfile.TemporaryDirectory() as tmp:
        runtime = MarkRuntime.local(store_path=Path(tmp) / "eval.db", embedder=provider)
        try:
            mem = runtime.memory("eval")
            ids = [mem.store_sync(fact) for fact in FACTS]
            position = {fid: i for i, fid in enumerate(ids)}
            mem.retrieve_sync("warm up")  # index sync outside the timed loop
            for query, gold, _ in QUERIES:
                t0 = time.perf_counter()
                result = mem.retrieve_sync(query)
                retrieve_times.append(time.perf_counter() - t0)
                order = [position[f.id] for f in result.fragments if f.id in position]
                hybrid_ranks.append(_rank(order, gold))
        finally:
            runtime.shutdown()

    return {
        "spec": spec,
        "fingerprint": provider.fingerprint,
        "index_ms_per_fact": round(index_seconds * 1000 / len(FACTS), 2),
        "embed_ms": _ms(embed_times),
        "retrieve_ms": _ms(retrieve_times),
        "scores": {mode: _score(ranks) for mode, ranks in (("vector", vector_ranks), ("hybrid", hybrid_ranks))},
        "misses": {
            "hybrid@5": [q for (q, _, _), r in zip(QUERIES, hybrid_ranks) if not r or r > 5],
        },
    }


def _score(ranks: list[int | None]) -> dict[str, dict[str, str]]:
    """hit@1 / hit@5 per category and overall, as 'hits/total'."""
    table: dict[str, dict[str, str]] = {}
    for category in (*CATEGORIES, "all"):
        rows = [r for r, (_, _, c) in zip(ranks, QUERIES) if category in ("all", c)]
        hit1 = sum(r == 1 for r in rows)
        hit5 = sum(bool(r and r <= 5) for r in rows)
        table[category] = {"hit@1": f"{hit1}/{len(rows)}", "hit@5": f"{hit5}/{len(rows)}"}
    return table


def _markdown(results: list[dict]) -> str:
    header = "| embedder | mode | " + " | ".join(f"{c} @1/@5" for c in (*CATEGORIES, "all")) + " | embed p50 ms | retrieve p50 ms |"
    lines = [header, "|" + "---|" * (len(CATEGORIES) + 5)]
    for res in results:
        for mode in ("vector", "hybrid"):
            cells = [
                f"{res['scores'][mode][c]['hit@1']} / {res['scores'][mode][c]['hit@5']}"
                for c in (*CATEGORIES, "all")
            ]
            label = res["spec"] if res["spec"] == res["fingerprint"] else f"{res['spec']} ({res['fingerprint']})"
            lines.append(
                f"| {label} | {mode} | " + " | ".join(cells)
                + f" | {res['embed_ms']['p50']} | {res['retrieve_ms']['p50'] if mode == 'hybrid' else '-'} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedders", nargs="+", default=["hash", "auto"], help="embedder spec strings")
    parser.add_argument("--allow-network", action="store_true", help="permit ollama:/openai: specs")
    parser.add_argument("--allow-download", action="store_true", help="let 'auto' download models")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--check", action="store_true", help="only validate the dataset")
    args = parser.parse_args(argv)

    problems = check_dataset()
    if args.check or problems:
        print("\n".join(problems) or f"dataset ok: {len(FACTS)} facts, {len(QUERIES)} queries")
        return 1 if problems else 0

    results = []
    for spec in args.embedders:
        if is_network_spec(spec) and not args.allow_network:
            print(f"skipping {spec!r}: network embedder needs --allow-network", file=sys.stderr)
            continue
        try:
            provider = embedder_from_spec(spec, allow_download=args.allow_download)
        except Exception as exc:  # report and keep going: one missing extra is not fatal
            print(f"skipping {spec!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(evaluate(spec, provider))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(_markdown(results))
        for res in results:
            misses = res["misses"]["hybrid@5"]
            print(f"\n{res['fingerprint']}: index {res['index_ms_per_fact']} ms/fact, "
                  f"embed p95 {res['embed_ms']['p95']} ms, retrieve p95 {res['retrieve_ms']['p95']} ms; "
                  f"hybrid misses@5: {len(misses)} (see --json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
