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
  fusion, scoring) with the default ``FusionConfig`` (RRF for the lexical
  hash embedder, score fusion for semantic embedders);
- ``hybrid-rrf``: the same pipeline forced to equal-weight RRF, the fusion
  every embedder used before 0.2.0a9 (the "before" column).

``--subset tune|holdout`` restricts the table to one half of the queries
(alternating within each category; see ``tune_fusion.py``, which picks the
fusion weights on the tune half).

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
from mark.intelligence.retrieval_policy import FusionConfig  # noqa: E402
from mark.memory.runtime import MarkRuntime  # noqa: E402

# Hybrid variants reported per embedder: (mode label, fusion config).
FUSIONS = (("hybrid-rrf", FusionConfig(mode="rrf")), ("hybrid", FusionConfig()))
SUBSETS = ("all", "tune", "holdout")


def in_subset(index: int, subset: str) -> bool:
    """Whether query ``index`` belongs to the subset (halves alternate per category)."""
    if subset == "all":
        return True
    category = QUERIES[index][2]
    position = sum(1 for _, _, c in QUERIES[:index] if c == category)
    return (position % 2 == 0) == (subset == "tune")


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


def hybrid_ranks(provider: EmbeddingProvider, fusion: FusionConfig) -> tuple[list[int | None], list[float]]:
    """Gold rank and latency of every query through the full pipeline under ``fusion``.

    Each fusion gets a fresh store, so retrieval feedback from one variant
    cannot change the priors seen by the next.
    """
    ranks, times = [], []
    with tempfile.TemporaryDirectory() as tmp:
        runtime = MarkRuntime.local(store_path=Path(tmp) / "eval.db", embedder=provider)
        runtime.pipeline.configure_fusion(fusion)
        try:
            mem = runtime.memory("eval")
            position = {mem.store_sync(fact): i for i, fact in enumerate(FACTS)}
            mem.retrieve_sync("warm up")  # index sync outside the timed loop
            for query, gold, _ in QUERIES:
                t0 = time.perf_counter()
                result = mem.retrieve_sync(query)
                times.append(time.perf_counter() - t0)
                order = [position[f.id] for f in result.fragments if f.id in position]
                ranks.append(_rank(order, gold))
        finally:
            runtime.shutdown()
    return ranks, times


def evaluate(spec: str, provider: EmbeddingProvider) -> dict:
    """Vector-only and hybrid ranks for every query, plus latencies."""
    start = time.perf_counter()
    fact_vectors = provider.embed_batch(FACTS)
    index_seconds = time.perf_counter() - start

    ranks: dict[str, list[int | None]] = {}
    embed_times: list[float] = []
    ranks["vector"] = []
    for query, gold, _ in QUERIES:
        t0 = time.perf_counter()
        q = provider.embed(query)
        embed_times.append(time.perf_counter() - t0)
        scores = [sum(a * b for a, b in zip(q, v)) for v in fact_vectors]
        order = sorted(range(len(FACTS)), key=lambda i: -scores[i])
        ranks["vector"].append(_rank(order, gold))

    retrieve_ms: dict[str, dict[str, float]] = {}
    for label, fusion in FUSIONS:
        ranks[label], times = hybrid_ranks(provider, fusion)
        retrieve_ms[label] = _ms(times)

    return {
        "spec": spec,
        "fingerprint": provider.fingerprint,
        "index_ms_per_fact": round(index_seconds * 1000 / len(FACTS), 2),
        "embed_ms": _ms(embed_times),
        "retrieve_ms": retrieve_ms,
        "ranks": ranks,
        "misses": {
            "hybrid@5": [q for (q, _, _), r in zip(QUERIES, ranks["hybrid"]) if not r or r > 5],
        },
    }


def _score(ranks: list[int | None], subset: str = "all") -> dict[str, dict[str, str]]:
    """hit@1 / hit@5 per category and overall, as 'hits/total'."""
    table: dict[str, dict[str, str]] = {}
    for category in (*CATEGORIES, "all"):
        rows = [
            r for i, (r, (_, _, c)) in enumerate(zip(ranks, QUERIES))
            if category in ("all", c) and in_subset(i, subset)
        ]
        hit1 = sum(r == 1 for r in rows)
        hit5 = sum(bool(r and r <= 5) for r in rows)
        table[category] = {"hit@1": f"{hit1}/{len(rows)}", "hit@5": f"{hit5}/{len(rows)}"}
    return table


def _markdown(results: list[dict], subset: str = "all") -> str:
    header = "| embedder | mode | " + " | ".join(f"{c} @1/@5" for c in (*CATEGORIES, "all")) + " | embed p50 ms | retrieve p50 ms |"
    lines = [header, "|" + "---|" * (len(CATEGORIES) + 5)]
    for res in results:
        for mode, ranks in res["ranks"].items():
            table = _score(ranks, subset)
            cells = [f"{table[c]['hit@1']} / {table[c]['hit@5']}" for c in (*CATEGORIES, "all")]
            label = res["spec"] if res["spec"] == res["fingerprint"] else f"{res['spec']} ({res['fingerprint']})"
            retrieve = res["retrieve_ms"].get(mode, {}).get("p50", "-")
            lines.append(
                f"| {label} | {mode} | " + " | ".join(cells) + f" | {res['embed_ms']['p50']} | {retrieve} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedders", nargs="+", default=["hash", "auto"], help="embedder spec strings")
    parser.add_argument("--allow-network", action="store_true", help="permit ollama:/openai: specs")
    parser.add_argument("--allow-download", action="store_true", help="let 'auto' download models")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--check", action="store_true", help="only validate the dataset")
    parser.add_argument("--subset", choices=SUBSETS, default="all", help="report one half of the queries")
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
        for res in results:
            res["scores"] = {mode: _score(ranks, args.subset) for mode, ranks in res["ranks"].items()}
        print(json.dumps(results, indent=2))
    else:
        print(_markdown(results, args.subset))
        for res in results:
            misses = res["misses"]["hybrid@5"]
            print(f"\n{res['fingerprint']}: index {res['index_ms_per_fact']} ms/fact, "
                  f"embed p95 {res['embed_ms']['p95']} ms, retrieve p95 {res['retrieve_ms']['hybrid']['p95']} ms; "
                  f"hybrid misses@5: {len(misses)} (see --json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
