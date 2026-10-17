"""Pick the score-fusion lexical weight on half the queries; report the other half.

Usage::

    python benchmarks/embeddings/tune_fusion.py \\
        --embedders model2vec fastembed:BAAI/bge-small-en-v1.5

For every semantic embedder given, the full pipeline is run with
``FusionConfig(mode="score", lexical_weight=w)`` for each ``w`` in
``LEXICAL_WEIGHTS``. Selection uses only one half of ``dataset.QUERIES``
(``--tune-on``, halves alternate within each category): the highest summed
hit@1 across embedders wins and ties go to the *largest* weight, so lexical
evidence is kept as strong as quality allows rather than tuned away. The
chosen weight is then reported on the other half and on the full set.

``anchor_lexical_weight`` cannot be tuned here: only a handful of these
queries name an anchor, and every model finds those. It was set on the
retention A/B log (benchmarks/retention_ab, all questions name a service
or ticket). ``STRESS`` adds near-duplicate identifiers (JIRA-4521 vs
JIRA-4512 vs JIRA-5421), where an embedder alone is unreliable, and reports
hit@1 for several anchor weights.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from dataset import CATEGORIES, FACTS, QUERIES  # noqa: E402
from eval_embedders import in_subset  # noqa: E402

from mark.embeddings import CachedEmbeddingProvider, EmbeddingProvider  # noqa: E402
from mark.embeddings.auto import embedder_from_spec  # noqa: E402
from mark.intelligence.retrieval_policy import FusionConfig  # noqa: E402
from mark.memory.runtime import MarkRuntime  # noqa: E402

LEXICAL_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5)
ANCHOR_WEIGHTS = (0.3, 0.5, 0.7, 0.9)

# Near-duplicate identifiers: the gold fact differs from its distractors only
# in digit order, which subword embeddings barely see.
STRESS_FACTS = [
    "Ticket JIRA-4521 tracks the memory leak in the image resizer.",
    "Ticket JIRA-4512 tracks the memory spike in the thumbnail service.",
    "Ticket JIRA-5421 tracks the memory leak in the video encoder.",
    "Invoice INV-2024-0087 was paid late by Northwind Traders.",
    "Invoice INV-2024-0078 was paid early by Northwind Traders.",
    "Invoice INV-2023-0087 was disputed by Northwind Traders.",
    "Build agent A7731 has a failing disk.",
    "Build agent A7713 has a failing fan.",
    "Build agent A1773 has a failing power supply.",
    "Port 9443 serves the billing API.",
    "Port 9434 serves the admin API.",
    "Port 4943 serves the metrics endpoint.",
]
STRESS_QUERIES = [
    ("JIRA-4521", 0), ("What is JIRA-4512 about?", 1), ("JIRA-5421 status", 2),
    ("INV-2024-0087", 3), ("Who paid INV-2024-0078?", 4), ("INV-2023-0087", 5),
    ("A7731", 6), ("What is wrong with A7713?", 7), ("A1773", 8),
    ("port 9443", 9), ("What runs on 9434?", 10), ("4943", 11),
]


def run(provider: EmbeddingProvider, fusion: FusionConfig, facts: list[str], queries) -> list[int | None]:
    """Gold rank per query through the full pipeline (fresh store per call)."""
    ranks: list[int | None] = []
    with tempfile.TemporaryDirectory() as tmp:
        runtime = MarkRuntime.local(store_path=Path(tmp) / "tune.db", embedder=provider)
        runtime.pipeline.configure_fusion(fusion)
        try:
            mem = runtime.memory("tune")
            position = {mem.store_sync(fact): i for i, fact in enumerate(facts)}
            for query, gold, *_ in queries:
                order = [position[f.id] for f in mem.retrieve_sync(query).fragments if f.id in position]
                ranks.append(order.index(gold) + 1 if gold in order else None)
        finally:
            runtime.shutdown()
    return ranks


def hits(ranks: list[int | None], subset: str, category: str = "all") -> int:
    """hit@1 count within a query subset (and optionally one category)."""
    return sum(
        r == 1 for i, r in enumerate(ranks)
        if in_subset(i, subset) and category in ("all", QUERIES[i][2])
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedders", nargs="+", required=True, help="semantic embedder specs")
    parser.add_argument("--tune-on", choices=("tune", "holdout"), default="tune", help="half used for selection")
    args = parser.parse_args(argv)
    report_on = "holdout" if args.tune_on == "tune" else "tune"

    providers = {}
    for spec in args.embedders:
        # Cached: the grid re-embeds the same 60 facts many times.
        providers[spec] = CachedEmbeddingProvider(embedder_from_spec(spec), maxsize=4096)
    anchor = FusionConfig().anchor_lexical_weight
    ranks = {
        (spec, w): run(p, FusionConfig("score", w, anchor), FACTS, QUERIES)
        for spec, p in providers.items() for w in LEXICAL_WEIGHTS
    }
    totals = {w: sum(hits(ranks[(s, w)], args.tune_on) for s in providers) for w in LEXICAL_WEIGHTS}
    best = max(totals.values())
    chosen = max(w for w in LEXICAL_WEIGHTS if totals[w] == best)  # ties: keep lexical strong
    print(f"selection on {args.tune_on!r} (anchor weight {anchor}): hit@1 sum per lexical weight")
    print("  " + "  ".join(f"{w}:{totals[w]}" for w in LEXICAL_WEIGHTS))
    print(f"chosen lexical_weight={chosen}\n")

    print(f"| embedder | subset | {' | '.join(CATEGORIES)} | all |")
    print("|---|---|" + "---|" * (len(CATEGORIES) + 1))
    for spec in providers:
        r = ranks[(spec, chosen)]
        for subset in (report_on, "all"):
            cells = [str(hits(r, subset, c)) for c in (*CATEGORIES, "all")]
            print(f"| {spec} | {subset} | " + " | ".join(cells) + " |")

    print("\nnear-duplicate id stress, hit@1 of 12 (lexical_weight fixed at chosen):")
    print("| embedder | rrf | no anchor boost | " + " | ".join(f"anchor {a}" for a in ANCHOR_WEIGHTS) + " |")
    print("|---|---|---|" + "---|" * len(ANCHOR_WEIGHTS))
    for spec, p in providers.items():
        cells = [sum(r == 1 for r in run(p, FusionConfig("rrf"), STRESS_FACTS, STRESS_QUERIES))]
        for weight in (chosen, *ANCHOR_WEIGHTS):
            fusion = FusionConfig("score", chosen, weight)
            cells.append(sum(r == 1 for r in run(p, fusion, STRESS_FACTS, STRESS_QUERIES)))
        print(f"| {spec} | " + " | ".join(map(str, cells)) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
