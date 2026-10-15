"""Minimal MARK command-line memory for the retention A/B experiment.

    mark ingest FILE [--session S]     store every non-empty line as one memory
    mark store "TEXT" [--session S]    store one memory (e.g. your own summary)
    mark recall "QUERY" [-k N]         hybrid recall, ranked by relevance (N <= 15)
    mark stats                         number of stored memories

Backed by MARK v0.2.0a7 (``MarkRuntime.local``) with the default zero-dependency
embedder, a SQLite store at $MARK_DB, and numpy-vectorised search.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mark.intelligence.retrieval import RetrievalPolicy
from mark.memory.runtime import MarkRuntime

AGENT = "relay"


def main() -> None:
    parser = argparse.ArgumentParser(prog="mark", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--session", default=None)
    p_store = sub.add_parser("store")
    p_store.add_argument("text")
    p_store.add_argument("--session", default=None)
    p_recall = sub.add_parser("recall")
    p_recall.add_argument("query")
    p_recall.add_argument("-k", type=int, default=10)
    sub.add_parser("stats")
    args = parser.parse_args()

    db = os.environ.get("MARK_DB")
    if not db:
        sys.exit("MARK_DB is not set")
    runtime = MarkRuntime.local(store_path=db)
    memory = runtime.memory(AGENT)
    try:
        if args.cmd == "ingest":
            lines = [ln.strip() for ln in Path(args.file).read_text().splitlines() if ln.strip()]
            for line in lines:
                memory.store_sync(line, session_id=args.session)
            print(f"stored {len(lines)} memories")
        elif args.cmd == "store":
            fid = memory.store_sync(args.text, session_id=args.session, importance=0.8)
            print(f"stored {fid}")
        elif args.cmd == "recall":
            k = max(1, min(args.k, 15))
            policy = RetrievalPolicy.DEEP if k > 10 else RetrievalPolicy.BALANCED
            result = memory.retrieve_sync(args.query, policy=policy)
            rows = list(zip(result.fragments, result.scores))[:k]
            if not rows:
                print("(no memories found)")
            for rank, (fragment, score) in enumerate(rows, 1):
                print(f"{rank:2d}. [{score:.3f}] {fragment.content}")
            if result.gap_report.severity.value not in {"none", "low"}:
                print(f"-- memory gap: {result.gap_report.severity.value} (weak evidence for this query)")
        elif args.cmd == "stats":
            print(f"{len(runtime.store.retrievable_ids(AGENT))} memories")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
