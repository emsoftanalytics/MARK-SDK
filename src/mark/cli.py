"""``mark`` command line: local memory for agents and scripts.

    mark [--db PATH] [--agent ID] <command> ...

    ingest FILE [--session S] [--tag T ...]      store each non-empty line as a memory
    store TEXT [--session S] [--occurred-at T]   store one memory
    recall QUERY [-k N] [--json]                 hybrid recall (facts first, then memories)
    recall-many [FILE|-] [-k N] [--json]         one query per line (or a JSON list)
    fact set SUBJECT ATTRIBUTE VALUE [--valid-from T]
    fact get SUBJECT [ATTRIBUTE]                 current value(s)
    fact history SUBJECT ATTRIBUTE               every value over time
    stats                                        number of retrievable memories
    serve [--port N]                             keep MARK warm for fast calls

The database comes from ``--db`` or ``$MARK_DB`` (default ``./.mark/memory.db``),
the agent namespace from ``--agent`` or ``$MARK_AGENT`` (default ``__mark__``).

When ``mark serve`` is running for the same database, every other command is
sent to it over loopback HTTP instead of loading MARK in-process: a call then
costs one small Python start-up plus milliseconds of search, instead of
importing and opening MARK each time. The server binds to loopback only and
requires a random token stored next to the database in a 0600 state file.

This module imports only the standard library at start-up; MARK itself is
loaded lazily, and never on the client path when a server answers.

Layout: this module holds the console entry point (argument parsing and
output); configuration, the command engine, the server and the client live
in :mod:`mark._cli` and are re-exported here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mark._cli.client import (  # noqa: F401
    _base_url,
    _call_server,
    _ingest_in_chunks,
    _live_server,
    execute,
)
from mark._cli.config import (  # noqa: F401
    _LOOPBACK,
    DEFAULT_AGENT,
    INGEST_CHUNK_BYTES,
    MAX_REQUEST_BYTES,
    _db_path,
    _lock_path,
    _proof,
    _state_path,
)
from mark._cli.engine import _Engine, _fact_json, _result_json  # noqa: F401
from mark._cli.server import _acquire_lock, _pid_alive, _release_lock, _write_state, serve  # noqa: F401


# ── output ────────────────────────────────────────────────────────────────────

def _print_recall(result: dict[str, Any]) -> None:
    for fact in result["facts"]:
        print(f" F. {fact['text']}{'' if fact['current'] else '  [historical]'}")
    if not result["memories"] and not result["facts"]:
        print("(no memories found)")
    for rank, memory in enumerate(result["memories"], 1):
        print(f"{rank:2d}. [{memory['score']:.3f}] {memory['content']}")
    if result["gap"] not in {"none", "low"}:
        reason = f": {result['gap_reason']}" if result["gap_reason"] else ""
        print(f"-- memory gap: {result['gap']}{reason}")


# ── entry point ───────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mark", description="Local memory for agents and scripts.")
    parser.add_argument("--db", help="SQLite store path (default: $MARK_DB or ./.mark/memory.db)")
    parser.add_argument("--agent", default=os.environ.get("MARK_AGENT", DEFAULT_AGENT), help="agent namespace")
    parser.add_argument("--no-server", action="store_true", help="always run in-process")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="store each non-empty line of FILE")
    p.add_argument("file")
    p.add_argument("--session")
    p.add_argument("--tag", action="append", dest="tags")
    p = sub.add_parser("store", help="store one memory")
    p.add_argument("text")
    p.add_argument("--session")
    p.add_argument("--occurred-at")
    p = sub.add_parser("recall", help="search memory")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("recall-many", help="search memory for many queries at once")
    p.add_argument("file", nargs="?", default="-")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    fact = sub.add_parser("fact", help="bi-temporal facts").add_subparsers(dest="fact_cmd", required=True)
    p = fact.add_parser("set")
    p.add_argument("subject")
    p.add_argument("attribute")
    p.add_argument("value")
    p.add_argument("--valid-from")
    p = fact.add_parser("get")
    p.add_argument("subject")
    p.add_argument("attribute", nargs="?")
    p = fact.add_parser("history")
    p.add_argument("subject")
    p.add_argument("attribute")
    sub.add_parser("stats", help="count memories and facts")
    p = sub.add_parser("serve", help="keep MARK warm for fast calls")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    return parser


def _read_queries(source: str) -> list[str]:
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("["):
        queries = []
        for item in json.loads(stripped):
            if isinstance(item, dict):
                if "question" not in item:
                    raise ValueError("each JSON query object needs a 'question' field")
                item = item["question"]
            queries.append(str(item))
        return queries
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``mark``); returns the process exit code."""
    args = _parser().parse_args(argv)
    db = _db_path(args.db)
    if args.cmd == "serve":
        serve(db, args.host, args.port)
        return 0

    agent = args.agent
    try:
        cmd, payload = _request(args, agent)
    except (OSError, ValueError) as exc:
        print(f"mark: {exc}", file=sys.stderr)
        return 1
    try:
        result = execute(db, cmd, payload, use_server=not args.no_server)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"mark: {exc}", file=sys.stderr)
        return 1
    _print_result(args, cmd, result)
    return 0


def _request(args: argparse.Namespace, agent: str) -> tuple[str, dict[str, Any]]:
    """Translate parsed arguments into an engine command and payload."""
    if args.cmd == "ingest":
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        cmd, payload = "ingest", {"agent": agent, "lines": lines, "session": args.session, "tags": args.tags}
    elif args.cmd == "store":
        cmd, payload = "store", {"agent": agent, "text": args.text, "session": args.session,
                                 "occurred_at": args.occurred_at}
    elif args.cmd == "recall":
        cmd, payload = "recall", {"agent": agent, "query": args.query, "k": args.k}
    elif args.cmd == "recall-many":
        cmd, payload = "recall-many", {"agent": agent, "queries": _read_queries(args.file), "k": args.k}
    elif args.cmd == "fact":
        cmd = f"fact-{args.fact_cmd}"
        payload = {"agent": agent, "subject": args.subject}
        if args.fact_cmd == "set":
            payload.update(attribute=args.attribute, value=args.value, valid_from=args.valid_from)
        else:
            payload["attribute"] = args.attribute
    else:
        cmd, payload = "stats", {"agent": agent}
    return cmd, payload


def _print_result(args: argparse.Namespace, cmd: str, result: dict[str, Any]) -> None:
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    elif cmd == "recall":
        _print_recall(result)
    elif cmd == "recall-many":
        for result_item in result["results"]:
            print(f"### {result_item['query']}")
            _print_recall(result_item)
    elif cmd.startswith("fact-"):
        facts = [result["fact"]] if "fact" in result else result["facts"]
        for fact in facts:
            print(fact["text"])
        if not facts:
            print("(no facts)")
    elif cmd == "ingest":
        print(f"stored {result['stored']} memories")
    elif cmd == "store":
        print(f"stored {result['id']}")
    else:
        print(f"{result['memories']} memories, {result['facts']} current facts")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
