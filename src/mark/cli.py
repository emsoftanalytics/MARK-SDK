"""``mark`` command line: local memory for agents and scripts.

    mark [--db PATH] [--agent ID] [--no-server] <command> ...

Write and revise
    remember TEXT [--kind K] [--tag T ...] [--importance X] [--extract-facts]
                  [--session S] [--occurred-at T]
                                                 store one memory (`store` is an alias)
    ingest FILE [--session S] [--tag T ...] [--kind K] [--extract-facts]
                                                 store each non-empty line as a memory
    forget ID [--secure] | forget --match TEXT   forget memories and the facts taken from
                                                 them (--secure erases the text from disk)
    resolve ID [--note TEXT]                     mark a TODO/bug as done
    supersede ID TEXT                            replace a memory, keeping its history
Read
    recall QUERY [-k N] [--chronological] [--json] [--session S] [--since T]
                 [--until T] [--tag T ...] [--kind K]
                                                 hybrid recall (facts first, then memories)
    recall-many [FILE|-] [-k N] [--chronological] [--json]
                                                 one query per line (or a JSON list)
    list [--kind K] [--tag T ...] [--open] [--limit N] [--offset N] [--json]
                                                 memories, most important first (--open
                                                 hides resolved and superseded ones)
    brief [--budget N] [--json]                  session-start brief: facts, open TODOs,
                                                 recent decisions and bugs
    timeline SUBJECT [ATTRIBUTE] [--limit N] [--since T] [--until T] [--json]
                                                 everything about SUBJECT, oldest first
Facts
    fact set SUBJECT ATTRIBUTE VALUE [--valid-from T] [--correction]
                                                 record a value (--correction replaces
                                                 the value(s) recorded for that time)
    fact get SUBJECT [ATTRIBUTE]                 current value(s)
    fact history SUBJECT ATTRIBUTE               every value over time
    fact retract SUBJECT [ATTRIBUTE] [--value V] withdraw facts recorded by mistake
                                                 (kept for audit)
Data and processes
    export FILE [--decrypt]                      back up the store (every agent unless
                                                 --agent is given) as JSON Lines
    import FILE                                  restore a backup (safe to repeat)
    agents                                       agent namespaces in the store
    stats                                        number of memories and current facts,
                                                 and which embedder is active
    serve [start|stop] [--host H] [--port N]     keep MARK warm (loopback only)
    mcp [--db P] [--agent ID] [--briefing] [--preset default|small|tiny]
        [--tools T,...] [--allow-agent ID ...] [--user-subject NAME]
                                                 run the MCP server (stdio) on this store
    config show [--json] | config user-subject NAME
                                                 store settings; the user subject (who
                                                 "I"/"my" is) is stored with the store

KINDS for --kind: decision, todo, bug, fix, preference, snippet, api, note.
IDs: recall and list print 8-character short ids; any unambiguous prefix works.

The database comes from ``--db`` or ``$MARK_DB`` (default ``./.mark/memory.db``):
a ``.db`` file, a ``.mark`` folder (its ``memory.db``) or a project folder (its
``.mark/memory.db``) -- the same rule as ``Mark.local(path)`` and ``mark mcp``.
The agent namespace comes from ``--agent`` or ``$MARK_AGENT`` (default ``__mark__``).

When ``mark serve`` is running for the same database, every other command is
sent to it over loopback HTTP instead of loading MARK in-process (``--no-server``
forces in-process): a call then costs one small Python start-up plus
milliseconds of search. The server binds to loopback only and requires a
random token stored next to the database in a 0600 state file.

This module imports only the standard library at start-up; MARK itself is
loaded lazily, and never on the client path when a server answers.

Layout: this module holds the console entry point; the parser, output,
configuration, command engine, server and client live in :mod:`mark._cli`
and are re-exported here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Everything below is re-exported: ``mark.cli.<name>`` is the historical path.
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
from mark._cli.output import print_recall as _print_recall  # noqa: F401
from mark._cli.output import print_result as _print_result
from mark._cli.parser import build_parser as _parser
from mark._cli.server import (  # noqa: F401
    _acquire_lock,
    _pid_alive,
    _release_lock,
    _write_state,
    serve,
)

# Errors a command may raise that are the user's to fix: print, exit 1.
_USER_ERRORS = (RuntimeError, ValueError, OSError, KeyError, TypeError)
# Commands that only read: on a missing store they say so instead of creating it.
_READ_ONLY = {"recall", "recall-many", "list", "brief", "stats", "timeline", "agents", "export",
              "config-show", "fact-get", "fact-history", "fact-list"}


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``mark``); returns the process exit code."""
    args = _parser().parse_args(argv)
    db = _db_path(args.db)
    agent = args.agent or os.environ.get("MARK_AGENT") or DEFAULT_AGENT
    try:
        if args.cmd == "serve":
            return _serve(args, db)
        if args.cmd == "mcp":
            from mark._cli.extras import run_mcp

            run_mcp(_db_path(args.mcp_db) if args.mcp_db else db, args.mcp_agent or agent, args)
            return 0
        cmd, payload = _request(args, agent)
        if cmd in _READ_ONLY and not db.exists():
            raise FileNotFoundError(f"no store at {db} (`mark remember TEXT` creates one; "
                                    "--db or $MARK_DB picks another)")
        result = execute(db, cmd, payload, use_server=not args.no_server)
    except _USER_ERRORS as exc:  # MARK errors subclass these builtins too
        print(f"mark: {_message(exc)}", file=sys.stderr)
        return 1
    _print_result(args, cmd, result)
    return 0


def _message(exc: BaseException) -> str:
    # KeyError's str() is the repr of its key, FileNotFoundError's carries an
    # errno prefix; MARK errors define their own text.
    if type(exc) in (KeyError, FileNotFoundError) and len(exc.args) == 1:
        return str(exc.args[0])
    return str(exc)


def _serve(args: argparse.Namespace, db: Path) -> int:
    if args.action == "stop":
        from mark._cli.extras import stop_server

        message = stop_server(db)
        if message.startswith("no mark serve running"):
            print(f"mark: {message}", file=sys.stderr)  # nothing was stopped: exit 1
            return 1
        print(message)
        return 0
    serve(db, args.host, args.port)  # RuntimeError (e.g. encrypted store) is reported by main
    return 0


def _request(args: argparse.Namespace, agent: str) -> tuple[str, dict[str, Any]]:
    """Translate parsed arguments into an engine command and payload."""
    cmd = args.cmd
    if cmd == "ingest":
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        return cmd, {"agent": agent, "lines": lines, "session": args.session, "tags": args.tags,
                     "extract_facts": args.extract_facts, "kind": args.kind}
    if cmd in {"remember", "store"}:
        return "store", {"agent": agent, "text": args.text, "session": args.session,
                         "occurred_at": args.occurred_at, "kind": args.kind, "tags": args.tags,
                         "importance": args.importance, "extract_facts": args.extract_facts}
    if cmd in {"recall", "recall-many"}:
        order = "chronological" if args.chronological else "relevance"
        payload: dict[str, Any] = {"agent": agent, "k": args.k, "order": order}
        if cmd == "recall-many":
            return cmd, {**payload, "queries": _read_queries(args.file)}
        payload["query"] = args.query
        # A kind filter means "open ones" (a resolved TODO is not a TODO) unless --all.
        open_only = args.open or (bool(args.kind) and not args.all)
        filters = {"session": args.session, "since": args.since, "until": args.until,
                   "tags": args.tags, "kind": args.kind, "open": open_only}
        return cmd, {**payload, **{k: v for k, v in filters.items() if v}}
    if cmd == "forget":
        if (args.id is None) == (args.match is None):
            raise ValueError("forget needs a memory ID or --match TEXT (not both)")
        return cmd, {"agent": agent, "id": args.id, "match": args.match, "secure": args.secure}
    if cmd == "resolve":
        return cmd, {"agent": agent, "id": args.id, "note": args.note}
    if cmd == "supersede":
        return cmd, {"agent": agent, "id": args.id, "text": args.text}
    if cmd == "list":
        return cmd, {"agent": agent, "kind": args.kind, "open": args.open, "limit": args.limit,
                     "offset": args.offset, "tags": args.tags}
    if cmd == "brief":
        return cmd, {"agent": agent, "budget": args.budget}
    if cmd in {"export", "import"}:
        # Absolute path: a running server resolves relative paths from its own cwd.
        payload = {"path": str(Path(args.file).expanduser().resolve()), "agent": args.agent}
        if cmd == "export":
            payload["decrypt"] = args.decrypt
        return cmd, payload
    if cmd == "agents":
        return cmd, {}
    if cmd == "config":
        if args.config_cmd == "show":
            return "config-show", {}
        return "config-user-subject", {"name": args.name}
    if cmd == "timeline":
        return cmd, {"agent": agent, "subject": args.subject, "attribute": args.attribute,
                     "limit": args.limit, "since": args.since, "until": args.until}
    if cmd == "fact":
        payload = {"agent": agent, "subject": args.subject}
        if args.fact_cmd == "set":
            payload.update(attribute=args.attribute, value=args.value, valid_from=args.valid_from,
                           correction=args.correction)
        elif args.fact_cmd != "list":
            payload["attribute"] = args.attribute
            if args.fact_cmd == "retract":
                payload["value"] = args.value
        return f"fact-{args.fact_cmd}", payload
    return "stats", {"agent": agent}


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
