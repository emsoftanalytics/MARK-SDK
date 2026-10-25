"""``mark`` command line: local memory for agents and scripts.

    mark [--db PATH] [--agent ID] [--no-server] <command> ...
    mark --version                       prints the installed version (mark-sdk X)

Write and revise
    remember TEXT [--kind K] [--tag T ...] [--importance X] [--extract-facts]
                  [--session S] [--occurred-at T] [--json]
                                                 store one memory; prints its short id
                                                 (`store` is an alias)
    ingest FILE [--session S] [--tag T ...] [--kind K] [--extract-facts]
                                                 store each non-empty line as a memory
                                                 (safe to repeat: counts new vs present)
    forget ID [--secure]                         forget a memory and the facts taken from
                                                 it (--secure erases the text from disk)
    forget --match TEXT [--dry-run] [--yes] [--all] [--secure]
                                                 forget every memory containing these whole
                                                 words: lists them, then asks (--yes skips
                                                 the question; without a terminal it refuses);
                                                 over 50 matches also needs --all
    resolve ID [--note TEXT]                     mark a TODO/bug as done
    supersede ID TEXT                            replace a memory, keeping its history
Read
    recall QUERY [-k N] [--chronological] [--json] [--session S] [--since T]
                 [--until T] [--tag T ...] [--kind K [--all]] [--open]
                                                 hybrid recall (facts first, then memories;
                                                 --kind/--open hide resolved/superseded
                                                 memories, --all keeps them)
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
    fact list [SUBJECT]                          every current fact (of SUBJECT)
    fact retract SUBJECT [ATTRIBUTE] [--value V] withdraw facts recorded by mistake
                                                 (kept for audit)
Data and processes
    export FILE [--decrypt]                      back up the store (every agent unless
                                                 --agent is given) as JSON Lines
    import FILE                                  restore a backup (safe to repeat)
    agents                                       agent namespaces in the store
    stats                                        number of memories and current facts,
                                                 and which embedder is active
    serve [start|stop|status] [--detach] [--host H] [--port N]
                                                 keep MARK warm (loopback only); agents
                                                 use --detach (returns once it answers)
    mcp [--db P] [--agent ID] [--briefing] [--preset default|small|tiny]
        [--tools T,...] [--allow-agent ID ...] [--user-subject NAME] [--extract-facts]
                                                 run the MCP server (stdio) on this store
    config show [--json] | config user-subject NAME
                                                 store settings; the user subject (who
                                                 "I"/"my" is) is stored with the store

KINDS for --kind: decision, todo, bug, fix, preference, snippet, api, note. Without
--kind, a leading "TODO:", "DECISION:"/"ADR:", "BUG:", "FIX:", "DONE:"/"NOTE:", "API:"
or "SNIPPET:" label sets it. Times (--occurred-at, --valid-from, --since, --until)
are ISO dates or words such as "yesterday" or "3 May 2026". Fact commands take
--json too. Read commands on a missing store say "no store at PATH" and create
nothing.
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
from mark._cli.request import _read_queries, _request  # noqa: F401

# The server module imports http.server (and with it http.client, ssl and the
# email parser): ~50 ms that a served `mark recall` must not pay. Its names
# stay importable from here, loaded on first use (PEP 562).
_SERVER_NAMES = {"_acquire_lock", "_pid_alive", "_release_lock", "_write_state", "serve"}


def __getattr__(name: str) -> Any:
    if name in _SERVER_NAMES:
        from mark._cli import server

        return getattr(server, name)
    raise AttributeError(f"module 'mark.cli' has no attribute {name!r}")

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
        if cmd == "forget" and payload.get("match") is not None:
            from mark._cli.confirm import forget_matching

            return forget_matching(args, db, payload)
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
    if args.action == "status" or args.detach:
        from mark._cli import daemon

        ok, message = daemon.status(db) if args.action == "status" else daemon.detach(db, args.host, args.port)
        print(message if ok or args.action == "status" else f"mark: {message}",
              file=sys.stdout if ok or args.action == "status" else sys.stderr)
        return 0 if ok else 1
    from mark._cli.server import serve

    serve(db, args.host, args.port)  # RuntimeError (e.g. encrypted store) is reported by main
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
