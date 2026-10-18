"""Argument parser of the ``mark`` command line (documented in :mod:`mark.cli`).

Standard library only. ``--kind`` choices come from :mod:`mark.kinds`, which
is standard-library only too, so building the parser never loads MARK.
"""
from __future__ import annotations

import argparse
from typing import Any

from mark._cli.config import DEFAULT_AGENT
from mark.kinds import KINDS


def _recall_options(p: argparse.ArgumentParser, *, filters: bool) -> None:
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--chronological", action="store_true",
                   help="list the results oldest first instead of by relevance")
    p.add_argument("--json", action="store_true")
    if filters:
        p.add_argument("--session", help="only memories of this session")
        p.add_argument("--since", help="only events on/after this date (ISO or words: 'last week')")
        p.add_argument("--until", help="only events on/before this date")
        p.add_argument("--tag", action="append", dest="tags", help="only memories with this tag (repeatable)")
        p.add_argument("--kind", choices=KINDS, help="only open memories of this kind (see --all)")
        p.add_argument("--open", action="store_true", help="hide resolved and superseded memories")
        p.add_argument("--all", action="store_true", help="with --kind: include resolved/superseded ones")


def build_parser() -> argparse.ArgumentParser:
    """The full ``mark`` parser (sub-commands and options)."""
    parser = argparse.ArgumentParser(prog="mark", description="Local memory for agents and scripts.")
    parser.add_argument("--db", help="SQLite store path (default: $MARK_DB or ./.mark/memory.db)")
    parser.add_argument("--agent", help=f"agent namespace (default: $MARK_AGENT or {DEFAULT_AGENT})")
    parser.add_argument("--no-server", action="store_true", help="always run in-process")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="store each non-empty line of FILE")
    p.add_argument("file")
    p.add_argument("--session")
    p.add_argument("--tag", action="append", dest="tags")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--extract-facts", action="store_true",
                   help="record facts stated in the lines (built-in patterns; dated by '[YYYY-MM-DD]' stamps)")

    p = sub.add_parser("remember", aliases=["store"], help="store one memory")
    p.add_argument("text")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--tag", action="append", dest="tags")
    p.add_argument("--importance", type=float, help="0..1 (default 0.5)")
    p.add_argument("--extract-facts", action="store_true", help="record facts stated in TEXT")
    p.add_argument("--session")
    p.add_argument("--occurred-at", help="when it happened: ISO date/time or words ('yesterday', '3 May')")
    p.add_argument("--json", action="store_true", help="print the full id as JSON")

    p = sub.add_parser("forget", help="forget a memory (and the facts extracted from it)")
    p.add_argument("id", nargs="?", help="memory id (the 8-character short id is enough)")
    p.add_argument("--match", help="forget every memory containing these whole words instead "
                                   "(lists them and asks first)")
    p.add_argument("--secure", action="store_true", help="also erase the text from the database file")
    p.add_argument("--yes", "-y", action="store_true", help="with --match: do not ask for confirmation")
    p.add_argument("--dry-run", action="store_true", help="with --match: only list what would be forgotten")
    p.add_argument("--all", action="store_true", help="with --match: allow more than 50 matches")
    p = sub.add_parser("resolve", help="mark a TODO/bug memory as done")
    p.add_argument("id")
    p.add_argument("--note")
    p = sub.add_parser("supersede", help="replace a memory with new text (history kept)")
    p.add_argument("id")
    p.add_argument("text")
    p = sub.add_parser("list", help="list memories, most important first")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--tag", action="append", dest="tags")
    p.add_argument("--open", action="store_true", help="hide resolved and superseded memories")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("recall", help="search memory")
    p.add_argument("query")
    _recall_options(p, filters=True)
    p = sub.add_parser("recall-many", help="search memory for many queries at once")
    p.add_argument("file", nargs="?", default="-")
    _recall_options(p, filters=False)
    p = sub.add_parser("brief", help="session-start brief: facts, open TODOs, decisions, bugs")
    p.add_argument("--budget", type=int, default=400, help="approximate token budget")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("timeline", help="every fact and memory about a subject, oldest first")
    p.add_argument("subject")
    p.add_argument("attribute", nargs="?")
    p.add_argument("--limit", type=int, help="keep only the newest N events")
    p.add_argument("--since", help="earliest event time (ISO date or words: 'last week')")
    p.add_argument("--until", help="latest event time (ISO date or words)")
    p.add_argument("--json", action="store_true")
    _fact_parser(sub)

    p = sub.add_parser("export", help="back up memories, facts and documents to a JSONL file")
    p.add_argument("file")
    p.add_argument("--decrypt", action="store_true", help="allow a plaintext export of an encrypted store")
    p = sub.add_parser("import", help="restore a JSONL backup (idempotent)")
    p.add_argument("file")
    sub.add_parser("agents", help="list agent namespaces in the store")
    sub.add_parser("stats", help="count memories and facts; show the embedder")
    p = sub.add_parser("serve", help="keep MARK warm for fast calls (`mark serve stop` stops it)")
    p.add_argument("action", nargs="?", choices=("start", "stop"), default="start")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    _config_parser(sub)
    _mcp_parser(sub)
    _accept_after_command(sub)
    return parser


def _accept_after_command(sub: Any) -> None:
    """Accept ``--no-server`` after the sub-command too (``mark recall Q --no-server``).

    ``default=SUPPRESS`` keeps a sub-parser from resetting the global flag
    when the option is given before the command.
    """
    for command in set(sub.choices.values()):
        nested = [a for a in command._actions if isinstance(a, argparse._SubParsersAction)]
        if nested:
            _accept_after_command(nested[0])
        elif command.prog != "mark mcp":
            command.add_argument("--no-server", action="store_true", default=argparse.SUPPRESS,
                                 help="always run in-process")


def _fact_parser(sub: Any) -> None:
    fact = sub.add_parser("fact", help="bi-temporal facts: values that change over time, with history",
                          description="Record and read facts (subject, attribute, value) over time.")
    # metavar + help per verb: `mark fact --help` lists every verb, not just one.
    commands = fact.add_subparsers(dest="fact_cmd", required=True, metavar="{set,get,history,list,retract}")
    p = commands.add_parser("set", help="record a value (a future --valid-from schedules it)")
    p.add_argument("subject")
    p.add_argument("attribute")
    p.add_argument("value")
    p.add_argument("--valid-from", help="when it became true: ISO date or words ('yesterday', '1 May')")
    p.add_argument("--correction", action="store_true",
                   help="replace the value(s) recorded for this time instead of adding a step")
    p = commands.add_parser("get", help="current value(s) of a subject (and attribute)")
    p.add_argument("subject")
    p.add_argument("attribute", nargs="?")
    p = commands.add_parser("history", help="every value an attribute has had, oldest first")
    p.add_argument("subject")
    p.add_argument("attribute")
    p = commands.add_parser("list", help="every current fact (of one SUBJECT)")
    p.add_argument("subject", nargs="?")
    p = commands.add_parser("retract", help="retract facts recorded by mistake (kept for audit)")
    p.add_argument("subject")
    p.add_argument("attribute", nargs="?")
    p.add_argument("--value", help="only retract this value")
    for parser in (commands.choices.values()):
        parser.add_argument("--json", action="store_true")


def _config_parser(sub: Any) -> None:
    config = sub.add_parser("config", help="show or change store settings (user subject)")
    commands = config.add_subparsers(dest="config_cmd", required=True, metavar="{show,user-subject}")
    p = commands.add_parser("show", help="store path, user subject, schema version and embedder")
    p.add_argument("--json", action="store_true")
    p = commands.add_parser("user-subject", help="who 'I'/'my' is in this store's memories "
                                                 "(stored with the store; default 'user')")
    p.add_argument("name")


def _mcp_parser(sub: Any) -> None:
    """``mark mcp``: the options of ``python -m mark.adapters.mcp`` (same launcher)."""
    p = sub.add_parser("mcp", help="run the MARK MCP server on stdio for this database")
    # Own dests: argparse lets sub-command defaults overwrite the global --db/--agent.
    p.add_argument("--db", dest="mcp_db", help="store (same rule as the global --db)")
    p.add_argument("--agent", dest="mcp_agent", help="default memory namespace")
    p.add_argument("--briefing", action="store_true", help="also expose the mark_briefing tool")
    p.add_argument("--preset", default="default", choices=("default", "small", "tiny"),
                   help="briefing size for the client's model")
    p.add_argument("--tools", help="comma-separated subset of the MCP tools to expose")
    p.add_argument("--allow-agent", action="append", dest="allowed", metavar="ID",
                   help="restrict clients to these namespaces (repeatable)")
    p.add_argument("--user-subject", help="who 'I'/'my' is (stored with the store)")
    p.add_argument("--extract-facts", action="store_true",
                   help="record facts stated in every MCP write (built-in patterns, stored user subject)")
