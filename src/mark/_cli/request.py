"""Translate parsed ``mark`` arguments into an engine command and its payload.

Standard library only (the client path must not load MARK). Split from
:mod:`mark.cli` to keep the entry point small.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
        payload = {"agent": agent, "id": args.id, "match": args.match, "secure": args.secure}
        if args.match is not None and args.all:
            payload["all"] = True
        return cmd, payload
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
