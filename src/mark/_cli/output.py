"""Human-readable output of the ``mark`` command line (``--json`` prints raw results).

Recall and list lines show what an agent needs to act on a memory and no
more: its short id (for ``forget``/``resolve``/``supersede``), its date (the
declared event date, else the day it was stored) and the text; retired
memories are flagged ``(superseded)``/``(resolved)``. Results sharing a date
keep their order without a rank marker (``order`` stays in ``--json``).
Standard library only.
"""
from __future__ import annotations

import json
from typing import Any


def _memory_line(memory: dict[str, Any], prefix: str) -> str:
    date = memory.get("date") or ""
    content = memory["content"]
    # The text already starts with "[<date>]": don't print the date twice.
    if date and content.startswith(f"[{date}"):
        date = ""
    flag = f" ({memory['lifecycle']})" if memory.get("lifecycle") else ""
    kind = f"[{memory['kind']}] " if memory.get("kind") else ""
    head = " ".join(part for part in (prefix, memory.get("short_id") or memory["id"][:8], date) if part)
    return f"{head} {kind}{content}{flag}"


def print_recall(result: dict[str, Any]) -> None:
    """Facts first (current unless marked), then ranked memories, then any gap."""
    for fact in result["facts"]:
        print(f" F. {fact['text']}{'' if fact['current'] else '  [historical]'}")
    if not result["memories"] and not result["facts"]:
        print("(no memories found)")
    for rank, memory in enumerate(result["memories"], 1):
        print(_memory_line(memory, f"{rank:2d}. [{memory['score']:.3f}]"))
    if result["gap"] not in {"none", "low"}:
        reason = f": {result['gap_reason']}" if result["gap_reason"] else ""
        print(f"-- memory gap: {result['gap']}{reason}")


def print_list(result: dict[str, Any]) -> None:
    for memory in result["memories"]:
        print(_memory_line(memory, ""))
    shown, total, offset = len(result["memories"]), result["total"], result.get("offset", 0)
    if not shown:
        print("(no memories)")
    elif offset + shown < total or offset:
        print(f"-- showing {offset + 1}-{offset + shown} of {total} (use --offset/--limit for more)")


def print_result(args: Any, cmd: str, result: dict[str, Any]) -> None:
    """Print *result* of *cmd* the way a person (or an agent reading text) wants it."""
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif cmd == "recall":
        print_recall(result)
    elif cmd == "recall-many":
        for result_item in result["results"]:
            print(f"### {result_item['query']}")
            print_recall(result_item)
    elif cmd in {"timeline", "brief"}:
        print(result["text"])
    elif cmd == "list":
        print_list(result)
    elif cmd == "fact-set" and result["fact"].get("scheduled"):
        fact = result["fact"]
        print(f"{fact['text']}  (not current until then)")
    elif cmd == "fact-retract":
        print(f"retracted {result['retracted']} fact(s)")
    elif cmd.startswith("fact-"):
        facts = [result["fact"]] if "fact" in result else result["facts"]
        for fact in facts:
            print(fact["text"])
        if not facts:
            print("(no facts)")
    elif cmd == "ingest":
        print(_ingest_line(result))
    elif cmd in {"store", "remember"}:
        extracted = f" ({result['extracted']} facts extracted)" if "extracted" in result else ""
        print(f"stored {result['id'][:8]}{extracted}")  # the short id every command accepts
    elif cmd == "forget":
        secure = " and erased from disk" if result["secure"] and result["forgotten"] else ""
        print(f"forgot {result['forgotten']} memor{'y' if result['forgotten'] == 1 else 'ies'}{secure}")
    elif cmd == "resolve":
        print(_resolve_line(result))
    elif cmd == "supersede":
        line = f"superseded {result['superseded'][:8]} with {result['id'][:8]}"
        if result.get("followed_from"):
            line += (f" ({result['followed_from'][:8]} was already superseded; "
                     f"its current version {result['superseded'][:8]} was replaced)")
        print(line)
    elif cmd in {"export", "import"}:
        skipped = f", {result['skipped']} skipped" if result.get("skipped") else ""
        invalid = f", {result['invalid']} invalid line(s)" if result.get("invalid") else ""
        verb = "exported" if cmd == "export" else "imported"
        print(f"{verb} {result['memories']} memories, {result['facts']} facts, "
              f"{result['documents']} documents{skipped}{invalid} ({result['path']})")
        for problem in result.get("problems", []):
            print(f"  {problem}")
    elif cmd == "config-show":
        print(f"store: {result['db']}\nuser subject: {result['user_subject']}\n"
              f"schema version: {result['schema_version']}\nembedder: {_embedder(result['embedder'])}")
    elif cmd == "config-user-subject":
        print(f"user subject: {result['user_subject']} (stored with the store)")
    elif cmd == "agents":
        for item in result["agents"]:
            print(f"{item['agent']}  ({item['memories']} memories)")
        if not result["agents"]:
            print("(no agents)")
    else:
        embedder = result.get("embedder")
        line = f"{result['memories']} memories, {result['facts']} current facts"
        if embedder:
            line += f"; embedder {_embedder(embedder)}"
        print(line)


def _resolve_line(result: dict[str, Any]) -> str:
    if result.get("already"):
        when = str(result.get("resolved_at") or "")[:10]
        return f"already resolved: {result['id'][:8]}{f' (on {when})' if when else ''}; nothing changed"
    line = f"resolved {result['id'][:8]}"
    if result.get("followed_from"):
        line += f" (the current version of {result['followed_from'][:8]}, which was superseded)"
    return line


def _ingest_line(result: dict[str, Any]) -> str:
    stored, existing = result["stored"], result.get("existing", 0)
    line = f"stored {stored} new memor{'y' if stored == 1 else 'ies'}"
    details = [f"{existing} already present"] if existing else []
    if result.get("skipped"):
        details.append(f"{result['skipped']} skipped")
    if "extracted" in result:
        details.append(f"{result['extracted']} facts extracted")
    return line + (f" ({', '.join(details)})" if details else "")


def _embedder(info: dict[str, Any]) -> str:
    kind = "semantic" if info["semantic"] else "lexical"
    return f"{info['name']} ({kind}, {info['source']})"
