"""Human-readable output of the ``mark`` command line (``--json`` prints raw results).

Recall and list lines show what an agent needs to act on a memory and no
more: its short id (for ``forget``/``resolve``/``supersede``), its date (the
declared event date, else the day it was stored) and the text. A ``#k``
marker appears only when two shown results share a date, giving their
order on that day; retired memories are flagged ``(superseded)``/``(resolved)``.
Standard library only.
"""
from __future__ import annotations

import json
from typing import Any


def _memory_line(memory: dict[str, Any], prefix: str) -> str:
    date = memory.get("date") or ""
    if memory.get("order"):
        date += f"#{memory['order']}"
    content = memory["content"]
    # The text already starts with "[<date>]": don't print the date twice.
    if memory.get("date") and content.startswith(f"[{memory['date']}"):
        date = f"#{memory['order']}" if memory.get("order") else ""
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
    elif cmd == "fact-retract":
        print(f"retracted {result['retracted']} fact(s)")
    elif cmd.startswith("fact-"):
        facts = [result["fact"]] if "fact" in result else result["facts"]
        for fact in facts:
            print(fact["text"])
        if not facts:
            print("(no facts)")
    elif cmd == "ingest":
        extracted = f", {result['extracted']} facts extracted" if "extracted" in result else ""
        print(f"stored {result['stored']} memories{extracted}")
    elif cmd in {"store", "remember"}:
        extracted = f" ({result['extracted']} facts extracted)" if "extracted" in result else ""
        print(f"stored {result['id']}{extracted}")
    elif cmd == "forget":
        secure = " and erased from disk" if result["secure"] and result["forgotten"] else ""
        print(f"forgot {result['forgotten']} memor{'y' if result['forgotten'] == 1 else 'ies'}{secure}")
    elif cmd == "resolve":
        print(f"resolved {result['id'][:8]}")
    elif cmd == "supersede":
        print(f"superseded {result['superseded'][:8]} with {result['id']}")
    elif cmd in {"export", "import"}:
        skipped = f", {result['skipped']} skipped" if result.get("skipped") else ""
        verb = "exported" if cmd == "export" else "imported"
        print(f"{verb} {result['memories']} memories, {result['facts']} facts, "
              f"{result['documents']} documents{skipped} ({result['path']})")
    elif cmd == "agents":
        for item in result["agents"]:
            print(f"{item['agent']}  ({item['memories']} memories)")
        if not result["agents"]:
            print("(no agents)")
    else:
        embedder = result.get("embedder")
        line = f"{result['memories']} memories, {result['facts']} current facts"
        if embedder:
            kind = "semantic" if embedder["semantic"] else "lexical"
            line += f"; embedder {embedder['name']} ({kind}, {embedder['source']})"
        print(line)
