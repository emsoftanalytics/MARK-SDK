# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Which recalled memories are no longer current, and why.

Readers of a recall (the CLI, the MCP ``mark_retrieve`` tool) label each
memory so an agent never takes an old statement for the current one:

* ``"resolved"``   a TODO/bug closed with ``resolve()``;
* ``"superseded"`` replaced with ``supersede()``/``update()``;
* ``"outdated"``   still open, but replaced: either retrieval found a later
  statement that replaces it (``RetrievalResult.outdated``), or it states
  only an earlier value of a current fact in the result ("lives in Porto"
  while ``user city = Lisbon`` is current and Porto is in its history).

``recall --open`` drops all three.
"""
from __future__ import annotations

import re
from typing import Any

from mark.kinds import TAG_RESOLVED, TAG_SUPERSEDED


def _mentions(text: str, value: str) -> bool:
    value = " ".join(str(value).split())
    if len(value) < 2:
        return False
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE) is not None


def fact_outdated(result: Any, store: Any, agent_id: str) -> set[str]:
    """Ids of memories in *result* that state only an earlier value of a current fact."""
    history_of = getattr(store, "fact_history", None)
    if history_of is None:
        return set()
    current: list[str] = []
    stale: list[str] = []
    for fact in getattr(result, "facts", None) or []:
        if not getattr(fact, "is_current", False):
            continue
        current.append(str(fact.value))
        stale += [str(row["value"]) for row in history_of(agent_id, fact.subject, fact.attribute)
                  if str(row["value"]) != str(fact.value)]
    stale = [v for v in stale if v not in current]
    if not stale:
        return set()
    out = set()
    for fragment in getattr(result, "fragments", None) or []:
        text = fragment.content or ""
        if any(_mentions(text, v) for v in stale) and not any(_mentions(text, v) for v in current):
            out.add(fragment.id)
    return out


def states(result: Any, store: Any = None, agent_id: str | None = None) -> dict[str, str]:
    """``{memory id: "resolved" | "superseded" | "outdated"}`` for the memories that are not current."""
    replaced = set(getattr(result, "outdated", None) or ())
    if store is not None and agent_id:
        replaced |= fact_outdated(result, store, agent_id)
    out: dict[str, str] = {}
    for fragment in getattr(result, "fragments", None) or []:
        tags = fragment.tags or ()
        if TAG_RESOLVED in tags:
            out[fragment.id] = "resolved"
        elif TAG_SUPERSEDED in tags:
            out[fragment.id] = "superseded"
        elif fragment.id in replaced:
            out[fragment.id] = "outdated"
    return out
