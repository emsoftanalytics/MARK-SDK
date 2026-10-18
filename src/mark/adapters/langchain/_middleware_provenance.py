"""Provenance and injection helpers shared by the LangChain middleware modules.

* **What gets stored, and how it is tagged.** :func:`store_options` maps one
  conversation message to ``observe()`` keyword arguments using
  :mod:`mark.context.provenance`, so every store path (history archival, run
  end, tool wrapper) tags the same message the same way: user questions and
  requests are never stored as plain statements, assistant replies are low
  trust, tool output is ``mark:tool`` and failed tool calls also ``mark:error``.
* **What may be shown as a note.** :func:`notes_only` removes the model's own
  replies, bare questions/requests and tool errors from a retrieval result
  before the middleware renders it, so earlier questions are never echoed back
  as "related notes" (a10 integration finding N3).
* **The injected wrapper.** :func:`memory_block` is the one place that wraps
  memory for the system prompt; the budget code measures this exact text, so
  the wrapper counts inside ``budget_tokens`` (finding N4).
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any

from mark.context.history import message_role, message_text
from mark.context.provenance import observe_options, tool_options
from mark.context.watch_lines import trusted

#: Opening line of the injected block: short and domain-neutral (a10 N4; the
#: old "[MARK project memory] ... file contents" wording was coding-specific).
MEMORY_OPEN = "[MARK memory] Notes from earlier conversations; use them if relevant."
MEMORY_CLOSE = "[/MARK memory]"
#: Separator between system-prompt blocks when a provider flattens them to text;
#: charged to the budget so the *added* text, not just the block, fits.
BLOCK_JOINT = "\n"


def memory_block(context: str) -> str:
    """The exact text injected for *context* ("" when there is nothing to inject)."""
    return f"{MEMORY_OPEN}\n{context}\n{MEMORY_CLOSE}" if context else ""


def wrapper_chars() -> int:
    """Characters the wrapper (and its joint) adds around the content."""
    return len(BLOCK_JOINT) + len(memory_block("x")) - 1


def is_tool_error(message: Any) -> bool:
    """True for a ToolMessage (or dict) whose ``status`` is ``"error"``."""
    status = message.get("status") if isinstance(message, dict) else getattr(message, "status", None)
    return status == "error"


def tool_observe_options(*, error: bool = False) -> dict[str, Any]:
    """``observe()`` kwargs for a tool result. ``observe`` takes no ``confidence``,
    so a failed call is marked low trust through metadata instead."""
    options = tool_options(error=error)
    if error:
        options.pop("confidence", None)
        options["metadata"] = {"trust": "low"}
    return options


def store_options(message: Any) -> dict[str, Any]:
    """``observe()`` kwargs for one conversation message, by role."""
    role = message_role(message)
    if role == "tool":
        return tool_observe_options(error=is_tool_error(message))
    return observe_options(role, message_text(message))


def notes_only(value: Any) -> Any:
    """*value* (a retrieval result) without fragments that must not be shown as notes.

    Scores/relevance lists stay aligned with the kept fragments. Results that
    cannot be copied are returned unchanged (rendering must never fail here).
    """
    fragments = getattr(value, "fragments", None)
    if not fragments:
        return value
    keep = [i for i, fragment in enumerate(fragments) if trusted(fragment)]
    if len(keep) == len(fragments):
        return value
    changes: dict[str, Any] = {"fragments": [fragments[i] for i in keep]}
    for name in ("scores", "relevance"):
        values = getattr(value, name, None)
        if isinstance(values, list) and len(values) == len(fragments):
            changes[name] = [values[i] for i in keep]
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.replace(value, **changes)
        clone = copy.copy(value)
        for name, item in changes.items():
            setattr(clone, name, item)
        return clone
    except Exception:
        return value
