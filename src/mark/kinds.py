# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Memory kinds and lifecycle tags shared by the Python API, CLI and MCP (``mark.kinds``).

Standard library only, so the ``mark`` command line can validate ``--kind``
without loading MARK.

A *kind* says what a memory is for (``remember(text, kind="todo")``) and is
stored as the tag ``kind:<kind>``, so every tag filter (recall, list, the
pipeline's tag pre-filter) can scope by kind with no schema change.

Lifecycle tags (set by ``supersede``/``update``/``resolve``):

* ``mark:superseded`` — the memory is no longer current (replaced, edited or
  resolved). Retrieval demotes it but keeps it findable for history questions.
* ``mark:resolved``   — a TODO/bug that was closed; ``list(kind="todo",
  open=True)`` leaves it out.

A memory written without ``kind`` takes it from a leading label
(:func:`kind_from_text`: ``TODO:``, ``DECISION:``/``ADR-7:``, ``BUG:``,
``FIX:``, ``DONE:``/``NOTE:``, ``API:``, ``SNIPPET:``), so the brief, ``list
--kind`` and ``recall --kind`` agree on what agents actually write.
"""
from __future__ import annotations

import functools
from typing import Any

#: Every accepted memory kind, in documentation order.
KINDS: tuple[str, ...] = ("decision", "todo", "bug", "fix", "preference", "snippet", "api", "note")

KIND_TAG_PREFIX = "kind:"
TAG_SUPERSEDED = "mark:superseded"
TAG_RESOLVED = "mark:resolved"

#: Defaults shared by every write API (simple blocks, MarkMemory, CLI, MCP).
DEFAULT_IMPORTANCE = 0.5
DEFAULT_CONFIDENCE = 1.0


def normalize_kind(kind: str | None) -> str | None:
    """Validate *kind* and return its canonical form (None stays None).

    Plurals and case are forgiven (``"TODOs"`` -> ``"todo"``); anything else
    raises :class:`mark.errors.InvalidInputError` listing the valid kinds.
    """
    if kind is None:
        return None
    text = str(kind).strip().lower()
    if text in KINDS:
        return text
    if text.endswith("s") and text[:-1] in KINDS:
        return text[:-1]
    from mark.errors import InvalidInputError

    raise InvalidInputError(f"unknown memory kind {kind!r}; expected one of: {', '.join(KINDS)}")


def kind_tag(kind: str) -> str:
    """The tag that stores *kind* (``"todo"`` -> ``"kind:todo"``)."""
    return KIND_TAG_PREFIX + kind


#: Leading labels and the kind they mean. DONE is a progress note: the brief
#: still reads its "DONE:" prefix to close the TODO it names.
_LABEL_KINDS = {"todo": "todo", "decision": "decision", "adr": "decision", "bug": "bug",
                "fix": "fix", "fixed": "fix", "done": "note", "note": "note", "api": "api",
                "snippet": "snippet"}


@functools.cache
def _label_re() -> Any:
    # Compiled on first use, not at import: the CLI imports this module on
    # every start-up and most commands never read a label.
    import re

    # An optional "[2026-10-02]" stamp, the label, then ":" or a spaced dash
    # ("TODO - x"); "API-key rotation" is prose, not a label.
    return re.compile(r"^\s*(?:\[[^\]]{1,40}\]\s*)?(TODO|DECISION|ADR(?:[-\s]?\d+)?|BUG|FIX(?:ED)?|DONE|"
                      r"NOTE|API|SNIPPET)(?:\s*:|\s+-\s)", re.IGNORECASE)


def kind_from_text(text: str | None) -> str | None:
    """The kind a leading label names ("TODO: add retries" -> "todo"), or None."""
    match = _label_re().match(text or "")
    if match is None:
        return None
    word = match.group(1).lower()
    return _LABEL_KINDS.get(word) or _LABEL_KINDS.get(word[:3])  # "adr-7" -> "adr"


def kind_of(tags_or_fragment: Any) -> str | None:
    """The kind of a memory: its ``kind:<k>`` tag, else (for a fragment) its leading label.

    Accepts a tag list or anything with ``tags`` (and ``content``), such as a
    :class:`mark.types.MemoryFragment`: ``kind_of(memory.get(id))``.
    """
    tags = getattr(tags_or_fragment, "tags", tags_or_fragment) or ()
    for tag in tags:
        if tag.startswith(KIND_TAG_PREFIX):
            return tag[len(KIND_TAG_PREFIX):]
    content = getattr(tags_or_fragment, "content", None)
    return kind_from_text(content) if isinstance(content, str) else None


def is_open(tags: list[str] | tuple[str, ...]) -> bool:
    """True when a memory is neither resolved nor superseded."""
    return TAG_RESOLVED not in tags and TAG_SUPERSEDED not in tags
