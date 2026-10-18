"""Line labels in notes: "DECISION:", "TODO:", "BUG:", "HAL-1:" and what they mean.

Agents and developers prefix notes with an upper-case label. The pattern
extractor strips labels before matching; this module says what a label
was, so extraction can report it:

  - a *kind* label ("DECISION:", "TODO:", "BUG:", "FIX:", "DONE:") tags every
    candidate from that line (``FactCandidate.tags``), and :func:`note_kinds`
    exposes the kinds of a whole text for memory kinds and briefings;
  - "TODO:"/"DONE:"/"FIXED:" followed by a ticket id record that ticket's
    status ("TODO: HAL-9 add retries" -> HAL-9.status = todo);
  - an *identifier* label ("HAL-1: plugin hook bug. status: resolved") is the
    subject of later subject-less "key: value" clauses on the same line.
"""
from __future__ import annotations

import re
from mark.security.lazy_regex import lazy_compile

#: Upper-case label word -> kind. Several spellings fold onto one kind.
KIND_LABELS: dict[str, str] = {
    "DECISION": "decision", "DECIDED": "decision", "ADR": "decision",
    "TODO": "todo", "TO-DO": "todo", "TASK": "todo", "ACTION": "todo", "ACTION ITEM": "todo",
    "DONE": "done", "COMPLETED": "done", "SHIPPED": "done",
    "BUG": "bug", "ISSUE": "bug", "DEFECT": "bug", "REGRESSION": "bug",
    "FIX": "fix", "FIXED": "fix", "HOTFIX": "fix", "RESOLVED": "fix",
    "NOTE": "note", "NB": "note", "FYI": "note",
    "LESSON": "lesson", "LEARNED": "lesson", "GOTCHA": "lesson",
    "PREFERENCE": "preference", "PREF": "preference",
    "SNIPPET": "snippet", "API": "api", "INCIDENT": "incident", "CHANGE": "change",
}
#: Status recorded for a ticket id that follows a TODO/DONE/FIX label.
STATUS_BY_KIND: dict[str, str] = {"todo": "todo", "done": "done", "fix": "fixed"}

_LABEL_WORD = lazy_compile(r"^\s*(?:\[[^\]]*\]\s*)?(?P<label>[A-Z][A-Z0-9_/&-]*(?:\s+[A-Z][A-Z0-9_/&-]*){0,2})\s*:\s+")
_ID_LABEL = lazy_compile(r"^[A-Z][A-Z0-9_]*-?\d[A-Z0-9_-]*$")
_LEADING_ID = lazy_compile(r"^(?P<id>[A-Za-z][A-Za-z0-9_]*-\d+[A-Za-z0-9_-]*|#\d+)(?=[\s:.,;]|$)")


def label_kind(label: str) -> str | None:
    """Kind named by one label ("DECISION ADR-7" -> "decision"), or None."""
    text = label.strip().upper()
    if text in KIND_LABELS:
        return KIND_LABELS[text]
    first = text.split()[0] if text.split() else ""
    return KIND_LABELS.get(first)


def is_id_label(label: str) -> bool:
    """True for a ticket-like label ("HAL-1", "OPS-17", "INC2207")."""
    return bool(_ID_LABEL.match(label.strip()))


def leading_id(text: str) -> str | None:
    """A ticket id at the start of *text* ("HAL-9 add retries" -> "HAL-9")."""
    match = _LEADING_ID.match(text.strip())
    return match.group("id") if match else None


def note_kinds(text: str) -> list[str]:
    """Kinds of every labelled line in *text*, in order and without repeats.

    Example: ``note_kinds("DECISION: use SQLite\\nTODO: add WAL")`` returns
    ``["decision", "todo"]``. A leading "[2026-10-09]" date stamp is skipped.
    """
    kinds: list[str] = []
    for line in (text or "").splitlines():
        match = _LABEL_WORD.match(line)
        kind = label_kind(match.group("label")) if match else None
        if kind and kind not in kinds:
            kinds.append(kind)
    return kinds
