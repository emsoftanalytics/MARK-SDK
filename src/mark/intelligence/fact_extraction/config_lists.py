"""Config blocks and per-entity attribute lists, read before the generic structured records.

Two line shapes that name no subject in the usual way:

  - **environment-style config** ``MAX_UPLOAD_MB: 250`` / ``SESSION_TIMEOUT_MIN=45``:
    an upper-case snake_case key is a setting of the configuration
    (subject ``config``, or the Markdown heading above the block). Keys
    without an underscore count only inside a block of two or more such
    lines, and never when they are note labels ("TODO:", "NOTE:");
  - **attribute lists** under a plural attribute header::

        Service ports:
        - auth-svc: 7001          -> auth-svc.port = 7001
        - search-svc: 7002

    The header names the attribute and each key the subject. Ports must be
    port numbers. A lone ``atlas-core: 8080`` (no header) says nothing about
    what 8080 is, so it is left alone: precision first.

Lines used here are removed from the text handed on, so nothing is read twice.
"""
from __future__ import annotations

import re

from mark.intelligence.fact_extraction.labels import KIND_LABELS
from mark.intelligence.fact_extraction.rules import ATTRIBUTE_SYNONYMS, ATTRIBUTE_WORDS

Triple = tuple[str, str, str]
_ENV = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*)\s*[:=]\s*(?P<value>\S.*?)\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_LIST_HEADER = re.compile(r"^\s*(?:#{1,6}\s+)?(?P<text>(?:[A-Za-z][\w-]*\s+){0,2}(?P<noun>[A-Za-z]+))\s*:?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+)?(?P<key>[^\W_][\w.@/-]*)\s*[:=]\s*(?P<value>\S[^:]*?)\s*$")
_PORT = re.compile(r"^\d{2,5}$")
_NOUNS = frozenset(ATTRIBUTE_WORDS)


def config_records(text: str) -> tuple[list[Triple], str]:
    """Triples from config blocks and attribute lists, plus the text left for other readers."""
    if ":" not in (text or "") and "=" not in (text or ""):
        return [], text or ""  # cheap exit: no key/value line at all
    lines = (text or "").splitlines()
    used: set[int] = set()
    triples = _env_blocks(lines, used) + _attribute_lists(lines, used)
    return triples, "\n".join(line for i, line in enumerate(lines) if i not in used)


def _env_blocks(lines: list[str], used: set[int]) -> list[Triple]:
    out: list[Triple] = []
    subject, block = "config", []  # block: (index, key, value)

    def flush() -> None:
        strong = [b for b in block if "_" in b[1]]
        # Plain upper-case keys ("PORT: 8080") only in a block of 2+ with single-token values.
        if len(block) >= 2 and all(" " not in b[2] for b in block):
            strong = block
        for index, key, value in strong:
            used.add(index)
            out.append((subject, key, value.strip("\"'")))
        block.clear()

    for index, line in enumerate(lines):
        heading = _HEADING.match(line)
        if heading is not None:
            flush()
            subject = heading.group("text").strip().strip("*_`") or "config"
            continue
        match = _ENV.match(line)
        if match is None or match.group("key").split("_")[0] in KIND_LABELS or match.group("key") in KIND_LABELS:
            flush()
            continue
        block.append((index, match.group("key"), match.group("value")))
    flush()
    return out


def _attribute_lists(lines: list[str], used: set[int]) -> list[Triple]:
    out: list[Triple] = []
    index = 0
    while index < len(lines):
        header = _LIST_HEADER.match(lines[index]) if index not in used else None
        attribute = _singular_attribute(header.group("noun")) if header else None
        if attribute is None:
            index += 1
            continue
        items, end = [], index + 1
        while end < len(lines) and (item := _LIST_ITEM.match(lines[end])) is not None and end not in used:
            items.append((end, item.group("key"), item.group("value")))
            end += 1
        valid = [i for i in items if attribute != "port" or _PORT.match(i[2]) and int(i[2]) <= 65535]
        if items and len(valid) == len(items):
            used.update([index, *(i for i, _, _ in items)])
            out.extend((key, attribute, value) for _, key, value in items)
        index = max(end, index + 1)
    return out


def _singular_attribute(noun: str) -> str | None:
    """'ports' -> 'port', 'Owners' -> 'owner'; None unless it is a known attribute noun in the plural."""
    word = noun.lower()
    if not word.endswith("s"):
        return None
    for singular in (word[:-1], word[:-2] if word.endswith("es") else ""):
        if singular in _NOUNS:
            return ATTRIBUTE_SYNONYMS.get(singular, singular)
    return None
