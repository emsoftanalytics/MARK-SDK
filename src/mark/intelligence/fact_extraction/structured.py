"""Structured content: JSON objects, YAML maps, key: value blocks, k=v pairs, numbered steps.

A field becomes ``subject.attribute = value`` only when the record names its
subject, so a lone "port: 8080" is never a fact. The subject comes from, in
order:

  1. a subject field in the record (``name``, ``service``, ``id``, ...);
  2. the record's parent: a YAML key with nested fields ("billing-api:"),
     a Markdown heading ("## search-api") or a short "Header:" line;
  3. for k=v pairs, a trailing "for <subject>" ("port=8080 for atlas-core").

Lines used by a resolved record are removed from the text handed to the
sentence patterns, so nothing is extracted twice; unresolved lines are left
for them ("tasky version: 0.2.0" is matched there). Numbered steps under a
header ("Deploy procedure: 1. Freeze merges. 2. Tag ...") become
``<header>.step N`` facts; number words count ("Step four: ...").
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

from mark.intelligence.fact_extraction.numbers import NUMBER_WORD, parse_number

Triple = tuple[str, str, str]

#: Field names that name the record itself, most specific first.
SUBJECT_KEYS = ("service", "name", "id", "app", "application", "project", "host", "hostname", "repo",
                "repository", "package", "component", "ticket", "product", "server", "device", "username",
                "user", "title", "key")
# Field names that label prose, not attributes ("Note: call the vendor").
_PROSE_KEYS = frozenset((
    "note notes nb summary description desc comment comments todo done decision bug fix fixed error warning warn"
    " info debug usage example examples question answer q a re subject from to cc bcc sent date time http https"
    " step tip hint important caution update fyi ps edit source reason why how what when where who result"
    " output input response request traceback details context background agenda action action items"
    " correction problem solution lesson caveat issue changelog"
).split())
_ID_KEY = re.compile(r"\d.*-|-.*\d")
# Section headings that are not subjects ("## Notes").
_GENERIC_HEADERS = frozenset((
    "notes summary overview details background config configuration settings status todo todos decisions"
    " introduction appendix faq changelog misc links references contacts team people meeting notes agenda"
    " action items next steps example examples usage install installation setup requirements"
).split())
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_KV = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+]\s+)?(?P<key>[A-Za-z][\w .\-/]{0,40}?)\s*:\s*(?P<value>.*?)\s*$")
_PAIR = re.compile(r"(?<![\w.-])(?P<key>[A-Za-z][\w.-]{0,30})=(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)")
_PAIR_SUBJECT = re.compile(r"\b(?:for|on|of|in)\s+(?P<subject>[^\W_][\w.@/-]*)\s*\.?\s*$", re.IGNORECASE)
_HEADER_LINE = re.compile(r"^\s*(?P<text>[^:\n]{2,60}?)\s*:\s*$")
_INLINE_STEPS = re.compile(r"^\s*(?P<head>[^:\n]{2,60}?)\s*:\s*(?=1[.)]\s)(?P<body>.+)$")
_STEP_SPLIT = re.compile(r"(?:^|\s)(\d{1,2})[.)]\s+")
_STEP_LINE = re.compile(rf"^\s*(?:(?P<num>\d{{1,2}})[.)]|step\s+(?P<word>{NUMBER_WORD})\s*[:.)\-–—]|(?P<ord>{NUMBER_WORD})"
                        r"\s*[:)\-–—])\s*(?P<text>\S.*?)\s*$", re.IGNORECASE)
_STEP_SENTENCE = re.compile(
    rf"^(?:the\s+)?(?:step\s+(?P<a>{NUMBER_WORD})|(?P<b>{NUMBER_WORD})\s+step)\s+(?:of|in)\s+(?:the\s+|our\s+)?"
    r"(?P<subject>[^,;]{2,60}?)\s+is\s+(?:to\s+)?(?P<value>[^;]+?)\.?\s*$", re.IGNORECASE)


def attribute_key(key: str) -> str:
    """Field name as an attribute: "test_command" -> "test command"."""
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", key)).strip().lower()


def _clean(value: Any) -> str:
    text = str(value).strip().strip("\"'").strip().rstrip(",")
    return "" if text.lower() in {"", "null", "none", "~"} or text[:1] in "{[|>" else text


def usable_key(key: str) -> bool:
    """A field name that can be an attribute (short, not a prose label or ticket id)."""
    words = attribute_key(key).split()
    return 0 < len(words) <= 4 and attribute_key(key) not in _PROSE_KEYS and not _ID_KEY.search(key)


def _field_value(value: str) -> bool:
    """A field value is short and not prose ("Standup: Dana moved the deploy. X runs ..." is prose)."""
    return len(value.split()) <= 8 and not re.search(r"[.!?]\s+\S", value)


def _header_subject(text: str) -> str | None:
    """Subject named by a heading or "Header:" line, or None for generic sections.

    A document chunk's heading path ("Guide › billing-api") names its last part.
    """
    text = text.split(" › ")[-1]
    text = re.sub(r"^\d+[.)]\s*", "", text.strip()).strip("*_` ")
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    for_match = re.search(r"\b(?:for|of)\s+(?:the\s+)?(?P<s>[^\W_][\w.@/-]*)$", text)
    if for_match and len(text.split()) > 1:
        text = for_match.group("s")
    if not text or len(text.split()) > 5 or text.lower() in _GENERIC_HEADERS:
        return None
    return text


def extract_records(text: str) -> tuple[list[Triple], str]:
    """Triples from structured parts of *text*, plus the text left for sentence patterns."""
    triples: list[Triple] = []
    remaining = text
    obj, span = _find_json(text)
    if obj is not None:
        head = _header_subject(text[:span[0]].strip().rstrip(":").strip()) if text[:span[0]].strip() else None
        found = list(_json_triples(obj, head))
        if found:
            triples.extend(found)
            remaining = text[:span[0]] + text[span[1]:]
    lines = remaining.splitlines()
    used: set[int] = set()
    triples.extend(_line_records(lines, used))
    triples.extend(_steps(lines, used))
    return triples, "\n".join(line for i, line in enumerate(lines) if i not in used)


def step_sentence(sentence: str) -> Triple | None:
    """"Step four of the backup runbook is to verify checksums" -> (backup runbook, step 4, verify checksums)."""
    match = _STEP_SENTENCE.match(sentence.strip())
    if match is None:
        return None
    number = parse_number(match.group("a") or match.group("b"))
    if number is None:
        return None
    return match.group("subject").strip(), f"step {number}", match.group("value").strip()


# ── JSON ─────────────────────────────────────────────────────────────────────

def _find_json(text: str) -> tuple[Any, tuple[int, int]]:
    start = text.find("{")
    if start < 0:
        return None, (0, 0)
    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return None, (0, 0)
    return (obj, (start, end)) if isinstance(obj, dict) else (None, (0, 0))


def _json_triples(obj: dict[str, Any], parent: str | None) -> Iterator[Triple]:
    subject_key = next((k for k in SUBJECT_KEYS for key in obj if key.lower() == k and _clean(obj[key])), None)
    subject = next((_clean(obj[key]) for key in obj if key.lower() == subject_key), None) if subject_key else parent
    for key, value in obj.items():
        if isinstance(value, dict):
            if subject is None:
                yield from _json_triples(value, key)  # {"billing-api": {"port": 1}}: the key names it
            else:
                # One level of nesting folds into the attribute: "db": {"engine": ...} -> "db engine".
                yield from ((subject, f"{attribute_key(key)} {attribute_key(k)}", _clean(v))
                            for k, v in value.items()
                            if not isinstance(v, (dict, list)) and _clean(v) and usable_key(f"{key} {k}"))
            continue
        if subject is None or key.lower() == subject_key or not usable_key(key):
            continue
        if isinstance(value, list):
            items = [_clean(v) for v in value if not isinstance(v, (dict, list))]
            if not items or len(items) != len(value) or len(items) > 5:
                continue
            value = ", ".join(items)
        cleaned = _clean(value)
        if cleaned:
            yield subject, attribute_key(key), cleaned


# ── key: value lines and k=v pairs ───────────────────────────────────────────

def _line_records(lines: list[str], used: set[int]) -> Iterator[Triple]:
    """Records of consecutive key: value lines, with YAML parents and headings."""
    heading: tuple[int, str | None] | None = None   # (line index, subject)
    parents: list[tuple[int, int, str]] = []        # (indent, line index, key) for YAML parents
    block: list[tuple[int, int, str, str]] = []     # (line index, indent, key, value)

    def flush() -> Iterator[Triple]:
        if block:
            yield from _resolve(block, parents, heading, used)
            block.clear()

    for index, line in enumerate(lines):
        if not line.strip():
            yield from flush()
            continue
        if (match := _HEADING.match(line)) is not None:
            yield from flush()
            heading, parents = (index, _header_subject(match.group("text"))), []
            continue
        pairs = list(_PAIR.finditer(line))
        if pairs and _mostly_pairs(line, pairs):
            yield from flush()
            yield from _pair_line(index, line, pairs, heading, used)
            continue
        kv = _KV.match(line)
        if kv is None or (kv.group("value") and not (usable_key(kv.group("key")) and _field_value(kv.group("value")))):
            yield from flush()
            continue
        indent = len(kv.group("indent").expandtabs(4))
        if not kv.group("value"):
            yield from flush()
            parents = [p for p in parents if p[0] < indent] + [(indent, index, kv.group("key").strip())]
            continue
        block.append((index, indent, kv.group("key").strip(), kv.group("value")))
    yield from flush()


def _resolve(block: list[tuple[int, int, str, str]], parents: list[tuple[int, int, str]],
             heading: tuple[int, str | None] | None, used: set[int]) -> Iterator[Triple]:
    fields = {attribute_key(key): _clean(value) for _, _, key, value in block}
    subject_key = next((k for k in SUBJECT_KEYS if fields.get(k)), None)
    subject, owner_lines = (fields[subject_key], []) if subject_key else (None, [])
    if subject is None:
        indent = block[0][1]
        parent = next((p for p in reversed(parents) if p[0] < indent), None)
        if parent is not None:
            subject, owner_lines = parent[2], [parent[1]]
        elif heading is not None and heading[1]:
            subject, owner_lines = heading[1], [heading[0]]
    if not subject:
        return
    for index, _, key, value in block:
        attribute, cleaned = attribute_key(key), _clean(value)
        used.add(index)
        if attribute != subject_key and cleaned:
            yield subject, attribute, cleaned
    used.update(owner_lines)


def _mostly_pairs(line: str, pairs: list[re.Match[str]]) -> bool:
    """True when a line is k=v pairs plus at most "for <subject>" (not prose quoting "a=b")."""
    rest = _PAIR.sub(" ", line)
    rest = _PAIR_SUBJECT.sub(" ", rest) if _PAIR_SUBJECT.search(line[pairs[-1].end():]) else rest
    return len(re.findall(r"[^\W_]+", rest)) <= 2


def _pair_line(index: int, line: str, pairs: list[re.Match[str]], heading: tuple[int, str | None] | None,
               used: set[int]) -> Iterator[Triple]:
    fields = {attribute_key(p.group("key")): _clean(p.group("value")) for p in pairs}
    subject_key = next((k for k in SUBJECT_KEYS if fields.get(k)), None)
    tail = _PAIR_SUBJECT.search(line[pairs[-1].end():])
    subject = (fields[subject_key] if subject_key else tail.group("subject") if tail
               else heading[1] if heading is not None else None)
    if not subject:
        return
    used.add(index)
    for attribute, value in fields.items():
        if attribute != subject_key and value and usable_key(attribute):
            yield subject, attribute, value


# ── numbered steps ───────────────────────────────────────────────────────────

def _steps(lines: list[str], used: set[int]) -> Iterator[Triple]:
    header: str | None = None
    for index, line in enumerate(lines):
        if index in used:
            continue
        inline = _INLINE_STEPS.match(line)
        if inline and (head := _header_subject(inline.group("head"))):
            parts = _STEP_SPLIT.split(" " + inline.group("body"))
            steps = [(int(parts[i]), parts[i + 1].strip().rstrip(".").strip()) for i in range(1, len(parts) - 1, 2)]
            if len(steps) >= 2 and [n for n, _ in steps] == list(range(1, len(steps) + 1)):
                used.add(index)
                yield from ((head, f"step {n}", step) for n, step in steps if step)
            continue
        heading = _HEADING.match(line) or _HEADER_LINE.match(line)
        if heading is not None:
            header = _header_subject(heading.group("text"))
            continue
        step = _STEP_LINE.match(line)
        number = parse_number(step.group("num") or step.group("word") or step.group("ord")) if step else None
        if header and step and number and number <= 50:
            used.add(index)
            yield header, f"step {number}", step.group("text").rstrip(".").strip()
        elif line.strip() and not step:
            header = None if not line.startswith((" ", "\t", "-", "*")) else header
