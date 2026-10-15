"""PatternFactExtractor: fast, deterministic fact extraction from rule packs.

Pipeline for one text:
  1. split into sentences (newlines, ``. ! ?`` followed by space);
  2. drop sentences that are questions or conditionals ("did X move to Y?",
     "if X moves to Y ...", "X could move to Y");
  3. strip a leading date stamp and upper-case log label ("[2026-03-30]
     CHANGE: ..."), then split into clauses (``;`` and ", ");
  4. drop clauses with a negation or a future/intent marker ("did not move",
     "will not be owned by", "plans to move");
  5. match every pack pattern at the clause start; normalise subject,
     attribute and value; validate; de-duplicate.

The guards trade recall for precision on purpose: a missed fact costs a
lookup in the raw memories, a wrong "current value" misleads the agent.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, is_redacted, valid_candidate
from mark.intelligence.fact_extraction.packs import (
    ATTRIBUTE_SYNONYMS,
    ATTRIBUTE_WORDS,
    BUILTIN_PACKS,
    PACKS_BY_NAME,
    RulePack,
)

_SENTENCE_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r";\s*|,\s+")
_STAMP = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?)Z?)?\]?\s*[:\-–—]?\s*")
# Upper-case log labels: "CHANGE:", "HR:", "DECISION ADR-7:" (all-caps tokens only).
_LABEL = re.compile(r"^[A-Z][A-Z0-9_/&-]*(?:\s+[A-Z][A-Z0-9_/&-]*){0,2}\s*:\s+")
_LEADING_FILLER = re.compile(
    r"^(?:today|yesterday|tonight|this\s+(?:morning|afternoon|evening|week)|earlier\s+today|now|then"
    r"|finally|also|meanwhile|later|afterwards|after\s+that|subsequently|eventually|update|fyi|note)\b[\s,:-]*",
    re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^(?:did|does|do|is|are|was|were|will|would|should|can|could|has|have|had|shall|may|might"
    r"|when|what|which|who|whom|whose|why|how|where)\b", re.IGNORECASE)
_CONDITIONAL = re.compile(
    r"\b(?:if|unless|whether|suppose|supposing|assuming|in\s+case|hypothetically|imagine"
    r"|would|could|might|should)\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|never|no\s+longer|no\s+more|neither|nor|cannot|failed\s+to|refused\s+to)\b|n't\b",
    re.IGNORECASE)
_INTENT = re.compile(
    r"\b(?:will|shall|going\s+to|plan(?:s|ned)?\s+to|planning|intends?\s+to|expected\s+to|expects?\s+to"
    r"|scheduled\s+to|about\s+to|propose[sd]?|proposal|wants?\s+to|would\s+like|to\s+be|soon)\b",
    re.IGNORECASE)
_INLINE_DATE = re.compile(r"\b(?:on|effective|as\s+of|since)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
# Leading kind nouns dropped from subjects when an identifier follows
# ("service atlas-core" -> "atlas-core", "TICKET OPS-12" -> "OPS-12").
_KIND_PREFIX = re.compile(
    r"^(?:the\s+)?(?:service|server|ticket|issue|bug|project|user|employee|customer|host|app|application"
    r"|system|repo|repository|job|task|incident|cluster|node|account|component|module)\s+(?=\S*[\d_.A-Z-])",
    re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TRAILING_FILLER = re.compile(
    r"\s+(?:now|today|yesterday|tonight|instead|permanently|temporarily|again|as\s+well|too"
    r"|immediately|effective\s+immediately|in\s+production|last\s+(?:night|week)|this\s+(?:morning|week))$",
    re.IGNORECASE)
_WORDS = sorted(ATTRIBUTE_WORDS, key=len, reverse=True)
_ALT = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in _WORDS)
_VALUE_PREFIX = re.compile(rf"^(?:the\s+)?({_ALT})\s+(?!of\b)(.+)$", re.IGNORECASE)
_VALUE_SUFFIX = re.compile(rf"^(.+?)\s+({_ALT})$", re.IGNORECASE)
_SUBJECT_SUFFIX = re.compile(rf"^(.+?)(?:['’]s)?\s+({_ALT})$", re.IGNORECASE)


class PatternFactExtractor:
    """Extract facts with regular-expression rule packs (fast, runs inline).

    ``packs`` accepts :class:`RulePack` objects and/or built-in pack names
    (``"changes"``, ``"ownership"``, ``"status"``, ``"membership"``,
    ``"assignment"``, ``"key_value"``). Default: every built-in pack.
    """

    name = "patterns"
    background = False

    def __init__(self, packs: Iterable[RulePack | str] | None = None) -> None:
        chosen = list(BUILTIN_PACKS) if packs is None else [
            PACKS_BY_NAME[p] if isinstance(p, str) else p for p in packs
        ]
        self.packs: tuple[RulePack, ...] = tuple(chosen)
        self._compiled = [(pack, pack.compiled()) for pack in self.packs]

    def extract(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Facts stated in *text*, in text order; ``valid_from`` from inline dates only."""
        found: list[FactCandidate] = []
        seen: set[tuple[str, str, str, datetime | None]] = set()
        when, first = occurred_at, True
        for raw_sentence in _SENTENCE_SPLIT.split(text or ""):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            stamp = _STAMP.match(sentence)
            if stamp:
                # The first stamp is the memory's own event time, which the caller
                # resolves (explicit occurred_at wins). A later line's stamp dates
                # that line and the ones after it (a multi-line log in one memory).
                if not first:
                    when = _parse_stamp(stamp.group(1), stamp.group(2)) or when
                sentence = sentence[stamp.end():]
            first = False
            while (label := _LABEL.match(sentence)) is not None:
                sentence = sentence[label.end():]
            if not sentence or _is_question(sentence) or _CONDITIONAL.search(sentence):
                continue
            for clause in _CLAUSE_SPLIT.split(sentence):
                for candidate in self._clause_candidates(clause, when):
                    key = (candidate.subject.lower(), candidate.attribute.lower(),
                           candidate.value.lower(), candidate.valid_from)
                    if key not in seen:
                        seen.add(key)
                        found.append(candidate)
        return found

    def _clause_candidates(self, clause: str, when: datetime | None) -> list[FactCandidate]:
        clause = _LEADING_FILLER.sub("", clause.strip()).rstrip(".!")
        if not clause or _NEGATION.search(clause) or _INTENT.search(clause):
            return []
        inline = _INLINE_DATE.search(clause)
        valid_from = _parse_stamp(inline.group(1), None) if inline else when
        out: list[FactCandidate] = []
        claimed: set[tuple[str, str]] = set()
        for pack, patterns in self._compiled:
            for pattern in patterns:
                match = pattern.match(clause)
                if match is None:
                    continue
                candidate = _candidate(pack, match.groupdict(), valid_from)
                if candidate is None:
                    continue
                # Earlier (more specific) packs win for the same subject/attribute.
                slot = (candidate.subject.lower(), candidate.attribute.lower())
                if slot not in claimed:
                    claimed.add(slot)
                    out.append(candidate)
        return out


def _candidate(pack: RulePack, groups: dict[str, str | None], valid_from: datetime | None) -> FactCandidate | None:
    """Normalise one regex match into a validated candidate (or None)."""
    if is_redacted(groups.get("subject") or "") or is_redacted(groups.get("value") or ""):
        return None
    subject = _ARTICLE.sub("", clean_text(groups.get("subject") or ""))
    value = clean_text(groups.get("value") or "")
    attribute = clean_text(groups.get("attribute") or groups.get("attribute_free") or "").lower()
    # Attribute nouns written next to the value ("port 4444", "the Data team").
    value, value_attr = _peel_value(value) if pack.attribute_from_value else (value, "")
    old_attr = _peel_value(clean_text(groups.get("old") or ""))[1]
    subject_match = _SUBJECT_SUFFIX.match(subject) if len(subject.split()) > 1 else None
    subject_attr = ""
    if subject_match:
        subject, subject_attr = subject_match.group(1), subject_match.group(2).lower()
    subject = re.sub(r"['’]s$", "", _KIND_PREFIX.sub("", subject).strip())
    verb = re.sub(r"\s+", " ", (groups.get("verb") or "").lower())
    attribute = (pack.attribute or attribute or value_attr or old_attr or subject_attr
                 or pack.verb_attributes.get(verb) or pack.verb_attributes.get("", ""))
    attribute = ATTRIBUTE_SYNONYMS.get(re.sub(r"\s+", " ", attribute), re.sub(r"\s+", " ", attribute))
    value = _TRAILING_FILLER.sub("", _ARTICLE.sub("", value)).strip()
    if pack.value_case == "lower":
        value = value.lower()
    value = pack.value_map.get(value.lower(), value)
    candidate = FactCandidate(subject=subject, attribute=attribute, value=value,
                              valid_from=valid_from, confidence=pack.confidence, rule=pack.name)
    return candidate if valid_candidate(candidate) else None


def _peel_value(value: str) -> tuple[str, str]:
    """Split an attribute noun off a value: 'port 4444' -> ('4444', 'port')."""
    prefix = _VALUE_PREFIX.match(value)
    if prefix:
        return clean_text(prefix.group(2)), prefix.group(1).lower()
    suffix = _VALUE_SUFFIX.match(value)
    if suffix:
        return clean_text(suffix.group(1)), suffix.group(2).lower()
    return value, ""


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(_QUESTION_START.match(sentence))


def _parse_stamp(day: str, clock: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(day + (f"T{clock}" if clock else ""))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)
