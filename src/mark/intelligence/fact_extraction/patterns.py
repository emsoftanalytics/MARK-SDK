"""PatternFactExtractor: fast, deterministic fact extraction from rule packs.

Pipeline for one text:
  1. structured parts first (JSON, YAML, key: value blocks, k=v pairs,
     numbered steps; see :mod:`.structured`); their lines are not re-read;
  2. split the rest into sentences; read date stamps and labels
     ("[2026-03-30] CHANGE:", "TODO:", "HAL-1:"; see :mod:`.sentences`);
  3. drop questions and conditionals; note corrections ("Correction: ...");
     rewrite first person as the user ("I live in X", "My sister is Y");
  4. split into clauses (``;``, ", " and "and" before a verb); a clause that
     starts with a verb borrows the previous clause's subject;
  5. drop intents ("plans to move"); a negated clause ("no longer runs on
     port 8080") is matched in its positive form and becomes a retraction;
  6. match every pack at the clause start; normalise and validate.

The guards trade recall for precision on purpose: a missed fact costs a
lookup in the raw memories, a wrong "current value" misleads the agent.
``extract`` returns assertions; ``extract_changes`` adds retractions.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import Iterable

from mark.intelligence.fact_extraction import negation, personal
from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, is_redacted, valid_candidate
from mark.intelligence.fact_extraction.clauses import carried_subject, is_continuation, split_clauses
from mark.intelligence.fact_extraction.labels import STATUS_BY_KIND, leading_id
from mark.intelligence.fact_extraction.normalise import build_candidate, peel_value
from mark.intelligence.fact_extraction.packs import BUILTIN_PACKS, PACKS_BY_NAME, RulePack
from mark.intelligence.fact_extraction.sentences import (
    CONDITIONAL,
    INLINE_DATE,
    INTENT,
    LEADING_FILLER,
    NEGATION,
    Sentence,
    is_question,
    iter_sentences,
    parse_stamp,
)
from mark.intelligence.fact_extraction.structured import attribute_key, extract_records, step_sentence, usable_key

#: Non-regex features that can be named in ``packs`` like a pack.
FEATURES = frozenset({"structured", "steps"})
_ORPHAN_FIELD = re.compile(r"^(?P<key>[A-Za-z][\w .-]{0,30}?)\s*[:=]\s*(?P<value>\S.*)$")


class PatternFactExtractor:
    """Extract facts with regular-expression rule packs (fast, runs inline).

    ``packs`` accepts :class:`RulePack` objects and/or built-in pack names
    (``"changes"``, ``"ownership"``, ``"status"``, ``"location"``,
    ``"preference"``, ``"price"``, ...) plus the features ``"structured"``
    (JSON/YAML/key-value records) and ``"steps"`` (numbered lists).
    Default: every built-in pack and feature.

    ``user_subject`` is the subject of first-person statements ("I moved to
    Porto" -> ``user.location = Porto``).
    """

    name = "patterns"
    background = False

    def __init__(self, packs: Iterable[RulePack | str] | None = None, *, user_subject: str = "user") -> None:
        names = set() if packs is None else {p for p in packs if isinstance(p, str)}
        chosen = list(BUILTIN_PACKS) if packs is None else [
            PACKS_BY_NAME[p] if isinstance(p, str) else p for p in packs if not (isinstance(p, str) and p in FEATURES)
        ]
        self.features = set(FEATURES) if packs is None else names & FEATURES
        if not str(user_subject or "").strip():
            raise ValueError("user_subject must be a non-empty name")
        self.user_subject = str(user_subject).strip()
        self.packs: tuple[RulePack, ...] = tuple(chosen)
        self._compiled = [(pack, pack.compiled()) for pack in self.packs]

    def extract(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Facts stated in *text*, in text order; ``valid_from`` from inline dates only."""
        return [c for c in self.extract_changes(text, occurred_at=occurred_at) if not c.retract]

    def extract_changes(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Like :meth:`extract`, plus ``retract=True`` candidates for values the text denies."""
        found: list[FactCandidate] = []
        seen: set[tuple[str, str, str, datetime | None, bool]] = set()

        def add(candidate: FactCandidate | None) -> None:
            # Validate before the user subject is filled in: it may be a word the
            # hygiene rules reject elsewhere ("me").
            if candidate is None or not valid_candidate(candidate):
                return
            candidate = self._for_user(candidate)
            if candidate is None:
                return
            key = (candidate.subject.lower(), candidate.attribute.lower(), candidate.value.lower(),
                   candidate.valid_from, candidate.retract)
            if key not in seen:
                seen.add(key)
                found.append(candidate)

        remaining = text or ""
        if "structured" in self.features or "steps" in self.features:
            records, remaining = extract_records(remaining)
            for subject, attribute, value in records:
                is_step = attribute.startswith("step ")
                if ("steps" if is_step else "structured") in self.features:
                    add(_triple(subject, attribute, value, occurred_at, "steps" if is_step else "structured"))
        for sentence in iter_sentences(remaining, occurred_at):
            for candidate in self._sentence_candidates(sentence, [c.subject for c in found]):
                add(candidate)
        return found

    # ── one sentence ─────────────────────────────────────────────────────────

    def _sentence_candidates(self, sent: Sentence, earlier: list[str]) -> list[FactCandidate]:
        text = sent.text
        if not text or is_question(text) or CONDITIONAL.search(text):
            return []
        out: list[FactCandidate] = []
        status = next((STATUS_BY_KIND[k] for k in sent.kinds if k in STATUS_BY_KIND), None)
        ticket = leading_id(sent.labelled_body) if status and sent.labelled_body else None
        if ticket:
            out.append(_triple(ticket, "status", status, sent.when, "labels"))
        if "steps" in self.features and (step := step_sentence(text)) is not None:
            out.append(_triple(*step, sent.when, "steps"))
        text, corrected = negation.strip_correction(text)
        text = personal.rewrite_first_person(text)
        carried: str | None = None
        previous: list[FactCandidate] = []
        for raw in split_clauses(text):
            clause = LEADING_FILLER.sub("", raw.strip())
            denied = negation.denied_value(clause)
            if denied is not None:
                # "X runs on port 9443, not 9444": withdraw 9444 for the same slot.
                out.extend(_retraction(c, peel_value(clean_text(denied))[0]) for c in previous[:1])
                continue
            clause = personal.resolve_pronoun(clause, personal.single_person(earlier + [c.subject for c in out]))
            continuation = is_continuation(clause)
            if continuation and carried:
                clause = f"{carried} {clause}"
            clause = personal.rewrite_diet(personal.conjugate_user_clause(clause))
            apposed, clause = personal.split_apposition(clause)
            clause, rather = negation.split_rather(clause)
            candidates = self._clause_candidates(clause, sent.when, sent.record_subject)
            if apposed is not None:
                candidates.insert(0, _triple(*apposed, sent.when, "personal_relation"))
            if rather and candidates and candidates[-1] and not candidates[-1].retract:
                candidates.append(_retraction(candidates[-1], peel_value(clean_text(rather))[0]))
            candidates = [c for c in candidates if c is not None]
            if not continuation:
                carried = carried_subject(clause, candidates)
            previous = [c for c in candidates if not c.retract] or previous
            out.extend(candidates)
        tags = tuple(sent.kinds)
        return [replace(c, correction=corrected and not c.retract, tags=tags or c.tags) for c in out if c is not None]

    def _clause_candidates(self, clause: str, when: datetime | None, record: str | None) -> list[FactCandidate]:
        clause = LEADING_FILLER.sub("", clause.strip()).rstrip(".!")
        if not clause or INTENT.search(clause):
            return []
        retract = False
        if NEGATION.search(clause):
            positive = negation.positive_form(clause)
            if positive is None:
                return []
            clause, retract = positive, True
        inline = INLINE_DATE.search(clause)
        valid_from = parse_stamp(inline.group(1), None) if inline else when
        out: list[FactCandidate] = []
        claimed: set[tuple[str, str]] = set()
        for pack, patterns in self._compiled:
            for pattern in patterns:
                match = pattern.match(clause)
                if match is None:
                    continue
                candidate = build_candidate(pack, match.groupdict(), valid_from)
                if candidate is None:
                    continue
                # Earlier (more specific) packs win for the same subject/attribute.
                slot = (candidate.subject.lower(), candidate.attribute.lower())
                if slot not in claimed:
                    claimed.add(slot)
                    out.append(candidate)
        if not out and record and (field := _ORPHAN_FIELD.match(clause)) and usable_key(field.group("key")):
            # "HAL-1: plugin hook bug. status: resolved": the line's id label is the subject.
            orphan = _triple(record, attribute_key(field.group("key")), clean_text(field.group("value")),
                             valid_from, "key_value")
            out = [orphan] if orphan is not None else []
        return [replace(c, retract=True) for c in out] if retract else out

    def _for_user(self, candidate: FactCandidate) -> FactCandidate | None:
        """Replace the first-person placeholder with the configured user subject."""
        subject, value = candidate.subject, candidate.value
        if personal.USER_TOKEN in subject and subject != personal.USER_TOKEN:
            return None  # "<user>'s favourite" is a mis-split, not a subject
        if personal.USER_TOKEN in value and value != personal.USER_TOKEN:
            return None
        if personal.is_user(subject):
            subject = self.user_subject
        if value == personal.USER_TOKEN:
            value = self.user_subject
        if (subject, value) == (candidate.subject, candidate.value):
            return candidate
        return replace(candidate, subject=subject, value=value)


def _triple(subject: str, attribute: str, value: str, when: datetime | None, rule: str) -> FactCandidate | None:
    """A candidate built outside the regex packs (records, steps, labels), cleaned and validated."""
    subject, value = clean_text(subject), personal.restore(clean_text(value))
    if is_redacted(subject) or is_redacted(value):
        return None
    candidate = FactCandidate(subject=subject, attribute=clean_text(attribute).lower(), value=value,
                              valid_from=when, confidence=0.7, rule=rule)
    return candidate if valid_candidate(candidate) else None


def _retraction(candidate: FactCandidate | None, value: str) -> FactCandidate | None:
    """A retraction of *value* in the slot of *candidate*."""
    if candidate is None or not value:
        return None
    return replace(candidate, value=value, retract=True, correction=False)
