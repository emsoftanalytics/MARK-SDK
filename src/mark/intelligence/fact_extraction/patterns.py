"""PatternFactExtractor: fast, deterministic fact extraction from rule packs.

Pipeline for one text:
  1. structured parts first (config blocks and attribute lists, see
     :mod:`.config_lists`; JSON, YAML, key: value blocks, k=v pairs, numbered
     steps, see :mod:`.structured`); their lines are not re-read;
  2. split the rest into sentences; read date stamps and labels
     ("[2026-03-30] CHANGE:", "TODO:", "HAL-1:"; see :mod:`.sentences`) and a
     leading "Starting 2027-01-01," (see :mod:`.dates`);
  3. drop questions and conditionals; note corrections ("Correction: ...");
     rewrite first person as the user ("I live in X", "My sister is Y");
     a list sentence ("I speak X, Y and Z") becomes one value per item;
  4. split into clauses (``;``, ", " and "and" before a verb); a clause that
     starts with a verb borrows the previous clause's subject;
  5. drop intents ("plans to move"); a negated clause ("no longer runs on
     port 8080") is matched in its positive form and becomes a retraction
     ("used to" a *historical* one);
  6. match every pack at the clause start; normalise and validate; the "A"
     of "from A to B" becomes a ``previous`` candidate before B;
  7. a subject that starts with a name known from this text is re-split
     (see :mod:`.subjects`).

The guards trade recall for precision on purpose: a missed fact costs a
lookup in the raw memories, a wrong "current value" misleads the agent.
``extract`` returns assertions; ``extract_changes`` adds retractions and
previous values.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from typing import Iterable

from mark.intelligence.fact_extraction import negation, personal
from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, valid_candidate
from mark.intelligence.fact_extraction.candidates import (for_user, keep_attribute_case, previous_value, retraction,
                                                          triple)
from mark.intelligence.fact_extraction.clauses import carried_subject, is_continuation, split_clauses
from mark.intelligence.fact_extraction.config_lists import config_records
from mark.intelligence.fact_extraction.dates import month_date, split_leading_date
from mark.intelligence.fact_extraction.everyday import sentence_triples
from mark.intelligence.fact_extraction.labels import STATUS_BY_KIND, leading_id
from mark.intelligence.fact_extraction.lists import list_triples
from mark.intelligence.fact_extraction.normalise import build_candidate, peel_value
from mark.intelligence.fact_extraction.provenance import user_text
from mark.intelligence.fact_extraction.packs import ATTRIBUTE_SYNONYMS, BUILTIN_PACKS, PACKS_BY_NAME, RulePack
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
from mark.intelligence.fact_extraction.subjects import defined_names, split_known, subject_key

#: Non-regex features that can be named in ``packs`` like a pack.
FEATURES = frozenset({"structured", "steps"})
_WORD = re.compile(r"[\w'’-]+")
_PREFERS = re.compile(r"\bprefers?\b", re.IGNORECASE)
# Structured keys are folded only for where a config lives ("config_path" -> "config file");
# other keys keep their own spelling ("db engine").
_CONFIG_SYNONYMS = {key: value for key, value in ATTRIBUTE_SYNONYMS.items() if value == "config file"}
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
        self._compiled = [(pack, frozenset(pack.trigger_words), pack.compiled_trigger(), pack.compiled())
                          for pack in self.packs]

    def extract(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Facts stated in *text*, in text order; ``valid_from`` from inline dates only."""
        return [c for c in self.extract_changes(text, occurred_at=occurred_at) if not (c.retract or c.previous)]

    def extract_changes(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Like :meth:`extract`, plus retractions and the earlier values of stated changes."""
        found: list[FactCandidate] = []
        seen: set[tuple[str, str, str, datetime | None, bool, bool]] = set()
        text = user_text(text)  # chat turns: the user's words only, never the assistant's or a tool's
        known = defined_names(text)

        def add(candidate: FactCandidate | None) -> None:
            # Validate before the user subject is filled in: it may be a word the
            # hygiene rules reject elsewhere ("me").
            if candidate is None or not valid_candidate(candidate):
                return
            candidate = for_user(candidate, self.user_subject)
            if candidate is None:
                return
            candidate = split_known(candidate, known, self.user_subject)
            key = (candidate.subject.lower(), candidate.attribute.lower(), candidate.value.lower(),
                   candidate.valid_from, candidate.retract, candidate.previous)
            if key not in seen:
                seen.add(key)
                found.append(candidate)
                known.add(subject_key(candidate.subject))

        remaining = text or ""
        if "structured" in self.features:
            configs, remaining = config_records(remaining)
            for subject, attribute, value in configs:
                add(keep_attribute_case(triple(subject, attribute, value, occurred_at, "structured"), attribute))
        if "structured" in self.features or "steps" in self.features:
            records, remaining = extract_records(remaining)
            for subject, attribute, value in records:
                is_step = attribute.startswith("step ")
                if ("steps" if is_step else "structured") in self.features:
                    attribute = attribute if is_step else _CONFIG_SYNONYMS.get(attribute, attribute)
                    add(triple(subject, attribute, value, occurred_at, "steps" if is_step else "structured"))
        for sentence in iter_sentences(remaining, occurred_at):
            procedures = [c.subject for c in found if c.attribute.startswith("step ")]
            for candidate in self._sentence_candidates(sentence, [c.subject for c in found], procedures):
                add(candidate)
        return found

    # ── one sentence ─────────────────────────────────────────────────────────

    def _sentence_candidates(self, sent: Sentence, earlier: list[str],
                             procedures: list[str] | None = None) -> list[FactCandidate]:
        text, scheduled = split_leading_date(sent.text)
        when = scheduled or sent.when
        if not text or is_question(text) or CONDITIONAL.search(text):
            return []
        out: list[FactCandidate | None] = []
        status = next((STATUS_BY_KIND[k] for k in sent.kinds if k in STATUS_BY_KIND), None)
        ticket = sent.label_id or (leading_id(sent.labelled_body) if status and sent.labelled_body else None)
        if status and ticket:
            out.append(triple(ticket, "status", status, when, "labels"))
        if "steps" in self.features and (step := step_sentence(text)) is not None:
            out.append(triple(*step, when, "steps"))
        text, corrected = negation.strip_correction(text)
        text = personal.rewrite_first_person(text)
        items = list_triples(text.rstrip())
        whole = None if items else sentence_triples(text.rstrip(), procedures or [])
        if items:
            # "I speak X, Y and Z": one multi-valued value per item, no clause splitting.
            out.extend(triple(*item, when, "lists", multi=True) for item in items)
            text = ""
        elif whole:
            # "Kemi changed her number, the new one is N": one fact across two clauses.
            out.extend(triple(*item, when, "everyday") for item in whole)
            text = ""
        carried: str | None = None
        previous: list[FactCandidate] = []
        for raw in split_clauses(text):
            clause = LEADING_FILLER.sub("", raw.strip())
            denied = negation.denied_value(clause)
            if denied is not None:
                # "X runs on port 9443, not 9444": withdraw 9444 for the same slot.
                out.extend(retraction(c, peel_value(clean_text(denied))[0]) for c in previous[:1])
                continue
            clause = personal.resolve_pronoun(clause, personal.single_person(earlier + [c.subject for c in out if c]))
            continuation = is_continuation(clause)
            if continuation and carried:
                clause = f"{carried} {clause}"
            clause = personal.rewrite_diet(personal.conjugate_user_clause(clause))
            apposed, clause = personal.split_apposition(clause)
            rather = None
            if not _PREFERS.search(clause):  # "prefers tea rather than coffee" keeps its comparison
                clause, rather = negation.split_rather(clause)
            candidates = self._clause_candidates(clause, when, sent.record_subject, dated=scheduled is not None)
            if apposed is not None:
                candidates.insert(0, triple(*apposed, when, "personal_relation"))
            if rather and candidates and candidates[-1] and not candidates[-1].retract:
                candidates.append(retraction(candidates[-1], peel_value(clean_text(rather))[0]))
            candidates = [c for c in candidates if c is not None]
            if not continuation:
                carried = carried_subject(clause, candidates)
            previous = [c for c in candidates if not (c.retract or c.previous)] or previous
            out.extend(candidates)
        tags = tuple(sent.kinds)
        return [replace(c, correction=corrected and not (c.retract or c.previous), tags=tags or c.tags)
                for c in out if c is not None]

    def _clause_candidates(self, clause: str, when: datetime | None, record: str | None, *,
                           dated: bool = False) -> list[FactCandidate]:
        clause = LEADING_FILLER.sub("", clause.strip()).rstrip(".!")
        if not clause or INTENT.search(clause):
            return []
        clause = negation.rewrite_stopped(clause)
        retract = historical = False
        if NEGATION.search(clause):
            positive = negation.positive_form(clause)
            if positive is None:
                return []
            historical = negation.is_cessation(clause)
            clause, retract = positive, True
        inline = INLINE_DATE.search(clause)
        valid_from = parse_stamp(inline.group(1), None) if inline else (month_date(clause, when) or when)
        dated = dated or inline is not None
        out: list[FactCandidate] = []
        claimed: set[tuple[str, str]] = set()
        words = set(_WORD.findall(clause.lower()))
        for pack, trigger_words, trigger, patterns in self._compiled:
            # Cheap pre-checks: skip packs none of whose patterns can match.
            if (trigger_words and trigger_words.isdisjoint(words)) or (trigger and not trigger.search(clause)):
                continue
            if pack.needs_date and not dated:
                continue
            for pattern in patterns:
                match = pattern.match(clause)
                if match is None:
                    continue
                groups = match.groupdict()
                candidate = build_candidate(pack, groups, valid_from)
                if candidate is None:
                    continue
                # Earlier (more specific) packs win for the same subject/attribute.
                slot = (candidate.subject.lower(), candidate.attribute.lower())
                if slot not in claimed:
                    claimed.add(slot)
                    before = None if retract else previous_value(candidate, groups.get("old"))
                    out.extend(c for c in (before, candidate) if c is not None)
        if not out and record and (field := _ORPHAN_FIELD.match(clause)) and usable_key(field.group("key")):
            # "HAL-1: plugin hook bug. status: resolved": the line's id label is the subject.
            orphan = triple(record, attribute_key(field.group("key")), clean_text(field.group("value")),
                            valid_from, "key_value")
            out = [orphan] if orphan is not None else []
        if retract:
            return [replace(c, retract=True, historical=historical) for c in out]
        return out
