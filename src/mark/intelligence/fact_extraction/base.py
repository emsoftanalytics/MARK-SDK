"""Core types for automatic fact extraction: candidates and the extractor protocol.

An extractor reads the text of one stored memory and proposes
``(subject, attribute, value)`` triples. It never writes anything itself:
the extraction service decides *when* to run it (inline or in the durable
background queue) and turns candidates into bi-temporal facts linked to the
memory they came from.

Invariants shared by every extractor:
  - ``extract`` is pure with respect to MARK state (no store access), so it
    can run on any thread and be retried safely;
  - it returns candidates in text order; when one text changes the same
    (subject, attribute) twice, the later candidate is the newer value;
  - ``valid_from`` is only set when the text itself says when the change
    happened; otherwise the service uses the memory's event time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FactCandidate:
    """One proposed fact: ``subject.attribute = value`` from ``valid_from`` on."""
    subject: str
    attribute: str
    value: str
    valid_from: datetime | None = None
    confidence: float = 1.0
    source_fragment_id: str | None = None
    #: Which rule pack (or extractor) produced the candidate, for auditing.
    rule: str = ""

    def with_source(self, fragment_id: str) -> "FactCandidate":
        """Copy linked to the memory it was extracted from."""
        return replace(self, source_fragment_id=fragment_id)


@runtime_checkable
class FactExtractor(Protocol):
    """Anything that turns memory text into fact candidates.

    Optional attributes the service reads with ``getattr``:
      ``name``        stable identifier, recorded on facts and queue jobs;
      ``background``  True for slow extractors (e.g. an LLM): writes only
                      enqueue a job and a worker thread runs the extractor.
    """

    def extract(self, text: str, *, occurred_at: datetime | None = None) -> list[FactCandidate]:
        """Return the facts *text* states, in text order."""
        ...


def extractor_name(extractor: object) -> str:
    """Stable name used to key queue jobs and audit metadata."""
    return str(getattr(extractor, "name", "") or type(extractor).__name__)


def runs_in_background(extractor: object) -> bool:
    """True when writes should enqueue jobs instead of extracting inline."""
    return bool(getattr(extractor, "background", False))


# ── candidate hygiene shared by the pattern and LLM extractors ───────────────

#: Words that are never a useful subject on their own (they refer elsewhere).
_PRONOUNS = frozenset(
    "it its he she they them we i you this that these those which who there here someone everyone".split()
)
_MAX_SUBJECT_CHARS = 80
_MAX_VALUE_CHARS = 120
_MAX_VALUE_WORDS = 10
_EDGE_PUNCT = " \t\"'`“”‘’*_.,;:!?()[]{}"


# Placeholders left by redaction ("[EMAIL]", "[REDACTED]"): a fact must never
# record a placeholder as if it were the real value.
_REDACTED = re.compile(r"\[(?:REDACTED[A-Z_]*|EMAIL|PHONE|CC|SSN)\]")


def is_redacted(text: str) -> bool:
    """True when *text* contains a redaction placeholder."""
    return bool(_REDACTED.search(text or ""))


def clean_text(value: str) -> str:
    """Collapse whitespace and strip quotes/punctuation from both ends."""
    return re.sub(r"\s+", " ", str(value)).strip(_EDGE_PUNCT).strip()


def valid_candidate(candidate: FactCandidate) -> bool:
    """Reject candidates that would make poor fact keys or values.

    Subjects must name something (not a pronoun) and both keys must contain
    letters or digits; values must be short — long values are almost always
    a whole clause captured by mistake.
    """
    subject, attribute, value = candidate.subject, candidate.attribute, candidate.value
    if not (subject and attribute and value):
        return False
    if subject.lower() in _PRONOUNS or len(subject) > _MAX_SUBJECT_CHARS:
        return False
    if not re.search(r"[^\W_]", subject) or not re.search(r"[^\W_]", attribute):
        return False
    if len(value) > _MAX_VALUE_CHARS or len(value.split()) > _MAX_VALUE_WORDS:
        return False
    return bool(re.search(r"[^\W_]", value))
