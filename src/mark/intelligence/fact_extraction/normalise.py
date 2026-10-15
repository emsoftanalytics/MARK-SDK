"""Turn one rule-pack regex match into a clean, validated :class:`FactCandidate`.

Normalisation steps, in order: drop redacted matches; strip articles and
quotes; read an attribute noun written next to the value ("port 4444", "the
Data team") or after the subject ("svc-a port"); drop kind nouns around
identifiers ("service atlas-core", "the docs-site service"); pick the
attribute (pack fixed > text > verb fallback > product type for a change);
fold synonyms; normalise the value case; swap subject and value for
assignments to an identifier. Anything that still looks wrong is rejected
by :func:`~.base.valid_candidate`.
"""
from __future__ import annotations

import re
from datetime import datetime

from mark.intelligence.fact_extraction.base import FactCandidate, clean_text, is_redacted, valid_candidate
from mark.intelligence.fact_extraction.personal import restore
from mark.intelligence.fact_extraction.rules import ATTRIBUTE_SYNONYMS, ATTRIBUTE_WORDS, RulePack
from mark.intelligence.fact_extraction.vocab import change_attribute

# Leading kind nouns dropped from subjects when an identifier follows
# ("service atlas-core" -> "atlas-core", "TICKET OPS-12" -> "OPS-12").
_KIND_PREFIX = re.compile(
    r"^(?:the\s+)?(?:service|server|ticket|issue|bug|project|user|employee|customer|host|app|application"
    r"|system|repo|repository|job|task|incident|cluster|node|account|component|module)\s+(?=\S*[\d_.A-Z-])",
    re.IGNORECASE)
# Trailing kind nouns after an identifier ("the docs-site service" -> "docs-site").
_KIND_SUFFIX = re.compile(
    r"^(\S*[\d_.-]\S*)\s+(?:service|server|repo|repository|api|app|cluster|project|job|component|module)$",
    re.IGNORECASE)
# One identifier-like token with letters and digits ("OPS-17", "web2").
ID_TOKEN = re.compile(r"^(?=\S*\d)(?=\S*[^\W\d_])\S+$")
_ARTICLE = re.compile(r"^(?:the|a|an|our|my|your|their)\s+", re.IGNORECASE)
_TRAILING_FILLER = re.compile(
    r"\s+(?:now|today|yesterday|tonight|instead|permanently|temporarily|again|as\s+well|too"
    r"|immediately|effective\s+immediately|in\s+production|last\s+(?:night|week|year|spring|summer|autumn|winter)"
    r"|this\s+(?:morning|week|year))$",
    re.IGNORECASE)
# Attribute nouns as word tuples: peeling compares the first/last 1-3 words
# with a set, which is much cheaper than a regex alternation over all nouns.
_NOUNS = frozenset(tuple(w.lower().split()) for w in ATTRIBUTE_WORDS)
_LONGEST = max(len(n) for n in _NOUNS)
# "Maya's office is in Oslo": a preposition before a place name is not part of it.
_PLACE_PREPOSITION = re.compile(r"^(?:in|at)\s+(?=(?-i:[A-Z]))", re.IGNORECASE)
# An infinitive is a plan, never a value ("My plan is to finish by Friday").
_INFINITIVE = re.compile(r"^to\s+[a-z]", re.IGNORECASE)
_SPELLING = (("favorite", "favourite"), ("color", "colour"))


def build_candidate(pack: RulePack, groups: dict[str, str | None], valid_from: datetime | None) -> FactCandidate | None:
    """Normalise one regex match into a validated candidate (or None)."""
    if is_redacted(groups.get("subject") or "") or is_redacted(groups.get("value") or ""):
        return None
    subject = _ARTICLE.sub("", clean_text(groups.get("subject") or ""))
    value = restore(clean_text(groups.get("value") or ""))
    attribute = clean_text(groups.get("attribute") or groups.get("attribute_free") or "").lower()
    # Attribute nouns written next to the value ("port 4444", "the Data team").
    value, value_attr = peel_value(value) if pack.attribute_from_value else (value, "")
    old = clean_text(groups.get("old") or "")
    old_attr = peel_value(old)[1]
    # "svc-a port changed ..." names the attribute after the subject; packs whose
    # verb fixes the attribute keep the subject whole ("Ana leads the Mobile squad").
    peel_subject = pack.attribute_from_value and len(subject.split()) > 1
    subject_match = _peel_suffix(subject, possessive=True) if peel_subject else None
    subject_attr = ""
    if subject_match:
        subject, subject_attr = subject_match
    subject = re.sub(r"['’]s$", "", _KIND_PREFIX.sub("", subject).strip())
    subject = _KIND_SUFFIX.sub(r"\1", subject)
    verb = re.sub(r"\s+", " ", (groups.get("verb") or "").lower())
    attribute = (pack.attribute or attribute or value_attr or old_attr or subject_attr
                 or pack.verb_attributes.get(verb) or pack.verb_attributes.get("", ""))
    if not attribute and old:
        # "switched from npm to pnpm": the products say what changed.
        attribute = change_attribute(old, value)
    attribute = re.sub(r"\s+", " ", attribute)
    for us, uk in _SPELLING:
        attribute = re.sub(rf"\b{us}\b", uk, attribute)
    attribute = ATTRIBUTE_SYNONYMS.get(attribute, attribute)
    value = _TRAILING_FILLER.sub("", _ARTICLE.sub("", value)).strip()
    if pack.name == "key_value" or pack.name.startswith("personal"):
        value = _PLACE_PREPOSITION.sub("", value)
    if _INFINITIVE.match(value):
        return None
    if pack.value_case == "lower":
        value = value.lower()
    value = pack.value_map.get(value.lower(), value)
    if pack.swap_when_value_is_id:
        # "Bo Kim is assigned to OPS-17": the identifier is the thing assigned.
        bare_value = _KIND_PREFIX.sub("", value).strip()
        if ID_TOKEN.match(bare_value) and not ID_TOKEN.match(subject):
            subject, value = bare_value, subject
    candidate = FactCandidate(subject=subject, attribute=attribute, value=value,
                              valid_from=valid_from, confidence=pack.confidence, rule=pack.name)
    return candidate if valid_candidate(candidate) else None


def peel_value(value: str) -> tuple[str, str]:
    """Split an attribute noun off a value: 'port 4444' -> ('4444', 'port'), 'the Data team' -> ('Data', 'team')."""
    words = value.split()
    start = 1 if words and words[0].lower() == "the" else 0
    for size in range(min(_LONGEST, len(words) - start - 1), 0, -1):   # prefix, longest noun first
        head = tuple(w.lower() for w in words[start:start + size])
        if head in _NOUNS and words[start + size].lower() != "of":
            return clean_text(" ".join(words[start + size:])), " ".join(head)
    return _peel_suffix(value) or (value, "")


def _peel_suffix(text: str, possessive: bool = False) -> tuple[str, str] | None:
    """'<rest> <noun>' -> (rest, noun) for a trailing attribute noun; None if there is none."""
    words = text.split()
    for size in range(min(_LONGEST, len(words) - 1), 0, -1):
        tail = tuple(w.lower() for w in words[-size:])
        if tail in _NOUNS:
            rest = " ".join(words[:-size])
            if possessive:
                rest = re.sub(r"['’]s$", "", rest)
            return (clean_text(rest) if not possessive else rest), " ".join(tail)
    return None
