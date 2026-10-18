"""Known subjects: keep "<known subject> <rest>" as subject + attribute.

Patterns take the attribute noun from the end of a long subject:
"The shorty plugin entry-point group is shorty.plugins" first splits as
``shorty plugin entry-point.group``. When the sentence starts with a subject
MARK already knows (a fact subject in the store, or a project/service named
earlier in the same text), the known name is the subject and the rest of
the words belong to the attribute: ``shorty.plugin entry-point group``.

Precision guards: the rest must be lower-case words (so "Mei Lin" is never
split even when "Mei" is known), the user subject is never split, and the
new attribute stays short (at most five words).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import replace

from mark.intelligence.fact_extraction.base import FactCandidate
from mark.security.lazy_regex import lazy_compile

_MAX_ATTRIBUTE_WORDS = 5
# "shorty is a URL shortener": a lower-case identifier defined at the start of a sentence.
_DEFINED = lazy_compile(r"(?:^|(?<=[.!?]\s)|(?<=\n))\s*(?P<name>[a-z][\w.-]*)\s+(?:is|are)\s+(?:a|an|the|our|my)\s",
                      re.MULTILINE)
_NOT_NAMES = frozenset("it this that there here what which who everything nothing something everyone".split())


def subject_key(text: str) -> str:
    """Normalised subject key, the same folding the store applies ("Atlas-Core" == "atlas core")."""
    return re.sub(r"[\s_\-]+", " ", str(text).lower().replace("’", "'")).strip(" \"'()[]{}").rstrip(" .,;:!?")


def defined_names(text: str) -> set[str]:
    """Keys of names the text introduces ("shorty is a URL shortener" -> {"shorty"})."""
    if " is " not in (text or "") and " are " not in (text or ""):
        return set()
    return {subject_key(m.group("name")) for m in _DEFINED.finditer(text or "")
            if m.group("name").lower() not in _NOT_NAMES}


def split_known(candidate: FactCandidate, known: Iterable[str] | Callable[[str], bool],
                user_subject: str = "user") -> FactCandidate:
    """Re-split ``candidate`` on the longest known subject its subject starts with (else unchanged).

    *known* is a collection of subject keys, or a predicate called with each
    possible prefix (longest first) so a store can answer with one indexed
    lookup instead of listing every subject.
    """
    words = candidate.subject.split()
    if len(words) < 2 or subject_key(candidate.subject) == subject_key(user_subject):
        return candidate
    if callable(known):
        is_known = known
    else:
        keys = known if isinstance(known, (set, frozenset)) else set(known)
        is_known = lambda prefix: subject_key(prefix) in keys  # noqa: E731
    for size in range(len(words) - 1, 0, -1):
        rest = words[size:]
        # Checked first (cheap): the rest must be lower-case words, never part of a name.
        if not all(re.match(r"^[a-z][\w.-]*$", word) for word in rest):
            return candidate
        attribute = " ".join([*rest, candidate.attribute])
        if len(attribute.split()) > _MAX_ATTRIBUTE_WORDS:
            return candidate  # shorter prefixes only make the attribute longer
        if is_known(" ".join(words[:size])):
            return replace(candidate, subject=" ".join(words[:size]), attribute=attribute)
    return candidate
