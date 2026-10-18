"""Current vs stale in free text: which retrieved memories supersede which.

People record changes in prose ("I moved from Lisbon to Porto", "Correction:
my favourite city is actually Lisbon, not Berlin", "I no longer drink
coffee", "switched storage from JSON to SQLite"). Relevance ranking often
puts the older statement first because it states the old value plainly.

A newer memory *supersedes* an older one when it reads as an update (it
uses change wording) and is about the same thing: it names a replaced value
that occurs in the older memory ("from Lisbon", "not Berlin", "no longer
drink coffee"), or the two share most of their content words, including at
least one that is not a name (the same subject alone is not the same
attribute). Memories
explicitly retired by the API (tag ``mark:superseded`` / ``mark:resolved``,
metadata ``superseded_by`` / ``resolved_by``) are stale as well.

:func:`order_by_supersession` only swaps such pairs: for present-tense
queries the newer memory moves directly ahead of the one it supersedes;
for "original/previously" queries the older one moves ahead. Every other
result keeps its rank, and scores stay non-increasing.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache

from mark.intelligence.query_anchors import content_entities, query_anchors
from mark.intelligence.query_understanding import (
    TemporalIntent,
    _anchor_norm,
    _contains_anchor,
    _content_terms,
    _stem,
)
from mark.intelligence.temporal_order import event_time
from mark.types import MemoryFragment

# Change wording: a memory using one of these reads as an update of an earlier state.
_UPDATE_RE = re.compile(
    r"\b(?:moved|switched|migrated|changed|replaced|relocated|transferred|renamed|upgraded|downgraded"
    r"|updated|correction|corrected|actually|no\s+longer|anymore|any\s+more|used\s+to|now|nowadays|these\s+days|instead"
    r"|resolved|removed|deprecated|superseded|retired|reverted)\b"
    # Status labels count only as labels: "done" / "fixed" inside prose or
    # code ("(title, done, priority)") are not updates.
    r"|\b(?:DONE|FIXED|RESOLVED|UPDATE|CORRECTION)\s*:",
    re.I,
)
# Phrases naming the value that was replaced.
_REPLACED_RES = (
    re.compile(r"\bfrom\s+(.{2,40}?)\s+(?:to|into|with|for)\b", re.I),
    re.compile(r"\b(?:not|instead\s+of|rather\s+than|replac\w*|no\s+longer)\s+(.{2,40}?)(?:[,;.!?)]|$|\s+(?:but|and|any\s?more)\b)", re.I),
    re.compile(r"\bused\s+to\s+(.{2,40}?)(?:[,;.!?)]|$|\s+but\b)", re.I),
)
# Words of change, not of topic: excluded when comparing what two memories are about.
_CHANGE_STEMS = set(_content_terms(
    "moved switched migrated changed replaced relocated transferred renamed upgraded downgraded updated "
    "correction corrected actually longer instead resolved fixed done removed deprecated superseded "
    "retired dropped reverted stopped change decision status todo note",
))
# Date stamps say when, not what: "[2026-03-31]" must not make two records look alike.
_ISO_STAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?\S*)?")
_RETIRED_TAGS = {"mark:superseded", "mark:resolved"}
_RETIRED_META = ("superseded_by", "resolved_by")
_MIN_SHARED = 2
_PROFILE_CHARS = 1500
_MIN_SHARED_RATIO = 0.4


def is_update_statement(content: str) -> bool:
    """True when a memory is worded as a change of an earlier state."""
    return _UPDATE_RE.search(content) is not None


def replaced_values(content: str) -> list[str]:
    """Normalised phrases a memory says were replaced ("Lisbon" in "moved from Lisbon to Porto")."""
    values: list[str] = []
    for pattern in _REPLACED_RES:
        for match in pattern.finditer(content):
            norm = _anchor_norm(match.group(1))
            if norm and norm not in values and len(norm) >= 2:
                values.append(norm)
    return values


def is_retired(fragment: MemoryFragment) -> bool:
    """True when the API marked a memory superseded or resolved (tag or metadata)."""
    if _RETIRED_TAGS.intersection(fragment.tags or ()):
        return True
    metadata = fragment.metadata or {}
    return any(metadata.get(key) for key in _RETIRED_META)


def is_superseded(fragment: MemoryFragment) -> bool:
    """True when a newer statement replaced this one; a resolved bug or TODO is not "replaced".

    (Resolving through the CLI tags both ``mark:resolved`` and ``mark:superseded``.)
    """
    tags, metadata = fragment.tags or (), fragment.metadata or {}
    if "mark:resolved" in tags or metadata.get("resolved_by") or metadata.get("lifecycle") == "resolved":
        return False
    return "mark:superseded" in tags or bool(metadata.get("superseded_by"))


def _topic(content: str) -> set[str]:
    return set(_content_terms(_ISO_STAMP_RE.sub(" ", content))) - _CHANGE_STEMS


def _stored_at(fragment: MemoryFragment) -> datetime:
    stamp = fragment.created_at
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _is_newer(new: MemoryFragment, old: MemoryFragment) -> bool:
    """Event time first; equal event times fall back to the order of storage."""
    a, b = event_time(new), event_time(old)
    if a != b:
        return a > b
    return _stored_at(new) > _stored_at(old)


class _Profile:
    """What supersession needs from one memory, computed once per retrieval."""
    __slots__ = ("fragment", "norm", "topic", "entities", "entity_stems", "update", "replaced")

    def __init__(self, fragment: MemoryFragment) -> None:
        self.fragment = fragment
        (self.norm, self.topic, self.entities, self.entity_stems,
         self.update, self.replaced) = _features(fragment.content[:_PROFILE_CHARS])


@lru_cache(maxsize=4096)
def _features(text: str) -> tuple:
    """Text features of a memory head, cached: the same memories recur across questions.

    State changes are short statements, so only the head of a long memory
    (a document) is compared, which keeps this stage cheap.
    """
    entities = frozenset(content_entities(_ISO_STAMP_RE.sub(" ", text)))
    update = is_update_statement(text)
    return (
        _anchor_norm(text),
        frozenset(_topic(text)),
        entities,
        frozenset(_stem(word) for entity in entities for word in entity.split()),
        update,
        tuple(replaced_values(text)) if update else (),
    )


class _Query:
    """What supersession needs from the query: its anchors and attribute words."""
    __slots__ = ("norm", "anchors", "terms")

    def __init__(self, query: str) -> None:
        self.norm = _anchor_norm(query)
        self.anchors = query_anchors(query)
        anchor_stems = {_stem(word) for anchor in self.anchors for word in anchor.split()}
        self.terms = {t for t in _content_terms(query) if t not in anchor_stems}


def _supersedes(new: _Profile, old: _Profile, query: _Query) -> bool:
    if new.fragment.id == old.fragment.id or not new.update or not _is_newer(new.fragment, old.fragment):
        return False
    # Chunks of one document are parts of one text, never updates of each other
    # ("change to the plan" in one room's notes does not retire another room's).
    doc = (new.fragment.metadata or {}).get("document_id")
    if doc and doc == (old.fragment.metadata or {}).get("document_id"):
        return False
    # Two records naming different subjects (atlas-gate vs atlas-core) are
    # never about the same thing, however alike their wording; and a query
    # about a subject only compares records about that subject.
    if new.entities and old.entities and not new.entities & old.entities:
        return False
    if query.anchors and not all(_contains_anchor(p.norm, a) for p in (new, old) for a in query.anchors):
        return False
    # The query's words that the old record answers with must be about the
    # same attribute: an owner change does not supersede a region record.
    asked = query.terms & old.topic
    for value in new.replaced:
        if _contains_anchor(old.norm, value) and not (query.norm and _contains_anchor(query.norm, value)) \
                and (not asked or _near(old.norm, value, asked)):
            return True
    if asked and not asked & new.topic:
        return False
    shared = new.topic & old.topic
    smaller = min(len(new.topic), len(old.topic)) or 1
    # Sharing only the subject's name ("atlas-core", "Jae Brandt") is not
    # enough: a datastore change does not supersede a port change.
    about_same = bool(shared - new.entity_stems - old.entity_stems)
    return about_same and len(shared) >= _MIN_SHARED and len(shared) / smaller >= _MIN_SHARED_RATIO


def _near(norm: str, value: str, terms: set[str], window: int = 5) -> bool:
    """True when *value* occurs within *window* tokens of one of the query's words."""
    tokens = norm.split()
    value_tokens = value.split()
    stems = [_stem(token) for token in tokens]
    for i in range(len(tokens) - len(value_tokens) + 1):
        if tokens[i:i + len(value_tokens)] == value_tokens:
            lo, hi = max(0, i - window), min(len(tokens), i + len(value_tokens) + window)
            if any(stem in terms for stem in stems[lo:hi]):
                return True
    return False


def supersedes(new: MemoryFragment, old: MemoryFragment, query: str = "") -> bool:
    """Whether *new* reads as an update of *old* (see module docstring).

    With a *query*: a replaced value only counts when the query does not
    name it (asking about "rent in Lisbon" is not answered by "I moved from
    Lisbon") and when it sits next to the attribute asked about; records
    must name the query's subject, and the update must speak to the
    attribute the old record answers (an owner change does not supersede a
    region record).
    """
    return _supersedes(_Profile(new), _Profile(old), _Query(query))


def order_by_supersession(
    query: str,
    intent: TemporalIntent,
    fragments: list[MemoryFragment],
    scores: list[float],
    relevances: list[float],
    outdated: set[str] | None = None,
) -> tuple[list[MemoryFragment], list[float], list[float]]:
    """Swap superseding / superseded pairs for the query's point in time (see module docstring).

    ``outdated`` (optional) collects the ids of superseded memories for
    present-tense questions, so renderers can mark them.
    """
    if len(fragments) < 2:
        return fragments, scores, relevances
    asked = _Query(query)
    profiles = [_Profile(f) for f in fragments]
    order = list(range(len(fragments)))
    pairs = [
        (j, i) for j in order if profiles[j].update for i in order
        if i != j and _supersedes(profiles[j], profiles[i], asked)
    ]
    if not pairs:
        return fragments, scores, relevances
    original = intent == TemporalIntent.ORIGINAL
    if outdated is not None and not original:
        outdated.update(fragments[old].id for _, old in pairs)
    for new, old in pairs:
        first, second = (old, new) if original else (new, old)
        if order.index(first) > order.index(second):
            order.remove(first)
            order.insert(order.index(second), first)
    return _apply(order, fragments, scores, relevances)


def order_by_fact_agreement(
    fragments: list[MemoryFragment],
    scores: list[float],
    relevances: list[float],
    current_values: list[str],
    stale_values: list[str],
    outdated: set[str] | None = None,
) -> tuple[list[MemoryFragment], list[float], list[float]]:
    """Move memories stating a current fact value ahead of ones stating only an old value.

    ``current_values`` / ``stale_values`` are the current and earlier values
    of the facts the query asks about ("Porto" / "Lisbon"). Memories that
    state only an earlier value are added to ``outdated`` when given.
    """
    current = [_anchor_norm(v) for v in current_values if _anchor_norm(v)]
    stale = [_anchor_norm(v) for v in stale_values if _anchor_norm(v) and _anchor_norm(v) not in current]
    if not current or not stale:
        return fragments, scores, relevances
    norms = [_anchor_norm(f.content) for f in fragments]
    agrees = [any(_contains_anchor(n, v) for v in current) for n in norms]
    is_stale = [not agrees[i] and any(_contains_anchor(n, v) for v in stale) for i, n in enumerate(norms)]
    if outdated is not None:
        # Outdated only by a value the current one replaced, not one it still
        # contains: "csv" -> "csv, json" grew a list, "csv" is still true.
        items = {_anchor_norm(item) for value in current_values for item in re.split(r"[,;]|\band\b", value)}
        replaced = [v for v in stale if v not in items]
        outdated.update(f.id for f, n, flag in zip(fragments, norms, is_stale)
                        if flag and any(_contains_anchor(n, v) for v in replaced))
    order = list(range(len(fragments)))
    first_stale = next((i for i in order if is_stale[i]), None)
    if first_stale is None:
        return fragments, scores, relevances
    movers = [i for i in order if agrees[i] and i > first_stale]
    if not movers:
        return fragments, scores, relevances
    rest = [i for i in order if i not in movers]
    at = rest.index(first_stale)
    return _apply(rest[:at] + movers + rest[at:], fragments, scores, relevances)


def _apply(order, fragments, scores, relevances):
    """Permute fragments and relevances; scores are re-assigned so they stay non-increasing."""
    ranked = sorted(scores, reverse=True)
    return [fragments[i] for i in order], ranked, [relevances[i] for i in order]
