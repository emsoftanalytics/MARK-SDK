"""Questions that ask for a kind of note: "What TODOs are still open?", "open bugs", "which decisions".

Such a question is a listing: its answer is every memory of that kind (only
the open ones when it says "open", "still open", "unresolved",
"outstanding", "pending", "left"), not the memories that share its words.
Lexically, "open" matched "opens the TOML file" in a fix while the open
TODOs, which never say "open", ranked lower.

:func:`asked_kind` reads the ask; :func:`kind_candidates` fetches memories
of the kind so they join the candidate pool; :func:`apply_kind` gives them
full evidence and halves every other candidate (soft: nothing is dropped).
"""
from __future__ import annotations

import re
from functools import cache
from typing import Any, NamedTuple

from mark.kinds import is_open, kind_of, kind_tag

# Shown score factor for candidates that are not of the asked kind (or not open when asked).
OFF_KIND = 0.5
# Word (hyphens removed) prefix -> kind; tasks and action items are TODOs.
_KIND_PREFIXES = (("todo", "todo"), ("task", "todo"), ("action item", "todo"), ("bug", "bug"),
                  ("decision", "decision"), ("fix", "fix"), ("snippet", "snippet"), ("preference", "preference"))
_FETCH_LIMIT = 50
_HINTS = ("todo", "to-do", "task", "action item", "bug", "decision", "fix", "snippet", "preference")


class KindAsk(NamedTuple):
    kind: str
    open_only: bool


@cache
def _ask_re() -> re.Pattern[str]:
    kinds = r"(?P<kind>to-?dos?|tasks?|action items?|bugs?|decisions?|fix(?:es)?|snippets?|preferences?)"
    state = r"(?:open|unresolved|outstanding|pending|remaining|left|not\s+(?:done|fixed|resolved))"
    return re.compile(
        rf"\b(?P<state1>{state})\s+{kinds}\b"
        rf"|\b{kinds.replace('kind', 'kind2')}\s+(?:are|is|do\s+we\s+have|have\s+we\s+got)?\s*(?:still\s+)?(?P<state2>{state})\b"
        rf"|^\s*(?:what|which|list|show)(?:\s+(?:are|were))?(?:\s+(?:the|all|my|our))?\s+"
        rf"{kinds.replace('kind', 'kind3')}(?:\s+(?:do|did)\s+(?:we|i)\s+(?:have|make|take))?\s*\??\s*$",
        re.I,
    )


def asked_kind(query: str) -> KindAsk | None:
    """The kind (and whether only open ones) a listing question asks for, or None."""
    low = (query or "").lower()
    if not any(word in low for word in _HINTS):
        return None  # most questions: no regex (its first compile costs ~30 ms)
    match = _ask_re().search(query or "")
    if match is None:
        return None
    word = (match.group("kind") or match.group("kind2") or match.group("kind3")).lower().replace("-", "")
    kind = next((kind for prefix, kind in _KIND_PREFIXES if word.startswith(prefix)), None)
    if kind is None:
        return None
    open_only = bool(match.group("state1") or match.group("state2"))
    return KindAsk(kind, open_only)


def _matches(fragment: Any, ask: KindAsk) -> bool:
    return kind_of(fragment) == ask.kind and (not ask.open_only or is_open(fragment.tags or ()))


def kind_candidates(store: Any, agent_id: str, allowed_ids: set[str], ask: KindAsk | None) -> dict[str, Any]:
    """Memories of the asked kind (open ones when asked) from the allowed scope, by id."""
    list_by_agent = getattr(store, "list_by_agent", None)
    if ask is None or list_by_agent is None:
        return {}
    try:
        rows = list_by_agent(agent_id, tags=[kind_tag(ask.kind)], limit=_FETCH_LIMIT)
    except TypeError:  # legacy stores without tag filtering
        return {}
    return {f.id: f for f in rows if f.id in allowed_ids and _matches(f, ask)}


def apply_kind(ask: KindAsk, fragments_by_id: dict[str, Any], relevance: dict[str, float],
               seed_scores: dict[str, float]) -> None:
    """Full evidence for memories of the asked kind, ``OFF_KIND`` for the rest (in place)."""
    for fid, fragment in fragments_by_id.items():
        if fid not in seed_scores:
            continue
        if _matches(fragment, ask):
            relevance[fid] = 1.0
            seed_scores[fid] = max(seed_scores[fid], 1.0)
        else:
            relevance[fid] = relevance.get(fid, 0.0) * OFF_KIND
            seed_scores[fid] *= OFF_KIND
