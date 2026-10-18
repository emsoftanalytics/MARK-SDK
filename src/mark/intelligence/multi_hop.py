"""Multi-hop recall: follow entities named in the first results to the next fact.

"Who manages the owner of ledger-sync?" needs two memories: "Tomasz owns
the ledger-sync service" (found by the question's words) and "Priya
manages Tomasz" (which shares no word with the question). Single-shot
retrieval finds only the first.

After ranking, the top results that mention what the question is about
(its anchors: "ledger-sync", "Tomasz") are scanned for *bridge* entities
the question does not mention ("Tomasz", "Priya"). Each bridge is looked up
in the keyword index together with the question's words, and the best
memories that really contain it are placed right behind the memory that led
to them (hops never outrank their source). A second hop repeats this from
the first hop's memories. Bounded and cheap: at most ``_MAX_SOURCES``
sources and ``_MAX_BRIDGES`` bridges per hop, one keyword lookup per bridge,
no model calls, and nothing at all for questions without anchors or whose
best match already contains every word of the question.

The same lookup finds *updates* of a top result that the question's words
miss: "Which city do I live in?" matches "I live in Lisbon" but not "I
moved from Lisbon to Porto"; looking up "Lisbon" finds the move, which
joins the results right behind the memory it supersedes (supersession
ordering then puts it first). See :meth:`_MultiHopMixin._pull_superseders`.

Expects the host class to provide ``_lexical_search`` and ``_get_many``.
"""
from __future__ import annotations

from mark.intelligence.query_anchors import (
    _anchor_norm,
    _contains_anchor,
    anchor_present,
    content_entities,
    gap_anchors,
    query_anchors,
)
from mark.intelligence.query_understanding import _coverage, _relevance
from mark.intelligence.supersession import _near, is_update_statement, supersedes
from mark.types import MemoryFragment

_MAX_SOURCES = 3
_MAX_BRIDGES = 3
_LOOKUP_DEPTH = 8
_PER_BRIDGE = 2
# Share of its source's evidence a hop inherits when competing for a top-k slot,
# and the calibrated evidence below which a direct match counts as weak.
_HOP_EVIDENCE = 0.7
_WEAK_MATCH = 0.25
# A result must carry some query evidence before its entities are followed
# for superseding updates.
_MIN_SOURCE_RELEVANCE = 0.25
# Results whose entities are followed to superseding updates.
_SUPERSEDER_SOURCES = 3

_Ranked = list[tuple[MemoryFragment, float, float]]  # (fragment, score, relevance)


class _MultiHopMixin:
    """Entity-bridged hops and superseding-update lookups (see module docstring)."""

    def _multi_hop(
        self,
        hops: int,
        query: str,
        query_terms: list[str],
        fragments: list[MemoryFragment],
        scores: list[float],
        relevances: list[float],
        agent_id: str,
        allowed_ids: set[str],
        scope_ids: set[str] | None,
        top_k: int,
        cache: dict,
    ) -> tuple[list[MemoryFragment], list[float], list[float]]:
        """Place memories reached through bridge entities behind their sources."""
        anchors = gap_anchors(query)
        if hops <= 0 or not anchors or not fragments:
            return fragments, scores, relevances
        # One memory already contains every word of the question: nothing to bridge.
        if _coverage(query_terms, fragments[0].content) >= 1.0:
            return fragments, scores, relevances
        query_norm = _anchor_norm(query)
        out: _Ranked = list(zip(fragments, scores, relevances))
        # (memory, evidence it carries): a hop inherits a damped share of its source's.
        sources = [(f, r) for f, r in zip(fragments[:_MAX_SOURCES], relevances)
                   if any(anchor_present(_anchor_norm(f.content), a) for a in anchors)]
        seen: set[str] = set()
        for _ in range(hops):
            reached: list[tuple[MemoryFragment, float]] = []
            bridges = 0
            for source, evidence in sources:
                for entity in content_entities(source.content):
                    if bridges >= _MAX_BRIDGES:
                        break
                    if entity in seen or _contains_anchor(query_norm, entity):
                        continue
                    seen.add(entity)
                    bridges += 1
                    inherited = evidence * _HOP_EVIDENCE
                    for candidate in self._lookup(cache, entity, query, query_terms, agent_id, allowed_ids, scope_ids):
                        if _place_after(out, source, candidate, query_terms, top_k, inherited):
                            reached.append((candidate, inherited))
            if not reached:
                break
            sources = reached[:_MAX_SOURCES]
        out = out[:top_k]
        return [f for f, _, _ in out], [s for _, s, _ in out], [r for _, _, r in out]

    def _lookup(self, cache, bridge, query, query_terms, agent_id, allowed_ids, scope_ids) -> list[MemoryFragment]:
        """The best few memories containing *bridge*, those matching the question's words first.

        The keyword query is the bridge alone (a rare term: a cheap index
        lookup); the hits are then ordered by how many of the question's
        words they contain. *cache* lives for one question: hops and
        superseder lookups often follow the same entity.
        """
        if bridge in cache:
            return cache[bridge]
        hits = self._lexical_search(bridge, agent_id, allowed_ids, scope_ids, _LOOKUP_DEPTH)
        fetched = self._get_many([fid for fid, _ in hits])
        found = [fetched[fid] for fid, _ in hits
                 if fid in fetched and _contains_anchor(_anchor_norm(fetched[fid].content), bridge)]
        found.sort(key=lambda f: _coverage(query_terms, f.content), reverse=True)  # stable: BM25 breaks ties
        cache[bridge] = found[:_PER_BRIDGE]
        return cache[bridge]

    def _pull_superseders(
        self,
        query: str,
        query_terms: list[str],
        fragments: list[MemoryFragment],
        scores: list[float],
        relevances: list[float],
        agent_id: str,
        allowed_ids: set[str],
        scope_ids: set[str] | None,
        top_k: int,
        cache: dict,
    ) -> tuple[list[MemoryFragment], list[float], list[float]]:
        """Insert newer updates of the top result that the query's words did not reach.

        Only for questions without anchors: a question naming its subject
        ("atlas-core") already retrieves that subject's updates by name.
        """
        if query_anchors(query):
            return fragments, scores, relevances
        query_norm = _anchor_norm(query)
        out: _Ranked = list(zip(fragments, scores, relevances))
        # The best three: equally strong matches (ties are common) may each have an update.
        for source, relevance in list(zip(fragments, relevances))[:_SUPERSEDER_SOURCES]:
            if relevance < _MIN_SOURCE_RELEVANCE:
                continue
            entities = [e for e in content_entities(source.content) if not _contains_anchor(query_norm, e)]
            # Values next to the asked attribute first: "live in Porto" -> "porto".
            source_norm = _anchor_norm(source.content)
            entities.sort(key=lambda e: not _near(source_norm, e, set(query_terms), window=2))
            for entity in entities[:2]:
                for candidate in self._lookup(cache, entity, query, query_terms, agent_id, allowed_ids, scope_ids):
                    if is_update_statement(candidate.content) and supersedes(candidate, source, query):
                        # An update of a best match always earns a slot (evidence 1.0) and
                        # shows the evidence of the memory it updates.
                        _place_after(out, source, candidate, query_terms, top_k, 1.0, move=False, shown=relevance)
        out = out[:top_k]
        return [f for f, _, _ in out], [s for _, s, _ in out], [r for _, _, r in out]


def _place_after(out: _Ranked, source: MemoryFragment, candidate: MemoryFragment, query_terms: list[str],
                 top_k: int, evidence: float, *, move: bool = True, shown: float | None = None) -> bool:
    """Put *candidate* right behind *source* (after earlier hops of it), sharing its score.

    A candidate already ranked above its source stays put; one ranked below
    is moved up when ``move`` is set. Direct matches at least as strong as
    the *evidence* the candidate inherits from its source stay ahead of it,
    and a new candidate may push the last result out of the top-k only when
    that result is a weak match or weaker than that evidence: hops fill weak
    tails, they never evict strong direct matches.
    Returns True when the list changed.
    """
    ids = [f.id for f, _, _ in out]
    at = ids.index(source.id)
    if candidate.id in ids:
        current = ids.index(candidate.id)
        if current <= at or not move:
            return False
        entry = out.pop(current)
    else:
        tail = out[top_k - 1][2] if len(out) >= top_k else 0.0
        if tail >= _WEAK_MATCH and tail >= evidence:
            return False
        # A hop shows the evidence it inherits when that beats its own words:
        # the displayed score is what put it in the list.
        inherited = evidence if shown is None else shown
        entry = (candidate, 0.0, max(inherited, _relevance(query_terms, candidate.content, 0.0)))
    # Behind the source, the hops already placed behind it (same score), and
    # any direct match at least as strong as the evidence the hop inherits.
    score = out[at][1]
    position = at + 1
    while position < len(out) and (out[position][1] == score or out[position][2] >= evidence):
        position += 1
    out.insert(position, (entry[0], out[position - 1][1], entry[2]))
    return True
