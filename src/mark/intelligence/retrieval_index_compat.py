"""Store / index compatibility helpers for the retrieval pipeline's index sync.

Custom and legacy stores or indexes may lack batch reads, agent partitions
or id listings; every helper here probes with ``getattr`` and falls back
(``TypeError`` on an older signature) so ``_IndexSyncMixin`` can stay
focused on keeping the index in sync. Split out of ``retrieval_index``.
"""
from __future__ import annotations

from typing import Iterable

from mark.index import VectorSearchResult
from mark.intelligence.retrieval_policy import _RETRIEVABLE_STATES
from mark.types import MemoryFragment


class _StoreCompatMixin:
    """Reads and index operations that tolerate legacy/custom stores and indexes."""

    def _stored_embeddings(self, agent_id: str, ids: set[str] | None) -> dict[str, list[float] | None]:
        fetch = getattr(self._store, "embeddings_for_agent", None)
        if fetch is not None and (ids is None or len(ids) > 2_000):  # a few ids: read just their rows
            rows = fetch(agent_id)
            return {fid: vec for fid, vec in rows if ids is None or fid in ids}
        if ids is None:
            try:
                fragments = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=None)
            except TypeError:
                fragments = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=1_000_000)
            return {f.id: f.embedding for f in fragments}
        found = self._get_many(list(ids))
        return {fid: f.embedding for fid, f in found.items()}

    def _get_many(self, ids: Iterable[str]) -> dict[str, MemoryFragment]:
        ids = list(ids)
        get_many = getattr(self._store, "get_many", None)
        if get_many is not None:
            return dict(get_many(ids))
        found: dict[str, MemoryFragment] = {}
        for fid in ids:
            fragment = self._store.get(fid)
            if fragment is not None:
                found[fid] = fragment
        return found

    def _index_add(self, fragment_id: str, vector: list[float], content: str, agent_id: str) -> None:
        try:
            self._index.add(fragment_id, vector, content, agent_id=agent_id)
        except TypeError:  # legacy/custom index without agent partitions
            self._index.add(fragment_id, vector, content)

    def _index_ids(self, agent_id: str) -> set[str]:
        ids = getattr(self._index, "ids", None)
        if ids is not None:
            try:
                return set(ids(agent_id))
            except TypeError:
                return set(ids())
        return {fid for fid in getattr(self._index, "_vectors", {})}

    def _index_search(
        self,
        q_vec: list[float],
        agent_id: str,
        scope_ids: set[str] | None,
        depth: int,
        allowed_ids: set[str],
    ) -> list[VectorSearchResult]:
        try:
            return self._index.search(q_vec, top_k=depth, include_ids=scope_ids, agent_id=agent_id)
        except TypeError:
            # Legacy index without agent partitions: always restrict to the agent's ids.
            return self._index.search(q_vec, top_k=depth, include_ids=allowed_ids)

    def _remove_agent_from_index(self, agent_id: str, ids: Iterable[str]) -> None:
        clear = getattr(self._index, "clear", None)
        if clear is not None:
            try:
                clear(agent_id)
                return
            except TypeError:
                pass
        for fid in ids:
            self._index.remove(fid)
