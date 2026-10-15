"""Index side of the retrieval pipeline: scoping, index sync and channel search.

``_IndexSyncMixin`` keeps the vector index consistent with the store:

- the allowed id set is computed in SQL first, so filters apply *before*
  scoring and other agents/sessions can never crowd the top-k;
- the index is lazily synced with stored embeddings under ``_sync_lock``
  (bulk-loaded in numpy blocks on a cold start, see ``index_bulk_load``;
  re-embedding only what is missing or has the wrong dimension); an
  unchanged scope and index revision skip the sync, so steady-state recall
  does no O(store) work;
- a changed embedder fingerprint re-embeds the agent's memories once.

Legacy stores/indexes are supported via ``getattr`` probes and ``TypeError`` fallbacks.

Expects the host class to provide ``_store``, ``_index``, ``_embedder``,
``_sync_lock``, ``_verified_agents`` and ``EMBEDDER_META_KEY``.
"""
from __future__ import annotations

import logging
from typing import Iterable

from mark.index import VectorSearchResult
from mark.intelligence.index_bulk_load import (
    _same_embedding_space, bulk_index, changed_ids, index_each, restore_snapshot, save_snapshot, scope_ids,
    wants_numpy,
)
from mark.intelligence.retrieval_policy import _RETRIEVABLE_STATES, SessionFilter
from mark.types import MemoryFragment

# Same logger name as before the split, so log filtering keeps working.
logger = logging.getLogger("mark.retrieval")


class _IndexSyncMixin:
    """Write path, scope filtering, index synchronisation and the two search channels."""

    # ── Write path ────────────────────────────────────────────────────────────

    def index_fragment(self, fragment: MemoryFragment) -> MemoryFragment:
        """Embed, persist, and index a fragment."""
        embedding = self._embedder.embed(fragment.content)
        indexed = fragment.model_copy(update={"embedding": embedding})
        self._store.store(indexed)
        self._index_add(indexed.id, embedding, indexed.content, indexed.agent_id)
        return indexed

    def index_fragments(self, fragments: list[MemoryFragment], batch_size: int = 64) -> list[MemoryFragment]:
        """Embed (in batches), persist (one transaction per batch), and index fragments."""
        indexed_all: list[MemoryFragment] = []
        store_many = getattr(self._store, "store_many", None)
        for start in range(0, len(fragments), batch_size):
            chunk = fragments[start:start + batch_size]
            vectors = self._embedder.embed_batch([fragment.content for fragment in chunk])
            indexed = [f.model_copy(update={"embedding": v}) for f, v in zip(chunk, vectors)]
            if store_many is not None:
                store_many(indexed)
            else:
                for fragment in indexed:
                    self._store.store(fragment)
            for fragment in indexed:
                self._index_add(fragment.id, fragment.embedding, fragment.content, fragment.agent_id)
            indexed_all.extend(indexed)
        return indexed_all

    # ── Scope & index sync ────────────────────────────────────────────────────

    def _allowed_ids(self, agent_id: str, sf: SessionFilter) -> set[str]:
        kwargs: dict = dict(
            session_id=sf.session_id,
            session_prefix=sf.session_prefix,
            tags=sf.tags,
            tier=sf.tier,
            scope=sf.scope,
        )
        if sf.block_id is not None or sf.block_ids:
            kwargs["block_id"] = sf.block_id
            kwargs["block_ids"] = sf.block_ids
        id_set = getattr(self._store, "retrievable_id_set", None)
        if id_set is not None:
            # The store returns its cached frozenset (same object while nothing
            # changed); wrap it once per change, not once per recall.
            cache = self.__dict__.setdefault("_scope_cache", {})
            return scope_ids(cache, (agent_id, repr(sorted(kwargs.items()))), id_set(agent_id, **kwargs))
        retrievable_ids = getattr(self._store, "retrievable_ids", None)
        if retrievable_ids is not None:
            return set(retrievable_ids(agent_id, **kwargs))
        # Legacy/custom stores: uncapped list, TTL filtered in Python.
        try:
            rows = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=None, **kwargs)
        except TypeError:
            rows = self._store.list_by_agent(agent_id, states=_RETRIEVABLE_STATES, limit=1_000_000, **kwargs)
        return {fragment.id for fragment in rows if not fragment.is_expired()}

    def _sync_index(self, agent_id: str, allowed_ids: set[str]) -> set[str]:
        """Make sure every allowed fragment is indexed with the current embedder.

        Returns the indexed ids — ``allowed_ids`` itself when equal, so the
        pipeline's equality check is O(1). Unchanged scope + index revision: no work.
        """
        with self._sync_lock:
            if agent_id not in self._verified_agents:
                self._verify_embedder(agent_id)
                self._verified_agents.add(agent_id)
            synced = self.__dict__.setdefault("_synced", {})
            revision = getattr(self._index, "revision", None)
            last = synced.get(agent_id)
            if revision is not None and last is not None and last[0] is allowed_ids and last[1] == revision:
                return last[2]
            complete = self._load_missing(agent_id, allowed_ids)
            count = getattr(self._index, "count", None)
            if complete and count is not None and count(agent_id) == len(allowed_ids):
                result = allowed_ids  # allowed ⊆ indexed and equal sizes: equal sets
            else:
                indexed = self._index_ids(agent_id)
                result = allowed_ids if indexed == allowed_ids else indexed
            synced[agent_id] = (allowed_ids, getattr(self._index, "revision", None), result)
            return result

    def _load_missing(self, agent_id: str, allowed_ids: set[str]) -> bool:
        """Index every allowed id the index lacks: stored vectors in bulk, re-embed the rest.

        Returns True when every allowed id is indexed afterwards. An empty
        index (cold start) skips set operations (~0.1 s each at 1M ids).
        """
        count = getattr(self._index, "count", None)
        before = count(agent_id) if count is not None else None
        find_missing = getattr(self._index, "missing", None)  # C-level set difference
        missing = allowed_ids if before == 0 else (
            find_missing(allowed_ids, agent_id) if find_missing else allowed_ids - self._index_ids(agent_id))
        if not missing:
            return True
        # Large cold start: snapshot first; (re)save one after a full row load.
        missing, before, snapshot_due, version = restore_snapshot(
            self._store, self._index, self._embedder, agent_id, missing, before)
        pages, add_many = getattr(self._store, "embedding_pages", None), getattr(self._index, "add_many", None)
        if pages is not None and add_many is not None and before is not None and len(missing) > 256:
            numpy = wants_numpy(self._index, before + len(missing))
            to_embed = bulk_index(pages(agent_id), missing, getattr(self._embedder, "dim", None), add_many, agent_id, numpy)
            if count(agent_id) - before + len(to_embed) != len(missing):
                # Some ids were not in embedding_pages (state changed meanwhile).
                to_embed = list(missing - self._index_ids(agent_id))
            if snapshot_due and not to_embed:
                save_snapshot(self._store, self._index, self._embedder, agent_id, version)
        else:  # small gaps (or legacy stores/indexes): per-vector path
            stored = self._stored_embeddings(agent_id, set(missing))
            to_embed = index_each(stored, missing, getattr(self._embedder, "dim", None), self._index_add, agent_id)
        if to_embed:
            self._embed_and_index(agent_id, to_embed)
        return not to_embed or set(to_embed) <= self._index_ids(agent_id)

    def _verify_embedder(self, agent_id: str) -> None:
        """Re-embed an agent's memories when the embedder changed since they were stored."""
        get_meta = getattr(self._store, "get_meta", None)
        set_meta = getattr(self._store, "set_meta", None)
        fingerprint = getattr(self._embedder, "fingerprint", None)
        if get_meta is None or set_meta is None or not fingerprint:
            return
        key = f"{self.EMBEDDER_META_KEY}:{agent_id}"
        stored = get_meta(key)
        if stored is not None and _same_embedding_space(stored, fingerprint):
            return
        if fingerprint.endswith(":None") and stored is not None:
            # A different model whose dimension is still unknown (HTTP/ONNX
            # providers learn it on first call): probe once so the switch is
            # detected now, instead of mixing old vectors into the new space.
            self._embedder.embed("embedding dimension probe")
            fingerprint = getattr(self._embedder, "fingerprint", fingerprint)
            if _same_embedding_space(stored, fingerprint):
                return
        if fingerprint.endswith(":None"):
            # Dimension not known until the first embed call: never persist a
            # partial fingerprint (it would force a re-embed on every restart).
            return
        if stored is not None:
            logger.info("embedder changed for agent %r (%s -> %s); re-embedding", agent_id, stored, fingerprint)
            # A compatible upgrade (hash v2 -> v3) re-embeds only the texts that changed.
            ids = changed_ids(self._store, self._embedder, agent_id, stored)
            if ids is None:
                ids = list(self._stored_embeddings(agent_id, None))
                self._remove_agent_from_index(agent_id, ids)
            else:
                for fragment_id in ids:
                    self._index.remove(fragment_id)
            self._embed_and_index(agent_id, ids)
        set_meta(key, fingerprint)

    def _embed_and_index(self, agent_id: str, fragment_ids: list[str], batch_size: int = 64) -> None:
        update = getattr(self._store, "update_embedding", None)
        for start in range(0, len(fragment_ids), batch_size):
            chunk = fragment_ids[start:start + batch_size]
            fragments = self._get_many(chunk)
            ordered = [fragments[fid] for fid in chunk if fid in fragments]
            if not ordered:
                continue
            vectors = self._embedder.embed_batch([f.content for f in ordered])
            for fragment, vector in zip(ordered, vectors):
                if update is not None:
                    update(fragment.id, vector)
                self._index_add(fragment.id, vector, fragment.content, agent_id)

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

    # ── Search channels ───────────────────────────────────────────────────────

    def _vector_search(
        self, text: str, agent_id: str, allowed_ids: set[str], scope_ids: set[str] | None, depth: int,
    ) -> list[VectorSearchResult]:
        q_vec = self._embedder.embed(text)
        try:
            return self._index_search(q_vec, agent_id, scope_ids, depth, allowed_ids)
        except ValueError as exc:  # DimensionMismatchError subclasses ValueError
            logger.warning("vector search failed (%s); rebuilding index for agent %r", exc, agent_id)
            with self._sync_lock:
                ids = list(allowed_ids)
                self._remove_agent_from_index(agent_id, ids)
                self._embed_and_index(agent_id, ids)
            return self._index_search(q_vec, agent_id, allowed_ids, depth, allowed_ids)

    def _lexical_search(
        self, text: str, agent_id: str, allowed_ids: set[str], scope_ids: set[str] | None, depth: int,
    ) -> list[tuple[str, float]]:
        search = getattr(self._store, "lexical_search", None)
        if search is None:
            return []
        try:
            hits = search(agent_id, text, limit=depth, include_ids=scope_ids)
        except Exception as exc:  # lexical channel is best-effort; vector still answers
            logger.warning("lexical search failed: %s", exc)
            return []
        return [(fid, score) for fid, score in hits if fid in allowed_ids]

    # ── Store / index compatibility helpers ───────────────────────────────────

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
