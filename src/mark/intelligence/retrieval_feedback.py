"""Plasticity feedback: how a retrieval reinforces what it returned.

Every returned fragment has its access recorded. Importance boosts (Hebbian)
and edge co-activation are applied only to results backed by real query
evidence (``relevance >= min_confidence``) in the default RELEVANT mode;
otherwise whatever happens to land in the top-k would grow more important on
every read, a self-reinforcing loop.

Expects the host class to provide ``_store``, ``_hebbian``, ``_coact`` and
``_get_many``.
"""
from __future__ import annotations

from mark.intelligence.retrieval_policy import ReinforceMode, RetrievalPolicySpec
from mark.types import MemoryFragment


class _ReinforcementMixin:
    """Access tracking, importance reinforcement and co-retrieval edge strengthening."""

    def _reinforce_results(
        self,
        fragments: list[MemoryFragment],
        relevances: list[float],
        spec: RetrievalPolicySpec,
        mode: ReinforceMode,
    ) -> None:
        if mode == ReinforceMode.NONE or not fragments:
            return
        ids = [f.id for f in fragments]
        touch_many = getattr(self._store, "touch_many", None)
        if touch_many is not None:
            touch_many(ids)
        elif hasattr(self._store, "touch"):
            for fid in ids:
                self._store.touch(fid)

        # Only results backed by real query evidence are reinforced, otherwise
        # whatever lands in the top-k grows more important on every read.
        if mode == ReinforceMode.ALL:
            reinforced = list(fragments)
        else:
            reinforced = [f for f, rel in zip(fragments, relevances) if rel >= spec.min_confidence]
        if not reinforced:
            return
        refreshed = self._get_many([f.id for f in reinforced])
        updates: dict[str, float] = {}
        for fragment in reinforced:
            current = refreshed.get(fragment.id, fragment)
            new_imp = self._hebbian.on_access(current)
            if abs(new_imp - current.importance) > 1e-9:
                updates[fragment.id] = new_imp
        if updates:
            update_many = getattr(self._store, "update_importance_many", None)
            if update_many is not None:
                update_many(updates)
            elif hasattr(self._store, "update_importance"):
                for fid, value in updates.items():
                    self._store.update_importance(fid, value)

        self._coactivate({f.id for f in reinforced})

    def _coactivate(self, frag_ids: set[str]) -> None:
        """Strengthen edges between co-retrieved fragments (undirected, once per edge)."""
        if len(frag_ids) < 2 or not hasattr(self._store, "update_edge_weight"):
            return
        node_to_fragment: dict[str, str] = {}
        for fid in frag_ids:
            for node in self._store.nodes_for_fragment(fid):
                node_to_fragment[node.id] = fid
        seen_edges: set[str] = set()
        for node_id, fid in node_to_fragment.items():
            edges = list(self._store.edges_from_node(node_id))
            if hasattr(self._store, "edges_to_node"):
                edges += self._store.edges_to_node(node_id)
            for edge in edges:
                if edge.id in seen_edges:
                    continue
                other = edge.target_id if edge.source_id == node_id else edge.source_id
                other_fid = node_to_fragment.get(other)
                if other_fid is None or other_fid == fid:
                    continue
                seen_edges.add(edge.id)
                new_w = self._coact.on_co_retrieval(edge.weight)
                if abs(new_w - edge.weight) > 1e-9:
                    self._store.update_edge_weight(edge.id, new_w)
