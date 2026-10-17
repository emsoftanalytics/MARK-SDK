"""Knowledge graph storage: documents, nodes and edges.

Edges may be scoped to a block via ``block_id``. Edge-weight updates are
non-structural writes (they never change which fragments are retrievable,
so they do not bump ``write_version``).
"""
from __future__ import annotations

import json

from mark.types import MemoryDocument, MemoryEdge, MemoryNode
from mark.types.graph import EdgeRelation


class GraphMixin:
    """CRUD for documents, graph nodes and graph edges."""

    def store_document(self, doc: MemoryDocument) -> str:
        """Persist a document."""
        with self._writing() as conn:
            self._note_changed(())  # graph rows never change which fragments are retrievable
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id,
                    doc.title,
                    doc.content,
                    doc.source,
                    doc.agent_id,
                    doc.scope.value,
                    json.dumps(doc.fragment_ids),
                    json.dumps(doc.tags),
                    json.dumps(doc.metadata),
                    doc.created_at.isoformat(),
                ),
            )
        return doc.id

    def get_document(self, doc_id: str) -> MemoryDocument | None:
        """Fetch a document by id, or None."""
        row = self._fetchone("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return None if row is None else self._row_to_document(dict(row))

    def store_node(self, node: MemoryNode) -> str:
        """Persist a graph node."""
        with self._writing() as conn:
            self._note_changed(())  # graph rows never change which fragments are retrievable
            conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.id,
                    node.node_type.value,
                    node.label,
                    node.agent_id,
                    node.fragment_id,
                    node.document_id,
                    node.weight,
                    json.dumps(node.metadata),
                    node.created_at.isoformat(),
                ),
            )
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a node by id, or None."""
        row = self._fetchone("SELECT * FROM nodes WHERE id = ?", (node_id,))
        return None if row is None else self._row_to_node(dict(row))

    def get_node_by_label(self, label: str, agent_id: str) -> MemoryNode | None:
        """Find the most-recently-created node with this label for this agent."""
        row = self._fetchone(
            "SELECT * FROM nodes WHERE label = ? AND agent_id = ? ORDER BY created_at DESC LIMIT 1",
            (label, agent_id),
        )
        return None if row is None else self._row_to_node(dict(row))

    def delete_edges_between(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | None = None,
    ) -> int:
        """Delete edges from source_id to target_id. Returns count of deleted rows."""
        sql    = "DELETE FROM edges WHERE source_id = ? AND target_id = ?"
        params: list[object] = [source_id, target_id]
        if relation is not None:
            sql += " AND relation = ?"
            params.append(relation.value)
        with self._writing() as conn:
            self._note_changed(())  # graph rows never change which fragments are retrievable
            cursor = conn.execute(sql, params)
        return cursor.rowcount

    def nodes_for_fragment(self, fragment_id: str) -> list[MemoryNode]:
        """Return nodes attached to a fragment."""
        rows = self._fetchall("SELECT * FROM nodes WHERE fragment_id = ?", (fragment_id,))
        return [self._row_to_node(dict(row)) for row in rows]

    def nodes_for_agent(self, agent_id: str) -> list[MemoryNode]:
        """Return an agent's nodes by weight."""
        rows = self._fetchall("SELECT * FROM nodes WHERE agent_id = ? ORDER BY weight DESC", (agent_id,))
        return [self._row_to_node(dict(row)) for row in rows]

    def store_edge(self, edge: MemoryEdge) -> str:
        """Insert or replace an edge row; block_id scopes the edge to a block."""
        with self._writing() as conn:
            self._note_changed(())  # graph rows never change which fragments are retrievable
            conn.execute(
                """
                INSERT OR REPLACE INTO edges
                    (id, source_id, target_id, relation, weight, agent_id,
                     metadata, created_at, block_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                    edge.weight,
                    edge.agent_id,
                    json.dumps(edge.metadata),
                    edge.created_at.isoformat(),
                    edge.block_id,
                ),
            )
        return edge.id

    def edges_from_node(
        self,
        node_id: str,
        relation_filter: list[EdgeRelation] | None = None,
    ) -> list[MemoryEdge]:
        """Return outgoing edges from a node."""
        sql = "SELECT * FROM edges WHERE source_id = ?"
        params: list[object] = [node_id]
        if relation_filter:
            placeholders = ",".join("?" for _ in relation_filter)
            sql += f" AND relation IN ({placeholders})"
            params.extend(relation.value for relation in relation_filter)
        rows = self._fetchall(sql, params)
        return [self._row_to_edge(dict(row)) for row in rows]

    def edges_to_node(self, node_id: str) -> list[MemoryEdge]:
        """Return all edges whose target is node_id (incoming edges)."""
        rows = self._fetchall("SELECT * FROM edges WHERE target_id = ?", (node_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def contradiction_edges(self, node_id: str) -> list[MemoryEdge]:
        """Return CONTRADICTS edges touching a node."""
        rows = self._fetchall(
            "SELECT * FROM edges WHERE relation = ? AND (source_id = ? OR target_id = ?)",
            (EdgeRelation.CONTRADICTS.value, node_id, node_id),
        )
        return [self._row_to_edge(dict(row)) for row in rows]

    def list_edges(self, agent_id: str) -> list[MemoryEdge]:
        """Return all edges owned by an agent, ordered by weight descending."""
        rows = self._fetchall("SELECT * FROM edges WHERE agent_id = ? ORDER BY weight DESC", (agent_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by id. Returns True if a row was deleted."""
        with self._writing() as conn:
            self._note_changed(())  # graph rows never change which fragments are retrievable
            cursor = conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        return cursor.rowcount > 0

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update the weight of an edge, clamped to [0, 1]."""
        with self._writing(structural=False) as conn:
            conn.execute(
                "UPDATE edges SET weight = ? WHERE id = ?",
                (max(0.0, min(1.0, weight)), edge_id),
            )
