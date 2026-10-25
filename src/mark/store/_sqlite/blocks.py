"""Memory blocks: block rows, fragment/node membership, links and the seal chain.

Sealed blocks form a per-agent hash chain (``prev_block_id`` /
``prev_block_hash``) and are immutable: a fragment cannot be moved out of a
SEALED block. Members of a QUARANTINED block are excluded from every
fragment listing (see ``FragmentReadMixin._fragment_filters``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from mark.store._sqlite.codec import _dt_to_text
from mark.types import BlockLink, BlockStatus, MemoryBlock, MemoryEdge, MemoryFragment, MemoryNode


class BlockMixin:
    """CRUD for blocks, block membership (fragments, nodes) and block links."""

    def store_block(self, block: MemoryBlock) -> str:
        """Insert or replace a block row, including provenance/chain columns."""
        with self._writing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO blocks
                    (id, name, agent_id, scope, block_type, metadata, created_at,
                     session_id, status, content_hash, prev_block_id, prev_block_hash, sealed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.id,
                    block.name,
                    block.agent_id,
                    block.scope.value,
                    block.block_type,
                    json.dumps(block.metadata),
                    block.created_at.isoformat(),
                    block.session_id,
                    block.status.value,
                    block.content_hash,
                    block.prev_block_id,
                    block.prev_block_hash,
                    _dt_to_text(block.sealed_at),
                ),
            )
        return block.id

    def get_block(self, block_id: str) -> MemoryBlock | None:
        """Fetch a block by id, or None."""
        row = self._fetchone("SELECT * FROM blocks WHERE id = ?", (block_id,))
        return None if row is None else self._row_to_block(dict(row))

    def get_block_by_name(self, name: str, agent_id: str) -> MemoryBlock | None:
        """Find an agent's block by name, or None."""
        row = self._fetchone("SELECT * FROM blocks WHERE name = ? AND agent_id = ?", (name, agent_id))
        return None if row is None else self._row_to_block(dict(row))

    def list_blocks(self, agent_id: str) -> list[MemoryBlock]:
        """Return all blocks."""
        rows = self._fetchall("SELECT * FROM blocks WHERE agent_id = ? ORDER BY created_at DESC", (agent_id,))
        return [self._row_to_block(dict(row)) for row in rows]

    def fragments_for_block(self, block_id: str) -> list[MemoryFragment]:
        """Return fragments belonging to a block."""
        rows = self._fetchall("SELECT * FROM fragments WHERE block_id = ? ORDER BY importance DESC", (block_id,))
        return [self._row_to_fragment(dict(row)) for row in rows]

    # ── graph-scoped block API ────────────────────────────────────────────────

    def assign_fragment_block(self, fragment_id: str, block_id: str | None) -> None:
        """Move a fragment into a block (or out, with block_id=None).

        Raises ValueError when the fragment currently belongs to a SEALED
        block and the move would take it out of that block.
        """
        with self._writing() as conn:
            self._note_changed((fragment_id,))
            row = conn.execute(
                """
                SELECT f.block_id AS block_id, b.status AS status
                FROM fragments f LEFT JOIN blocks b ON b.id = f.block_id
                WHERE f.id = ?
                """,
                (fragment_id,),
            ).fetchone()
            if (
                row is not None
                and row["block_id"] is not None
                and row["block_id"] != block_id
                and row["status"] == BlockStatus.SEALED.value
            ):
                raise ValueError(
                    f"Fragment {fragment_id!r} belongs to sealed block {row['block_id']!r}; "
                    "sealed blocks are immutable."
                )
            conn.execute(
                "UPDATE fragments SET block_id = ?, updated_at = ? WHERE id = ?",
                (block_id, datetime.now(timezone.utc).isoformat(), fragment_id),
            )

    def add_node_to_block(self, block_id: str, node_id: str) -> None:
        """Register a node as a member of a block (idempotent, many-to-many)."""
        with self._writing() as conn:
            self._note_changed(())  # membership of nodes/links: no fragment changes
            conn.execute(
                "INSERT OR IGNORE INTO block_nodes (block_id, node_id, added_at) VALUES (?, ?, ?)",
                (block_id, node_id, datetime.now(timezone.utc).isoformat()),
            )

    def remove_node_from_block(self, block_id: str, node_id: str) -> bool:
        """Remove a node's membership in a block. Returns True if removed."""
        with self._writing() as conn:
            self._note_changed(())  # membership of nodes/links: no fragment changes
            cursor = conn.execute(
                "DELETE FROM block_nodes WHERE block_id = ? AND node_id = ?",
                (block_id, node_id),
            )
        return cursor.rowcount > 0

    def nodes_in_block(self, block_id: str) -> list[MemoryNode]:
        """Return all member nodes of a block."""
        rows = self._fetchall(
            """
            SELECT n.* FROM nodes n
            JOIN block_nodes bn ON bn.node_id = n.id
            WHERE bn.block_id = ?
            ORDER BY bn.added_at
            """,
            (block_id,),
        )
        return [self._row_to_node(dict(row)) for row in rows]

    def block_ids_for_node(self, node_id: str) -> list[str]:
        """Return ids of every block this node belongs to."""
        rows = self._fetchall("SELECT block_id FROM block_nodes WHERE node_id = ? ORDER BY added_at", (node_id,))
        return [str(row[0]) for row in rows]

    def edges_in_block(self, block_id: str) -> list[MemoryEdge]:
        """Return all edges scoped to a block."""
        rows = self._fetchall("SELECT * FROM edges WHERE block_id = ? ORDER BY created_at", (block_id,))
        return [self._row_to_edge(dict(row)) for row in rows]

    def store_block_link(self, link: BlockLink) -> str:
        """Insert or replace a directed link between two blocks."""
        with self._writing() as conn:
            self._note_changed(())  # membership of nodes/links: no fragment changes
            conn.execute(
                """
                INSERT OR REPLACE INTO block_links
                    (id, source_block_id, target_block_id, relation, weight,
                     agent_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.source_block_id,
                    link.target_block_id,
                    link.relation,
                    link.weight,
                    link.agent_id,
                    json.dumps(link.metadata),
                    link.created_at.isoformat(),
                ),
            )
        return link.id

    def links_from_block(self, block_id: str) -> list[BlockLink]:
        """Return outgoing (forward) links from a block."""
        rows = self._fetchall(
            "SELECT * FROM block_links WHERE source_block_id = ? ORDER BY created_at", (block_id,)
        )
        return [self._row_to_block_link(dict(row)) for row in rows]

    def links_to_block(self, block_id: str) -> list[BlockLink]:
        """Return incoming (backward) links to a block."""
        rows = self._fetchall(
            "SELECT * FROM block_links WHERE target_block_id = ? ORDER BY created_at", (block_id,)
        )
        return [self._row_to_block_link(dict(row)) for row in rows]

    def delete_block_link(self, link_id: str) -> bool:
        """Delete a block link by id. Returns True if a row was deleted."""
        with self._writing() as conn:
            self._note_changed(())  # membership of nodes/links: no fragment changes
            cursor = conn.execute("DELETE FROM block_links WHERE id = ?", (link_id,))
        return cursor.rowcount > 0

    def list_blocks_by_status(self, agent_id: str, status: BlockStatus) -> list[MemoryBlock]:
        """Return an agent's blocks filtered by lifecycle status."""
        rows = self._fetchall(
            "SELECT * FROM blocks WHERE agent_id = ? AND status = ? ORDER BY created_at",
            (agent_id, status.value),
        )
        return [self._row_to_block(dict(row)) for row in rows]

    def latest_sealed_block(self, agent_id: str, *, exclude_id: str | None = None) -> MemoryBlock | None:
        """Return the most recently sealed block in this agent's chain, if any."""
        sql = "SELECT * FROM blocks WHERE agent_id = ? AND status = ? AND sealed_at IS NOT NULL"
        params: list[object] = [agent_id, BlockStatus.SEALED.value]
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        sql += " ORDER BY sealed_at DESC, created_at DESC LIMIT 1"
        row = self._fetchone(sql, params)
        return None if row is None else self._row_to_block(dict(row))

    def sealed_chain(self, agent_id: str) -> list[MemoryBlock]:
        """Return this agent's sealed blocks in chain (seal-time) order."""
        rows = self._fetchall(
            """
            SELECT * FROM blocks
            WHERE agent_id = ? AND sealed_at IS NOT NULL AND status IN (?, ?)
            ORDER BY sealed_at, created_at
            """,
            (agent_id, BlockStatus.SEALED.value, BlockStatus.QUARANTINED.value),
        )
        return [self._row_to_block(dict(row)) for row in rows]
