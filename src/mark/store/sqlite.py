"""Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mark.security import ContentHasher, EncryptionProvider, NoOpEncryptionProvider
from mark.types import BlockLink, BlockStatus, MemoryBlock, MemoryDocument, MemoryEdge, MemoryFragment, MemoryNode
from mark.types import MemoryScope, MemoryState, MemoryTier  # MemoryTier used in list_by_agent filter
from mark.types.graph import EdgeRelation, NodeType


class LocalMemoryStore:
    """Thread-safe SQLite store for fragments, blocks, documents, nodes, and edges."""

    CURRENT_SCHEMA_VERSION = 2

    # Maps schema version N to the SQL statements needed to upgrade from N-1 to N.
    # Add an entry here whenever a schema change is made and bump CURRENT_SCHEMA_VERSION.
    _MIGRATIONS: dict[int, list[str]] = {
        # 1: []  — baseline; all tables created fresh, no ALTER TABLE needed
        # 2: graph-scoped memory blocks — block provenance columns, block-scoped
        #    edges, node membership, and block links. Column order here must match
        #    the v2 _DDL_BLOCKS/_DDL_EDGES definitions (ALTER TABLE appends).
        2: [
            "ALTER TABLE blocks ADD COLUMN session_id TEXT",
            "ALTER TABLE blocks ADD COLUMN status TEXT DEFAULT 'open'",
            "ALTER TABLE blocks ADD COLUMN content_hash TEXT",
            "ALTER TABLE blocks ADD COLUMN prev_block_id TEXT",
            "ALTER TABLE blocks ADD COLUMN prev_block_hash TEXT",
            "ALTER TABLE blocks ADD COLUMN sealed_at TEXT",
            "ALTER TABLE edges ADD COLUMN block_id TEXT",
            """
            CREATE TABLE IF NOT EXISTS block_nodes (
                block_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (block_id, node_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS block_links (
                id TEXT PRIMARY KEY,
                source_block_id TEXT NOT NULL,
                target_block_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                agent_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_blink_source ON block_links(source_block_id)",
            "CREATE INDEX IF NOT EXISTS idx_blink_target ON block_links(target_block_id)",
            "CREATE INDEX IF NOT EXISTS idx_bnode_node   ON block_nodes(node_id)",
            "CREATE INDEX IF NOT EXISTS idx_edge_block   ON edges(block_id)",
            "CREATE INDEX IF NOT EXISTS idx_block_agent  ON blocks(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(status)",
        ],
    }

    _DDL_SCHEMA_INFO = """
        CREATE TABLE IF NOT EXISTS schema_info (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """
    _DDL_BLOCKS = """
        CREATE TABLE IF NOT EXISTS blocks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            block_type TEXT DEFAULT 'ltm',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            session_id TEXT,
            status TEXT DEFAULT 'open',
            content_hash TEXT,
            prev_block_id TEXT,
            prev_block_hash TEXT,
            sealed_at TEXT
        )
    """
    _DDL_BLOCK_NODES = """
        CREATE TABLE IF NOT EXISTS block_nodes (
            block_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (block_id, node_id)
        )
    """
    _DDL_BLOCK_LINKS = """
        CREATE TABLE IF NOT EXISTS block_links (
            id TEXT PRIMARY KEY,
            source_block_id TEXT NOT NULL,
            target_block_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            agent_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_FRAGMENTS = """
        CREATE TABLE IF NOT EXISTS fragments (
            id TEXT PRIMARY KEY,
            content BLOB NOT NULL,
            block_id TEXT,
            scope TEXT NOT NULL,
            tier TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            importance REAL DEFAULT 0.5,
            state TEXT DEFAULT 'raw',
            confidence REAL DEFAULT 1.0,
            ttl_seconds INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT,
            tags TEXT DEFAULT '[]',
            source TEXT,
            metadata TEXT DEFAULT '{}',
            embedding TEXT,
            contradiction_ids TEXT DEFAULT '[]'
        )
    """
    _DDL_DOCUMENTS = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            agent_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            fragment_ids TEXT DEFAULT '[]',
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_NODES = """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label TEXT NOT NULL,
            agent_id TEXT,
            fragment_id TEXT,
            document_id TEXT,
            weight REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _DDL_EDGES = """
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            agent_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            block_id TEXT
        )
    """
    _DDL_SESSION_EVENTS = """
        CREATE TABLE IF NOT EXISTS session_events (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            event_type TEXT NOT NULL,
            fragment_id TEXT,
            query TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    _INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_frag_agent   ON fragments(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_state   ON fragments(state)",
        "CREATE INDEX IF NOT EXISTS idx_frag_scope   ON fragments(scope)",
        "CREATE INDEX IF NOT EXISTS idx_frag_tier    ON fragments(tier)",
        "CREATE INDEX IF NOT EXISTS idx_frag_session ON fragments(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_node_frag    ON nodes(fragment_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_source  ON edges(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_target  ON edges(target_id)",
        "CREATE INDEX IF NOT EXISTS idx_edge_block   ON edges(block_id)",
        "CREATE INDEX IF NOT EXISTS idx_sevt_agent   ON session_events(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_sevt_session ON session_events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_blink_source ON block_links(source_block_id)",
        "CREATE INDEX IF NOT EXISTS idx_blink_target ON block_links(target_block_id)",
        "CREATE INDEX IF NOT EXISTS idx_bnode_node   ON block_nodes(node_id)",
        "CREATE INDEX IF NOT EXISTS idx_block_agent  ON blocks(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_block_status ON blocks(status)",
    ]

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        encryption: EncryptionProvider | None = None,
    ) -> None:
        self.db_path = str(path)
        self.encryption = encryption or NoOpEncryptionProvider()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._shared_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._shared_conn = self._open(":memory:")
        else:
            Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def close(self) -> None:
        """Close open connections."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def store(self, fragment: MemoryFragment) -> str:
        """Insert or replace a fragment row (content encrypted at rest)."""
        fragment = ContentHasher.attach(fragment)
        encrypted_content = self.encryption.encrypt_str(fragment.content)
        with self._lock:
            self._conn().execute(
                """
                INSERT OR REPLACE INTO fragments VALUES (
                    :id, :content, :block_id, :scope, :tier, :agent_id, :session_id,
                    :importance, :state, :confidence, :ttl_seconds, :created_at,
                    :updated_at, :last_accessed_at, :tags, :source, :metadata,
                    :embedding, :contradiction_ids
                )
                """,
                {
                    "id": fragment.id,
                    "content": encrypted_content,
                    "block_id": fragment.block_id,
                    "scope": fragment.scope.value,
                    "tier": fragment.tier.value,
                    "agent_id": fragment.agent_id,
                    "session_id": fragment.session_id,
                    "importance": fragment.importance,
                    "state": fragment.state.value,
                    "confidence": fragment.confidence,
                    "ttl_seconds": fragment.ttl_seconds,
                    "created_at": fragment.created_at.isoformat(),
                    "updated_at": fragment.updated_at.isoformat(),
                    "last_accessed_at": _dt_to_text(fragment.last_accessed_at),
                    "tags": json.dumps(fragment.tags),
                    "source": fragment.source,
                    "metadata": json.dumps(fragment.metadata),
                    "embedding": json.dumps(fragment.embedding) if fragment.embedding is not None else None,
                    "contradiction_ids": json.dumps(fragment.contradiction_ids),
                },
            )
            self._conn().commit()
        return fragment.id

    def get(self, fragment_id: str) -> MemoryFragment | None:
        """Fetch a fragment by id, or None."""
        row = self._conn().execute("SELECT * FROM fragments WHERE id = ?", (fragment_id,)).fetchone()
        return None if row is None else self._row_to_fragment(dict(row))

    def list_by_agent(
        self,
        agent_id: str,
        *,
        scope: MemoryScope | None = None,
        states: Iterable[MemoryState] | None = None,
        tier: MemoryTier | None = None,
        session_id: str | None = None,
        session_prefix: str | None = None,
        tags: list[str] | None = None,
        block_id: str | None = None,
        block_ids: list[str] | None = None,
        limit: int = 200,
    ) -> list[MemoryFragment]:
        """Query fragments for agent_id with optional session, tag, tier, scope, and block filters.

        session_id      — exact match on session_id column
        session_prefix  — prefix match: session_id LIKE '<prefix>%'
        tags            — all listed tags must be present (AND semantics)
        tier            — exact match on tier column
        scope           — exact match on scope column
        block_id        — restrict to fragments belonging to one block
        block_ids       — restrict to fragments belonging to any listed block

        Fragments belonging to a QUARANTINED block are always excluded, so a
        corrupted block is isolated without affecting the rest of the agent's
        memory.
        """
        sql = "SELECT * FROM fragments WHERE agent_id = ?"
        params: list[object] = [agent_id]
        # Block-level quarantine: a quarantined block's members never surface.
        sql += (
            " AND (block_id IS NULL OR block_id NOT IN"
            " (SELECT id FROM blocks WHERE status = ?))"
        )
        params.append(BlockStatus.QUARANTINED.value)
        if block_id is not None:
            sql += " AND block_id = ?"
            params.append(block_id)
        elif block_ids:
            placeholders = ",".join("?" for _ in block_ids)
            sql += f" AND block_id IN ({placeholders})"
            params.extend(block_ids)
        if scope is not None:
            sql += " AND scope = ?"
            params.append(scope.value)
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier.value)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        elif session_prefix is not None:
            sql += " AND session_id LIKE ?"
            params.append(f"{session_prefix}%")
        if tags:
            for tag in tags:
                sql += " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
                params.append(tag)
        if states:
            state_list = list(states)
            placeholders = ",".join("?" for _ in state_list)
            sql += f" AND state IN ({placeholders})"
            params.extend(state.value for state in state_list)
        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [self._row_to_fragment(dict(row)) for row in rows]

    def list_sessions(self, agent_id: str) -> list[str]:
        """Return distinct session_ids for agent_id, ordered by most recent activity."""
        rows = self._conn().execute(
            """
            SELECT session_id
            FROM fragments
            WHERE agent_id = ? AND session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            """,
            (agent_id,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def list_all(self) -> list[MemoryFragment]:
        """Return every stored fragment."""
        rows = self._conn().execute("SELECT * FROM fragments").fetchall()
        return [self._row_to_fragment(dict(row)) for row in rows]

    def update_state(self, fragment_id: str, new_state: MemoryState) -> None:
        """Update a fragment's lifecycle state."""
        with self._lock:
            self._conn().execute(
                "UPDATE fragments SET state = ?, updated_at = ? WHERE id = ?",
                (new_state.value, datetime.now(timezone.utc).isoformat(), fragment_id),
            )
            self._conn().commit()

    def update_embedding(self, fragment_id: str, embedding: list[float]) -> None:
        """Persist a fragment's embedding."""
        with self._lock:
            self._conn().execute(
                "UPDATE fragments SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), fragment_id),
            )
            self._conn().commit()

    def delete(self, fragment_id: str) -> bool:
        """Delete by id; returns True when a row was removed."""
        with self._lock:
            cursor = self._conn().execute("DELETE FROM fragments WHERE id = ?", (fragment_id,))
            self._conn().commit()
        return cursor.rowcount > 0

    def clear(self, agent_id: str | None = None) -> None:
        """Delete fragments, optionally only for one agent."""
        with self._lock:
            if agent_id is None:
                self._conn().execute("DELETE FROM fragments")
            else:
                self._conn().execute("DELETE FROM fragments WHERE agent_id = ?", (agent_id,))
            self._conn().commit()

    def count(self, agent_id: str) -> int:
        """Return the number of fragments stored for an agent."""
        return int(
            self._conn().execute("SELECT COUNT(*) FROM fragments WHERE agent_id = ?", (agent_id,)).fetchone()[0]
        )

    def store_block(self, block: MemoryBlock) -> str:
        """Insert or replace a block row, including provenance/chain columns."""
        with self._lock:
            self._conn().execute(
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
            self._conn().commit()
        return block.id

    def get_block(self, block_id: str) -> MemoryBlock | None:
        """Fetch a block by id, or None."""
        row = self._conn().execute("SELECT * FROM blocks WHERE id = ?", (block_id,)).fetchone()
        return None if row is None else self._row_to_block(dict(row))

    def get_block_by_name(self, name: str, agent_id: str) -> MemoryBlock | None:
        """Find an agent's block by name, or None."""
        row = self._conn().execute(
            "SELECT * FROM blocks WHERE name = ? AND agent_id = ?", (name, agent_id)
        ).fetchone()
        return None if row is None else self._row_to_block(dict(row))

    def list_blocks(self, agent_id: str) -> list[MemoryBlock]:
        """Return all blocks."""
        rows = self._conn().execute(
            "SELECT * FROM blocks WHERE agent_id = ? ORDER BY created_at DESC", (agent_id,)
        ).fetchall()
        return [self._row_to_block(dict(row)) for row in rows]

    def fragments_for_block(self, block_id: str) -> list[MemoryFragment]:
        """Return fragments belonging to a block."""
        rows = self._conn().execute(
            "SELECT * FROM fragments WHERE block_id = ? ORDER BY importance DESC", (block_id,)
        ).fetchall()
        return [self._row_to_fragment(dict(row)) for row in rows]

    # ── graph-scoped block API ────────────────────────────────────────────────

    def assign_fragment_block(self, fragment_id: str, block_id: str | None) -> None:
        """Move a fragment into a block (or out, with block_id=None)."""
        with self._lock:
            self._conn().execute(
                "UPDATE fragments SET block_id = ?, updated_at = ? WHERE id = ?",
                (block_id, datetime.now(timezone.utc).isoformat(), fragment_id),
            )
            self._conn().commit()

    def add_node_to_block(self, block_id: str, node_id: str) -> None:
        """Register a node as a member of a block (idempotent, many-to-many)."""
        with self._lock:
            self._conn().execute(
                "INSERT OR IGNORE INTO block_nodes (block_id, node_id, added_at) VALUES (?, ?, ?)",
                (block_id, node_id, datetime.now(timezone.utc).isoformat()),
            )
            self._conn().commit()

    def remove_node_from_block(self, block_id: str, node_id: str) -> bool:
        """Remove a node's membership in a block. Returns True if removed."""
        with self._lock:
            cursor = self._conn().execute(
                "DELETE FROM block_nodes WHERE block_id = ? AND node_id = ?",
                (block_id, node_id),
            )
            self._conn().commit()
        return cursor.rowcount > 0

    def nodes_in_block(self, block_id: str) -> list[MemoryNode]:
        """Return all member nodes of a block."""
        rows = self._conn().execute(
            """
            SELECT n.* FROM nodes n
            JOIN block_nodes bn ON bn.node_id = n.id
            WHERE bn.block_id = ?
            ORDER BY bn.added_at
            """,
            (block_id,),
        ).fetchall()
        return [self._row_to_node(dict(row)) for row in rows]

    def block_ids_for_node(self, node_id: str) -> list[str]:
        """Return ids of every block this node belongs to."""
        rows = self._conn().execute(
            "SELECT block_id FROM block_nodes WHERE node_id = ? ORDER BY added_at", (node_id,)
        ).fetchall()
        return [str(row[0]) for row in rows]

    def edges_in_block(self, block_id: str) -> list[MemoryEdge]:
        """Return all edges scoped to a block."""
        rows = self._conn().execute(
            "SELECT * FROM edges WHERE block_id = ? ORDER BY created_at", (block_id,)
        ).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]

    def store_block_link(self, link: BlockLink) -> str:
        """Insert or replace a directed link between two blocks."""
        with self._lock:
            self._conn().execute(
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
            self._conn().commit()
        return link.id

    def links_from_block(self, block_id: str) -> list[BlockLink]:
        """Return outgoing (forward) links from a block."""
        rows = self._conn().execute(
            "SELECT * FROM block_links WHERE source_block_id = ? ORDER BY created_at", (block_id,)
        ).fetchall()
        return [self._row_to_block_link(dict(row)) for row in rows]

    def links_to_block(self, block_id: str) -> list[BlockLink]:
        """Return incoming (backward) links to a block."""
        rows = self._conn().execute(
            "SELECT * FROM block_links WHERE target_block_id = ? ORDER BY created_at", (block_id,)
        ).fetchall()
        return [self._row_to_block_link(dict(row)) for row in rows]

    def delete_block_link(self, link_id: str) -> bool:
        """Delete a block link by id. Returns True if a row was deleted."""
        with self._lock:
            cursor = self._conn().execute("DELETE FROM block_links WHERE id = ?", (link_id,))
            self._conn().commit()
        return cursor.rowcount > 0

    def list_blocks_by_status(self, agent_id: str, status: BlockStatus) -> list[MemoryBlock]:
        """Return an agent's blocks filtered by lifecycle status."""
        rows = self._conn().execute(
            "SELECT * FROM blocks WHERE agent_id = ? AND status = ? ORDER BY created_at",
            (agent_id, status.value),
        ).fetchall()
        return [self._row_to_block(dict(row)) for row in rows]

    def latest_sealed_block(self, agent_id: str, *, exclude_id: str | None = None) -> MemoryBlock | None:
        """Return the most recently sealed block in this agent's chain, if any."""
        sql = "SELECT * FROM blocks WHERE agent_id = ? AND status = ? AND sealed_at IS NOT NULL"
        params: list[object] = [agent_id, BlockStatus.SEALED.value]
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        sql += " ORDER BY sealed_at DESC, created_at DESC LIMIT 1"
        row = self._conn().execute(sql, params).fetchone()
        return None if row is None else self._row_to_block(dict(row))

    def sealed_chain(self, agent_id: str) -> list[MemoryBlock]:
        """Return this agent's sealed blocks in chain (seal-time) order."""
        rows = self._conn().execute(
            """
            SELECT * FROM blocks
            WHERE agent_id = ? AND sealed_at IS NOT NULL AND status IN (?, ?)
            ORDER BY sealed_at, created_at
            """,
            (agent_id, BlockStatus.SEALED.value, BlockStatus.QUARANTINED.value),
        ).fetchall()
        return [self._row_to_block(dict(row)) for row in rows]

    def store_document(self, doc: MemoryDocument) -> str:
        """Persist a document."""
        with self._lock:
            self._conn().execute(
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
            self._conn().commit()
        return doc.id

    def get_document(self, doc_id: str) -> MemoryDocument | None:
        """Fetch a document by id, or None."""
        row = self._conn().execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return None if row is None else self._row_to_document(dict(row))

    def store_node(self, node: MemoryNode) -> str:
        """Persist a graph node."""
        with self._lock:
            self._conn().execute(
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
            self._conn().commit()
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a node by id, or None."""
        row = self._conn().execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return None if row is None else self._row_to_node(dict(row))

    def get_node_by_label(self, label: str, agent_id: str) -> MemoryNode | None:
        """Find the most-recently-created node with this label for this agent."""
        row = self._conn().execute(
            "SELECT * FROM nodes WHERE label = ? AND agent_id = ? ORDER BY created_at DESC LIMIT 1",
            (label, agent_id),
        ).fetchone()
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
        with self._lock:
            cursor = self._conn().execute(sql, params)
            self._conn().commit()
        return cursor.rowcount

    def nodes_for_fragment(self, fragment_id: str) -> list[MemoryNode]:
        """Return nodes attached to a fragment."""
        rows = self._conn().execute("SELECT * FROM nodes WHERE fragment_id = ?", (fragment_id,)).fetchall()
        return [self._row_to_node(dict(row)) for row in rows]

    def nodes_for_agent(self, agent_id: str) -> list[MemoryNode]:
        """Return an agent's nodes by weight."""
        rows = self._conn().execute("SELECT * FROM nodes WHERE agent_id = ? ORDER BY weight DESC", (agent_id,)).fetchall()
        return [self._row_to_node(dict(row)) for row in rows]

    def store_edge(self, edge: MemoryEdge) -> str:
        """Insert or replace an edge row; block_id scopes the edge to a block."""
        with self._lock:
            self._conn().execute(
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
            self._conn().commit()
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
        rows = self._conn().execute(sql, params).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]

    def edges_to_node(self, node_id: str) -> list[MemoryEdge]:
        """Return all edges whose target is node_id (incoming edges)."""
        rows = self._conn().execute(
            "SELECT * FROM edges WHERE target_id = ?", (node_id,)
        ).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]

    def contradiction_edges(self, node_id: str) -> list[MemoryEdge]:
        """Return CONTRADICTS edges touching a node."""
        rows = self._conn().execute(
            "SELECT * FROM edges WHERE relation = ? AND (source_id = ? OR target_id = ?)",
            (EdgeRelation.CONTRADICTS.value, node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]

    def _conn(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        if not getattr(self._local, "conn", None):
            self._local.conn = self._open(self.db_path)
        return self._local.conn

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def update_importance(self, fragment_id: str, importance: float) -> None:
        """Update the importance of a fragment, clamped to [0, 1]."""
        with self._lock:
            self._conn().execute(
                "UPDATE fragments SET importance = ?, updated_at = ? WHERE id = ?",
                (max(0.0, min(1.0, importance)), datetime.now(timezone.utc).isoformat(), fragment_id),
            )
            self._conn().commit()

    def touch(self, fragment_id: str) -> None:
        """Record an access: update last_accessed_at and increment access_count in metadata."""
        with self._lock:
            row = self._conn().execute(
                "SELECT metadata FROM fragments WHERE id = ?", (fragment_id,)
            ).fetchone()
            if row is None:
                return
            meta = json.loads(str(row[0]))
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            now = datetime.now(timezone.utc).isoformat()
            self._conn().execute(
                "UPDATE fragments SET last_accessed_at = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (now, json.dumps(meta), now, fragment_id),
            )
            self._conn().commit()

    def list_edges(self, agent_id: str) -> list[MemoryEdge]:
        """Return all edges owned by an agent, ordered by weight descending."""
        rows = self._conn().execute(
            "SELECT * FROM edges WHERE agent_id = ? ORDER BY weight DESC", (agent_id,)
        ).fetchall()
        return [self._row_to_edge(dict(row)) for row in rows]

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by id. Returns True if a row was deleted."""
        with self._lock:
            cursor = self._conn().execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self._conn().commit()
        return cursor.rowcount > 0

    def update_edge_weight(self, edge_id: str, weight: float) -> None:
        """Update the weight of an edge, clamped to [0, 1]."""
        with self._lock:
            self._conn().execute(
                "UPDATE edges SET weight = ? WHERE id = ?",
                (max(0.0, min(1.0, weight)), edge_id),
            )
            self._conn().commit()

    def integrity_check(self) -> list[str]:
        """Run PRAGMA integrity_check. Returns [] on a healthy database."""
        rows = self._conn().execute("PRAGMA integrity_check").fetchall()
        results = [str(row[0]) for row in rows]
        return [] if results == ["ok"] else results

    def export(self, dest_path: str | Path) -> None:
        """Backup the database to dest_path using the SQLite backup API."""
        if self.db_path == ":memory:":
            raise ValueError("Cannot export an in-memory database")
        dest = str(Path(dest_path).expanduser().resolve())
        with sqlite3.connect(dest) as dest_conn:
            self._conn().backup(dest_conn)

    def schema_version(self) -> int:
        """Return the current schema version recorded in schema_info."""
        row = self._conn().execute(
            "SELECT value FROM schema_info WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run incremental SQL migrations from from_version up to CURRENT_SCHEMA_VERSION.

        Additive steps that were already applied (e.g. a column created by the
        current DDL on a partially pre-versioned database) are skipped instead
        of failing, so mixed-state databases upgrade cleanly.
        """
        for v in range(from_version + 1, self.CURRENT_SCHEMA_VERSION + 1):
            for sql in self._MIGRATIONS.get(v, []):
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" in str(exc).lower():
                        continue
                    raise

    def log_event(
        self,
        agent_id: str,
        event_type: str,
        *,
        session_id: str | None = None,
        fragment_id: str | None = None,
        query: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a session event. Returns the new event id."""
        event_id = str(uuid.uuid4())
        now      = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn().execute(
                "INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, agent_id, session_id, event_type, fragment_id, query,
                 json.dumps(metadata or {}), now),
            )
            self._conn().commit()
        return event_id

    def list_events(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return session events newest-first."""
        sql    = "SELECT * FROM session_events WHERE agent_id = ?"
        params: list[object] = [agent_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._conn().execute(sql, params).fetchall()
        return [
            {
                "id":          str(row["id"]),
                "agent_id":    str(row["agent_id"]),
                "session_id":  _optional_str(row["session_id"]),
                "event_type":  str(row["event_type"]),
                "fragment_id": _optional_str(row["fragment_id"]),
                "query":       _optional_str(row["query"]),
                "metadata":    json.loads(str(row["metadata"])),
                "created_at":  str(row["created_at"]),
            }
            for row in rows
        ]

    def _init_schema(self) -> None:
        conn = self._conn()
        for ddl in [self._DDL_SCHEMA_INFO, self._DDL_BLOCKS, self._DDL_FRAGMENTS,
                    self._DDL_DOCUMENTS, self._DDL_NODES, self._DDL_EDGES,
                    self._DDL_SESSION_EVENTS, self._DDL_BLOCK_NODES, self._DDL_BLOCK_LINKS]:
            conn.execute(ddl)
        row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
        if row is None:
            # Determine whether this is a truly fresh DB or a pre-versioned one
            # (created before schema_info existed). A pre-versioned DB has data in
            # fragments/blocks/edges; a fresh DB has none.
            has_data = conn.execute(
                "SELECT EXISTS(SELECT 1 FROM fragments LIMIT 1)"
            ).fetchone()[0]
            if has_data:
                # Pre-versioned DB: treat as schema version 0 and migrate forward
                starting_version = 0
                self._run_migrations(conn, starting_version)
            # Either way, stamp the current version now that tables are up-to-date
            conn.execute(
                "INSERT INTO schema_info VALUES ('schema_version', ?)",
                (str(self.CURRENT_SCHEMA_VERSION),),
            )
        else:
            stored = int(row[0])
            if stored < self.CURRENT_SCHEMA_VERSION:
                self._run_migrations(conn, stored)
                conn.execute(
                    "UPDATE schema_info SET value = ? WHERE key = 'schema_version'",
                    (str(self.CURRENT_SCHEMA_VERSION),),
                )
        # Indexes are created after migrations so indexes over migrated columns
        # (e.g. edges.block_id) can be built on upgraded databases.
        for index in self._INDEXES:
            conn.execute(index)
        conn.commit()

    def _row_to_fragment(self, row: dict[str, object]) -> MemoryFragment:
        fragment = MemoryFragment(
            id=str(row["id"]),
            content=self.encryption.decrypt_str(row["content"] if isinstance(row["content"], bytes) else str(row["content"]).encode("utf-8")),
            block_id=_optional_str(row["block_id"]),
            scope=MemoryScope(str(row["scope"])),
            tier=MemoryTier(str(row["tier"])),
            agent_id=str(row["agent_id"]),
            session_id=_optional_str(row["session_id"]),
            importance=float(row["importance"]),
            state=MemoryState(str(row["state"])),
            confidence=float(row["confidence"]),
            ttl_seconds=None if row["ttl_seconds"] is None else int(row["ttl_seconds"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            last_accessed_at=_parse_optional_dt(row["last_accessed_at"]),
            tags=json.loads(str(row["tags"])),
            source=_optional_str(row["source"]),
            metadata=json.loads(str(row["metadata"])),
            embedding=json.loads(str(row["embedding"])) if row["embedding"] else None,
            contradiction_ids=json.loads(str(row["contradiction_ids"])),
        )
        if not ContentHasher.check(fragment):
            return fragment.model_copy(update={"state": MemoryState.CONTRADICTED})
        return fragment

    @staticmethod
    def _row_to_block(row: dict[str, object]) -> MemoryBlock:
        return MemoryBlock(
            id=str(row["id"]),
            name=str(row["name"]),
            agent_id=str(row["agent_id"]),
            scope=MemoryScope(str(row["scope"])),
            block_type=str(row["block_type"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
            session_id=_optional_str(row.get("session_id")),
            status=BlockStatus(str(row.get("status") or "open")),
            content_hash=_optional_str(row.get("content_hash")),
            prev_block_id=_optional_str(row.get("prev_block_id")),
            prev_block_hash=_optional_str(row.get("prev_block_hash")),
            sealed_at=_parse_optional_dt(row.get("sealed_at")),
        )

    @staticmethod
    def _row_to_document(row: dict[str, object]) -> MemoryDocument:
        return MemoryDocument(
            id=str(row["id"]),
            title=str(row["title"]),
            content=str(row["content"]),
            source=_optional_str(row["source"]),
            agent_id=str(row["agent_id"]),
            scope=MemoryScope(str(row["scope"])),
            fragment_ids=json.loads(str(row["fragment_ids"])),
            tags=json.loads(str(row["tags"])),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_node(row: dict[str, object]) -> MemoryNode:
        return MemoryNode(
            id=str(row["id"]),
            node_type=NodeType(str(row["node_type"])),
            label=str(row["label"]),
            agent_id=_optional_str(row["agent_id"]),
            fragment_id=_optional_str(row["fragment_id"]),
            document_id=_optional_str(row["document_id"]),
            weight=float(row["weight"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_block_link(row: dict[str, object]) -> BlockLink:
        return BlockLink(
            id=str(row["id"]),
            source_block_id=str(row["source_block_id"]),
            target_block_id=str(row["target_block_id"]),
            relation=str(row["relation"]),
            weight=float(row["weight"]),
            agent_id=str(row["agent_id"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_edge(row: dict[str, object]) -> MemoryEdge:
        return MemoryEdge(
            id=str(row["id"]),
            source_id=str(row["source_id"]),
            target_id=str(row["target_id"]),
            relation=EdgeRelation(str(row["relation"])),
            weight=float(row["weight"]),
            agent_id=str(row["agent_id"]),
            metadata=json.loads(str(row["metadata"])),
            created_at=_parse_dt(row["created_at"]),
            block_id=_optional_str(row.get("block_id")),
        )


def _dt_to_text(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_optional_dt(value: object) -> datetime | None:
    if value is None:
        return None
    return _parse_dt(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
