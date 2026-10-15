"""Schema definitions for the SQLite store (data only, no behaviour).

CREATE statements and per-version migrations consumed by ``SchemaMixin``,
kept on a class so ``LocalMemoryStore`` exposes them as class attributes.

Invariants:
  - every statement is idempotent (IF NOT EXISTS), so opening an existing
    database re-runs them safely;
  - the _DDL_* column order matches what ALTER TABLE produces when the
    _COLUMN_MIGRATIONS are applied to an old database, so fresh and
    upgraded databases have identical shapes.
"""
from __future__ import annotations

from mark.store._sqlite.codec import _FTS_ML_TABLE, _FTS_TABLE
from mark.store._sqlite.vector_snapshot import SNAPSHOT_DDL


class SchemaDDL:
    """Table, index and migration definitions shared by the store mixins."""

    CURRENT_SCHEMA_VERSION = 6

    # Column additions per schema version: (table, column, declaration).
    # Applied by inspecting PRAGMA table_info, never by trusting the stamped
    # version or row counts, so partially-migrated databases converge.
    # Column order must match the _DDL_* definitions (ALTER TABLE appends).
    _COLUMN_MIGRATIONS: dict[int, list[tuple[str, str, str]]] = {
        # 1: baseline
        # 2: graph-scoped memory blocks — block provenance columns, block-scoped edges.
        2: [
            ("blocks", "session_id", "TEXT"),
            ("blocks", "status", "TEXT DEFAULT 'open'"),
            ("blocks", "content_hash", "TEXT"),
            ("blocks", "prev_block_id", "TEXT"),
            ("blocks", "prev_block_hash", "TEXT"),
            ("blocks", "sealed_at", "TEXT"),
            ("edges", "block_id", "TEXT"),
        ],
        # 3: binary float32 embeddings (JSON column kept for legacy rows).
        3: [
            ("fragments", "embedding_blob", "BLOB"),
        ],
    }

    # Non-column statements per schema version. All must be idempotent.
    _MIGRATIONS: dict[int, list[str]] = {
        2: [
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
        ],
        # 3: indexes are created from _INDEXES after migrations; FTS backfill
        #    is driven by the fts_backfilled meta key.
        3: [],
        # 4: bi-temporal facts (subject, attribute) -> value with validity intervals.
        4: [
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                attribute TEXT NOT NULL,
                attribute_key TEXT NOT NULL,
                value BLOB NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                recorded_at TEXT NOT NULL,
                retracted_at TEXT,
                source_fragment_id TEXT,
                metadata TEXT DEFAULT '{}'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_fact_key ON facts(agent_id, subject_key, attribute_key, valid_from)",
            "CREATE INDEX IF NOT EXISTS idx_fact_current ON facts(agent_id, valid_to, retracted_at)",
        ],
        # 5: automatic fact extraction — a durable job queue for slow extractors
        #    (one job per fragment and extractor) and the ledger of extracted
        #    assertions that makes re-extraction idempotent.
        5: [
            """
            CREATE TABLE IF NOT EXISTS fact_jobs (
                fragment_id TEXT NOT NULL,
                extractor TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                occurrence INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (fragment_id, extractor)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_fact_jobs_status ON fact_jobs(status, updated_at)",
            """
            CREATE TABLE IF NOT EXISTS fact_extractions (
                agent_id TEXT NOT NULL,
                key TEXT NOT NULL,
                fragment_id TEXT,
                fact_id TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (agent_id, key)
            )
            """,
        ],
        # 6: a partial index finds the next TTL expiry (only TTL rows pay for it; a
        #    covering retrievable-id index cost ~25% of write speed: not used).
        #    The multilingual FTS table is set up by _init_fts.
        6: [
            "CREATE INDEX IF NOT EXISTS idx_frag_ttl ON fragments(agent_id, created_at, ttl_seconds)"
            " WHERE ttl_seconds IS NOT NULL",
            *SNAPSHOT_DDL,  # vector-index snapshots (see vector_snapshot)
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
            contradiction_ids TEXT DEFAULT '[]',
            embedding_blob BLOB
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
    # Lexical channel. Kept in sync from Python (not triggers) because the
    # fragments.content column may hold ciphertext. rowid == fragments.rowid.
    _DDL_FTS = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
            content,
            tags,
            fragment_id UNINDEXED,
            agent_id UNINDEXED,
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
    """
    # Multilingual channel (fts_text): no stemmer; marks/symbols are token chars.
    _DDL_FTS_ML = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_ML_TABLE} USING fts5(
            content, fragment_id UNINDEXED, agent_id UNINDEXED,
            tokenize = "unicode61 remove_diacritics 2 categories 'L* N* Co M* S*'"
        )
    """
    _DDL_FTS_ML_FALLBACK = _DDL_FTS_ML.replace(" categories 'L* N* Co M* S*'", "")
    # Deletes need no plaintext: a trigger mirrors them. (An UPDATE OF content
    # trigger was measured to slow every upsert by ~40%; _fts_upsert deletes.)
    _DDL_FTS_ML_TRIGGER = f"""
        CREATE TRIGGER IF NOT EXISTS {_FTS_ML_TABLE}_delete AFTER DELETE ON fragments
        BEGIN DELETE FROM {_FTS_ML_TABLE} WHERE rowid = old.rowid; END
    """
    _INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_frag_agent   ON fragments(agent_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_state   ON fragments(state)",
        "CREATE INDEX IF NOT EXISTS idx_frag_scope   ON fragments(scope)",
        "CREATE INDEX IF NOT EXISTS idx_frag_tier    ON fragments(tier)",
        "CREATE INDEX IF NOT EXISTS idx_frag_session ON fragments(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_block   ON fragments(block_id)",
        "CREATE INDEX IF NOT EXISTS idx_frag_agent_state_imp"
        " ON fragments(agent_id, state, importance DESC)",
        "CREATE INDEX IF NOT EXISTS idx_node_frag    ON nodes(fragment_id)",
        "CREATE INDEX IF NOT EXISTS idx_node_label_agent ON nodes(label, agent_id)",
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
