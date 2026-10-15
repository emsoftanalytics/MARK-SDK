"""Decoding SQLite rows into MARK domain models.

Fragment decoding also decrypts content and verifies its content hash; a
mismatch (at-rest tampering or corruption) relabels the fragment
CONTRADICTED and reports it so the caller can persist that state.
"""
from __future__ import annotations

import json

from mark.security import ContentHasher
from mark.store._sqlite.codec import _as_bytes, _decode_embedding, _optional_str, _parse_dt, _parse_optional_dt
from mark.types import BlockLink, BlockStatus, MemoryBlock, MemoryDocument, MemoryEdge, MemoryFragment, MemoryNode
from mark.types import MemoryScope, MemoryState, MemoryTier
from mark.types.graph import EdgeRelation, NodeType


class RowDecodingMixin:
    """Row -> model converters shared by every read path."""

    def _decode_fragment(self, row: dict[str, object]) -> tuple[MemoryFragment, bool]:
        """Decode a row; returns (fragment, tampered). Tampered rows are relabelled CONTRADICTED."""
        fragment = MemoryFragment(
            id=str(row["id"]),
            content=self.encryption.decrypt_str(_as_bytes(row["content"])),
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
            embedding=_decode_embedding(row.get("embedding_blob"), row["embedding"]),
            contradiction_ids=json.loads(str(row["contradiction_ids"])),
        )
        # Report tampering only on the first detection: a row already stored as
        # CONTRADICTED needs no further write.
        if not ContentHasher.check(fragment, self.encryption):
            already = fragment.state == MemoryState.CONTRADICTED
            return fragment.model_copy(update={"state": MemoryState.CONTRADICTED}), not already
        return fragment, False

    def _row_to_fragment(self, row: dict[str, object]) -> MemoryFragment:
        return self._decode_fragment(row)[0]

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
