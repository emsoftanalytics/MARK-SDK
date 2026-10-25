# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Deterministic hashing rules for sealed memory blocks.

Only *immutable* fields are hashed. Plasticity fields that MARK legitimately
changes after sealing (importance, state, access counters, node/edge
weights) are excluded, so reinforcement and decay never break a seal while
any edit to content, identity, provenance, or creation time does.

Payloads are serialised as canonical JSON (sorted keys, no whitespace) so
the same member always yields the same SHA-256 digest across processes.
"""
from __future__ import annotations

import hashlib
import json

from mark.types import MemoryEdge, MemoryFragment, MemoryNode

#: Block-metadata key under which per-member hashes are recorded at seal time.
_MEMBER_HASHES_KEY = "mark_member_hashes"


def block_content_hash(member_hashes: dict[str, str]) -> str:
    """Deterministic block hash over sorted ``(member_id, hash)`` pairs."""
    canonical = json.dumps(sorted(member_hashes.items()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _hash_fragment(fragment: MemoryFragment) -> str:
    # importance/state/access metadata are mutable by design — excluded.
    return _canonical_hash({
        "kind": "fragment",
        "id": fragment.id,
        "content": fragment.content,
        "agent_id": fragment.agent_id,
        "session_id": fragment.session_id,
        "source": fragment.source,
        "created_at": fragment.created_at.isoformat(),
    })


def _hash_node(node: MemoryNode) -> str:
    # weight is plasticity-mutable — excluded.
    return _canonical_hash({
        "kind": "node",
        "id": node.id,
        "node_type": node.node_type.value,
        "label": node.label,
        "fragment_id": node.fragment_id,
        "created_at": node.created_at.isoformat(),
    })


def _hash_edge(edge: MemoryEdge) -> str:
    # weight is reinforced on co-activation — excluded.
    return _canonical_hash({
        "kind": "edge",
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation": edge.relation.value,
        "created_at": edge.created_at.isoformat(),
    })
