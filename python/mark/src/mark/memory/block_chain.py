# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Tamper-evident sealing and chaining of memory blocks.

Sealing a block computes a deterministic SHA-256 content hash over the
immutable fields of its member fragments, nodes, and block-scoped edges,
then links the block to the previously sealed block in the agent's chain
through ``prev_block_hash``. The result is an append-only, hash-chained
history of memory blocks:

* ``verify()`` recomputes member hashes and pinpoints the exact corrupted
  fragment, node, or edge inside a block.
* ``verify_chain()`` walks the agent's sealed blocks in order and checks
  both content hashes and chain linkage.
* ``quarantine()`` isolates a block — its members drop out of retrieval and
  traversal without affecting any other block or the agent's function.

Mutable plasticity fields (importance, access counts, edge weights) are
deliberately excluded from hashes so reinforcement and decay never break
the chain.

This is a local hash chain, not a consensus ledger: it makes tampering
evident, not impossible. Anchoring sealed hashes to an external ledger and
encrypting sealed blocks are hosted (cloud) concerns exposed only through
hook contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryBlock, MemoryEdge, MemoryFragment, MemoryNode

_MEMBER_HASHES_KEY = "mark_member_hashes"


@dataclass(frozen=True)
class BlockVerification:
    """Outcome of verifying one sealed block.

    valid                 True when every member hash and the block content
                          hash match what was recorded at seal time
    block_id              the verified block
    corrupted_member_ids  ids of fragments/nodes/edges whose current hash
                          differs from the sealed hash
    missing_member_ids    ids recorded at seal time that no longer exist
    added_member_ids      ids present now that were not part of the seal
    reason                human-readable summary
    """

    valid: bool
    block_id: str
    corrupted_member_ids: list[str] = field(default_factory=list)
    missing_member_ids: list[str] = field(default_factory=list)
    added_member_ids: list[str] = field(default_factory=list)
    reason: str = "ok"


@dataclass(frozen=True)
class ChainVerification:
    """Outcome of verifying an agent's whole sealed chain.

    valid             True when every block verifies and every prev link holds
    checked           number of sealed blocks examined
    broken_block_ids  blocks whose content failed verification
    broken_links      (block_id, reason) pairs for chain-linkage failures
    """

    valid: bool
    checked: int
    broken_block_ids: list[str] = field(default_factory=list)
    broken_links: list[tuple[str, str]] = field(default_factory=list)


class BlockChain:
    """Per-agent sealing, verification, and quarantine over memory blocks.

    Obtain one through :meth:`mark.memory.mark_memory.MarkMemory.chain` or
    construct directly::

        chain = BlockChain(store, agent_id="video-agent")
        sealed = chain.seal(block.id)
        report = chain.verify(block.id)
        summary = chain.verify_chain()
    """

    def __init__(self, store: LocalMemoryStore, agent_id: str) -> None:
        self._store = store
        self.agent_id = agent_id

    # ── sealing ───────────────────────────────────────────────────────────────

    def seal(self, block_id: str) -> MemoryBlock:
        """Seal an OPEN block: hash its members and append it to the chain.

        Member hashes are stored in block metadata so verification can later
        pinpoint exactly which member changed. Returns the sealed block.
        Raises KeyError for unknown blocks and ValueError when the block is
        not OPEN.
        """
        block = self._store.get_block(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id!r}")
        if block.status != BlockStatus.OPEN:
            raise ValueError(f"Only OPEN blocks can be sealed; {block.name!r} is {block.status.value}")

        member_hashes = self._member_hashes(block_id)
        content_hash = self._content_hash(member_hashes)
        prev = self._store.latest_sealed_block(self.agent_id, exclude_id=block_id)

        metadata = dict(block.metadata)
        metadata[_MEMBER_HASHES_KEY] = member_hashes
        sealed = block.model_copy(update={
            "status": BlockStatus.SEALED,
            "content_hash": content_hash,
            "prev_block_id": prev.id if prev else None,
            "prev_block_hash": prev.content_hash if prev else None,
            "sealed_at": datetime.now(timezone.utc),
            "metadata": metadata,
        })
        self._store.store_block(sealed)
        return sealed

    # ── verification ──────────────────────────────────────────────────────────

    def verify(self, block_id: str) -> BlockVerification:
        """Re-hash a sealed block's members and compare against the seal.

        Detects corrupted members (hash mismatch), deleted members, and
        members added after sealing. OPEN blocks report valid=False with an
        explanatory reason rather than raising.
        """
        block = self._store.get_block(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id!r}")
        if block.content_hash is None:
            return BlockVerification(
                valid=False, block_id=block_id,
                reason="block has never been sealed",
            )

        sealed_hashes: dict[str, str] = dict(block.metadata.get(_MEMBER_HASHES_KEY, {}))
        current_hashes = self._member_hashes(block_id)

        corrupted = sorted(
            member_id
            for member_id, sealed_hash in sealed_hashes.items()
            if member_id in current_hashes and current_hashes[member_id] != sealed_hash
        )
        missing = sorted(set(sealed_hashes) - set(current_hashes))
        added = sorted(set(current_hashes) - set(sealed_hashes))
        content_ok = self._content_hash(sealed_hashes) == block.content_hash

        valid = not corrupted and not missing and not added and content_ok
        if valid:
            reason = "ok"
        elif not content_ok and not (corrupted or missing or added):
            reason = "sealed content_hash does not match sealed member hashes (seal record tampered)"
        else:
            parts = []
            if corrupted:
                parts.append(f"{len(corrupted)} corrupted member(s)")
            if missing:
                parts.append(f"{len(missing)} missing member(s)")
            if added:
                parts.append(f"{len(added)} member(s) added after seal")
            reason = ", ".join(parts)
        return BlockVerification(
            valid=valid,
            block_id=block_id,
            corrupted_member_ids=corrupted,
            missing_member_ids=missing,
            added_member_ids=added,
            reason=reason,
        )

    def verify_chain(self) -> ChainVerification:
        """Verify every sealed block and the hash links between them."""
        chain = self._store.sealed_chain(self.agent_id)
        broken_blocks: list[str] = []
        broken_links: list[tuple[str, str]] = []
        prev_hash_by_id = {b.id: b.content_hash for b in chain}

        for block in chain:
            if not self.verify(block.id).valid:
                broken_blocks.append(block.id)
            if block.prev_block_id is not None:
                expected = prev_hash_by_id.get(block.prev_block_id)
                if expected is None:
                    broken_links.append((block.id, f"previous block {block.prev_block_id!r} is gone"))
                elif expected != block.prev_block_hash:
                    broken_links.append((block.id, "prev_block_hash does not match previous block"))
        return ChainVerification(
            valid=not broken_blocks and not broken_links,
            checked=len(chain),
            broken_block_ids=broken_blocks,
            broken_links=broken_links,
        )

    # ── quarantine ────────────────────────────────────────────────────────────

    def quarantine(self, block_id: str) -> MemoryBlock:
        """Isolate a block: its fragments leave retrieval, traversal skips it.

        Other blocks and the agent keep working untouched. The block's data
        stays inspectable through the store for diagnosis.
        """
        block = self._store.get_block(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id!r}")
        updated = block.model_copy(update={"status": BlockStatus.QUARANTINED})
        self._store.store_block(updated)
        return updated

    def release(self, block_id: str) -> MemoryBlock:
        """Release a quarantined block back to its pre-quarantine status.

        A block that had been sealed returns to SEALED; one that was never
        sealed returns to OPEN.
        """
        block = self._store.get_block(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id!r}")
        restored = BlockStatus.SEALED if block.content_hash is not None else BlockStatus.OPEN
        updated = block.model_copy(update={"status": restored})
        self._store.store_block(updated)
        return updated

    # ── hashing internals ─────────────────────────────────────────────────────

    def _member_hashes(self, block_id: str) -> dict[str, str]:
        """Hash every member of a block over its immutable fields only."""
        hashes: dict[str, str] = {}
        for fragment in self._store.fragments_for_block(block_id):
            hashes[fragment.id] = _hash_fragment(fragment)
        for node in self._store.nodes_in_block(block_id):
            hashes[node.id] = _hash_node(node)
        for edge in self._store.edges_in_block(block_id):
            hashes[edge.id] = _hash_edge(edge)
        return hashes

    @staticmethod
    def _content_hash(member_hashes: dict[str, str]) -> str:
        """Deterministic block hash over sorted member hashes."""
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
