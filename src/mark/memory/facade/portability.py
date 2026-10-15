# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Backup and restore: JSON Lines export/import of memories, facts and documents.

Format (one JSON object per line, ``type`` first)::

    {"type": "header", "format": "mark-jsonl", "version": 1, "agents": [...], ...}
    {"type": "memory", "id": ..., "agent_id": ..., "content": ..., "tags": [...], ...}
    {"type": "fact", "agent_id": ..., "subject": ..., "attribute": ..., "value": ..., "valid_from": ...}
    {"type": "document", "agent_id": ..., "id": ..., "title": ..., ...}

Embeddings are not exported: an import re-embeds with the *target* store's
embedder, so a backup can move between embedders. Import is idempotent —
memories whose id already exists are skipped and re-asserting a fact is a
no-op — so restoring the same file twice changes nothing. Exports from an
encrypted store hold plaintext, so they require ``decrypt=True``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from mark.memory.facade.core import MemoryCore

FORMAT = "mark-jsonl"
VERSION = 1
_BATCH = 256


def export_jsonl(store: Any, path: str | Path, agent_ids: Iterable[str], *, decrypt: bool = False) -> dict[str, int]:
    """Write every memory, live fact and document of *agent_ids* to *path*; returns counts."""
    from mark._version import __version__

    if getattr(store, "encrypted", False) and not decrypt:
        from mark.errors import StoreEncryptionError

        raise StoreEncryptionError(
            "this store is encrypted; an export writes plaintext, so pass decrypt=True "
            "(CLI: --decrypt) to confirm and keep the file somewhere safe"
        )
    agents = list(agent_ids)
    counts = {"memories": 0, "facts": 0, "documents": 0}
    target = Path(path).expanduser()
    # 0600 from the first byte: a backup holds the same text as the store.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        header = {"type": "header", "format": FORMAT, "version": VERSION, "mark_version": __version__,
                  "agents": agents, "exported_at": datetime.now().astimezone().isoformat()}
        out.write(json.dumps(header) + "\n")
        for agent in agents:
            for fragment in store.page_fragments(agent, limit=None):
                row = fragment.model_dump(mode="json", exclude={"embedding"})
                out.write(json.dumps({"type": "memory", **row}, ensure_ascii=False) + "\n")
                counts["memories"] += 1
            for fact in store.fact_rows(agent):
                row = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in fact.items()
                       if k not in {"id", "valid_to"}}  # valid_to is derived on import
                out.write(json.dumps({"type": "fact", "agent_id": agent, **row}, ensure_ascii=False) + "\n")
                counts["facts"] += 1
            for document in store.document_rows(agent):
                out.write(json.dumps({"type": "document", **document}, ensure_ascii=False, default=str) + "\n")
                counts["documents"] += 1
    return counts


def import_jsonl(path: str | Path, memory_for: Callable[[str], Any], *, agent_id: str | None = None) -> dict[str, int]:
    """Restore an export; ``agent_id`` forces every record into one namespace.

    Returns counts of what was added and of ``skipped`` records (already
    present, or of an unknown type). Raises :class:`mark.errors.InvalidInputError`
    for a file that is not a MARK export.
    """
    from mark.errors import InvalidInputError

    counts = {"memories": 0, "facts": 0, "documents": 0, "skipped": 0}
    pending: dict[str, list[dict[str, Any]]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as source:
        first = source.readline()
        try:
            header = json.loads(first) if first.strip() else {}
        except ValueError:
            header = {}
        if header.get("format") != FORMAT:
            raise InvalidInputError(f"{path} is not a MARK export (missing {FORMAT!r} header)")
        if int(header.get("version", 0)) > VERSION:
            raise InvalidInputError(f"{path} uses export version {header['version']}; upgrade mark-sdk")
        for number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise InvalidInputError(f"{path}:{number}: not valid JSON ({exc})") from None
            agent = agent_id or str(record.get("agent_id") or "__mark__")
            kind = record.pop("type", None)
            if kind == "memory":
                record["agent_id"] = agent
                pending.setdefault(agent, []).append(record)
                if len(pending[agent]) >= _BATCH:
                    _flush(memory_for(agent), pending.pop(agent), counts)
            elif kind == "fact":
                _flush(memory_for(agent), pending.pop(agent, []), counts)  # facts may cite memories
                memory = memory_for(agent)
                before = len(memory._store.fact_history(agent, record["subject"], record["attribute"]))
                memory._store.fact_set(
                    agent, record["subject"], record["attribute"], record["value"],
                    valid_from=datetime.fromisoformat(record["valid_from"]),
                    source_fragment_id=record.get("source_fragment_id"),
                    metadata=record.get("metadata") or {},
                )
                after = len(memory._store.fact_history(agent, record["subject"], record["attribute"]))
                counts["facts" if after > before else "skipped"] += 1
            elif kind == "document":
                record["agent_id"] = agent
                added = memory_for(agent)._store.import_document_row(record)
                counts["documents" if added else "skipped"] += 1
            else:
                counts["skipped"] += 1
    for agent, records in pending.items():
        _flush(memory_for(agent), records, counts)
    return counts


def _flush(memory: Any, records: list[dict[str, Any]], counts: dict[str, int]) -> None:
    """Re-embed and store a batch of exported memories, skipping ids already present."""
    if not records:
        return
    from mark.security.hasher import ContentHasher
    from mark.store._sqlite.codec import _RETRIEVABLE_VALUES
    from mark.types import MemoryFragment

    store = memory._store
    existing = store.get_many([r["id"] for r in records])
    indexable, plain = [], []
    for record in records:
        if record["id"] in existing:
            counts["skipped"] += 1
            continue
        # The digest may be keyed by the source store's key: recompute it here.
        record["metadata"] = {k: v for k, v in (record.get("metadata") or {}).items() if k != "content_hash"}
        fragment = ContentHasher.attach(MemoryFragment.model_validate(record), store.encryption)
        (indexable if fragment.state.value in _RETRIEVABLE_VALUES else plain).append(fragment)
    if plain:
        store.store_many(plain)
    if indexable:
        memory._pipeline.index_fragments(indexable)
    counts["memories"] += len(indexable) + len(plain)


class PortabilityMixin(MemoryCore):
    """``memory.export_jsonl(path)`` / ``memory.import_jsonl(path)`` for one agent."""

    def export_jsonl(self, path: str | Path, *, decrypt: bool = False) -> dict[str, int]:
        """Back up this agent's memories, facts and documents to a JSON Lines file."""
        return export_jsonl(self._store, path, [self.agent_id], decrypt=decrypt)

    def import_jsonl(self, path: str | Path) -> dict[str, int]:
        """Restore an export into this agent (whatever agent it came from)."""
        return import_jsonl(path, lambda _agent: self, agent_id=self.agent_id)
