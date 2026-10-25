# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Backup and restore: JSON Lines export/import of memories, facts and documents.

Format (one JSON object per line, ``type`` first)::

    {"type": "header", "format": "mark-jsonl", "version": 1, "agents": [...], ...}
    {"type": "memory", "id": ..., "agent_id": ..., "content": ..., "tags": [...], ...}
    {"type": "fact", "agent_id": ..., "subject": ..., "attribute": ..., "value": ..., "valid_from": ...}
    {"type": "document", "agent_id": ..., "id": ..., "title": ..., ...}

Embeddings are not exported: an import re-embeds with the *target* store's
embedder, so a backup can move between embedders. Import is idempotent —
memories whose id already exists are skipped, and a fact already stored with
the same value at the same time is skipped — so restoring the same file twice
changes nothing. Malformed lines are skipped and reported, never fatal. Exports from an
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
            for fragment in store.page_fragments(agent, limit=None, include_unreadable=decrypt):
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


def import_jsonl(path: str | Path, memory_for: Callable[[str], Any], *, agent_id: str | None = None) -> dict[str, Any]:
    """Restore an export; ``agent_id`` forces every record into one namespace.

    Returns counts of what was added, ``skipped`` records (already present, or
    of an unknown type) and, when any, ``invalid`` lines with one reason per
    line in ``problems`` (at most :data:`MAX_PROBLEMS`). A bad line never aborts the
    restore. Re-importing a file adds nothing: memories are matched by id and
    facts by (subject, attribute, value, valid_from). Raises
    :class:`mark.errors.InvalidInputError` only for a file that is not a MARK export.
    """
    from mark.errors import InvalidInputError

    counts: dict[str, Any] = {"memories": 0, "facts": 0, "documents": 0, "skipped": 0, "invalid": 0,
                              "problems": []}
    pending: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    known_facts: dict[str, set[tuple[str, str, str, str]]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as source:
        first = source.readline()
        try:
            header = json.loads(first) if first.strip() else {}
        except ValueError:
            header = {}
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            raise InvalidInputError(f"{path} is not a MARK export (missing {FORMAT!r} header)")
        if int(header.get("version", 0)) > VERSION:
            raise InvalidInputError(f"{path} uses export version {header['version']}; upgrade mark-sdk")
        for number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not a JSON object")
            except ValueError as exc:
                _invalid(counts, number, f"not valid JSON ({exc})")
                continue
            agent = agent_id or str(record.get("agent_id") or "__mark__")
            kind = record.pop("type", None)
            try:
                if kind == "memory":
                    record["agent_id"] = agent
                    pending.setdefault(agent, []).append((number, record))
                    if len(pending[agent]) >= _BATCH:
                        _flush(memory_for(agent), pending.pop(agent), counts)
                elif kind == "fact":
                    _flush(memory_for(agent), pending.pop(agent, []), counts)  # facts may cite memories
                    _import_fact(memory_for(agent), agent, record, known_facts, counts)
                elif kind == "document":
                    record["agent_id"] = agent
                    added = memory_for(agent)._store.import_document_row(record)
                    counts["documents" if added else "skipped"] += 1
                else:
                    counts["skipped"] += 1
            except (KeyError, TypeError, ValueError) as exc:
                _invalid(counts, number, f"bad {kind or 'record'} ({_reason(exc)})")
    for agent, records in pending.items():
        _flush(memory_for(agent), records, counts)
    if not counts["invalid"]:  # a clean import keeps the historical result shape
        del counts["invalid"], counts["problems"]
    return counts


#: Reasons kept in ``import_jsonl(...)["problems"]``; the ``invalid`` count is exact.
MAX_PROBLEMS = 20


def _invalid(counts: dict[str, Any], number: int, reason: str) -> None:
    counts["invalid"] += 1
    if len(counts["problems"]) < MAX_PROBLEMS:
        counts["problems"].append(f"line {number}: {reason}")


def _reason(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return f"missing field {exc.args[0]!r}"
    errors = getattr(exc, "errors", None)
    if callable(errors):  # pydantic: field + message, not the whole dump
        return "; ".join(f"{'.'.join(map(str, e.get('loc', ()))) or 'record'}: {e.get('msg')}" for e in errors())
    return str(exc).splitlines()[0][:200]


def _import_fact(memory: Any, agent: str, record: dict[str, Any],
                 known: dict[str, set[tuple[str, str, str, str]]], counts: dict[str, Any]) -> None:
    """Record one exported fact unless the same value at the same time is already stored.

    ``fact_set`` only skips re-asserting the *latest* value at a time, so a
    same-time sequence (6 then 7) re-imported used to become 6, 7, 6, 7.
    """
    from mark.store._sqlite.codec import _fact_key

    store = memory._store
    import_row = getattr(store, "import_fact_row", None)
    if import_row is not None:  # exact, idempotent restore (keeps recorded_at)
        counts["facts" if import_row(agent, record) else "skipped"] += 1
        return
    valid_from = datetime.fromisoformat(str(record["valid_from"]).replace("Z", "+00:00"))
    if agent not in known:
        known[agent] = {_fact_identity(_fact_key, row["subject"], row["attribute"], row["value"],
                                       row["valid_from"]) for row in store.fact_rows(agent)}
    identity = _fact_identity(_fact_key, record["subject"], record["attribute"], record["value"], valid_from)
    if identity in known[agent]:
        counts["skipped"] += 1
        return
    store.fact_set(agent, record["subject"], record["attribute"], record["value"], valid_from=valid_from,
                   source_fragment_id=record.get("source_fragment_id"), metadata=record.get("metadata") or {})
    known[agent].add(identity)
    counts["facts"] += 1


def _fact_identity(key: Callable[[str], str], subject: str, attribute: str, value: Any,
                   valid_from: Any) -> tuple[str, str, str, str]:
    from datetime import timezone

    when = valid_from if isinstance(valid_from, datetime) else datetime.fromisoformat(str(valid_from))
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return key(subject), key(attribute), json.dumps(value, sort_keys=True, default=str), \
        when.astimezone(timezone.utc).isoformat()


def _flush(memory: Any, records: list[tuple[int, dict[str, Any]]], counts: dict[str, Any]) -> None:
    """Re-embed and store a batch of exported memories, skipping ids already present.

    Each record is validated on its own, so one malformed memory is reported
    as an invalid line instead of failing the whole batch.
    """
    if not records:
        return
    from mark.security.hasher import ContentHasher
    from mark.store._sqlite.codec import _RETRIEVABLE_VALUES
    from mark.types import MemoryFragment

    store = memory._store
    existing = store.get_many([r.get("id") for _, r in records if isinstance(r.get("id"), str)])
    indexable, plain = [], []
    for number, record in records:
        if record.get("id") in existing:
            counts["skipped"] += 1
            continue
        try:
            # The digest may be keyed by the source store's key: recompute it here.
            record["metadata"] = {k: v for k, v in (record.get("metadata") or {}).items() if k != "content_hash"}
            fragment = ContentHasher.attach(MemoryFragment.model_validate(record), store.encryption)
        except (TypeError, ValueError, AttributeError) as exc:  # pydantic ValidationError is a ValueError
            _invalid(counts, number, f"bad memory ({_reason(exc)})")
            continue
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

    def import_jsonl(self, path: str | Path) -> dict[str, Any]:
        """Restore an export into this agent (whatever agent it came from)."""
        return import_jsonl(path, lambda _agent: self, agent_id=self.agent_id)
