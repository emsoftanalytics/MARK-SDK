# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Store documents and files in a memory: chunk, write chunk memories, record the parent.

Used by ``memory.store_document`` and ``memory.remember_file``. Chunk
memories are ordinary memories, so every write goes through the memory's
own ``store_many_sync`` (middleware, governance, redaction, consolidation
gate, fact extraction) — documents never bypass write-time policy.

Each chunk carries ``metadata = {"source": "document", "document_id",
"document_title"}`` and the tag ``mark:document``; the parent record keeps
the chunk ids in document order, the full text (encrypted like memories)
and a SHA-256 of the text, which makes storing the same text twice a no-op.

The parent record is a second copy of the text, so the memory's write
policy runs on it *before* anything is stored: secrets in the text, title,
source and caller metadata are redacted exactly as in the chunks, and
instruction-like text is flagged under ``metadata["mark_security"]``.
Without this the raw secret would survive in ``documents.content`` (and in
the SQLite files) even though every chunk was redacted.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mark.documents.chunking import chunk_text
from mark.documents.loaders import OFFICE_TYPES, LoadedFile, load_file
from mark.documents.records import DocumentRecord
from mark.errors import InvalidMemoryError
from mark.media.errors import TranscriberRequiredError, UnsupportedFileError
from mark.media.files import reject_binary, sha256_file, sniff
from mark.media.transcribers import run_transcriber
from mark.security.injection import UNTRUSTED_TAG
from mark.security.write_policy import DEFAULT_WRITE_POLICY, WritePolicy

DOCUMENT_TAG = "mark:document"
_RESERVED = ("source", "document_id", "document_title")


def looks_like_path(value: object) -> bool:
    """A Path, or a one-line string naming an existing file."""
    if isinstance(value, Path):
        return True
    if not isinstance(value, str) or "\n" in value or len(value) > 4096 or not value.strip():
        return False
    try:
        return Path(value).expanduser().is_file()
    except OSError:
        return False


def store_document(
    memory: Any,
    text_or_path: str | Path,
    *,
    title: str | None = None,
    session_id: str | None = None,
    occurred_at: datetime | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Iterable[str] = (),
    chunk_tokens: int = 300,
    overlap: int = 40,
    extract_facts: bool | None = None,
    source: str | None = None,
) -> DocumentRecord:
    """Chunk *text_or_path* into memories linked to one document record (see module docstring)."""
    loaded: LoadedFile | None = load_file(text_or_path) if looks_like_path(text_or_path) else None
    text = loaded.text if loaded else str(text_or_path)
    if not text.strip():
        raise InvalidMemoryError("document text is empty", reason="empty")
    extra = dict(metadata or {})
    reject_binary(extra)
    clash = sorted(set(extra) & set(_RESERVED))
    if clash:
        raise ValueError(f"metadata keys reserved for documents: {clash}")
    policy = _write_policy(memory)
    # Redact before hashing: the digest is stored too, and the same secret-free
    # text must map to the same record whether or not it was redacted.
    decision = policy.apply(text, metadata=extra or None)
    text, extra = decision.content, dict(decision.metadata or {})
    security = extra.pop("mark_security", None)
    title, title_kinds = policy.redact_text(
        (title or (loaded.title if loaded else _default_title(text))).strip())
    source, source_kinds = policy.redact_text(source or (loaded.source if loaded else None) or "")
    if title_kinds or source_kinds:
        security = dict(security or {})
        security["redacted"] = sorted({*security.get("redacted", ()), *title_kinds, *source_kinds})
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = memory._store.document_find(memory.agent_id, digest)
    if existing is not None:
        return _record(existing)
    chunks = chunk_text(text, title=title, kind=loaded.kind if loaded else "markdown",
                        chunk_tokens=chunk_tokens, overlap=overlap)
    document_id = str(uuid.uuid4())
    chunk_meta = {**extra, "source": "document", "document_id": document_id, "document_title": title}
    # A chunk of a flagged document stays untrusted even when its own text screens clean:
    # the planted instruction may sit in another chunk that is recalled alongside it.
    inherited = [UNTRUSTED_TAG] if UNTRUSTED_TAG in (decision.tags or ()) else []
    ids = memory.store_many_sync(
        [chunk.text for chunk in chunks], metadata=chunk_meta, tags=[DOCUMENT_TAG, *tags, *inherited],
        source="document",
        session_id=session_id, occurred_at=[occurred_at] * len(chunks) if occurred_at is not None else None,
        extract_facts=extract_facts,
    )
    facts = int(getattr(memory, "last_extracted", 0) or 0)
    row_meta = {**extra, "sha256": digest, "word_count": len(text.split()), "chunk_tokens": chunk_tokens,
                "overlap": overlap, "media_type": loaded.media_type if loaded else "text/plain"}
    if security:
        row_meta["mark_security"] = security
    row = {
        "id": document_id, "agent_id": memory.agent_id, "title": title, "content": text,
        "source": source or None, "fragment_ids": ids, "tags": list(tags),
        "created_at": datetime.now(timezone.utc), "metadata": row_meta,
    }
    memory._store.document_put(row)
    return _record(row, facts_extracted=facts)


def remember_file(memory: Any, path: str | Path, *, caption: str | None = None, subjects: Iterable[str] = (),
                  transcriber: Any = None, **options: Any) -> Any:
    """Remember any file by its content: text/PDF -> document, image -> image memory,
    audio/video -> transcript document (needs a transcriber). Returns the record."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f"file not found: {file}")
    media_type = sniff(file)
    if media_type.startswith("image/"):
        image_options = {k: options[k] for k in ("session_id", "occurred_at", "metadata", "tags") if k in options}
        return memory.images.remember(file, caption=caption, subjects=subjects, **image_options)
    if media_type in ("text/plain", "application/pdf") or media_type in OFFICE_TYPES:
        return store_document(memory, file, **options)
    if media_type.startswith(("audio/", "video/")):
        transcriber = transcriber or getattr(memory, "transcriber", None)
        if transcriber is None:
            raise TranscriberRequiredError(
                f"{file.name} is {media_type}; pass transcriber=... (e.g. mark.media.OpenAITranscriber(base_url=...)) "
                "or set memory.transcriber to remember what it says")
        text = run_transcriber(transcriber, file)
        if not text:
            raise UnsupportedFileError(f"{file.name}: the transcriber returned no text")
        meta = {**dict(options.pop("metadata", None) or {}), "modality": media_type.split("/")[0],
                "media_path": str(file), "media_sha256": sha256_file(file), "media_type": media_type}
        title = options.pop("title", None) or f"Transcript of {file.name}"
        return store_document(memory, text, title=title, metadata=meta, source=str(file), **options)
    raise UnsupportedFileError(
        f"{file.name} looks like {media_type}; MARK remembers text, Markdown, code, HTML, PDF, Word/Excel/"
        "PowerPoint (docx/xlsx/pptx), images, and "
        "audio/video with a transcriber")


def _write_policy(memory: Any) -> WritePolicy:
    """The memory's write policy (redaction + injection screening), else the default one."""
    resolve = getattr(memory, "_write_policy_in_effect", None)
    policy = resolve() if callable(resolve) else None
    return policy if isinstance(policy, WritePolicy) else DEFAULT_WRITE_POLICY


def _default_title(text: str) -> str:
    """First heading or first line, shortened."""
    for line in text.splitlines():
        line = re.sub(r"^\s*#+\s*", "", line).strip()
        if line:
            return line if len(line) <= 60 else line[:57].rstrip() + "..."
    return "document"


def _record(row: Mapping[str, Any], *, facts_extracted: int = 0) -> DocumentRecord:
    return DocumentRecord(id=row["id"], title=row["title"], chunk_ids=tuple(row["fragment_ids"]),
                          created_at=row["created_at"], source=row.get("source"), metadata=dict(row["metadata"]),
                          text=row.get("content"), facts_extracted=facts_extracted)
