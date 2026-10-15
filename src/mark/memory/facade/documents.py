# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Documents and files on MarkMemory: store_document, remember_file, documents.

Thin facade over :mod:`mark.documents`; the logic (chunking, loading,
routing by file content) lives there. Chunks are written with the memory's
own ``store_many_sync``, so middleware, governance and fact extraction
apply to documents exactly as to any other write.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mark.memory.facade.core import MemoryCore

if TYPE_CHECKING:
    from mark.documents import DocumentRecord, DocumentsView


class DocumentsMixin(MemoryCore):
    """store_document / remember_file / documents for one agent memory."""

    #: Default transcriber for audio/video in ``remember_file`` (anything with
    #: ``transcribe(path) -> str``, e.g. ``mark.media.OpenAITranscriber``).
    transcriber: Any = None

    def store_document(
        self,
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
    ) -> "DocumentRecord":
        """Remember a long text or a file as chunk memories linked to one document.

        text_or_path   the text itself, or a path to .txt/.md/code/.html, or a
                       PDF (needs ``pip install 'mark-sdk[pdf]'``)
        chunk_tokens   target chunk size in (estimated) tokens; ``overlap`` is
                       repeated between neighbouring chunks of a section
        extract_facts  as in store_sync; the count is ``record.facts_extracted``

        Chunks follow headings, paragraphs and code fences, and each starts
        with its heading path. Storing the same text again returns the
        existing record.
        """
        from mark.documents.ingest import store_document

        return store_document(
            self, text_or_path, title=title, session_id=session_id, occurred_at=occurred_at, metadata=metadata,
            tags=tags, chunk_tokens=chunk_tokens, overlap=overlap, extract_facts=extract_facts,
        )

    def remember_file(
        self,
        path: str | Path,
        *,
        caption: str | None = None,
        subjects: Iterable[str] = (),
        transcriber: Any = None,
        **options: Any,
    ) -> Any:
        """Remember any file, routed by its content (not its name).

        Text, Markdown, code, HTML and PDF become documents (see
        :meth:`store_document`; ``options`` are passed on); images go to
        ``memory.images`` (``caption``/``subjects``); audio and video are
        transcribed with ``transcriber`` (or ``memory.transcriber``) and the
        transcript is stored as a document linked to the media file. Other
        content raises :class:`mark.documents.UnsupportedFileError`.
        """
        from mark.documents.ingest import remember_file

        return remember_file(self, path, caption=caption, subjects=subjects, transcriber=transcriber, **options)

    @property
    def documents(self) -> "DocumentsView":
        """Stored documents: list / get / chunks / count / forget."""
        from mark.documents.view import DocumentsView

        return DocumentsView(self)
