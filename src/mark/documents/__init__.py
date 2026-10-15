# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Documents: remember long text, files and transcripts as linked chunk memories.

    record = memory.store_document("handbook.md")            # or text, or a PDF path
    memory.retrieve_sync("What is the on-call rota?")         # finds the right chunk
    memory.documents.list(); memory.documents.forget(record)
    memory.remember_file("standup.m4a", transcriber=OpenAITranscriber(base_url=...))

Modules:
  - ``chunking``  structure-aware chunking (headings, code fences, sentences, overlap)
  - ``loaders``   file -> text (text, Markdown, code, HTML, PDF via the ``pdf`` extra)
  - ``ingest``    store_document / remember_file on a memory
  - ``records``   DocumentRecord
  - ``view``      DocumentsView (``memory.documents``)
"""
from __future__ import annotations

from mark.documents.chunking import Chunk, chunk_text, estimate_tokens
from mark.documents.ingest import DOCUMENT_TAG, remember_file, store_document
from mark.documents.loaders import LoadedFile, load_file
from mark.documents.records import DocumentRecord
from mark.documents.view import DocumentsView
from mark.media.errors import MissingExtraError, TranscriberRequiredError, UnsupportedFileError

__all__ = [
    "DOCUMENT_TAG",
    "Chunk",
    "DocumentRecord",
    "DocumentsView",
    "LoadedFile",
    "MissingExtraError",
    "TranscriberRequiredError",
    "UnsupportedFileError",
    "chunk_text",
    "estimate_tokens",
    "load_file",
    "remember_file",
    "store_document",
]
