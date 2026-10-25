# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Load a file as document text: plain text, Markdown, source code, HTML, PDF, Office.

The file's *content* decides how it is read (see
:func:`mark.media.files.sniff`): text is decoded as UTF-8 (BOM-aware, with
a Latin-1 fallback), HTML is reduced to text with its headings kept as
Markdown headings, and PDF text is extracted with the optional ``pypdf``
dependency (``pip install 'mark-sdk[pdf]'``), one "## Page N" section per
page so recall can say where a passage came from. Word, Excel and
PowerPoint files (docx/xlsx/pptx) are read with the standard library (see
:mod:`mark.documents.office`). An empty file or anything else raises
:class:`~mark.media.errors.UnsupportedFileError` with a hint.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from mark.media.errors import MissingExtraError, UnsupportedFileError
from mark.media.files import sniff

#: Suffixes chunked as source code (split at top-level definitions).
CODE_SUFFIXES = frozenset(
    ".py .pyi .js .mjs .cjs .ts .tsx .jsx .go .rs .java .kt .kts .c .h .cc .cpp .hpp .cs .rb .php .swift"
    " .scala .sh .bash .zsh .sql .lua .r .dart .ex .exs .hs .clj .vue .svelte".split())
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
_MAX_BYTES = 50 * 1024 * 1024  # refuse to load huge files whole
#: Office Open XML media types read by :mod:`mark.documents.office` (no extra needed).
OFFICE_TYPES = frozenset({
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
})


@dataclass(frozen=True)
class LoadedFile:
    """Text read from a file, with what chunking and provenance need."""
    text: str
    title: str
    source: str
    kind: str          # "text" | "markdown" | "code"
    media_type: str
    sha256: str


def load_file(path: str | Path) -> LoadedFile:
    """Read *path* as document text (see module docstring)."""
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(f"file not found: {file}")
    if file.stat().st_size > _MAX_BYTES:
        raise UnsupportedFileError(f"{file} is larger than {_MAX_BYTES // (1024 * 1024)} MB; split it first")
    media_type = sniff(file)
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    suffix = file.suffix.lower()
    if file.stat().st_size == 0:
        raise UnsupportedFileError(f"{file.name} is empty; there is nothing to remember")
    if media_type == "application/pdf":
        return LoadedFile(read_pdf(file), file.name, str(file), "markdown", media_type, digest)
    if media_type in OFFICE_TYPES:
        from mark.documents.office import read_office  # stdlib zip + XML, loaded on first use

        return LoadedFile(read_office(file, media_type), file.name, str(file), "markdown", media_type, digest)
    if media_type != "text/plain":
        raise UnsupportedFileError(
            f"{file.name} looks like {media_type}, not text; use memory.remember_file() for images, "
            "PDFs and audio (audio needs a transcriber)")
    text = decode_text(file.read_bytes())
    if suffix in _HTML_SUFFIXES or re.match(r"\s*(?:<!doctype html|<html)", text[:200], re.IGNORECASE):
        return LoadedFile(html_to_text(text), file.name, str(file), "markdown", "text/html", digest)
    kind = "code" if suffix in CODE_SUFFIXES else "markdown"
    return LoadedFile(text, file.name, str(file), kind, media_type, digest)


def decode_text(data: bytes) -> str:
    """UTF-16/32 or UTF-8 by byte-order mark, UTF-8, else Windows-1252, else Latin-1 (never fails).

    Windows-1252 before Latin-1: it is what "Latin-1" files from Windows tools
    really are, and it maps 0x80-0x9F to quotes, dashes and the euro sign
    instead of invisible control characters.
    """
    from mark.media.files import bom_encoding

    found = bom_encoding(data)
    if found is not None and found[0] != "utf-8":
        return data[found[1]:].decode(found[0], errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_pdf(path: Path) -> str:
    """Text of every page as "## Page N" sections (needs the ``pdf`` extra)."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MissingExtraError("pdf", "Reading PDF text") from exc
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types for damaged files
        raise UnsupportedFileError(f"{path.name}: the PDF could not be read ({type(exc).__name__}: {exc})") from exc
    if not any(pages):
        raise UnsupportedFileError(f"{path.name} has no extractable text (a scanned PDF needs OCR first)")
    return "\n\n".join(f"## Page {number}\n\n{text}" for number, text in enumerate(pages, 1) if text)


class _HTMLText(HTMLParser):
    """Collect visible text; h1-h6 become Markdown headings, blocks become paragraphs."""

    _BLOCKS = frozenset("p div section article li tr br pre blockquote ul ol table header footer main".split())
    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self._BLOCKS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """Visible text of an HTML page, headings kept as Markdown headings."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    lines = [line.strip() for line in "".join(parser.parts).split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
