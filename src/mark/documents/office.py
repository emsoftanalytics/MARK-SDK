# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Text of Office Open XML files (docx, xlsx, pptx) with the standard library only.

An Office file is a zip of XML parts. MARK reads just the parts that hold
text, so no extra dependency is needed:

* **docx** ``word/document.xml``: paragraphs in order; "Heading N"/"Title"
  styles become Markdown headings (chunking follows them); table rows become
  ``cell | cell`` lines;
* **xlsx** ``xl/workbook.xml`` + ``xl/sharedStrings.xml`` + each sheet: one
  "## <sheet name>" section per sheet with its rows as a Markdown table
  (shared, inline, boolean and numeric cells; formulas give their cached value);
* **pptx** ``ppt/slides/slideN.xml``: one "## Slide N" section per slide.

Hostile files are refused with :class:`UnsupportedFileError`: a part larger
than :data:`MAX_PART_BYTES` uncompressed (zip bombs) and any XML with a
DOCTYPE/ENTITY declaration (entity-expansion bombs; Office never uses them).
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from mark.media.errors import UnsupportedFileError

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
#: Part that identifies each format inside the zip.
MARKER_PARTS = {"word/document.xml": DOCX, "xl/workbook.xml": XLSX, "ppt/presentation.xml": PPTX}
MAX_PART_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 20_000  # per sheet: a huge export is data, not a document to remember whole

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def office_type(names: list[str]) -> str | None:
    """The Office media type of a zip with these member names, or None."""
    return next((media for part, media in MARKER_PARTS.items() if part in names), None)


def read_office(path: Path, media_type: str) -> str:
    """Text of one docx/xlsx/pptx file (see module docstring)."""
    try:
        with zipfile.ZipFile(path) as archive:
            reader = {DOCX: _docx, XLSX: _xlsx, PPTX: _pptx}[media_type]
            text = reader(archive)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise UnsupportedFileError(f"{path.name}: the Office file could not be read ({type(exc).__name__})") from exc
    if not text.strip():
        raise UnsupportedFileError(f"{path.name} has no text to remember")
    return text


def _xml(archive: zipfile.ZipFile, name: str) -> ElementTree.Element:
    """Parse one part after the size and DOCTYPE guards."""
    info = archive.getinfo(name)
    if info.file_size > MAX_PART_BYTES:
        raise UnsupportedFileError(f"{name} is larger than {MAX_PART_BYTES // (1024 * 1024)} MB uncompressed")
    data = archive.read(info)
    if re.search(rb"<!(?:DOCTYPE|ENTITY)", data[:4096], re.IGNORECASE) or b"<!ENTITY" in data:
        raise UnsupportedFileError(f"{name} declares XML entities; refusing to expand them")
    return ElementTree.fromstring(data)


# ── docx ─────────────────────────────────────────────────────────────────────

def _docx(archive: zipfile.ZipFile) -> str:
    body = _xml(archive, "word/document.xml").find(f"{_W}body")
    blocks: list[str] = []
    for element in list(body) if body is not None else []:
        if element.tag == f"{_W}p":
            text = _paragraph(element)
            if text.strip():
                blocks.append(_heading_prefix(element) + text.strip())
        elif element.tag == f"{_W}tbl":
            rows = [" | ".join(" ".join(_paragraph(p).strip() for p in cell.iter(f"{_W}p")).strip()
                               for cell in row.iter(f"{_W}tc"))
                    for row in element.iter(f"{_W}tr")]
            if any(r.strip(" |") for r in rows):
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def _paragraph(paragraph: ElementTree.Element) -> str:
    """Runs of one paragraph: text, tabs and line breaks in document order."""
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _heading_prefix(paragraph: ElementTree.Element) -> str:
    """'# ' for Title/Heading1, '## ' for Heading2, ... so chunking follows the outline."""
    style = paragraph.find(f"{_W}pPr/{_W}pStyle")
    name = (style.get(f"{_W}val") or "") if style is not None else ""
    if name.lower() == "title":
        return "# "
    match = re.fullmatch(r"(?i)heading\s*([1-6])", name)
    return "#" * int(match.group(1)) + " " if match else ""


# ── xlsx ─────────────────────────────────────────────────────────────────────

def _xlsx(archive: zipfile.ZipFile) -> str:
    names = archive.namelist()
    shared = []
    if "xl/sharedStrings.xml" in names:
        shared = ["".join(t.text or "" for t in item.iter(f"{_S}t"))
                  for item in _xml(archive, "xl/sharedStrings.xml").iter(f"{_S}si")]
    sections = []
    for title, part in _sheets(archive, names):
        rows = _sheet_rows(_xml(archive, part), shared)
        if rows:
            lines = ["| " + " | ".join(row) + " |" for row in rows]
            lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
            sections.append(f"## {title}\n\n" + "\n".join(lines))
    return "\n\n".join(sections)


def _sheets(archive: zipfile.ZipFile, names: list[str]) -> list[tuple[str, str]]:
    """(sheet name, part path) in workbook order; rels resolve ids, else sheetN.xml by position."""
    workbook = _xml(archive, "xl/workbook.xml")
    targets: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in names:
        for rel in _xml(archive, "xl/_rels/workbook.xml.rels").iter(f"{_PKG}Relationship"):
            target = (rel.get("Target") or "").lstrip("/")
            targets[rel.get("Id") or ""] = target if target.startswith("xl/") else f"xl/{target}"
    out = []
    for index, sheet in enumerate(workbook.iter(f"{_S}sheet"), 1):
        part = targets.get(sheet.get(f"{_R}id") or "", f"xl/worksheets/sheet{index}.xml")
        if part in names:
            out.append((sheet.get("name") or f"Sheet {index}", part))
    return out


def _sheet_rows(sheet: ElementTree.Element, shared: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in sheet.iter(f"{_S}row"):
        cells: dict[int, str] = {}
        for position, cell in enumerate(row.iter(f"{_S}c")):
            column = _column(cell.get("r") or "", position)
            cells[column] = _cell_text(cell, shared)
        if any(v.strip() for v in cells.values()):
            width = max(cells) + 1
            rows.append([cells.get(i, "").replace("|", "/").replace("\n", " ") for i in range(width)])
        if len(rows) >= _MAX_ROWS:
            break
    # A Markdown table needs one width: pad short rows with empty cells.
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _column(ref: str, position: int) -> int:
    """Zero-based column of a cell reference ("C4" -> 2); the position when there is none."""
    letters = re.match(r"[A-Z]+", ref)
    if not letters:
        return position
    index = 0
    for char in letters.group(0):
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    kind = cell.get("t") or "n"
    if kind == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{_S}t"))
    value = cell.findtext(f"{_S}v") or ""
    if kind == "s":
        return shared[int(value)] if value.isdigit() and int(value) < len(shared) else ""
    if kind == "b":
        return "TRUE" if value == "1" else "FALSE" if value == "0" else value
    return value


# ── pptx ─────────────────────────────────────────────────────────────────────

def _pptx(archive: zipfile.ZipFile) -> str:
    """Slide text as "## Slide N" sections, each followed by its speaker notes ("Notes: ...")."""
    names = set(archive.namelist())
    slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    sections = []
    for number, name in enumerate(slides, 1):
        text = "\n".join(_paragraphs(_xml(archive, name)))
        notes = _slide_notes(archive, name, names)
        if notes:
            text = f"{text}\n\nNotes: {notes}" if text else f"Notes: {notes}"
        if text:
            sections.append(f"## Slide {number}\n\n{text}")
    return "\n\n".join(sections)


def _paragraphs(root: ElementTree.Element) -> list[str]:
    paragraphs = ["".join(t.text or "" for t in p.iter(f"{_A}t")).strip() for p in root.iter(f"{_A}p")]
    return [p for p in paragraphs if p]


def _slide_notes(archive: zipfile.ZipFile, slide: str, names: set[str]) -> str:
    """The speaker notes of *slide*: its notesSlide part (found through the slide's rels), body text only.

    Notes parts also hold the slide image, header/footer and slide-number
    placeholders; only the ``body`` placeholder is what the speaker wrote.
    """
    rels = f"ppt/slides/_rels/{slide.rsplit('/', 1)[1]}.rels"
    if rels not in names:
        return ""
    target = next((rel.get("Target", "") for rel in _xml(archive, rels).iter(f"{_PKG}Relationship")
                   if rel.get("Type", "").endswith("/notesSlide")), "")
    part = "ppt/" + target.split("../", 1)[1] if target.startswith("../") else target.lstrip("/")
    if part not in names:
        return ""
    lines: list[str] = []
    for shape in _xml(archive, part).iter(f"{_P}sp"):
        placeholder = next(shape.iter(f"{_P}ph"), None)
        if placeholder is not None and placeholder.get("type") == "body":
            lines += _paragraphs(shape)
    return " ".join(lines)
