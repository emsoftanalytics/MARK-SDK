# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""File ingestion gaps found by the 0.2.0a10 breadth re-test.

Latin-1 / Windows-1252 text is text (not octet-stream), an empty file is a
MARK error, ``remember(Path)`` points to ``remember_file``, and Office files
(docx, xlsx, pptx) are read with the standard library only.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from mark.documents import DocumentRecord, UnsupportedFileError
from mark.documents.loaders import load_file
from mark.embeddings import HashEmbeddingProvider
from mark.errors import InvalidMemoryError, MarkError
from mark.media.files import sniff
from mark.memory.runtime import MarkRuntime

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_P = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


def _memory():
    runtime = MarkRuntime.local(store_path=":memory:", embedder=HashEmbeddingProvider(dim=64))
    return runtime.memory("files")


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def _docx(path: Path) -> Path:
    body = (f'<w:document {_W}><w:body>'
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Ledger runbook</w:t></w:r></w:p>'
            '<w:p><w:r><w:t xml:space="preserve">ledger-sync runs on </w:t></w:r><w:r><w:t>port 7000.</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Dana owns ledger-sync.</w:t></w:r></w:p>'
            '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>env</w:t></w:r></w:p></w:tc>'
            '<w:tc><w:p><w:r><w:t>prod</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
            '</w:body></w:document>')
    return _zip(path, {"word/document.xml": body})


def _xlsx(path: Path) -> Path:
    shared = (f'<sst {_S} count="4"><si><t>Plan</t></si><si><t>Price</t></si>'
              '<si><t>Starter</t></si><si><r><t>Gro</t></r><r><t>wth</t></r></si></sst>')
    sheet = (f'<worksheet {_S}><sheetData>'
             '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
             '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>19</v></c></row>'
             '<row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>79</v></c></row>'
             '<row r="4"><c r="A4" t="inlineStr"><is><t>Team</t></is></c><c r="C4" t="b"><v>1</v></c></row>'
             '</sheetData></worksheet>')
    workbook = (f'<workbook {_S} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Pricing" sheetId="1" r:id="rId1"/></sheets></workbook>')
    return _zip(path, {"xl/workbook.xml": workbook, "xl/sharedStrings.xml": shared,
                       "xl/worksheets/sheet1.xml": sheet})


def _pptx(path: Path) -> Path:
    slide = (f'<p:sld {_P}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Q3 roadmap</a:t></a:r></a:p>'
             '<a:p><a:r><a:t>Ship the billing-api v2</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>')
    return _zip(path, {"ppt/presentation.xml": "<p:presentation/>", "ppt/slides/slide1.xml": slide,
                       "ppt/slides/slide2.xml": slide.replace("Q3 roadmap", "Risks")})


def test_windows_1252_text_is_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes("Café in São Paulo — résumé “quoted” costs 5 €.\n".encode("cp1252"))
    assert sniff(path) == "text/plain"
    loaded = load_file(path)
    assert "Café in São Paulo — résumé “quoted” costs 5 €." in loaded.text
    assert isinstance(_memory().remember_file(path), DocumentRecord)


def test_latin1_text_is_text(tmp_path: Path) -> None:
    path = tmp_path / "old.txt"
    path.write_bytes("Größe: 12 cm; Ångström; niño\n".encode("latin-1"))
    assert sniff(path) == "text/plain" and "Größe" in load_file(path).text


def test_binary_is_still_refused(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(1, 32)) * 50 + b"\x81\x8d\x8f\x90\x9d" * 40)
    assert sniff(path) == "application/octet-stream"


def test_empty_file_is_a_mark_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    memory = _memory()
    for call in (memory.store_document, memory.remember_file):
        with pytest.raises(MarkError) as info:
            call(empty)
        assert isinstance(info.value, ValueError) and "empty" in str(info.value)


def test_remember_a_path_points_to_remember_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidMemoryError, match="remember_file"):
        _memory().remember(tmp_path / "notes.txt")


def test_docx_text_with_headings_and_tables(tmp_path: Path) -> None:
    path = _docx(tmp_path / "runbook.docx")
    assert sniff(path) == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    text = load_file(path).text
    assert "# Ledger runbook" in text and "ledger-sync runs on port 7000." in text
    assert "env | prod" in text
    memory = _memory()
    record = memory.remember_file(path, extract_facts=True)
    assert memory.facts.get("ledger-sync", "port").value == "7000" and record.title == "runbook.docx"


def test_xlsx_shared_strings_and_values(tmp_path: Path) -> None:
    path = _xlsx(tmp_path / "pricing.xlsx")
    assert sniff(path) == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    text = load_file(path).text
    assert "## Pricing" in text
    assert "| Plan | Price |  |" in text and "| Starter | 19 |  |" in text and "| Growth | 79 |  |" in text
    assert "| Team |  | TRUE |" in text


def test_pptx_slide_text(tmp_path: Path) -> None:
    path = _pptx(tmp_path / "deck.pptx")
    assert sniff(path) == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    text = load_file(path).text
    assert "## Slide 1" in text and "Ship the billing-api v2" in text and "## Slide 2\n\nRisks" in text


def test_plain_zip_and_hostile_xml_are_refused(tmp_path: Path) -> None:
    plain = _zip(tmp_path / "archive.zip", {"readme.txt": "hello"})
    with pytest.raises(UnsupportedFileError):
        load_file(plain)
    bomb = _zip(tmp_path / "bomb.docx", {"word/document.xml": (
        '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
        f'<w:document {_W}><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>')})
    with pytest.raises(UnsupportedFileError):
        load_file(bomb)
