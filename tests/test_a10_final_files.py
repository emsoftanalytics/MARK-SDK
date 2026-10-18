# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Final a10 files: UTF-16/UTF-32 text with a BOM is text; pptx speaker notes are ingested."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from mark.documents.loaders import load_file
from mark.media.files import sniff

BOM = "\ufeff"
TEXT = "Ọ̀yìnbó notes: the backup runs at 02:00 UTC.\nSecond line — résumé.\n"
_P = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-8-sig"])
def test_bom_text_is_text(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / "notes.txt"
    body = TEXT * 50
    if encoding in ("utf-16-le", "utf-16-be"):  # explicit-endian codecs write no BOM: add it
        body = BOM + body
    path.write_bytes(body.encode(encoding))
    assert sniff(path) == "text/plain"
    assert "the backup runs at 02:00 UTC" in load_file(path).text
    assert BOM not in load_file(path).text


def test_mpeg_audio_is_still_audio(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"\xff\xfb\x90\x64" + bytes(range(1, 250)) * 4)
    assert sniff(path) == "audio/mpeg"


def _slide(text: str) -> str:
    return (f'<p:sld {_P}><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p>'
            '</p:txBody></p:sp></p:spTree></p:cSld></p:sld>')


def _notes(text: str) -> str:
    return (f'<p:notes {_P}><p:cSld><p:spTree>'
            '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr></p:sp>'
            '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:txBody>'
            f'<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>'
            '<p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum" idx="5"/></p:nvPr></p:nvSpPr><p:txBody>'
            '<a:p><a:r><a:t>7</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>')


def _rels(target: str) -> str:
    return ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            f'notesSlide" Target="{target}"/></Relationships>')


def test_pptx_speaker_notes_are_ingested(tmp_path: Path) -> None:
    path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", "<p:presentation/>")
        archive.writestr("ppt/slides/slide1.xml", _slide("Q3 roadmap"))
        archive.writestr("ppt/slides/slide2.xml", _slide("Risks"))
        # Notes files numbered differently from slides: the slide's rels decide.
        archive.writestr("ppt/slides/_rels/slide2.xml.rels", _rels("../notesSlides/notesSlide1.xml"))
        archive.writestr("ppt/notesSlides/notesSlide1.xml", _notes("Mention the vendor delay to Dana."))
    text = load_file(path).text
    assert "## Slide 2\n\nRisks" in text
    assert "Notes: Mention the vendor delay to Dana." in text
    assert text.index("Mention the vendor") > text.index("## Slide 2")
    assert "\n7" not in text  # slide-number placeholder is not a note


def test_stacked_diacritics_fold_for_keyword_evidence() -> None:
    from mark.intelligence.query_understanding import _content_terms, _coverage

    content = "Ọjà Ọ̀yìnbó ní Èkó ti di ibi ìtajà tuntun."
    assert _coverage(_content_terms("Oja Oyinbo Eko"), content) == 1.0
    assert _coverage(_content_terms("Ọ̀yìnbó"), content) == 1.0
    assert _coverage(_content_terms("beträgt Budget"), "Das Budget für Q1 beträgt 120.000 €.") == 1.0
    assert _coverage(_content_terms("재고"), "부산 창고 재고 조사는 금요일입니다.") == 1.0
    assert _coverage(_content_terms("生产线"), "深圳工厂的新生产线将于三月投产。") == 1.0
