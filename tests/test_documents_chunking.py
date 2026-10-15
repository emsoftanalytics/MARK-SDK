# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Structure-aware chunking for documents.

Chunks follow the document's structure: Markdown headings start a new chunk
and their path prefixes every chunk below them; code fences stay whole;
paragraphs are packed up to the token budget; an oversized paragraph is
split at sentences (then words); neighbouring chunks of one section overlap.
"""
from __future__ import annotations

import pytest

from mark.documents import chunk_text, estimate_tokens


def test_short_text_is_one_chunk_with_its_title() -> None:
    (chunk,) = chunk_text("Just one line.", title="Note")
    assert chunk.text == "# Note\n\nJust one line." and chunk.heading_path == ("Note",)


def test_headings_prefix_their_chunks_and_start_new_ones() -> None:
    text = "# Guide\n\nIntro words here.\n\n## Install\n\nRun pip install.\n\n### Extras\n\nAdd [pdf].\n\n## Use\n\nCall it."
    chunks = chunk_text(text, title="guide.md")
    paths = [c.heading_path for c in chunks]
    assert paths == [("Guide",), ("Guide", "Install"), ("Guide", "Install", "Extras"), ("Guide", "Use")]
    assert chunks[2].text.startswith("# Guide › Install › Extras\n\n")
    assert all(c.text.count("#") == 1 for c in chunks)  # source heading lines become the prefix


def test_code_fences_are_never_split() -> None:
    code = "```python\n" + "\n".join(f"x{i} = {i}  # step {i}" for i in range(80)) + "\n```"
    text = "# Code\n\nBefore.\n\n" + code + "\n\nAfter."
    chunks = chunk_text(text, chunk_tokens=60, overlap=0)
    holding = [c for c in chunks if "```python" in c.text]
    assert len(holding) == 1 and holding[0].text.count("```") == 2 and "x79 = 79" in holding[0].text


def test_long_paragraphs_split_at_sentences_with_overlap() -> None:
    sentences = [f"Sentence number {i} talks about topic {i} in some detail." for i in range(120)]
    chunks = chunk_text(" ".join(sentences), chunk_tokens=100, overlap=20)
    assert len(chunks) > 5
    assert all(estimate_tokens(c.body) <= 100 + 25 for c in chunks)
    # Every sentence survives, and consecutive chunks share their boundary text.
    joined = " ".join(c.body for c in chunks)
    assert all(s in joined for s in sentences)
    assert chunks[0].body.split(". ")[-1].rstrip(".") in chunks[1].body


def test_one_huge_sentence_still_respects_the_budget() -> None:
    chunks = chunk_text("word " * 1000, chunk_tokens=50, overlap=0)
    assert len(chunks) >= 15 and all(estimate_tokens(c.body) <= 55 for c in chunks)


def test_code_files_split_at_top_level_definitions() -> None:
    source = "import os\n\n\ndef alpha():\n    return 1\n\n\nclass Beta:\n    def gamma(self):\n        return 2\n"
    chunks = chunk_text(source, title="mod.py", kind="code", chunk_tokens=12, overlap=0)
    assert any(c.text.startswith("# mod.py › def alpha") for c in chunks)
    assert any("class Beta" in c.text and "def gamma" in c.text for c in chunks)


@pytest.mark.parametrize("bad", [dict(chunk_tokens=0), dict(overlap=-1), dict(chunk_tokens=50, overlap=50)])
def test_invalid_budgets_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", **bad)
