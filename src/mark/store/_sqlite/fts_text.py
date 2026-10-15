"""Text preparation for the two FTS5 tables: English (porter) and multilingual.

The porter table indexes content as written; FTS5's English stemmer makes
"deploys" match "deployed". Applied to other languages it conflates
unrelated words, and SQLite's tokenizer cannot segment CJK (a whole
sentence becomes one token) or keep emoji (symbols are separators). So
fragments that contain non-ASCII text get a second row in the multilingual
table (no stemmer, combining marks kept), prepared here:

- runs of CJK characters (Han, kana, Hangul) become overlapping bigrams, in
  order, followed by their single characters: "上海仓库" -> "上海 海仓 仓库 上 海
  仓 库". A query uses the bigrams only (or the single character), so both
  two-character words and phrases match without a dictionary;
- non-ASCII symbols (emoji, currency, arrows, ...) become standalone tokens;
- everything else is left to the tokenizer (unicode61, diacritics folded).

Query routing: ASCII words go to the porter table, everything else to the
multilingual table, and the two BM25 scores are added per fragment.
"""
from __future__ import annotations

import re
import unicodedata

from mark.embeddings.base import _STOPWORDS
from mark.embeddings.segments import CJK_RUN as _CJK_RUN
from mark.embeddings.segments import SYMBOL_NOISE as _SYMBOL_NOISE
from mark.embeddings.segments import cjk_bigrams, is_symbol

_WORD = re.compile(r"\w+")
# A chunk of the query that is not plain ASCII (split on whitespace and ASCII punctuation).
_CHUNK_SPLIT = re.compile(r"[\s!-/:-@\[-`{-~]+")


# ASCII symbols are token characters in the multilingual table (so emoji are);
# blank them so "x=1" stays two tokens there, as in the porter table.
_ASCII_SYMBOLS = str.maketrans({char: " " for char in "$+<=>^`|~"})


def _spaced_symbols(text: str) -> str:
    """Isolate each non-ASCII symbol as its own token; drop emoji joiners/selectors."""
    if text.isascii():
        return text
    return "".join(
        f" {char} " if is_symbol(char) else " " if char in _SYMBOL_NOISE else char
        for char in text.translate(_ASCII_SYMBOLS)
    )


def multilingual_index_text(text: str) -> str | None:
    """Text for the multilingual table, or None when *text* is plain ASCII."""
    if text.isascii():
        return None
    singles: list[str] = []

    def segment(match: re.Match[str]) -> str:
        run = match.group(0)
        singles.extend(run)
        return " " + " ".join(cjk_bigrams(run)) + " "

    prepared = _CJK_RUN.sub(segment, _spaced_symbols(text))
    if singles and any(len(run) > 1 for run in _CJK_RUN.findall(text)):
        prepared += " " + " ".join(dict.fromkeys(singles))
    return prepared


def multilingual_query_chunks(text: str, limit: int = 16) -> list[str]:
    """Query pieces for the multilingual table: each becomes one quoted FTS phrase.

    Only pieces that contain non-ASCII characters are returned (ASCII words
    are searched in the porter table). CJK runs become bigrams.
    """
    if text.isascii():
        return []
    chunks: list[str] = []
    for piece in _CHUNK_SPLIT.split(_spaced_symbols(text).lower()):
        if not piece or piece.isascii():
            continue
        cjk = _CJK_RUN.fullmatch(piece)
        if cjk is not None:
            chunks.extend(cjk_bigrams(piece))
            continue
        if _CJK_RUN.search(piece):
            # Mixed run ("東京オフィスの会議室は"): segment the CJK parts, keep the rest.
            for part in _CJK_RUN.split(piece):
                if part and any(ch.isalnum() or is_symbol(ch) for ch in part):
                    chunks.append(part)
            for run in _CJK_RUN.findall(piece):
                chunks.extend(cjk_bigrams(run))
            continue
        if any(ch.isalnum() or is_symbol(ch) for ch in piece):
            chunks.append(piece)
    return list(dict.fromkeys(chunks))[:limit]


def multilingual_phrase(text: str) -> str | None:
    """A phrase as consecutive multilingual-table tokens (CJK as bigrams), or None."""
    if text.isascii():
        return None
    spaced = _CJK_RUN.sub(lambda m: " " + " ".join(cjk_bigrams(m.group(0))) + " ", _spaced_symbols(text.lower()))
    return " ".join(spaced.split()) or None


def ascii_query_terms(text: str, limit: int = 16) -> list[str]:
    """Distinct ASCII words of a query for the porter table, stopwords removed.

    Accented Latin words are folded ("café" -> "cafe") so they still match
    the porter table, which folds diacritics too; words that do not fold to
    ASCII are left to the multilingual table. Single characters are dropped
    unless they are digits. When every word is a stopword the stopwords are
    kept, so "who is it" still searches.
    """
    words: list[str] = []
    for word in _WORD.findall(text.lower()):
        if not word.isascii():
            word = "".join(ch for ch in unicodedata.normalize("NFKD", word) if not unicodedata.combining(ch))
            if not word.isascii():
                continue
        if len(word) >= 2 or word.isdigit():
            words.append(word)
    content = [word for word in words if word not in _STOPWORDS] or words
    return list(dict.fromkeys(content))[:limit]


def fts_quote(token: str) -> str:
    """Quote a token or phrase for an FTS5 MATCH expression."""
    return '"' + token.replace('"', '""') + '"'
