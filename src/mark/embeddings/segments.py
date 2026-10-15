"""Unicode-aware word segmentation shared by the hash embedder and keyword search.

Regular-expression words (``\\w+``) work for languages that put spaces
between words but fail two common cases:

- CJK text has no spaces, so a whole sentence becomes one "word" and no
  query ever matches it. Runs of Han, kana and Hangul are split into
  overlapping character bigrams ("上海仓库" -> 上海, 海仓, 仓库), the standard
  dictionary-free approach: any two-character word, and any longer phrase,
  is a sequence of bigrams;
- emoji and other symbols (€, ™, →) are not word characters and vanish.
  Each non-ASCII symbol becomes a token of its own.

ASCII text is returned exactly as ``\\w+`` would split it, so embeddings and
indexes of plain English text do not change.
"""
from __future__ import annotations

import re
import unicodedata

# Han (incl. extension A and compatibility), Hiragana, Katakana, Hangul.
CJK_CLASS = "぀-ヿ㐀-䶿一-鿿豈-﫿가-힯ᄀ-ᇿ㄰-㆏"
CJK_RUN = re.compile(f"[{CJK_CLASS}]+")
_TOKEN = re.compile(r"\w+|[^\w\s]")
_WORD = re.compile(r"\w+")
# Emoji presentation selectors, joiners and skin-tone modifiers: not searchable on their own.
SYMBOL_NOISE = frozenset("︎️‍") | {chr(c) for c in range(0x1F3FB, 0x1F400)}


def is_symbol(char: str) -> bool:
    """True for a non-ASCII symbol code point (emoji, currency, arrows, ...)."""
    return char > "\x7f" and char not in SYMBOL_NOISE and unicodedata.category(char).startswith("S")


def cjk_bigrams(run: str) -> list[str]:
    """Overlapping bigrams of a CJK run; a single character stays as it is."""
    return [run[i:i + 2] for i in range(max(1, len(run) - 1))]


def segment_words(text: str) -> list[str]:
    """Words of *text*: ``\\w+`` words, CJK runs as bigrams, symbols as single tokens."""
    if text.isascii():
        return _WORD.findall(text)
    words: list[str] = []
    for token in _TOKEN.findall(text):
        if _WORD.match(token):
            if CJK_RUN.search(token) is None:
                words.append(token)
                continue
            for part in re.split(f"([{CJK_CLASS}]+)", token):
                if not part:
                    continue
                words.extend(cjk_bigrams(part) if CJK_RUN.fullmatch(part) else [part])
        elif is_symbol(token):
            words.append(token)
    return words
