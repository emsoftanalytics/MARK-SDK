"""Structure-aware chunking: split a document where its author did.

A document becomes blocks first, then blocks are packed into chunks:

  - Markdown headings end the current chunk; the heading path ("Guide ›
    Install › Extras") prefixes every chunk below it as a "# ..." line, so
    each chunk carries its context into recall and fact extraction;
  - fenced code blocks are atomic (never split, up to a hard cap);
  - paragraphs are packed together up to ``chunk_tokens``; a paragraph that
    is too long is split at sentence ends, and a sentence that is too long
    at word boundaries;
  - consecutive chunks of one section overlap by about ``overlap`` tokens
    (whole sentences where possible), so a detail on a boundary is whole in
    at least one chunk;
  - ``kind="code"`` splits source files at top-level definitions instead.

Token counts are estimates (words and punctuation marks), which is all a
chunk budget needs; nothing here depends on a tokenizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+|[^\w\s]")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_TOP_LEVEL = re.compile(r"^(?:async\s+def|def|class|function|fn|func|pub\s+fn|impl|struct|interface|type|const|let"
                        r"|export\s+(?:default\s+)?(?:function|class|const))\b\s*([\w$]+)?")
_SEPARATOR = " › "


def estimate_tokens(text: str) -> int:
    """Approximate token count: words plus punctuation marks."""
    return len(_TOKEN.findall(text))


@dataclass(frozen=True)
class Chunk:
    """One chunk: ``text`` is what gets stored ("# path" line + body)."""
    index: int
    heading_path: tuple[str, ...]
    body: str

    @property
    def text(self) -> str:
        """The stored text: the heading path as a "# A › B" line, then the body."""
        return f"# {_SEPARATOR.join(self.heading_path)}\n\n{self.body}" if self.heading_path else self.body

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.body)


def chunk_text(text: str, *, title: str | None = None, kind: str = "text", chunk_tokens: int = 300,
               overlap: int = 40) -> list[Chunk]:
    """Split *text* into chunks (see module docstring); ``kind`` is "text"/"markdown" or "code"."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("overlap must be >= 0 and smaller than chunk_tokens")
    sections = _code_sections(text, title) if kind == "code" else _markdown_sections(text, title)
    chunks: list[Chunk] = []
    for path, blocks in sections:
        for body in _pack(blocks, chunk_tokens, overlap):
            chunks.append(Chunk(len(chunks), path, body))
    return chunks


# ── sections: (heading path, blocks) ─────────────────────────────────────────

def _markdown_sections(text: str, title: str | None) -> list[tuple[tuple[str, ...], list[str]]]:
    lines = text.replace("\r\n", "\n").split("\n")
    has_h1 = any((m := _HEADING.match(line)) and len(m.group(1)) == 1 for line in lines)
    root: tuple[str, ...] = (title,) if title and not has_h1 else ()
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    stack: list[tuple[int, str]] = []
    blocks: list[str] = []
    paragraph: list[str] = []
    fence: list[str] | None = None
    fence_mark = ""

    def path() -> tuple[str, ...]:
        return root + tuple(name for _, name in stack)

    def end_paragraph() -> None:
        if paragraph:
            blocks.append("\n".join(paragraph).strip())
            paragraph.clear()

    for line in lines:
        if fence is not None:
            fence.append(line)
            if line.strip().startswith(fence_mark):
                blocks.append("\n".join(fence))
                fence = None
            continue
        if (opener := _FENCE.match(line)) is not None:
            end_paragraph()
            fence, fence_mark = [line], opener.group(1)
            continue
        if (heading := _HEADING.match(line)) is not None:
            end_paragraph()
            if blocks:
                sections.append((path(), blocks))
            blocks = []
            level = len(heading.group(1))
            stack = [(lvl, name) for lvl, name in stack if lvl < level] + [(level, heading.group(2).strip())]
            continue
        if line.strip():
            paragraph.append(line)
        else:
            end_paragraph()
    if fence is not None:
        blocks.append("\n".join(fence))  # unterminated fence: keep what there is
    end_paragraph()
    if blocks:
        sections.append((path(), blocks))
    return sections


def _code_sections(text: str, title: str | None) -> list[tuple[tuple[str, ...], list[str]]]:
    """Source code: one section per top-level definition (the preamble keeps the file name)."""
    root: tuple[str, ...] = (title,) if title else ()
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    current: list[str] = []
    name: str | None = None
    for line in text.replace("\r\n", "\n").split("\n"):
        match = _TOP_LEVEL.match(line)
        if match is not None and not line[:1].isspace():
            if "\n".join(current).strip():
                sections.append((root + ((name,) if name else ()), ["\n".join(current).strip("\n")]))
            current = []
            name = " ".join(line.split("(")[0].split(":")[0].split()[:3])
        current.append(line)
    if "\n".join(current).strip():
        sections.append((root + ((name,) if name else ()), ["\n".join(current).strip("\n")]))
    return sections


# ── packing ──────────────────────────────────────────────────────────────────
# A piece is (separator, text): the separator goes before it when it follows
# another piece -- " " between sentences of one paragraph, a blank line
# between blocks -- so packing never merges two paragraphs into one line.
Piece = tuple[str, str]


def _pack(blocks: list[str], budget: int, overlap: int) -> list[str]:
    """Pack blocks into bodies of at most *budget* tokens, with *overlap* carried over."""
    pieces: list[Piece] = []
    for block in blocks:
        parts = _split_block(block, budget)
        pieces.extend([("\n\n", parts[0])] + [(" " if not _FENCE.match(block) else "\n\n", p) for p in parts[1:]])
    bodies: list[str] = []
    current: list[Piece] = []
    size = 0
    for piece in pieces:
        cost = estimate_tokens(piece[1])
        if current and size + cost > budget:
            bodies.append(_join(current))
            current = _tail(current, overlap, budget - cost)
            size = sum(estimate_tokens(p[1]) for p in current)
        current.append(piece)
        size += cost
    if current:
        bodies.append(_join(current))
    return bodies


def _split_block(block: str, budget: int) -> list[str]:
    """One block as parts that each fit *budget* (code fences stay whole up to a hard cap)."""
    if estimate_tokens(block) <= budget:
        return [block]
    if _FENCE.match(block):
        cap = max(budget * 8, 2000)
        return [block] if estimate_tokens(block) <= cap else _split_lines(block, cap)
    parts: list[str] = []
    for sentence in _SENTENCE_END.split(block):
        parts.extend([sentence] if estimate_tokens(sentence) <= budget else _split_words(sentence, budget))
    return parts


def _split_words(text: str, budget: int) -> list[str]:
    words, parts, current = text.split(), [], []
    for word in words:
        if current and estimate_tokens(" ".join(current + [word])) > budget:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _split_lines(block: str, cap: int) -> list[str]:
    """An oversized code fence, split at line boundaries (each part re-fenced)."""
    lines = block.split("\n")
    opener, body = lines[0], [line for line in lines[1:] if not _FENCE.match(line)]
    marker = _FENCE.match(opener).group(1)
    parts, current = [], []
    for line in body:
        if current and estimate_tokens("\n".join(current + [line])) > cap:
            parts.append("\n".join([opener, *current, marker]))
            current = []
        current.append(line)
    if current:
        parts.append("\n".join([opener, *current, marker]))
    return parts


def _tail(pieces: list[Piece], overlap: int, room: int) -> list[Piece]:
    """The last pieces (whole sentences) worth about *overlap* tokens, to repeat in the next chunk."""
    if overlap <= 0 or room <= 0:
        return []
    kept: list[Piece] = []
    size = 0
    for piece in reversed(pieces):
        cost = estimate_tokens(piece[1])
        if _FENCE.match(piece[1]) or size + cost > min(overlap, room):
            break
        kept.insert(0, piece)
        size += cost
    return kept


def _join(pieces: list[Piece]) -> str:
    return pieces[0][1] + "".join(separator + text for separator, text in pieces[1:])
