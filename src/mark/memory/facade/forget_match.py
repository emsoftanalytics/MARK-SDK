# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Safe text matching for ``forget(match=...)``.

A release re-test deleted 37 of 41 memories with ``--match spendlog`` and a
whole store with ``--match e``, because the match was a plain substring. The
rules now are:

* **whole words / phrases.** ``"cat"`` matches "the cat sleeps", never
  "concatenate"; a phrase matches its words in order with any whitespace
  between them. Case is ignored.
* **a minimum.** At least :data:`MIN_MATCH_CHARS` characters, and not only
  stopwords ("the", "and the").
* **a cap.** More than :data:`FORGET_MATCH_CAP` matches is refused unless the
  caller passes ``all=True`` (``--all`` on the CLI).
* **dry runs.** ``forget(match=..., dry_run=True)`` returns a
  :class:`ForgetPreview` (count, ids, short previews) and deletes nothing;
  the CLI and MCP tool use it to ask before deleting more than one memory.

Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: More matches than this are refused unless ``all=True``.
FORGET_MATCH_CAP = 50
#: Shortest accepted match text.
MIN_MATCH_CHARS = 3
#: Characters of memory text shown per preview line.
PREVIEW_CHARS = 80

#: Words that never identify a memory on their own.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its me my no not "
    "of on or our she so than that the their them then there these they this to too was we were "
    "what when where which who will with you your".split()
)


@dataclass(frozen=True)
class ForgetPreview:
    """What ``forget(match=..., dry_run=True)`` would delete (nothing was deleted).

    ``previews`` holds one line per memory: its 8-character short id and the
    start of its text. ``int(preview)`` is the count.
    """

    match: str
    count: int
    ids: list[str] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)

    def __int__(self) -> int:
        return self.count

    @property
    def over_cap(self) -> bool:
        """True when deleting these needs ``all=True`` (more than the cap)."""
        return self.count > FORGET_MATCH_CAP


def match_pattern(match: str) -> re.Pattern[str]:
    """The whole-word, case-insensitive pattern for *match*; ``InvalidInputError`` when unsafe."""
    from mark.errors import InvalidInputError

    words = str(match or "").split()
    text = " ".join(words)
    if len(text) < MIN_MATCH_CHARS:
        raise InvalidInputError(
            f"forget(match=...) needs at least {MIN_MATCH_CHARS} characters (got {text!r}); "
            "forget one memory by id instead")
    if all(word.casefold().strip(".,;:!?'\"") in _STOPWORDS for word in words):
        raise InvalidInputError(
            f"forget(match={text!r}) is only common words and would match almost everything; "
            "use a distinctive word or phrase, or forget by id")
    body = r"\s+".join(re.escape(word) for word in words)
    # (?<!\w)/(?!\w) rather than \b: works for texts that start/end with
    # punctuation ("TODO:", "--flag") as well as plain words.
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def matching(fragments: Iterable[Any], match: str) -> list[Any]:
    """The fragments whose text contains *match* as whole words."""
    pattern = match_pattern(match)
    return [f for f in fragments if pattern.search(f.content or "")]


def preview(match: str, fragments: list[Any]) -> ForgetPreview:
    """A :class:`ForgetPreview` of *fragments* (newest first, as ``list()`` returns them)."""
    lines = []
    for fragment in fragments:
        text = " ".join((fragment.content or "").split())
        if len(text) > PREVIEW_CHARS:
            text = text[: PREVIEW_CHARS - 1] + "…"
        lines.append(f"{fragment.id[:8]}  {text}")
    return ForgetPreview(match=" ".join(str(match).split()), count=len(fragments),
                         ids=[f.id for f in fragments], previews=lines)


def over_cap_error(found: ForgetPreview) -> Exception:
    """The error raised when a match exceeds the cap without ``all=True``."""
    from mark.errors import InvalidInputError

    return InvalidInputError(
        f"forget(match={found.match!r}) matches {found.count} memories, more than the safety cap "
        f"of {FORGET_MATCH_CAP}; check them with dry_run=True (CLI: --dry-run), then pass "
        "all=True (CLI: --all) to forget them all")


def plural(count: int, word: str = "memory") -> str:
    """``"1 memory"`` / ``"2 memories"`` (also ``fact``/``facts``)."""
    if count == 1:
        return f"1 {word}"
    return f"{count} {word[:-1] + 'ies' if word.endswith('y') else word + 's'}"
