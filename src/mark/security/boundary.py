# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Defuse prompt-boundary lookalikes inside stored text before it is rendered.

The XML rendering (``as_context``) escapes every ``<`` so a memory can never
close ``<memory_context>``. Plain-text renderings (``as_text()``, the session
brief) keep ordinary markup readable, so they instead defuse only what could
pass for a boundary or a chat-template role marker: ``</memory_context>``,
``<system>``, ``<mark_briefing>``, ``<|im_start|>``, ``[INST]``, ``<<SYS>>``,
``[MARK …]``. Angle and square brackets in those spans become look-alike
characters (``‹ › ⟦ ⟧``), so the text stays legible to a model but can no
longer end the block it is quoted in. Ordinary markup (``<div>``, ``a < b``)
is untouched, and the function is idempotent.
"""
from __future__ import annotations

from mark.security.lazy_regex import lazy_compile

# Tag names a model (or MARK) treats as a boundary or a role.
_NAMES = (r"memory_[\w-]*|mark[\w:-]*|system(?:_prompt|[-_]reminder)?|assistant|user|human|developer|"
          r"instructions?|context|documents?|tool[\w-]*|function[\w-]*|invoke|antml[\w:-]*|im_(?:start|end)|"
          r"untrusted[\w-]*|search_results?|result")
_BOUNDARY = lazy_compile(
    rf"<\s*/?\s*(?:{_NAMES})\b[^<>\n]{{0,200}}>"      # <system>, </memory_context>, <mark_x a="b">
    r"|<\|[^|<>\n]{1,40}\|>"                          # <|im_start|>, <|eot_id|>
    r"|<</?SYS>>|\[/?INST\]"                          # Llama-2 markers
    r"|\[\s*/?\s*mark\b[^\]\n]{0,200}\]",             # [MARK project memory] / [/MARK ...]
    flags=2,  # re.IGNORECASE (kept numeric so importing this module stays cheap)
)
_SWAP = str.maketrans({"<": "‹", ">": "›", "[": "⟦", "]": "⟧"})


def defuse_boundaries(text: str) -> str:
    """Return *text* with boundary-tag and role-marker lookalikes defused."""
    if not text or ("<" not in text and "[" not in text):
        return text or ""
    return _BOUNDARY.sub(lambda m: m.group(0).translate(_SWAP), text)


__all__ = ["defuse_boundaries"]
