# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Cheap, deterministic prompt-injection screening for memory writes.

Memories are replayed into future prompts, so text that *instructs* a model
("ignore all previous instructions", fake chat-template role markers,
smuggled tool calls, hidden zero-width payloads) is dangerous to recall as
if it were plain knowledge. The screen does not block anything by default:
the write path tags such memories ``mark:untrusted`` and records the
reasons, and context renderers label them so the model treats them as
quoted data. Patterns require an *instruction shape* (a verb aimed at the
model plus an instruction noun), so memories that merely discuss prompts
("the system prompt for bot X says be concise") are not flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNTRUSTED_TAG = "mark:untrusted"

_I = re.IGNORECASE
_QUAL = r"(?:all|any|every|the|your|my|these|those|of|prior|previous|above|earlier|preceding|existing|" \
        r"system|original|initial|other|safety)"
_NOUN = r"(?:instructions?|prompts?|directives?|guidelines|guardrails|rules|restrictions|policies|programming)"

# (reason, pattern). Ordered roughly by how common they are in the wild.
_PATTERN_LIST: list[tuple[str, "re.Pattern[str]"]] = [
    ("ignore_instructions", re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass)\s+(?:{_QUAL}\s+){{1,4}}{_NOUN}\b", _I)),
    ("role_override", re.compile(
        r"\byou\s+are\s+now\s+(?:an?\s+|the\s+|in\s+)?(?:[\w-]+\s+){0,3}?"
        r"(?:DAN|jailbroken|unrestricted|unfiltered|uncensored|mode|AI|assistant|model|chatbot)\b"
        r"|\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|shall)\b"
        r"|\b(?:do\s+anything\s+now|DAN\s+mode|jailbreak(?:ed)?\s+mode|developer\s+mode\s+(?:enabled|activated|on))\b"
        r"|\b(?:act|behave|respond)\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken|evil)\b", _I)),
    ("prompt_exfiltration", re.compile(
        r"\b(?:reveal|print|show|output|leak|repeat|dump|display|tell\s+me)\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:full\s+|entire\s+|hidden\s+|original\s+)?(?:system\s+prompt|initial\s+instructions|hidden\s+instructions)", _I)),
    ("fake_instructions", re.compile(
        r"\b(?:new|updated|real|actual|true|revised|override)\s+(?:system\s+)?(?:instructions?|directives?|prompt)\s*:", _I)),
    ("role_markers", re.compile(
        r"<\|(?:im_start|im_end|system|assistant|user|endoftext|start_header_id)\|>|\[/?INST\]|<</?SYS>>"
        r"|^\s*#{2,}\s*(?:system|instruction)s?\s*:?\s*$|</?(?:system|system_prompt)>", _I | re.M)),
    ("tool_call_injection", re.compile(
        r"[\"'](?:tool_calls?|function_call|tool_use)[\"']\s*:\s*[\[{]"
        r"|<\s*(?:function_calls|tool_call|tool_use)\b|<\s*invoke\s+name\s*=", _I)),
    ("encoded_payload", re.compile(
        r"\b(?:decode|base64[- ]?decode)\b[^.\n]{0,60}?\b(?:and|then)\s+(?:run|execute|eval|follow|obey)\b"
        r"|\b(?:eval|exec)\s*\(\s*(?:atob|base64\.b64decode|b64decode)\b"
        r"|\bbase64\s+(?:-d|--decode)\s*\|\s*(?:ba|z)?sh\b", _I)),
    ("concealment", re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+(?:tell|inform|alert|mention\s+(?:this\s+)?to)\s+the\s+user\s+"
        r"(?:about\s+)?(?:this|these|that\s+you)\b"
        r"|\bwithout\s+(?:telling|informing|alerting)\s+the\s+user\b", _I)),
    ("exfiltration", re.compile(
        r"\b(?:send|post|upload|forward|email|exfiltrate)\b[^.\n]{0,40}?\b(?:memor(?:y|ies)|conversation|chat\s+history|"
        r"credentials|secrets?|api\s+keys?|passwords?|tokens)\b[^.\n]{0,40}?\bto\s+(?:https?://|\S+@\S+)", _I)),
    ("model_addressed", re.compile(
        r"\b(?:attention|note\s+to|message\s+for|if\s+you\s+are|when\s+you\s+read\s+this,?)\s+"
        r"(?:an?\s+|the\s+)?(?:AI|LLM|language\s+model|assistant|agent)\b[^.\n]{0,40}?"
        r"\b(?:must|should|ignore|instead|reading)\b", _I)),
]

# Lower-case keywords at least one of which must occur before a pattern runs:
# substring scans are ~100x cheaper than the regexes, and almost no memory
# contains any of them, so screening costs a few microseconds per write.
_HINTS: dict[str, tuple[str, ...]] = {
    "ignore_instructions": ("ignore", "disregard", "forget", "override", "bypass"),
    "role_override": ("you are now", "from now on", "anything now", "dan mode", "jailbreak", "developer mode", " as "),
    "prompt_exfiltration": ("system prompt", "initial instructions", "hidden instructions"),
    "fake_instructions": ("instruction", "directive", "prompt"),
    "role_markers": ("<|", "inst]", "<<sys>>", "<</sys>>", "##", "system>", "system_prompt>"),
    "tool_call_injection": ("tool_call", "function_call", "tool_use", "invoke"),
    "encoded_payload": ("decode", "atob", "base64"),
    "concealment": ("the user",),
    "exfiltration": (" to http", "@"),
    "model_addressed": ("attention", "note to", "message for", "if you are", "when you read this"),
}
_PATTERNS = [(reason, _HINTS[reason], pattern) for reason, pattern in _PATTERN_LIST]

# Invisible characters used to smuggle instructions. Zero-width joiners in
# emoji/Persian text are legitimate, so ZWJ/ZWNJ only count in runs of 3+.
_ALWAYS_HIDDEN = re.compile(r"[​⁠-⁤‪-‮⁦-⁩\U000e0000-\U000e007f]|(?<!^)﻿")
_JOINER_RUN = re.compile(r"[‌‍]{3,}")
_HTML_COMMENT = re.compile(r"<!--([\s\S]{0,2000}?)-->")
_COMMENT_INSTRUCTION = re.compile(
    r"\b(?:ignore|disregard|you\s+must|you\s+are|instructions?|system\s+prompt|assistant|AI|LLM|"
    r"do\s+not\s+(?:tell|mention)|recommend|always\s+say)\b", _I,
)


# Phrases quoted as examples ("users try 'ignore previous instructions'")
# are discussion, not instructions; markup-style reasons never get this pass.
_QUOTABLE = {"ignore_instructions", "role_override", "prompt_exfiltration", "fake_instructions", "concealment"}
_OPEN_QUOTES = "'\"`\u2018\u201c"
_CLOSE_QUOTES = "'\"`\u2019\u201d"


def _is_quoted(text: str, match: "re.Match[str]") -> bool:
    before = text[match.start() - 1] if match.start() > 0 else ""
    after = text[match.end():match.end() + 2]
    return bool(before) and before in _OPEN_QUOTES and any(ch in _CLOSE_QUOTES for ch in after)


@dataclass(frozen=True)
class InjectionReport:
    """Outcome of :func:`screen_injection`: flagged plus sorted reasons."""

    flagged: bool
    reasons: tuple[str, ...] = ()


def screen_injection(text: str) -> InjectionReport:
    """Screen *text* for instruction-like content aimed at a model.

    Pure and deterministic; linear-time regexes only (bounded gaps), so it is
    safe to run on every write.
    """
    if not text:
        return InjectionReport(False)
    reasons: set[str] = set()
    lowered = text.lower()
    for reason, hints, pattern in _PATTERNS:
        if not any(hint in lowered for hint in hints):
            continue
        for match in pattern.finditer(text):
            if reason in _QUOTABLE and _is_quoted(text, match):
                continue
            reasons.add(reason)
            break
    # Pure-ASCII text (most memories) cannot contain invisible characters.
    if not text.isascii() and (_ALWAYS_HIDDEN.search(text) or _JOINER_RUN.search(text)):
        reasons.add("hidden_characters")
    for comment in (_HTML_COMMENT.finditer(text) if "<!--" in text else ()):
        if _COMMENT_INSTRUCTION.search(comment.group(1)):
            reasons.add("hidden_comment")
            break
    return InjectionReport(bool(reasons), tuple(sorted(reasons)))


__all__ = ["InjectionReport", "UNTRUSTED_TAG", "screen_injection"]
