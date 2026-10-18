# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Injection that hides in markup, links or encodings rather than in plain words.

* **concealment**: an element styled so a human never sees it
  (``display:none``, ``visibility:hidden``, ``font-size:0``, ``opacity:0``,
  white text, the ``hidden`` attribute, ``<font color=white>``) whose text
  talks to a model. Hidden UI scaffolding ("Loading…", an empty template) is
  ordinary markup and stays unflagged.
* **exfiltration links**: a Markdown/HTML link or image whose query string
  is filled from a template (``?d={conversation}``, ``?k=$TOKEN``,
  ``?data=USER_EMAIL_HERE``): the reader is meant to build the URL from data.
* **obfuscation**: the same screen runs again on de-obfuscated variants:
  letter-spaced text (``i g n o r e``, ``D-i-s-r-e-g-a-r-d``) collapsed,
  leetspeak (``1gn0r3 4ll``) mapped back to letters, and base64 blobs decoded.
  A hit only in a variant is reported as ``obfuscation`` (or
  ``encoded_payload`` for base64) plus the underlying reason.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Callable

from mark.security.lazy_regex import lazy_compile

Scan = Callable[[str], "set[str]"]
_I = re.IGNORECASE

_HIDDEN_STYLE = (r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:[;\"']|!)"
                 r"|opacity\s*:\s*0(?:\.0+)?\s*[;\"'!]|(?<![-\w])color\s*:\s*(?:#fff(?:fff)?\b|white\b"
                 r"|rgba?\(\s*255\s*,\s*255\s*,\s*255)|(?:height|width|max-height)\s*:\s*0(?:px)?\s*[;\"']"
                 r"|(?:left|text-indent|top)\s*:\s*-\d{3,}|clip\s*:\s*rect\(\s*0")
_HIDDEN_ELEMENT = lazy_compile(
    r"<(?P<tag>[a-z][a-z0-9]*)\b(?=[^>]*?(?:style\s*=\s*[\"'][^\"']*(?:" + _HIDDEN_STYLE + r")"
    r"|\shidden(?:\s|>|=)|aria-hidden\s*=\s*[\"']?true|color\s*=\s*[\"']?(?:#?fff(?:fff)?|white)\b))"
    r"[^>]*>(?P<body>[^<]{0,2000}(?:<(?!/(?P=tag)\s*>)[^<]{0,500}){0,20})</(?P=tag)\s*>", _I)
# Text inside hidden markup that addresses a model or gives it orders.
_TALKS_TO_MODEL = lazy_compile(
    r"\b(?:AI|A\.I\.|LLM|GPT|assistant|chatbot|language\s+model|agents?|model|ignore|disregard|you\s+must"
    r"|you\s+are|instructions?|system\s+prompt|do\s+not\s+(?:tell|mention)|recommend|always\s+(?:say|answer|reply)"
    r"|rate|score|forward|send|approve|reveal)\b", _I)

# Query value built from a template: {x}, {{x}}, ${x}, $X, %7B, <x>, USER_EMAIL_HERE.
_TEMPLATE = r"(?:\{\{?[^}\s]{1,60}\}|\$\{?[A-Za-z_]\w*|%7B|<[^>\s]{1,40}>|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b)"
_EXFIL_LINK = lazy_compile(
    r"!?\[[^\]\n]{0,200}\]\(\s*<?https?://[^)\s]*?[?&][\w.-]+=[^)\s&]*" + _TEMPLATE
    + r"|<img\b[^>]*\bsrc\s*=\s*[\"']?https?://[^\"'\s>]*?[?&][\w.-]+=[^\"'\s>&]*" + _TEMPLATE, _I)

# Five or more single letters joined by spaces or by - . · * (one style per run).
_LETTER_SPACED = lazy_compile(r"(?<!\w)(?:[^\W\d_](?:[ \t]{1,3}|[-.·*][ \t]?)){4,}[^\W\d_](?!\w)")
_SQUASHED = lazy_compile(
    r"(?:ignore|disregard|forget|override|bypass|skip)(?:all|any|every|the|your|my|of|prior|previous|above|earlier"
    r"|preceding|existing|system|original|initial|other|safety)+(?:instruction|prompt|directive|guideline|rule"
    r"|restriction|polic|direction|command|programming)")
# Cheap gates (each starts with a character class, which re skips through in C).
_SPACED_GATE = lazy_compile(r"[ \t.\-·*][^\W\d_][ \t.\-·*]{1,3}[^\W\d_][ \t.\-·*]{1,3}[^\W\d_][ \t.\-·*]")
_LEET_GATE = lazy_compile(r"[013457@$](?=[A-Za-z])")
# Two or more leet words close together ("1gn0r3 4ll pr3v10u5"): a lone "md5" or "0x1f" is not obfuscation.
_LEET_WORD = r"[A-Za-z013457@$]*(?:[A-Za-z][013457@$]|[013457@$][A-Za-z])[A-Za-z013457@$]*"
_LEET_RUN = lazy_compile(rf"(?<!\w){_LEET_WORD}(?:\W+(?:\w+\W+){{0,2}}?{_LEET_WORD}(?!\w))+")
_LEET_TOKEN = lazy_compile(r"[A-Za-z013457@$]+")
_LEET_MAP = str.maketrans("013457@$", "oieastas")
_B64 = lazy_compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{20,}={0,2}(?![A-Za-z0-9+/=])")
_RUN_DECODED = lazy_compile(
    r"\b(?:run|execute|eval|follow|obey|perform|carry\s+out)\b[^.\n]{0,40}?\b(?:decod\w*|base64)\b"
    r"|\b(?:decode|base64[- ]?decode)\b[^.\n]{0,60}?\b(?:and|then)\s+(?:run|execute|eval|follow|obey|do)\b", _I)


def _collapse_spaced(span: str) -> str:
    """``i g n o r e   a l l`` -> ``ignore all``; ``D-i-s-r-e-g-a-r-d a-l-l`` -> ``Disregard all``."""
    if re.search(r"\w[-.·*_]\s?\w", span):
        return re.sub(r"(?<=\w)[-.·*_]\s?(?=\w)", "", span)
    return " ".join(part.replace(" ", "").replace("\t", "") for part in re.split(r"[ \t]{2,}", span))


_CONTEXT = 80     # characters kept around a de-obfuscated span
_MAX_SPANS = 20   # spans re-screened per text (bounded cost on huge inputs)


def _snippet(text: str, match: "re.Match[str]", replacement: str) -> str:
    return text[max(0, match.start() - _CONTEXT):match.start()] + replacement + text[match.end():match.end() + _CONTEXT]


def _spaced_reasons(text: str, scan: Scan) -> set[str]:
    """Reasons found after collapsing letter-spaced runs (each run screened with its context)."""
    found: set[str] = set()
    for i, match in enumerate(_LETTER_SPACED.finditer(text)):
        if i >= _MAX_SPANS:
            break
        span = match.group(0)
        if _SQUASHED.search(re.sub(r"[^a-z]", "", span.lower())):
            found.add("ignore_instructions")
        found |= scan(_snippet(text, match, _collapse_spaced(span)))
    return found


def _leet_reasons(text: str, scan: Scan) -> set[str]:
    """Reasons found after mapping leetspeak runs back to letters."""
    found: set[str] = set()
    for i, match in enumerate(_LEET_RUN.finditer(text)):
        if i >= _MAX_SPANS:
            break
        plain = _LEET_TOKEN.sub(lambda m: m.group(0).translate(_LEET_MAP), match.group(0))
        found |= scan(_snippet(text, match, plain))
    return found


def _decoded_blobs(text: str) -> list[str]:
    out = []
    for match in _B64.finditer(text):
        blob = match.group(0)
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            decoded = raw.decode("utf-8")
        except (binascii.Error, ValueError):
            continue
        printable = sum(ch.isprintable() or ch in "\n\t" for ch in decoded)
        if decoded and printable / len(decoded) > 0.95 and " " in decoded:
            out.append(decoded)
    return out


def markup_reasons(text: str, scan: Scan, base: "set[str]") -> set[str]:
    """Reasons from hidden markup, exfiltration links and de-obfuscated variants of *text*.

    *scan* is the plain-text screen (phrase patterns only); it is re-run on
    variants, never recursively on its own output. *base* holds the reasons
    the plain text already gave, so only new ones count as obfuscation.
    """
    reasons: set[str] = set()
    lowered = text.lower()
    if "<" in text and any(h in lowered for h in ("style", "hidden", "color")):
        for match in _HIDDEN_ELEMENT.finditer(text):
            body = re.sub(r"<[^>]*>", " ", match.group("body"))
            if _TALKS_TO_MODEL.search(body) or scan(body):
                reasons.add("concealment")
                break
    if "http" in lowered and ("](" in text or "<img" in lowered) and _EXFIL_LINK.search(text):
        reasons.add("exfiltration")
    found: set[str] = set()
    if _SPACED_GATE.search(text):
        found |= _spaced_reasons(text, scan)
    gate = _LEET_GATE.finditer(text)
    if next(gate, None) and next(gate, None):   # two letter-digit joints, like "1gn0r3"
        found |= _leet_reasons(text, scan)
    if found - base:
        reasons |= (found - base) | {"obfuscation"}
    long_tokens = [token for token in text.split() if len(token) >= 20]
    if long_tokens:
        decoded = _decoded_blobs(" ".join(long_tokens))
        if any(scan(blob) for blob in decoded) or (decoded and _RUN_DECODED.search(text)):
            reasons.add("encoded_payload")
    return reasons


__all__ = ["markup_reasons"]
