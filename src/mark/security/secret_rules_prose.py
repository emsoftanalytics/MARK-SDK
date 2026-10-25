# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Secret rules for credentials written in prose rather than ``key=value`` form.

The generic assignment rules in :mod:`.secret_rules` need the keyword right
before a ``:``/``=``/"is" separator. People also write:

* "The root password on the router is admin1234!"  (words between keyword and value)
* "router pw: admin1234!", "wifi pass: Kx8!mPq2zR"  (short keywords)
* "The API key 9f8e…3210", "weather API key is 3kTz…6AeU"  (a long key after the label)
* "wallet seed: abandon ability able …"  (a 12/24-word recovery phrase)

Each rule is gated so ordinary prose stays untouched (see BENIGN_A10 in
tests/_security_corpus.py): prose values must look like a credential
(digits/symbols or a long run), short keywords need a ``:``/``=``, long keys
need 32+ characters mixing letters and digits, and seed phrases need exactly
12/15/18/21/24 lower-case words with no English function words.
"""
from __future__ import annotations

import math
import re

from mark.security.rules import Rule, value_of
from mark.security.secret_rules import SECRET_RULES, _VALUE, _plausible_value
from mark.security.lazy_regex import lazy_compile
from mark.security.secret_rules_i18n import INTL_PROSE_RULES
from mark.security.secret_rules_struct import STRUCT_SECRET_RULES
from mark.security.secret_rules_vendor import VENDOR_SECRET_RULES

# ── "password <for|on|of|to> <1-5 words> is <value>" ─────────────────────────

_PROSE_PASSWORD = lazy_compile(
    r"(?i)(?<![\w-])(?:password|passwd|passphrase|pwd)\s+(?:for|on|of|to|at)\s+(?:[\w.@'’/-]+\s+){1,5}?"
    rf"(?:is|was|=|:)\s+{_VALUE}")


def _credential_like(value: str) -> bool:
    """A prose value is a password only if it does not read as a word ("printed", "in")."""
    return bool(re.search(r"[\d!@#$%^&*+=?~]", value)) or (len(value) >= 10 and not value.isalpha())


def _prose_password_ok(match: "re.Match[str]") -> bool:
    return _plausible_value(match, strict_word=True, min_length=4) and _credential_like(value_of(match))


# ── "pw: X", "pass=X" ─────────────────────────────────────────────────────────

# Short keywords only with an explicit ":"/"=" (never "is"): "pass" is also a
# verb and an exam result ("pass: 45"), so its value must be 6+ characters and
# not only digits; "pw" is unambiguous enough for 4+.
# The separator must touch the keyword ("pw: x"), so docstrings such as
# "pass :data:`h11.CLIENT`" (the verb, then a reST role) stay untouched.
_SHORT_PASSWORD = lazy_compile(rf"(?i)(?<![\w-])(?P<kw>pw|pass)[:=](?![/=:])[ \t]*{_VALUE}")


def _short_password_ok(match: "re.Match[str]") -> bool:
    value = value_of(match)
    if "`" in value or value.startswith(":"):
        return False  # markup ("pass=`x`" in docs), not a credential
    if match.group("kw").lower() == "pass" and (len(value) < 6 or value.isdigit()):
        return False
    return _plausible_value(match, strict_word=True, min_length=4) and _credential_like(value)


# ── "API key <long key>", "token is <long key>", "bot token <key>" ──────────

_LONG_KEY = lazy_compile(
    r"(?i)(?<![\w-])(?P<kw>api[ _-]?key|access[ _-]?key|secret[ _-]?key|client[ _-]?secret|secret|token"
    r"|credentials?|password|passphrase)"
    # "the AWS secret for prod is <key>": a short "for/of/on <what>" may sit before "is".
    r"(?:[ \t]+(?:for|of|on|in|to)[ \t]+(?:[\w.@-]+[ \t]+){0,3}?(?:is|was)|[ \t]+(?:is|was)|[ \t]*[:=])?"
    r"[ \t]+[`\"']?(?P<v>[A-Za-z0-9_\-+/=]{20,})(?![\w/+=-])")
_UUID = lazy_compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _long_key_ok(match: "re.Match[str]") -> bool:
    value = match.group("v")
    # A key mixes letters and digits; identifiers ("token_refresh_handler_v2") are snake_case words.
    if not (re.search(r"\d", value) and re.search(r"[A-Za-z]", value)) or _UUID.match(value):
        return False
    if "/" in value and (value.startswith("/") or "//" in value or not (re.search(r"[a-z]", value)
                                                                     and re.search(r"[A-Z]", value))):
        return False  # a path or URL fragment, not base64 key material
    if re.fullmatch(r"[a-z]+(?:[_-][a-z0-9]+)+", value):
        return False
    # 32+ characters is enough; a 20-31 character key must also look random:
    # upper, lower and digits, with high per-character entropy.
    return len(value) >= 32 or (bool(re.search(r"[a-z]", value) and re.search(r"[A-Z]", value))
                                and _entropy(value) >= 3.5)


def _entropy(value: str) -> float:
    """Shannon entropy in bits per character."""
    counts = [value.count(ch) for ch in set(value)]
    return -sum(c / len(value) * math.log2(c / len(value)) for c in counts)


def _long_key_kind(match: "re.Match[str]") -> str:
    keyword = match.group("kw").lower()
    if "key" in keyword:
        return "api_key"
    if keyword.startswith("pass"):
        return "password"
    return "token" if keyword == "token" else "secret"


# ── BIP-39 style seed / recovery phrases ─────────────────────────────────────

_SEED_LEAD = (r"(?i)(?<![\w-])(?:(?:wallet|bip-?39)\s+)?(?:seed(?:\s+(?:phrase|words))?|recovery\s+(?:phrase|words|seed)"
              r"|mnemonic(?:\s+(?:phrase|words))?|secret\s+recovery\s+phrase)")
_SEED_PHRASE = lazy_compile(_SEED_LEAD + r"[ \t]*(?:[:=]|is|was)?\s+(?P<v>[a-z]{3,8}(?:[ \t\n]+[a-z]{3,8}){11,23})"
                          r"(?=\s*(?:[.,;)]|$))", re.MULTILINE)
# BIP-39 words are 3-8 letters and the English list has no function words, so
# any of these means the run is a sentence, not a recovery phrase.
_FUNCTION_WORDS = frozenset("the and for with but not then that this are was were from into have has its".split())


def _seed_ok(match: "re.Match[str]") -> bool:
    words = match.group("v").split()
    return len(words) in (12, 15, 18, 21, 24) and not _FUNCTION_WORDS.intersection(words)


# ── physical access codes: "the door code for the server room is 739154" ─────

# Only with an access noun (door, gate, alarm, safe, …): status, error, zip,
# postal, area and country codes never qualify. 4-10 digits that end the clause.
_ACCESS_NOUN = (r"(?:door|gate|alarm|safe|locker|lock|padlock|lockbox|key[ \t-]?box|keypad|vault|garage|entry"
                r"|entrance|access|building|room|cabinet)")
_CODE_WORD = r"(?:code|combination|combo|passcode|pin)"
_ACCESS_CODE = lazy_compile(
    rf"(?im)(?<![\w-])(?:{_ACCESS_NOUN}[ \t]+{_CODE_WORD}(?:[ \t]+(?:for|to|of|on|at)[ \t]+(?:[\w'’-]+[ \t]+){{0,3}}?)?"
    rf"|{_CODE_WORD}[ \t]+(?:for|to|of|on|at)[ \t]+(?:the[ \t]+|our[ \t]+|my[ \t]+|your[ \t]+)?(?:[\w'’-]+[ \t]+){{0,2}}?"
    rf"{_ACCESS_NOUN}[ \t]*)"
    r"(?:[ \t]+(?:is|was)[ \t]+|[ \t]*[:=][ \t]*)[#*]?(?P<v>\d{4,10})"
    # The number ends the clause ("safe code: 2024 rewrite" is about code).
    r"(?=[#*]?(?:[.,;!?)]*$|[.,;!?)]+\s|\s+(?:and|or|but|so|to|for|on|at|if|when|then|until|before|after"
    r"|which|please|[\u2014-])(?!\w)))")

PROSE_SECRET_RULES: list[Rule] = [
    Rule("seed_phrase", _SEED_PHRASE, validate=_seed_ok, hint=("seed", "recovery", "mnemonic")),
    Rule("password", _PROSE_PASSWORD, validate=_prose_password_ok, hint=("pass", "pwd")),
    Rule("password", _SHORT_PASSWORD, validate=_short_password_ok, hint=("pw", "pass")),
    Rule("api_key", _LONG_KEY, validate=_long_key_ok, kind_of=_long_key_kind, hint=("key", "secret", "token")),
    Rule("access_code", _ACCESS_CODE, hint=("code", "combination", "combo", "pin")),
]


def _with_prose_rules(rules: list[Rule]) -> list[Rule]:
    """The default rules with vendor shapes and prose rules before the bearer/card/assignment rules.

    Before ``bearer_token`` so "deploy token <key>" is labelled a token, and
    before the generic assignments so a seed phrase is redacted whole. The
    structural rules (any secret-named key, CLI flags, netrc) run last and
    only see values the specific rules left in place.
    """
    at = next((i for i, rule in enumerate(rules) if rule.kind == "bearer_token"), len(rules))
    return [*rules[:at], *VENDOR_SECRET_RULES, *PROSE_SECRET_RULES, *INTL_PROSE_RULES, *rules[at:],
            *STRUCT_SECRET_RULES]


#: Every default secret rule, in the order the engine applies them.
ALL_SECRET_RULES: list[Rule] = _with_prose_rules(SECRET_RULES)

__all__ = ["ALL_SECRET_RULES", "PROSE_SECRET_RULES"]
