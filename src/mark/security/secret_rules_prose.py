# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
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

import re

from mark.security.rules import Rule, value_of
from mark.security.secret_rules import SECRET_RULES, _VALUE, _plausible_value

# ── "password <for|on|of|to> <1-5 words> is <value>" ─────────────────────────

_PROSE_PASSWORD = re.compile(
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
_SHORT_PASSWORD = re.compile(rf"(?i)(?<![\w-])(?P<kw>pw|pass)[ \t]*[:=](?![/=])[ \t]*{_VALUE}")


def _short_password_ok(match: "re.Match[str]") -> bool:
    value = value_of(match)
    if match.group("kw").lower() == "pass" and (len(value) < 6 or value.isdigit()):
        return False
    return _plausible_value(match, strict_word=True, min_length=4) and _credential_like(value)


# ── "API key <32+ chars>", "token is <32+ chars>" ────────────────────────────

_LONG_KEY = re.compile(
    r"(?i)(?<![\w-])(?P<kw>api[ _-]?key|access[ _-]?key|secret[ _-]?key|client[ _-]?secret|secret|token)"
    r"(?:[ \t]+(?:is|was)|[ \t]*[:=])?[ \t]+[`\"']?(?P<v>[A-Za-z0-9_\-]{32,})(?![\w-])")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _long_key_ok(match: "re.Match[str]") -> bool:
    value = match.group("v")
    # A key mixes letters and digits; identifiers ("token_refresh_handler_v2") are snake_case words.
    if not (re.search(r"\d", value) and re.search(r"[A-Za-z]", value)) or _UUID.match(value):
        return False
    return not re.fullmatch(r"[a-z]+(?:[_-][a-z0-9]+)+", value)


def _long_key_kind(match: "re.Match[str]") -> str:
    keyword = match.group("kw").lower()
    if "key" in keyword:
        return "api_key"
    return "token" if keyword == "token" else "secret"


# ── BIP-39 style seed / recovery phrases ─────────────────────────────────────

_SEED_LEAD = (r"(?i)(?<![\w-])(?:(?:wallet|bip-?39)\s+)?(?:seed(?:\s+(?:phrase|words))?|recovery\s+(?:phrase|words|seed)"
              r"|mnemonic(?:\s+(?:phrase|words))?|secret\s+recovery\s+phrase)")
_SEED_PHRASE = re.compile(_SEED_LEAD + r"[ \t]*(?:[:=]|is|was)?\s+(?P<v>[a-z]{3,8}(?:[ \t\n]+[a-z]{3,8}){11,23})"
                          r"(?=\s*(?:[.,;)]|$))", re.MULTILINE)
# BIP-39 words are 3-8 letters and the English list has no function words, so
# any of these means the run is a sentence, not a recovery phrase.
_FUNCTION_WORDS = frozenset("the and for with but not then that this are was were from into have has its".split())


def _seed_ok(match: "re.Match[str]") -> bool:
    words = match.group("v").split()
    return len(words) in (12, 15, 18, 21, 24) and not _FUNCTION_WORDS.intersection(words)


PROSE_SECRET_RULES: list[Rule] = [
    Rule("seed_phrase", _SEED_PHRASE, validate=_seed_ok, hint=("seed", "recovery", "mnemonic")),
    Rule("password", _PROSE_PASSWORD, validate=_prose_password_ok, hint=("pass", "pwd")),
    Rule("password", _SHORT_PASSWORD, validate=_short_password_ok, hint=("pw", "pass")),
    Rule("api_key", _LONG_KEY, validate=_long_key_ok, kind_of=_long_key_kind, hint=("key", "secret", "token")),
]


def _with_prose_rules(rules: list[Rule]) -> list[Rule]:
    """The default rules with the prose rules inserted before the bearer/card/assignment rules.

    Before ``bearer_token`` so "deploy token <key>" is labelled a token, and
    before the generic assignments so a seed phrase is redacted whole.
    """
    at = next((i for i, rule in enumerate(rules) if rule.kind == "bearer_token"), len(rules))
    return [*rules[:at], *PROSE_SECRET_RULES, *rules[at:]]


#: Every default secret rule, in the order the engine applies them.
ALL_SECRET_RULES: list[Rule] = _with_prose_rules(SECRET_RULES)

__all__ = ["ALL_SECRET_RULES", "PROSE_SECRET_RULES"]
