# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Structural secret rules: decide by the *shape* of the text, not by example.

A value is a secret when the name it is bound to says so. These rules find
that binding in every common syntax and judge the *key name*, split into
words (``DATABASE_PASSWORD`` -> database/password, ``AccountKey`` ->
account/key, ``_authToken`` -> auth/token):

* assignments: ``KEY=V``, ``KEY: V``, ``"KEY": "V"``, ``KEY := V``, ``KEY => V``;
  semicolon/ampersand lists (connection strings, URL queries) end a value at
  the next ``;name=`` / ``&name=``;
* command-line flags: ``--api-key V``, ``--password=V``, and the short flags
  of tools known to take a secret inline (``mysql -pV``, ``sshpass -p V``,
  ``redis-cli -a V``, ``docker login -p V``, ``curl -u user:V``, ``ldapsearch -w V``);
* netrc entries: ``machine host login me password V``.

The last key word decides: ``password``, ``secret``, ``token``, ``psk``, …
are strong (any plausible value is redacted); ``sig``, ``auth``, ``cookie``,
``session`` are weak (the value must look like a token: 8+ characters with
letters and digits); ``key`` counts only with a qualifier (``api``,
``account``, ``private``, …), so ``sort_key`` or ``primary_key`` never do.
Placeholders (``your-key-here``, ``${VAR}``, ``changeme``) are always kept.
"""
from __future__ import annotations

import re

from mark.security.lazy_regex import lazy_compile
from mark.security.rules import Rule, value_of
from mark.security.secret_rules import _plausible_value, credential_like, is_placeholder

# ── key-name classification ──────────────────────────────────────────────────

_STRONG_SUFFIXES = ("password", "passwd", "passphrase", "passcode", "pwd", "secret", "token", "apikey",
                    "accesskey", "secretkey", "privatekey", "accountkey", "signingkey", "masterkey",
                    "credential", "credentials", "psk")
_STRONG_WORDS = frozenset({"pass", "pw", "authorization", "bearer", "sas", "sastoken"})
_WEAK_WORDS = frozenset({"sig", "signature", "auth", "cookie", "session", "sessionid", "sid", "hmac"})
_KEY_QUALIFIERS = frozenset(
    "api app application access account private secret signing sign encryption encrypt decryption master "
    "client auth shared storage service subscription admin crypto hmac jwt session deploy ssh gpg pgp "
    "consumer webhook server aes rsa license x".split())
_WORDS = lazy_compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def key_words(key: str) -> list[str]:
    """``DB_PASSWORD`` -> [db, password]; ``apiKey`` -> [api, key]; ``x-api-key`` -> [x, api, key]."""
    return [w.lower() for part in re.split(r"[\s_.\-/:]+", key) for w in _WORDS.findall(part)]


def key_strength(key: str) -> str | None:
    """``"strong"``, ``"weak"`` or None for a key name (see module docstring)."""
    words = key_words(key)
    if not words:
        return None
    last = words[-1]
    if last in _STRONG_WORDS or last.endswith(_STRONG_SUFFIXES):
        return "strong"
    if last == "key" and len(words) > 1 and words[-2] in _KEY_QUALIFIERS:
        return "strong"
    if last == "id" and words[-3:-1] == ["private", "key"]:
        return "strong"   # GCP service-account private_key_id
    if last in _WEAK_WORDS:
        return "weak"
    return None


_TOKEN_CHARS = lazy_compile(r"[A-Za-z0-9+/=_%.~-]{8,}")


def _token_like(value: str, key: str = "") -> bool:
    """One opaque token: 8+ token characters mixing letters and digits.

    Under ``sig``/``signature`` a bare hex digest is an integrity hash, not a
    credential, so those keys need mixed case or base64 characters.
    """
    value = value.rstrip(",;")
    if not (_TOKEN_CHARS.fullmatch(value) and re.search(r"\d", value) and re.search(r"[A-Za-z]", value)):
        return False
    if key_words(key)[-1:] in (["sig"], ["signature"]):
        return bool(re.search(r"[a-z]", value) and re.search(r"[A-Z]", value)) or bool(re.search(r"[+/%]", value))
    return True


def _kind_for(key: str) -> str:
    """Placeholder kind from the key name: password, private_key, cookie, credential, auth_token, token, api_key, secret."""
    last = (key_words(key) or [""])[-1]
    if last.endswith(("password", "passwd", "passphrase", "passcode", "pwd")) or last in ("pass", "pw", "psk"):
        return "password"
    lowered = key.lower()
    for word, kind in (("private", "private_key"), ("cookie", "cookie"), ("credential", "credential"),
                       ("authorization", "auth_token"), ("token", "token")):
        if word in lowered:
            return kind
    return "api_key" if "key" in lowered else "secret"


# ── assignments ──────────────────────────────────────────────────────────────

# Unquoted values end at whitespace, at trailing punctuation, or where the
# next ";name=" / "&name=" pair of a connection string or query begins.
_VALUE = (r"""(?:"(?P<vq>[^"\n]*)"|'(?P<vs>[^'\n]*)'|"""
          r"""(?P<v>[^\s"'`]+?)(?=[,;.)\]}`]*(?:\s|$)|[;&][\w .-]{1,40}=))""")
_KW = r"(?:pass|pwd|secret|token|key|auth|sig|psk|cred|cookie|session|sas|bearer|private|signature|sid|hmac)"
# Each keyword-anchored pattern starts with a lookahead on the keywords' first
# letters: re then skips non-candidate positions in C (~2x faster on prose).
_ASSIGN = lazy_compile(
    rf"(?i)(?=[pstkacbh])(?P<kw>{_KW})[\w.-]*?[\"']?(?P<sep>[ \t]*(?::=|=>|[:=](?![/=]))[ \t]*){_VALUE}")


def _key_span(match: "re.Match[str]") -> str | None:
    """The whole key around the keyword, or None inside a [PLACEHOLDER]."""
    text, start = match.string, match.start()
    while start and (text[start - 1].isalnum() or text[start - 1] in "_.-"):
        start -= 1
    if start and text[start - 1] == "[":
        return None
    return text[start:match.start("sep")].rstrip("\"'")


def _assign_ok(match: "re.Match[str]") -> bool:
    key = _key_span(match)
    strength = key_strength(key) if key else None
    if strength is None:
        return False
    value = value_of(match).strip()
    if strength == "weak" and not _token_like(value, key):
        return False
    if match.group("sep").strip() in (":=", "=>") and match.group("vq") is None and match.group("vs") is None:
        return False   # Go/PHP/walrus assignments are code unless the value is a string literal
    lone_word_key = key.isalpha() and key.islower() and match.group("sep").strip() == ":"
    return _plausible_value(match, strict_word=lone_word_key, min_length=4)


def _assign_kind(match: "re.Match[str]") -> str:
    return _kind_for(_key_span(match) or "")


# ── command-line flags ───────────────────────────────────────────────────────

_ARG = r"""(?:"(?P<vq>[^"\n]+)"|'(?P<vs>[^'\n]+)'|(?P<v>[^\s"'`;|&]+))"""
_LONG_FLAG = lazy_compile(rf"(?i)(?<![\w-])--(?P<key>[a-z][\w-]*?{_KW}[\w-]*)(?:=|[ \t]+){_ARG}")
# Tools that take a secret inline; the flag letters are case-sensitive (mysql -P
# is a port, ldapsearch -W prompts). mysql's -p must touch its value: "-p shop"
# prompts for the password and names a database.
_TOOL_FLAGS = lazy_compile(
    r"(?i)(?=[msrdpnohbtcwl])\b(?:"
    r"(?:mysql|mysqldump|mysqladmin|mysqlimport|mysqlcheck|mariadb[\w-]*)\b[^\n|;&]{0,200}?[ \t](?-i:-p)(?=[^\s-])"
    r"|sshpass\b[^\n|;&]{0,40}?[ \t](?-i:-p)[ \t]*"
    r"|redis-cli\b[^\n|;&]{0,200}?[ \t]-a[ \t]+"
    r"|(?:docker|podman|nerdctl|helm|oras|skopeo|buildah)\b[^\n|;&]{0,60}?\blogin\b[^\n|;&]{0,200}?[ \t](?-i:-p)[ \t]+"
    r"|twine\b[^\n|;&]{0,200}?[ \t](?-i:-p)[ \t]+"
    r"|(?:curl|wget|http|https)\b[^\n|;&]{0,200}?[ \t](?:-u|--user)[ \t]+[\"']?[^\s:\"']+:"
    r"|ldap\w+\b[^\n|;&]{0,200}?[ \t](?-i:-w)[ \t]+"
    rf"){_ARG}")


def _flag_key(match: "re.Match[str]") -> str:
    """The long flag's name (``api-key``), or "" for a tool's short flag."""
    return (match.group("key") or "") if "key" in match.re.groupindex else ""


def _flag_ok(match: "re.Match[str]") -> bool:
    key = _flag_key(match)
    strength = key_strength(key) if key else "strong"   # a tool's -p/-a/-w always takes a secret
    value = value_of(match).rstrip("'\"")
    if strength is None or not value or value.startswith("-") or is_placeholder(value):
        return False
    if strength == "weak" and not _token_like(value, key):
        return False
    return _plausible_value(match, strict_word=False, min_length=4)


def _flag_kind(match: "re.Match[str]") -> str:
    return _kind_for(_flag_key(match) or "password")


# ── netrc ────────────────────────────────────────────────────────────────────

_NETRC = lazy_compile(r"\b(?:login|account)[ \t]+\S+\s+password[ \t]+(?P<v>\S+)"
                      r"|^[ \t]*password[ \t]+(?P<vs>\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE)


def _netrc_ok(match: "re.Match[str]") -> bool:
    if match.group("vs") is not None and not re.search(r"(?i)\b(?:machine|default)\b", match.string):
        return False   # a bare "password X" line is only netrc inside a netrc file
    return credential_like(value_of(match)) and _plausible_value(match, strict_word=True, min_length=4)


# ── HTTP basic-auth tuples and .pgpass lines ─────────────────────────────────

# requests/httpx: auth=("admin", "V"), session.auth = ('u', 'V'), BasicAuth("u", "V").
_AUTH_TUPLE = lazy_compile(
    r"""(?:\bauth[ \t]*=[ \t]*\(|\b(?:HTTPBasicAuth|HTTPDigestAuth|BasicAuth|DigestAuth|HTTPProxyAuth)[ \t]*\()"""
    r"""[ \t]*(?P<q>["'])[\w.@+-]{1,64}(?P=q)[ \t]*,[ \t]*(?:"(?P<vq>[^"\n]{4,})"|'(?P<vs>[^'\n]{4,})')[ \t]*\)""")
# .pgpass: hostname:port:database:username:password, one entry per line.
_PGPASS = lazy_compile(r"^[ \t]*(?P<h>[\w.-]+|\*):(?:\d{1,5}|\*):[\w.*-]+:[\w.@*-]+:(?P<v>\S+)[ \t]*$", re.MULTILINE)


def _auth_tuple_ok(match: "re.Match[str]") -> bool:
    return _plausible_value(match, strict_word=True, min_length=4)


def _pgpass_ok(match: "re.Match[str]") -> bool:
    host = match.group("h")
    if host != "*" and not re.search(r"[A-Za-z.]", host):
        return False   # "12:30:45:00:15" is a time code, not a host
    return credential_like(value_of(match)) and _plausible_value(match, strict_word=True, min_length=4)


STRUCT_SECRET_RULES: list[Rule] = [
    # Hints are cheap substring checks; prefilters are cheap regexes that start
    # with a literal, so most memories skip these rules without a full scan.
    Rule("password", _TOOL_FLAGS, validate=_flag_ok,
         hint=("mysql", "mariadb", "sshpass", "redis-cli", "login", "twine", "curl", "wget", "http", "ldap"),
         prefilter=lazy_compile(r"[ \t]-(?:[pawu]|-user)")),
    Rule("secret", _LONG_FLAG, validate=_flag_ok, kind_of=_flag_kind, hint="--"),
    Rule("password", _NETRC, validate=_netrc_ok, hint=("login", "account", "machine", "default")),
    Rule("password", _AUTH_TUPLE, validate=_auth_tuple_ok, hint="auth"),
    Rule("password", _PGPASS, validate=_pgpass_ok, prefilter=lazy_compile(r":(?:\d{1,5}|\*):")),
    Rule("secret", _ASSIGN, validate=_assign_ok, kind_of=_assign_kind,
         hint=("pass", "pwd", "secret", "token", "key", "auth", "sig", "psk", "cred", "cookie", "session",
               "sas", "bearer", "private", "sid", "hmac"), prefilter=lazy_compile(r"[:=]")),
]

__all__ = ["STRUCT_SECRET_RULES", "key_strength", "key_words"]
