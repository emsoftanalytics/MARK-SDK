# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Default secret rules: credentials that must never be stored in clear.

Order matters. Whole-structure secrets (PEM blocks, URLs with credentials,
vendor tokens) run before the generic ``key = value`` rules so a value is
never half-redacted, and the generic rules then skip values that already
are placeholders. Every rule is tuned for low false positives on code,
hashes, ids, versions and prose (see tests/_security_corpus.py).
"""
from __future__ import annotations

import re

from mark.security.checksums import luhn_valid
from mark.security.rules import Rule, value_of

# ── value plausibility ────────────────────────────────────────────────────────

# References and placeholders that name a secret without containing it:
# $VAR, ${VAR}, %VAR%, os.environ[...], getenv(...), <your-key>, ****, …
_REFERENCE = re.compile(
    r"""(?ix)^(?:
        \$\{?[A-Za-z_][\w]*\}?[\w.\[\]'"()]*      # shell / compose variables
      | %[A-Za-z_]\w*%                            # windows variables
      | (?:os\.environ|os\.getenv|getenv|process\.env|env|ENV|config|settings|secrets)\b.*
      | <[^>]*>?                                  # <your-key-here>
      | \[[A-Z]+[:\]].*                           # existing placeholders
      | [*xX•]{3,} | \.+ | …
      | none|null|nil|undefined|true|false|yes|no|n/?a|empty|required|optional
      | bearer|basic|token
      | (?:str|bytes|int|float|bool|dict|list|object|Any|Optional\[.*|Secret\w*)    # type annotations
      | [rbfu]{1,2}["'].*                         # r"…" / f"…" / b"…" code literals
    )[,;.)\]}`]*$"""
)
_LOWER_WORD = re.compile(r"^[a-z]+$")
# Code identifiers passed as values: session_id=session_id, token=self._token,
# api_key=OPENAI_API_KEY, passphrase=args.passphrase. A real secret is
# rarely a dotted name or an all-lowercase/all-uppercase snake_case word.
_IDENTIFIER = re.compile(
    r"^(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|_*[a-z]+(?:_[a-z]+)+|[A-Z]+(?:_[A-Z]+)+|self|cls|kwargs|args)$"
)


def _plausible_value(match: "re.Match[str]", *, strict_word: bool, min_length: int = 1) -> bool:
    """Reject references/placeholders; optionally reject plain prose words.

    ``strict_word`` is used for weak contexts ("password is …", "token: …")
    where a lowercase dictionary word ("required", "expired") is prose.
    """
    value = value_of(match).strip()
    if not value or _REFERENCE.match(value):
        return False
    closes = match.string[match.end():match.end() + 1] in (")", "]")
    if re.match(r"[A-Za-z_][\w.]*[(\[]", value) and (closes or value[-1] in ")]"):
        return False  # a call or subscript such as get_secret("db") / token[:-1]
    if _IDENTIFIER.match(value.rstrip(",;)]}")):
        return False  # a variable or attribute reference in code
    if len(value) < min_length:
        return False  # "1 token = 4 chars"
    if strict_word and _LOWER_WORD.match(value) and len(value) <= 12:
        return False
    return True


# ── generic assignments ──────────────────────────────────────────────────────

# Unquoted values run to the next whitespace (so "Hunter2!xyz&3" is redacted
# whole — the old "[REDACTED]&3" leak), minus trailing sentence/JSON punctuation.
_VALUE = r"""(?:"(?P<vq>[^"\n]*)"|'(?P<vs>[^'\n]*)'|(?P<v>\S+?)(?=[,;.)\]}`]*(?:\s|$)))"""
_SEP = r"""["']?(?:[ \t]*[:=](?![/=])[ \t]*|[ \t]+(?:is|was)[ \t]+)"""
# Patterns start at the sensitive *keyword* (fast literal scan); the full key
# ("DB_PASSWORD", "client_secret") is recovered by scanning back in _key_of.
_PASSWORD_KW = r"(?:password|passwd|passphrase|pwd|(?<=[\w.-])[_.-]pass)"
_SECRET_KW = (
    r"(?:secret|token|api[ _-]?key|apikey|access[_-]?key|private[_-]?key|"
    r"signing[_-]?key|encryption[_-]?key|cookie|credentials?|authorization)"
)

PASSWORD_ASSIGNMENT = re.compile(rf"(?i)(?P<kw>{_PASSWORD_KW})(?P<sep>{_SEP}){_VALUE}")
SECRET_ASSIGNMENT = re.compile(rf"(?i)(?P<kw>{_SECRET_KW})(?P<sep>{_SEP}){_VALUE}")


def _key_of(match: "re.Match[str]") -> str | None:
    """Return the full key ending at the keyword, or None for a placeholder.

    "[SECRET:openai_key]" must not read as ``SECRET: openai_key]``.
    """
    text, start = match.string, match.start()
    while start and (text[start - 1].isalnum() or text[start - 1] in "_.-"):
        start -= 1
    if start and text[start - 1] == "[":
        return None
    return text[start:match.end("kw")]


def _password_ok(match: "re.Match[str]") -> bool:
    if _key_of(match) is None:
        return False
    # "password: hunter" is a password; "password is required" is prose.
    prose = match.group("sep").strip(" \t\"'") in {"is", "was"}
    return _plausible_value(match, strict_word=prose)


def _secret_ok(match: "re.Match[str]") -> bool:
    sep = match.group("sep").strip(" \t\"'")
    if sep in {"is", "was"}:
        return False  # "the secret to good sourdough is patience"
    # "token: expired" is prose; PRIVATE_KEY=zzzz / api_key: abc is config.
    key = _key_of(match)
    if key is None:
        return False
    prose_key = key.isalpha() and key.islower() and sep == ":"
    return _plausible_value(match, strict_word=prose_key, min_length=4)


def _secret_kind(match: "re.Match[str]") -> str:
    key = (_key_of(match) or "").lower()
    for word, kind in (("private", "private_key"), ("cookie", "cookie"), ("credential", "credential"),
                       ("authorization", "auth_token"), ("token", "token")):
        if word in key:
            return kind
    if "key" in key:
        return "api_key"
    return "secret"


# ── vendor-specific validators ───────────────────────────────────────────────

def _openai_ok(match: "re.Match[str]") -> bool:
    # Real keys carry digits; long all-letter bodies are still keys. This keeps
    # identifiers like "sk-loading-spinner-container" out.
    body = match.group(0)[3:]
    return any(ch.isdigit() for ch in body) or len(body) >= 32


def _bearer_ok(match: "re.Match[str]") -> bool:
    token = match.group("v")
    return bool(re.search(r"[0-9._~+/=\-]", token)) or len(token) >= 20


_AWS_CONTEXT = re.compile(r"(?i)aws|secret|access[ _-]?key")


def _aws_secret_ok(match: "re.Match[str]") -> bool:
    """40-char base64 is only an AWS secret near AWS/secret wording.

    Mixed case is required, which rules out git SHAs and hex digests.
    """
    value = match.group("v")
    if not (re.search(r"[a-z]", value) and re.search(r"[A-Z]", value)):
        return False
    if not re.search(r"[0-9/+]", value):
        return False
    start = match.start()
    window = match.string[max(0, start - 120):start]
    return bool(_AWS_CONTEXT.search(window))


# Labelled identifiers ("order #…", "invoice no. …") are never card numbers,
# even when ~1 in 10 of them happens to pass the Luhn check.
_ID_LABEL = re.compile(
    r"(?i)\b(?:order|invoice|tracking|ref(?:erence)?|ticket|case|booking|confirmation|serial|"
    r"shipment|parcel|transaction|txn|request|build|id)\b[\s#:.-]*(?:no\.?|num(?:ber)?|id)?[\s#:.-]*$"
)


def _card_ok(match: "re.Match[str]") -> bool:
    before = match.string[max(0, match.start() - 30):match.start()]
    if before and (before[-1].isalnum() or before[-1] in "_-"):
        return False  # inside a hex digest or an identifier
    return luhn_valid(match.group(0)) and not _ID_LABEL.search(before)


def _rx(pattern: str, flags: int = 0) -> "re.Pattern[str]":
    return re.compile(pattern, flags)


SECRET_RULES: list[Rule] = [
    Rule("private_key", _rx(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
        r"(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|\Z)"), hint="private key"),
    Rule("password", _rx(r"://[^\s/:@]+:(?P<v>[^\s/@]+)@"), hint="://"),  # scheme://user:pass@host
    Rule("slack_webhook", _rx(r"https://hooks\.slack\.com/(?:services|workflows)/(?P<v>[A-Za-z0-9/_-]{20,})"),
         hint="hooks.slack"),
    Rule("discord_webhook", _rx(r"https://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/(?P<v>\d+/[\w-]{20,})"),
         hint="discord"),
    Rule("jwt", _rx(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{10,}"), hint="eyj", word_start=True),
    Rule("anthropic_key", _rx(r"sk-ant-[A-Za-z0-9_-]{16,}"), hint="sk-ant-", word_start=True),
    Rule("openai_key", _rx(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{12,}"), validate=_openai_ok,
         hint="sk-", word_start=True),
    Rule("stripe_key", _rx(r"[sr]k_(?:live|test)_[A-Za-z0-9]{10,}"), hint=("sk_", "rk_"), word_start=True),
    Rule("stripe_key", _rx(r"whsec_[A-Za-z0-9]{20,}"), hint="whsec_", word_start=True),
    Rule("aws_access_key_id", _rx(r"(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b"),
         hint=("akia", "asia", "agpa", "aida", "aroa", "aipa", "anpa", "anva"), word_start=True),
    Rule("aws_secret_access_key", _rx(r"(?<![A-Za-z0-9/+=])(?P<v>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"),
         validate=_aws_secret_ok, hint=("aws", "secret", "access")),
    Rule("github_token", _rx(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
         hint=("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"), word_start=True),
    Rule("gitlab_token", _rx(r"gl(?:pat|dt|rt|ptt|cbt|soat|ft|imt)-[A-Za-z0-9_-]{20,}"),
         hint=("glpat-", "gldt-", "glrt-", "glptt-", "glcbt-", "glsoat-", "glft-", "glimt-"), word_start=True),
    Rule("slack_token", _rx(r"xox[abposr]-[A-Za-z0-9-]{10,}"), hint="xox", word_start=True),
    Rule("google_api_key", _rx(r"AIza[0-9A-Za-z_-]{30,}"), hint="aiza", word_start=True),
    Rule("huggingface_token", _rx(r"hf_[A-Za-z0-9]{30,}"), hint="hf_", word_start=True),
    Rule("npm_token", _rx(r"npm_[A-Za-z0-9]{36}\b"), hint="npm_", word_start=True),
    Rule("sendgrid_key", _rx(r"SG\.[\w-]{22}\.[\w-]{43}\b"), hint="sg.", word_start=True),
    Rule("api_key", _rx(r"(?:key|secret|token|tok)_(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{20,}\b"),
         hint=("key_", "secret_", "token_", "tok_"), word_start=True),
    Rule("bearer_token", _rx(r"(?i)(?:Bearer|Basic|Token)[ \t]+(?P<v>[A-Za-z0-9._~+/=\-]{8,})"),
         validate=_bearer_ok, hint=("bearer", "basic", "token"), word_start=True),
    # Card shapes: contiguous, 4-4-4-4(+), or Amex 4-6-5; then a Luhn check.
    Rule("card_number", _rx(r"(?:\d{13,19}|\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ -])\d{6}\2\d{4,5})"
                            r"(?!\w|-\d)"), validate=_card_ok,
         prefilter=re.compile(r"\d{4}[ -]?\d{4}")),
    # PINs only in an explicit "PIN: 1234" context: bare 4-digit numbers are
    # years, ports, quantities and codes far more often than PINs.
    Rule("pin", _rx(r"(?i)(?:PIN(?:[ \t]*(?:code|number))?|passcode)(?P<sep>[ \t]*(?:[:=#]|is|was)[ \t]*)"
                    r"(?P<v>\d{4,8})\b"), hint=("pin", "passcode"), word_start=True),
    Rule("password", PASSWORD_ASSIGNMENT, validate=_password_ok, hint=("pass", "pwd")),
    Rule("secret", SECRET_ASSIGNMENT, validate=_secret_ok, kind_of=_secret_kind,
         hint=("secret", "token", "key", "cookie", "credential", "authorization")),
]

__all__ = ["PASSWORD_ASSIGNMENT", "SECRET_ASSIGNMENT", "SECRET_RULES"]
