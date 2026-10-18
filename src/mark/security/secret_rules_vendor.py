# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Vendor token shapes beyond the core set in :mod:`.secret_rules`.

Each rule is a documented prefix or structure that ordinary text does not
produce: PyPI, Docker Hub, HashiCorp Vault, Slack app/refresh tokens,
Google OAuth access tokens, Discord and Telegram bot tokens, and a dozen
SaaS prefixes. Shapes without a literal prefix (Discord, Telegram) also need
letters and digits mixed in the token body, which identifiers and numbers
never have in that layout.
"""
from __future__ import annotations

import re

from mark.security.lazy_regex import lazy_compile
from mark.security.rules import Rule


def _mixed(match: "re.Match[str]") -> bool:
    body = match.group("t") if "t" in match.re.groupindex else match.group(0)
    return bool(re.search(r"\d", body) and re.search(r"[a-z]", body) and re.search(r"[A-Z]", body))


def _rule(kind: str, pattern: str, hint: str | tuple[str, ...], validate=None) -> Rule:
    return Rule(kind, lazy_compile(pattern), validate=validate, hint=hint, word_start=True)


VENDOR_SECRET_RULES: list[Rule] = [
    _rule("pypi_token", r"pypi-[A-Za-z0-9_-]{32,}", "pypi-"),
    _rule("docker_token", r"dckr_(?:pat|oat)_[A-Za-z0-9_-]{20,}", "dckr_"),
    _rule("vault_token", r"hv[sbr]\.[A-Za-z0-9_-]{20,}", ("hvs.", "hvb.", "hvr.")),
    _rule("slack_token", r"xapp-\d-[A-Za-z0-9-]{20,}|xoxe(?:\.xox[bp])?-\d-[A-Za-z0-9-]{20,}", ("xapp-", "xoxe")),
    _rule("google_oauth_token", r"ya29\.[A-Za-z0-9_-]{20,}", "ya29."),
    _rule("shopify_token", r"shp(?:at|ss|ca|pa)_[a-fA-F0-9]{32}", "shp"),
    _rule("atlassian_token", r"ATATT3[A-Za-z0-9_=-]{40,}", "atatt3"),
    _rule("age_secret_key", r"AGE-SECRET-KEY-1[0-9A-Z]{50,}", "age-secret-key-"),
    _rule("doppler_token", r"dp\.(?:pt|st|sa|ct|scim|audit)\.[A-Za-z0-9]{30,}", "dp."),
    _rule("linear_key", r"lin_(?:api|oauth)_[A-Za-z0-9]{32,}", "lin_"),
    _rule("notion_token", r"ntn_[A-Za-z0-9]{40,}", "ntn_"),
    _rule("mailgun_key", r"key-[0-9a-f]{32}\b", "key-"),
    _rule("postman_key", r"PMAK-[0-9a-f]{24}-[0-9a-f]{34}", "pmak-"),
    _rule("grafana_token", r"gl(?:sa|c)_[A-Za-z0-9_=+/-]{32,}", ("glsa_", "glc_")),
    _rule("databricks_token", r"dapi[0-9a-f]{32}(?:-\d)?\b", "dapi"),
    _rule("digitalocean_token", r"do[opr]_v1_[a-f0-9]{64}", ("dop_", "doo_", "dor_")),
    _rule("newrelic_key", r"NRA[KIJ]-[A-Z0-9]{27}", "nra"),
    _rule("groq_key", r"gsk_[A-Za-z0-9]{40,}", "gsk_"),
    _rule("replicate_token", r"r8_[A-Za-z0-9]{30,}", "r8_"),
    # Discord bot token: base64(user id) . 6-char timestamp . 27-40 char HMAC.
    Rule("discord_token", lazy_compile(
        r"[MNO][A-Za-z0-9_-]{23,27}\.[A-Za-z0-9_-]{6}\.(?P<t>[A-Za-z0-9_-]{27,40})(?![\w-])"),
        validate=_mixed, prefilter=lazy_compile(r"\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}"), word_start=True),
    # Telegram bot token: numeric bot id, colon, 35-char secret.
    Rule("telegram_token", lazy_compile(r"(?<![\d:])\d{8,10}:(?P<t>[A-Za-z0-9_-]{35})(?![\w-])"),
         validate=_mixed, prefilter=lazy_compile(r"\d{8}:")),
]

__all__ = ["VENDOR_SECRET_RULES"]
