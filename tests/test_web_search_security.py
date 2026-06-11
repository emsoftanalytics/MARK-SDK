# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""URL safety for the explicit local web lookup skill."""

import pytest

from mark.skills.web_search import WebSearchSkill, _validate_fetch_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://internal/archive.tar",
        "gopher://legacy",
        "http://localhost:8080/admin",
        "http://service.localhost/",
        "http://printer.local/status",
        "http://127.0.0.1/secrets",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data/",
        "not a url at all",
        "http://",
    ],
)
def test_unsafe_fetch_urls_rejected(url):
    assert _validate_fetch_url(url) is not None


@pytest.mark.parametrize(
    "url",
    [
        "https://en.wikipedia.org/wiki/Memory",
        "http://example.com/page",
        "https://93.184.216.34/",  # public IP literal
    ],
)
def test_public_fetch_urls_allowed(url):
    assert _validate_fetch_url(url) is None


def test_fetch_mode_returns_error_without_network(anyio_or_asyncio=None):
    import asyncio

    skill = WebSearchSkill()
    result = asyncio.run(skill.run({"mode": "fetch", "url": "file:///etc/passwd"}, {}))
    assert result.success is False
    assert "http/https" in (result.error or "")


def test_hostname_resolving_to_private_ip_rejected(monkeypatch):
    import socket

    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert _validate_fetch_url("https://public-looking.example.test/") is not None
