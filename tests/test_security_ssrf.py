# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Security hardening: SSRF protection in the web-search fetcher (address
validation, DNS pinning, redirect revalidation, size caps, TLS hostnames).
"""
from __future__ import annotations

import asyncio
import http.server
import socket
import threading
from typing import Any

import pytest

from mark.middlewares.skills import web_search


# ── 4. SSRF ───────────────────────────────────────────────────────────────────

class _Handler(http.server.BaseHTTPRequestHandler):
    hits: list[tuple[str, str]] = []
    body = b"<html><body>hello public world</body></html>"
    redirect_to: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        type(self).hits.append((self.path, self.headers.get("Host", "")))
        if type(self).redirect_to and self.path == "/start":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture()
def local_server():
    _Handler.hits = []
    _Handler.redirect_to = None
    _Handler.body = b"<html><body>hello public world</body></html>"
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_local_http_server_is_blocked(local_server) -> None:
    port = local_server.server_address[1]
    url = f"http://127.0.0.1:{port}/secret"
    with pytest.raises(web_search.FetchBlocked):
        web_search._safe_http_get(url, timeout=2)
    result = asyncio.run(web_search.WebSearchSkill().run({"mode": "fetch", "url": url}, {}))
    assert result.success is False
    assert _Handler.hits == []


@pytest.mark.parametrize(
    "address",
    ["100.64.0.1", "10.1.2.3", "172.16.0.1", "192.168.0.1", "169.254.169.254", "0.0.0.0",
     "224.0.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "192.0.2.10", "198.18.0.1"],
)
def test_non_global_resolutions_are_blocked(monkeypatch, address: str) -> None:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(family, socket.SOCK_STREAM, 6, "", (address, 80))])
    assert web_search._validate_fetch_url("http://innocent.example.test/") is not None


def test_mixed_public_and_private_resolution_is_blocked(monkeypatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80)),
    ])
    assert web_search._validate_fetch_url("http://rebind.example.test/") is not None


def test_resolution_failure_fails_closed(monkeypatch) -> None:
    def boom(*a, **k):
        raise socket.gaierror("no dns")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert web_search._validate_fetch_url("https://example.com/") is not None


def test_credentials_in_url_are_rejected() -> None:
    assert web_search._validate_fetch_url("http://user:pw@93.184.216.34/") is not None


def _route_public_name_to_loopback(monkeypatch) -> None:
    """Pretend public.example.test resolves to 127.0.0.1 and that 127.0.0.1 is public."""
    real = socket.getaddrinfo

    def fake(host, *a, **k):
        if host == "public.example.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", a[0] if a else 80))]
        return real(host, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    monkeypatch.setattr(web_search, "_is_internal_address", lambda addr: str(addr) != "127.0.0.1")


def test_connection_is_pinned_and_host_header_kept(monkeypatch, local_server) -> None:
    port = local_server.server_address[1]
    _route_public_name_to_loopback(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:9")
    monkeypatch.setenv("http_proxy", "http://10.255.255.1:9")
    body = web_search._safe_http_get(f"http://public.example.test:{port}/page?q=1", timeout=5)
    assert b"hello public world" in body
    assert _Handler.hits == [("/page?q=1", f"public.example.test:{port}")]


def test_redirect_hops_are_revalidated(monkeypatch, local_server) -> None:
    port = local_server.server_address[1]
    _route_public_name_to_loopback(monkeypatch)
    _Handler.redirect_to = f"http://127.0.0.2:{port}/internal"
    with pytest.raises(web_search.FetchBlocked):
        web_search._safe_http_get(f"http://public.example.test:{port}/start", timeout=5)
    assert [path for path, _ in _Handler.hits] == ["/start"]


def test_response_size_is_capped(monkeypatch, local_server) -> None:
    port = local_server.server_address[1]
    _route_public_name_to_loopback(monkeypatch)
    _Handler.body = b"x" * 100_000
    body = web_search._safe_http_get(f"http://public.example.test:{port}/", timeout=5, max_bytes=1_000)
    assert len(body) == 1_000


class _FakeResponse:
    def __init__(self, status: int, location: str | None = None) -> None:
        self.status = status
        self._location = location

    def getheader(self, name: str) -> str | None:
        return self._location if name == "Location" else None

    def read(self, n: int) -> bytes:
        return b"ok"


class _FakeConn:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def request(self, *a, **k) -> None:
        pass

    def getresponse(self) -> _FakeResponse:
        return self._response

    def close(self) -> None:
        pass


@pytest.mark.parametrize("location", ["ftp://93.184.216.34/x", "file:///etc/passwd", "http://10.0.0.1/"])
def test_redirect_to_bad_scheme_or_host_blocked(monkeypatch, location: str) -> None:
    monkeypatch.setattr(web_search, "_open_connection", lambda target, timeout: _FakeConn(_FakeResponse(302, location)))
    with pytest.raises(web_search.FetchBlocked):
        web_search._safe_http_get("http://93.184.216.34/", timeout=2)


def test_redirect_loop_is_capped(monkeypatch) -> None:
    opened: list[str] = []

    def fake_open(target, timeout):
        opened.append(target.path)
        return _FakeConn(_FakeResponse(301, "/again"))

    monkeypatch.setattr(web_search, "_open_connection", fake_open)
    with pytest.raises(web_search.FetchBlocked, match="too many redirects"):
        web_search._safe_http_get("http://93.184.216.34/", timeout=2)
    assert len(opened) == 6  # original request + 5 redirect hops


def test_https_connection_uses_pinned_ip_and_hostname_for_tls(monkeypatch) -> None:
    target = web_search._PinnedTarget("https", "example.com", 443, "93.184.216.34", "/")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(socket, "create_connection", lambda addr, timeout=None: seen.setdefault("addr", addr) and "sock")
    conn = web_search._PinnedHTTPSConnection(target, 3)
    monkeypatch.setattr(conn._ssl_context, "wrap_socket",
                        lambda sock, server_hostname=None: seen.setdefault("sni", server_hostname))
    conn.connect()
    assert seen == {"addr": ("93.184.216.34", 443), "sni": "example.com"}
    assert conn.host == "example.com"


@pytest.mark.parametrize("address", [
    "64:ff9b::7f00:1",        # NAT64 → 127.0.0.1
    "64:ff9b::a9fe:a9fe",     # NAT64 → 169.254.169.254 (cloud metadata)
    "64:ff9b:1::a00:1",       # local-use NAT64 → 10.0.0.1
    "2002:7f00:1::1",         # 6to4 → 127.0.0.1
    "::ffff:10.0.0.1",        # IPv4-mapped private
])
def test_ssrf_blocks_ipv6_translations_of_internal_ipv4(address: str) -> None:
    import ipaddress

    from mark.middlewares.skills.web_search import _is_internal_address

    assert _is_internal_address(ipaddress.ip_address(address))


@pytest.mark.parametrize("address", ["8.8.8.8", "2606:4700:4700::1111"])
def test_ssrf_allows_public_addresses(address: str) -> None:
    import ipaddress

    from mark.middlewares.skills.web_search import _is_internal_address

    assert not _is_internal_address(ipaddress.ip_address(address))
