# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""SSRF primitives for :mod:`mark.middlewares.skills.web_search`.

Address classification (with IPv4-mapped/NAT64/6to4/Teredo unwrapping), the
pinned fetch target, and HTTP(S) connections that dial the pre-validated IP
while keeping the hostname for ``Host``, SNI and certificate checks.

``web_search._pin_url`` and ``web_search._safe_http_get`` resolve
``_is_internal_address`` and ``_open_connection`` through the ``web_search``
namespace (where they are re-exported), so monkeypatching them on
``web_search`` keeps taking effect.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass

_USER_AGENT = "Mozilla/5.0 (compatible; MARK/1.0)"
_TIMEOUT_MS = 15_000
_MAX_PAGE_CHARS = 8_000
_MAX_RESPONSE_BYTES = _MAX_PAGE_CHARS * 4
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class FetchBlocked(Exception):
    """Raised when a fetch (or one of its redirect hops) fails URL validation."""


@dataclass(frozen=True)
class _PinnedTarget:
    """A validated fetch target: the hostname plus the single IP it is pinned to."""
    scheme: str
    host: str
    port: int
    ip: str
    path: str


def _is_internal_address(address: ipaddress._BaseAddress) -> bool:
    """Anything that is not globally routable is internal.

    ``is_global`` excludes loopback, RFC1918 private ranges, link-local,
    CGNAT (100.64.0.0/10), reserved, documentation, and unspecified ranges;
    multicast is checked explicitly. IPv4-mapped IPv6 addresses are unwrapped.
    """
    for embedded in _embedded_ipv4(address):
        if _is_internal_address(embedded):
            return True
    return (not address.is_global) or address.is_multicast or address.is_reserved


# Translation prefixes whose IPv6 addresses carry an IPv4 destination in
# their low 32 bits; Python reports some of them as globally routable.
_NAT64_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),    # RFC 6052 well-known prefix
    ipaddress.ip_network("64:ff9b:1::/48"),  # RFC 8215 local-use prefix
)


def _embedded_ipv4(address: ipaddress._BaseAddress) -> list[ipaddress.IPv4Address]:
    """IPv4 addresses an IPv6 address tunnels to (mapped, NAT64, 6to4, Teredo).

    Python reports several of these translation ranges as globally routable,
    so the embedded IPv4 destination must be checked as well.
    """
    if not isinstance(address, ipaddress.IPv6Address):
        return []
    found: list[ipaddress.IPv4Address] = []
    if address.ipv4_mapped is not None:
        found.append(address.ipv4_mapped)
    if any(address in network for network in _NAT64_NETWORKS):
        found.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    if address.sixtofour is not None:
        found.append(address.sixtofour)
    if address.teredo is not None:
        found.extend(address.teredo)
    return found


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection to a pre-validated IP; ``Host`` still carries the hostname."""

    def __init__(self, target: _PinnedTarget, timeout: float) -> None:
        super().__init__(target.host, target.port, timeout=timeout)
        self._pinned_ip = target.ip

    def connect(self) -> None:
        # Dial the validated IP instead of re-resolving the hostname, so DNS
        # rebinding between validation and connect cannot reach an internal host.
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pre-validated IP; SNI and certificate checks use the hostname."""

    def __init__(self, target: _PinnedTarget, timeout: float) -> None:
        context = ssl.create_default_context()
        super().__init__(target.host, target.port, timeout=timeout, context=context)
        self._pinned_ip = target.ip
        self._ssl_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


def _open_connection(target: _PinnedTarget, timeout: float) -> http.client.HTTPConnection:
    """Open a connection pinned to ``target.ip`` (HTTPS when the scheme says so)."""
    if target.scheme == "https":
        return _PinnedHTTPSConnection(target, timeout)
    return _PinnedHTTPConnection(target, timeout)
