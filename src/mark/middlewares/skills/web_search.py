# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# WebSearchSkill - lightweight DuckDuckGo, Wikipedia, and basic page fetch.
#
# Local MARK intentionally does not launch Playwright. LLM-guided browser lookup,
# source credibility scoring, and automatic self-healing belong in registered extensions.
"""Lightweight explicit web lookup skill (DuckDuckGo, Wikipedia, fetch)."""
from __future__ import annotations

import asyncio
import http.client
import ipaddress
import logging
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

from mark.types import MemoryFragment, MemoryScope, MemoryState

from .base import Skill, SkillManifest, SkillResult

try:
    from mark.types import SourceAttribution, SourceCredibility
    _HAS_ATTRIBUTION = True
except ImportError:
    _HAS_ATTRIBUTION = False
    SourceAttribution = None   # type: ignore[assignment,misc]
    SourceCredibility = None   # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

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
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return (not address.is_global) or address.is_multicast


def _pin_url(url: str) -> tuple[_PinnedTarget | None, str | None]:
    """Validate *url*, resolve it once, and return the pinned target or an error.

    Fails closed: resolution errors, unparseable addresses, and any resolved
    address that is not globally routable reject the URL.
    """
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return None, f"Invalid URL: {url!r}"
    if parsed.scheme not in {"http", "https"}:
        return None, f"Only http/https URLs can be fetched (got scheme {parsed.scheme!r})"
    host = parsed.hostname or ""
    if not host:
        return None, "URL has no host"
    if parsed.username is not None or parsed.password is not None:
        return None, "URLs with embedded credentials are not fetched"
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost") or lowered.endswith(".local"):
        return None, "Refusing to fetch loopback/internal hosts"
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_internal_address(literal):
            return None, "Refusing to fetch private/loopback/link-local addresses"
        return _PinnedTarget(parsed.scheme, host, port, str(literal), path), None

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        return None, f"Could not resolve host {host!r}: {type(exc).__name__}"
    if not infos:
        return None, f"Could not resolve host {host!r}"
    pinned: str | None = None
    for info in infos:
        try:
            resolved = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        except (ValueError, IndexError, TypeError):
            return None, "Could not validate resolved host address"
        if _is_internal_address(resolved):
            return None, "Refusing to fetch hosts that resolve to private/loopback/link-local addresses"
        if pinned is None:
            pinned = str(resolved)
    assert pinned is not None
    return _PinnedTarget(parsed.scheme, host, port, pinned, path), None


def _validate_fetch_url(url: str) -> str | None:
    """Return an error string when a fetch URL is unsafe, else None.

    Only http/https are fetchable — file://, ftp://, and friends would let a
    crafted observation read local files into agent memory. Hosts must
    resolve, and every resolved address must be globally routable, to keep
    gap healing from probing internal services.
    """
    return _pin_url(url)[1]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection to a pre-validated IP; ``Host`` still carries the hostname."""

    def __init__(self, target: _PinnedTarget, timeout: float) -> None:
        super().__init__(target.host, target.port, timeout=timeout)
        self._pinned_ip = target.ip

    def connect(self) -> None:
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
    if target.scheme == "https":
        return _PinnedHTTPSConnection(target, timeout)
    return _PinnedHTTPConnection(target, timeout)


def _safe_http_get(
    url: str,
    *,
    timeout: float,
    max_bytes: int = _MAX_RESPONSE_BYTES,
    max_redirects: int = _MAX_REDIRECTS,
) -> bytes:
    """GET *url* with SSRF protection.

    Every hop (including redirects) is re-validated and connected to its
    pinned IP. Proxy environment variables are never consulted. The body is
    capped at *max_bytes* and the whole exchange at *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    current = url
    for _hop in range(max_redirects + 1):
        target, error = _pin_url(current)
        if error or target is None:
            raise FetchBlocked(error or "blocked")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fetch deadline exceeded")
        conn = _open_connection(target, remaining)
        try:
            conn.request(
                "GET",
                target.path,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = conn.getresponse()
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not location:
                    raise FetchBlocked("redirect without Location header")
                current = urljoin(current, location.strip())
                continue
            if response.status >= 400:
                raise FetchBlocked(f"HTTP {response.status}")
            return response.read(max_bytes)
        finally:
            conn.close()
    raise FetchBlocked(f"too many redirects (> {max_redirects})")


class WebSearchSkill(Skill):
    """
    Lightweight web lookup for explicit local gap healing.

    Modes:
      search            DuckDuckGo text search
      wiki              Wikipedia summary
      fetch             Basic page extraction from `url`
      search_and_fetch  Search, then fetch the top result

    Registered extensions can add LLM-guided browser lookup, source credibility
    scoring, result caching, and governed self-healing.
    """

    name = "web_search"
    description = (
        "Search the web using lightweight DuckDuckGo, Wikipedia, and basic fetch "
        "modes. Use only after MARK misses current-session and agent-namespace "
        "memory. All results carry source attribution when attribution types are "
        "available."
    )
    permissions = {"web"}

    def __init__(
        self,
        num_results: int = 5,
        agent_id: str = "agent",
        timeout_ms: int = _TIMEOUT_MS,
        **_: Any,
    ) -> None:
        self.num_results = num_results
        self.agent_id = agent_id
        self.timeout_ms = timeout_ms

    @classmethod
    def manifest(cls) -> SkillManifest:
        """Return this skill's manifest."""
        return SkillManifest(
            name=cls.name,
            description=cls.description,
            version="1.0.0",
            author="mark-contributors",
            permissions=list(cls.permissions),
            tags=["search", "retrieval", "external", "web", "wikipedia"],
            install_hint='pip install "mark-sdk[web]"',
        )

    async def run(self, input: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute with the given input and context."""
        mode = input.get("mode", "search")
        agent_id = context.get("agent_id", self.agent_id)

        if mode == "fetch":
            url = input.get("url", "")
            if not url:
                return SkillResult(content="", success=False, error="url is required")
            scheme_error = _validate_fetch_url(url)
            if scheme_error:
                return SkillResult(content="", success=False, error=scheme_error)
            text = await self._fetch_page(url)
            return SkillResult(
                content=text or f"Could not fetch content from {url}",
                success=bool(text),
                metadata={"mode": "fetch", "url": url, "agent_id": agent_id},
            )

        query = input.get("query", "")
        if not query:
            return SkillResult(content="", success=False, error="query is required")

        try:
            raw = await self._dispatch(mode, query)
        except Exception as exc:
            return SkillResult(content="", success=False, error=f"Web search failed: {exc}")

        if mode == "search_and_fetch" and raw:
            top_url = raw[0].get("url", "")
            if top_url:
                full = await self._fetch_page(top_url)
                if full:
                    raw[0]["snippet"] = full[:2000]

        return self._build_result(query, raw, agent_id, mode)

    async def _dispatch(self, mode: str, query: str) -> List[Dict[str, str]]:
        if mode == "wiki":
            return await self._wiki_search(query)
        if mode in {"search", "search_and_fetch"}:
            return await self._ddg_search(query)
        return []

    async def _ddg_search(self, query: str) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import]
        except ImportError:
            return []

        def _sync() -> List[Dict[str, str]]:
            with DDGS() as ddgs:
                return [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                        "source": "duckduckgo",
                    }
                    for r in ddgs.text(query, max_results=self.num_results)
                ]

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _sync)
        except Exception:
            return []

    async def _wiki_search(self, query: str) -> List[Dict[str, str]]:
        try:
            import wikipediaapi  # type: ignore[import]
        except ImportError:
            return []

        def _sync() -> List[Dict[str, str]]:
            wiki = wikipediaapi.Wikipedia(language="en", user_agent="MARK/1.0")
            page = wiki.page(query)
            if not page.exists():
                return []
            snippet = ". ".join(page.summary.split(". ")[:5])
            return [
                {
                    "title": page.title,
                    "url": page.fullurl,
                    "snippet": snippet,
                    "source": "wikipedia",
                }
            ]

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _sync)
        except Exception:
            return []

    async def _fetch_page(self, url: str) -> str:
        # Validation (and DNS pinning) happens inside _safe_http_get for every hop.
        def _sync() -> str:
            raw = _safe_http_get(url, timeout=max(1.0, self.timeout_ms / 1000))
            html = raw.decode("utf-8", errors="ignore")
            try:
                from bs4 import BeautifulSoup  # type: ignore[import]

                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s{2,}", " ", text)[:_MAX_PAGE_CHARS]

        try:
            return await asyncio.get_running_loop().run_in_executor(None, _sync)
        except FetchBlocked as exc:
            logger.warning("web fetch blocked for %s: %s", url, exc)
            return ""
        except Exception as exc:
            logger.warning("web fetch failed for %s: %s: %s", url, type(exc).__name__, exc)
            return ""

    def _build_result(
        self,
        query: str,
        raw: List[Dict[str, str]],
        agent_id: str,
        mode: str,
    ) -> SkillResult:
        attributions: List[Any] = []
        fragments: List[Any] = []
        lines = [f"Web search results for: {query!r} [mode={mode}]"]

        for item in raw:
            url = item.get("url", "")
            title = item.get("title", "Result")
            snippet = item.get("snippet", "")
            source = item.get("source", "web")

            if _HAS_ATTRIBUTION:
                credibility = (
                    SourceCredibility.VERIFIED  # type: ignore[union-attr]
                    if source == "wikipedia"
                    else SourceCredibility.WEB  # type: ignore[union-attr]
                )
                attr: Any = SourceAttribution(  # type: ignore[misc]
                    source_url=url,
                    source_title=title,
                    snippet=snippet,
                    credibility=credibility,
                    via=f"web_search:{source}",
                )
            else:
                attr = {"url": url, "title": title, "snippet": snippet, "source": source}

            attributions.append(attr)
            lines.append(f"- [{source}] {title}: {snippet} ({url})")
            frag_kwargs: Dict[str, Any] = {
                "content": f"{title}: {snippet}",
                "agent_id": agent_id,
                "scope": MemoryScope.AGENT,
                "importance": 0.6 if source == "wikipedia" else 0.4,
                "state": MemoryState.UNVERIFIED,
                "source": url,
                "tags": ["web_search", f"source:{source}", "auto_acquired"],
            }
            if _HAS_ATTRIBUTION:
                frag_kwargs["attribution"] = attr
            fragments.append(MemoryFragment(**frag_kwargs))

        return SkillResult(
            content="\n".join(lines) if raw else f"No results for {query!r}",
            attributions=attributions,
            fragments=fragments,
            success=True,
            metadata={
                "query": query,
                "mode": mode,
                "result_count": len(raw),
                "agent_id": agent_id,
            },
        )
