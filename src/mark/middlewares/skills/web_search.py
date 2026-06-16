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
import ipaddress
import re
import socket
from typing import Any, Dict, List
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from mark.types import MemoryFragment, MemoryScope, MemoryState

from .base import Skill, SkillManifest, SkillResult

try:
    from mark.types import SourceAttribution, SourceCredibility
    _HAS_ATTRIBUTION = True
except ImportError:
    _HAS_ATTRIBUTION = False
    SourceAttribution = None   # type: ignore[assignment,misc]
    SourceCredibility = None   # type: ignore[assignment,misc]

_USER_AGENT = "Mozilla/5.0 (compatible; MARK/1.0)"
_TIMEOUT_MS = 15_000
_MAX_PAGE_CHARS = 8_000


def _is_internal_address(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _validate_fetch_url(url: str) -> str | None:
    """Return an error string when a fetch URL is unsafe, else None.

    Only http/https are fetchable — file://, ftp://, and friends would let a
    crafted observation read local files into agent memory. Loopback, private,
    and link-local addresses are rejected to keep gap healing from probing
    internal services.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return f"Invalid URL: {url!r}"
    if parsed.scheme not in {"http", "https"}:
        return f"Only http/https URLs can be fetched (got scheme {parsed.scheme!r})"
    host = parsed.hostname or ""
    if not host:
        return "URL has no host"
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return "Refusing to fetch loopback/internal hosts"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return None
        for info in infos:
            resolved_host = info[4][0]
            try:
                resolved = ipaddress.ip_address(resolved_host)
            except ValueError:
                return "Could not validate resolved host address"
            if _is_internal_address(resolved):
                return "Refusing to fetch hosts that resolve to private/loopback/link-local addresses"
        return None
    if _is_internal_address(address):
        return "Refusing to fetch private/loopback/link-local addresses"
    return None


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
        if _validate_fetch_url(url):
            return ""

        def _sync() -> str:
            request = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(request, timeout=max(1, self.timeout_ms / 1000)) as response:
                raw = response.read(_MAX_PAGE_CHARS * 4)
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
        except Exception:
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
