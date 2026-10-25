# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Search backends and result shaping for :class:`WebSearchSkill`.

DuckDuckGo / Wikipedia lookups (blocking calls run in the default executor,
failures degrade to no results), HTML to text extraction for fetched pages, and conversion of raw hits into a
:class:`~mark.middlewares.skills.contracts.SkillResult` whose fragments carry
source attribution. No network safety logic lives here: page fetches go
through ``web_search._safe_http_get``.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

from mark.middlewares.skills.contracts import SkillResult
from mark.middlewares.skills.web_net import _MAX_PAGE_CHARS
from mark.types import MemoryFragment, MemoryScope, MemoryState

try:
    from mark.types import SourceAttribution, SourceCredibility
    _HAS_ATTRIBUTION = True
except ImportError:
    _HAS_ATTRIBUTION = False
    SourceAttribution = None   # type: ignore[assignment,misc]
    SourceCredibility = None   # type: ignore[assignment,misc]


def ddg_text_search(ddgs_cls: Any, query: str, num_results: int) -> List[Dict[str, str]]:
    """Blocking DuckDuckGo text search using the given ``DDGS`` class."""
    with ddgs_cls() as ddgs:
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
                "source": "duckduckgo",
            }
            for r in ddgs.text(query, max_results=num_results)
        ]


def wiki_summary(wikipediaapi: Any, query: str) -> List[Dict[str, str]]:
    """Blocking Wikipedia lookup: the first five sentences of the page summary."""
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


async def ddg_search(query: str, num_results: int) -> List[Dict[str, str]]:
    """DuckDuckGo search off the event loop; ``[]`` when unavailable or failing."""
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        return []

    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, ddg_text_search, DDGS, query, num_results
        )
    except Exception:
        return []


async def wiki_search(query: str) -> List[Dict[str, str]]:
    """Wikipedia lookup off the event loop; ``[]`` when unavailable or failing."""
    try:
        import wikipediaapi  # type: ignore[import]
    except ImportError:
        return []

    try:
        return await asyncio.get_running_loop().run_in_executor(
            None, wiki_summary, wikipediaapi, query
        )
    except Exception:
        return []


def page_text(raw: bytes) -> str:
    """Extract readable text from a fetched page, capped at ``_MAX_PAGE_CHARS``.

    Uses BeautifulSoup (dropping script/style/navigation chrome) when it is
    installed, else a tag-stripping regex.
    """
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


def build_search_result(
    query: str,
    raw: List[Dict[str, str]],
    agent_id: str,
    mode: str,
) -> SkillResult:
    """Turn raw search hits into a :class:`SkillResult` with attributed fragments.

    Web content is stored UNVERIFIED with low importance (Wikipedia slightly
    higher) and tagged ``auto_acquired`` so downstream governance can tell it
    apart from first-party memory.
    """
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
