"""Core MCP tools: ``mark_retrieve``, ``mark_observe`` and ``mark_write``.

Descriptions are neutral (what the tool does, when it is useful) rather
than pushy ("call this as your VERY FIRST action"): a tool description is a
prompt, and over-eager ones made small models call memory on every turn
(usage test m2). Every parameter is documented with its bounds.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable

from mark.adapters.backend import call_backend_off_loop as _backend_call
from mark.adapters.mcp._common import (
    ITEM_CHARS,
    MAX_CONTENT,
    MAX_QUERY,
    AgentId,
    bounded,
    clean_error,
    fail,
    render_retrieval,
    require_range,
    require_text,
    short,
)
from mark.kinds import KINDS, kind_from_text, kind_tag

Query = Annotated[str, bounded("What to look for, in plain words or exact identifiers.",
                               minLength=1, maxLength=MAX_QUERY)]
Content = Annotated[str, bounded("The text to remember (one fact, decision or outcome).",
                                 minLength=1, maxLength=MAX_CONTENT)]
Tags = Annotated[list[str] | None, bounded("Optional labels, e.g. ['decision'] or ['kind:todo'].")]
Session = Annotated[str | None, bounded("Optional session id to scope the memory.")]
Importance = Annotated[float, bounded("0.0-1.0; higher ranks first in retrieval.", minimum=0.0, maximum=1.0)]
# A plain string (checked by _kind_tags, so a typo gets a MARK sentence, not
# a schema dump); the choices are still advertised in the JSON schema.
Kind = Annotated[str, bounded("What the memory is: " + ", ".join(KINDS) + ". Empty = from a leading "
                              "label such as 'TODO:' or 'DECISION:', else none.",
                              enum=["", *KINDS], maxLength=20)]
ExtractFacts = Annotated[bool, bounded("Also record facts stated in the text (\"I live in Lisbon\" -> "
                                       "<user> city = Lisbon), as values with history for mark_fact.")]
#: Default reply cap of mark_retrieve, in tokens (characters / 4).
RETRIEVE_BUDGET = 1500


def _kind_tags(tags: list[str] | None, kind: str, content: str) -> list[str]:
    """*tags* plus ``kind:<k>`` from *kind* or, when empty, the content's leading label."""
    from mark.kinds import normalize_kind

    out = list(dict.fromkeys(tags or []))
    if any(str(t).startswith("kind:") for t in out):
        return out
    try:
        chosen = normalize_kind(kind.strip()) if kind and kind.strip() else kind_from_text(content)
    except ValueError as exc:
        fail(str(exc))
    if chosen:
        out.append(kind_tag(chosen))
    return out


async def _extract(backend: Any, agent_id: str, fragment_id: str) -> str:
    """Extract facts from a just-written memory; the reply suffix (" (1 fact extracted)")."""
    from mark.adapters.backend import run_off_loop

    memory_for = getattr(backend, "memory_for", None)
    if memory_for is None or not fragment_id:
        fail("extract_facts needs a local MARK backend (memory_for); this backend has none")
    count = await run_off_loop(lambda: memory_for(agent_id)._extract_stored([fragment_id], True))
    return f" ({count} fact{'' if count == 1 else 's'} extracted)"


def register_core_tools(server: Any, backend: Any, resolve: Callable[[str], str], *,
                        default_blocks: list[str] | None, compress: bool, enabled: set[str]) -> None:
    """Add the enabled core tools to *server*. *resolve* maps agent_id or fails."""

    if "mark_retrieve" in enabled:
        @server.tool()
        async def mark_retrieve(
            query: Query,
            agent_id: AgentId = "",
            max_results: Annotated[int, bounded("Memories to return (1-50).", minimum=1, maximum=50)] = 10,
            budget_tokens: Annotated[int, bounded("Cap on the reply size in tokens (chars/4), 50-8000; "
                                                  "0 = no overall cap.", minimum=0, maximum=8000)] = RETRIEVE_BUDGET,
            max_item_chars: Annotated[int, bounded("Longest text shown per memory (100-4000); a longer "
                                                   "memory shows its best-matching passage.",
                                                   minimum=100, maximum=4000)] = ITEM_CHARS,
            session_id: Session = None,
            session_prefix: Annotated[str | None, bounded("Only sessions whose id starts with this.")] = None,
            tags: Annotated[list[str] | None, bounded("Only memories carrying all these tags.")] = None,
            block_id: Annotated[str | None, bounded("Only this memory block.")] = None,
            block_ids: Annotated[list[str] | None, bounded("Only these memory blocks.")] = None,
        ) -> str:
            """Search memory for notes relevant to a query.

            Returns current facts and the most relevant memories (with date and
            short id), most relevant first, within budget_tokens. Useful before
            answering questions about earlier work, decisions, conventions or the user.
            """
            text = require_text("query", query, MAX_QUERY)
            require_range("max_results", max_results, 1, 50)
            require_range("max_item_chars", max_item_chars, 100, 4000)
            if budget_tokens and not 50 <= budget_tokens <= 8000:
                fail("budget_tokens must be 0 (no cap) or between 50 and 8000")
            result = await _backend_call(
                backend, "retrieve", text, agent_id=resolve(agent_id), session_id=session_id,
                session_prefix=session_prefix, tags=tags, block_id=block_id,
                block_ids=block_ids or default_blocks, compress=compress,
            )
            if not result.ok:
                fail(f"mark_retrieve failed: {clean_error(result.error)}")
            value = result.value
            if hasattr(value, "fragments"):
                return render_retrieval(value, max_results=max_results, budget_tokens=budget_tokens,
                                        max_item_chars=max_item_chars)
            return str(value.as_text() if hasattr(value, "as_text") else value)

    if "mark_observe" in enabled:
        @server.tool()
        async def mark_observe(
            content: Content,
            agent_id: AgentId = "",
            session_id: Session = None,
            tags: Tags = None,
            source: Annotated[str, bounded("Who produced it: user, tool, document, mcp ...")] = "mcp",
            importance: Importance = 0.5,
            extract_facts: ExtractFacts = False,
        ) -> str:
            """Record something that happened (an event, outcome or observation).

            Use for episodic notes such as "Implemented Product CRUD in
            routers/products.py". Use mark_write for reference knowledge.
            """
            text = require_text("content", content, MAX_CONTENT)
            require_range("importance", importance, 0.0, 1.0)
            agent = resolve(agent_id)
            result = await _backend_call(backend, "observe", text, agent_id=agent,
                                         session_id=session_id, tags=_kind_tags(tags, "", text) or None,
                                         source=source, importance=importance)
            if not result.ok:
                fail(f"mark_observe failed: {clean_error(result.error)}")
            fragment_id = result.metadata.get("fragment_id", "")
            facts = await _extract(backend, agent, fragment_id) if extract_facts else ""
            return f"MARK observation stored{f' as {short(fragment_id)}' if fragment_id else ''}{facts}."

    if "mark_write" in enabled:
        @server.tool()
        async def mark_write(
            content: Content,
            agent_id: AgentId = "",
            session_id: Session = None,
            tags: Tags = None,
            source: Annotated[str, bounded("Who produced it: user, tool, document, mcp ...")] = "mcp",
            importance: Importance = 0.8,
            canonical: Annotated[bool, bounded("Mark as authoritative reference (ranks first).")] = False,
            kind: Kind = "",
            extract_facts: ExtractFacts = False,
        ) -> str:
            """Store durable reference knowledge: a convention, decision, TODO, bug or fact.

            Set kind (decision, todo, bug, fix, ...) or start the text with a
            label such as "DECISION:" or "TODO:" so mark_brief and mark_list
            find it. Unlike mark_observe, this is meant to be recalled across
            future sessions. Returns the memory's short id.
            """
            text = require_text("content", content, MAX_CONTENT)
            require_range("importance", importance, 0.0, 1.0)
            write_tags = _kind_tags(tags, kind, text)
            if canonical and "canonical" not in write_tags:
                write_tags.append("canonical")
            agent = resolve(agent_id)
            result = await _backend_call(backend, "write", text, agent_id=agent,
                                         session_id=session_id, tags=write_tags or None, source=source,
                                         importance=0.95 if canonical else importance)
            if not result.ok:
                fail(f"mark_write failed: {clean_error(result.error)}")
            fragment_id = result.metadata.get("fragment_id", "")
            facts = await _extract(backend, agent, fragment_id) if extract_facts else ""
            label = "canonical memory" if canonical else "memory"
            return f"MARK {label} written{f' as {short(fragment_id)}' if fragment_id else ''}{facts}."
