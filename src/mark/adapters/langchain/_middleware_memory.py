"""Standalone message-list memory helper (:class:`MarkMemoryMiddleware`).

For code that calls chat models directly rather than through a LangChain
agent graph: recall MARK memory for a query and inject it into a plain
message list, or remember an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.adapters.backend import MarkBackend, call_backend_off_loop
from mark.adapters.langchain._middleware_support import _run_sync


@dataclass
class MarkMemoryMiddleware:
    """
    Lightweight MARK memory helper for message-list augmentation.

    Use when you are *not* using a full LangChain agent graph — e.g. when
    calling a chat model directly.  For agent-graph integration use
    :class:`MarkAgentMiddleware` instead.

    Example::

        mw = MarkMemoryMiddleware(backend=backend, agent_id="__mark__")
        messages = [{"role": "user", "content": "Add a health check."}]
        augmented = mw.augment_messages(messages, "health check")
        response  = llm.invoke(augmented)
        mw.remember_sync(f"Response: {response.content}")
    """

    backend: MarkBackend
    agent_id: str
    session_id: str | None = None
    max_context_chars: int = 6000
    write_outcomes: bool = False

    # ── async ────────────────────────────────────────────────────────────

    async def recall(self, query: str) -> str:
        """Retrieve and format memory context for the query (``""`` if none)."""
        result = await call_backend_off_loop(
            self.backend, "retrieve", query,
            agent_id=self.agent_id, session_id=self.session_id,
        )
        if not result.ok:
            return ""
        # Resolved at call time: the facade module imports this one.
        from mark.adapters.langchain.middleware import MarkAgentMiddleware

        text, _ = MarkAgentMiddleware._render(result.value)
        return text[: self.max_context_chars]

    async def remember(self, content: str, **meta: Any) -> None:
        """Store content into MARK memory."""
        if not self.write_outcomes:
            return
        await call_backend_off_loop(
            self.backend, "observe", content, agent_id=self.agent_id, metadata=meta
        )

    # ── sync ─────────────────────────────────────────────────────────────

    def recall_sync(self, query: str) -> str:
        """Synchronous variant of recall()."""
        return _run_sync(self.recall(query))

    def remember_sync(self, content: str, **meta: Any) -> None:
        """Synchronous variant of remember()."""
        _run_sync(self.remember(content, **meta))

    # ── message augmentation ─────────────────────────────────────────────

    def augment_messages(
        self,
        messages: list[Any],
        query: str,
        *,
        inject_as: str = "system",
    ) -> list[Any]:
        """Return a new message list with MARK context prepended.

        Retrieves memory for *query* and injects it as a system message
        (default) or into the first human message (``inject_as="human"``).
        If memory is empty the original list is returned unchanged.
        """
        context = self.recall_sync(query)
        if not context:
            return list(messages)

        block = f"[MARK project memory]\n{context}\n[/MARK project memory]"
        out: list[Any] = []
        injected = False

        for msg in messages:
            role = (msg.get("role") if isinstance(msg, dict)
                    else getattr(msg, "type", getattr(msg, "role", None)))
            if role in ("system", "SystemMessage") and not injected:
                existing = (msg.get("content") if isinstance(msg, dict)
                            else getattr(msg, "content", "")) or ""
                new_content = f"{existing}\n\n{block}".lstrip()
                out.append({**msg, "content": new_content} if isinstance(msg, dict)
                           else type(msg)(content=new_content))
                injected = True
            else:
                out.append(msg)

        if not injected:
            if inject_as == "system":
                out.insert(0, {"role": "system", "content": block})
            else:
                for i, msg in enumerate(out):
                    role = (msg.get("role") if isinstance(msg, dict)
                            else getattr(msg, "type", getattr(msg, "role", None)))
                    if role in ("user", "human", "HumanMessage"):
                        existing = (msg.get("content") if isinstance(msg, dict)
                                    else getattr(msg, "content", "")) or ""
                        new_content = f"{block}\n\n{existing}"
                        out[i] = ({**msg, "content": new_content} if isinstance(msg, dict)
                                  else type(msg)(content=new_content))
                        break

        return out
