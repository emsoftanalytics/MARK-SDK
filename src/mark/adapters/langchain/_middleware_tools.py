"""Tool call wrapper mixin for :class:`MarkAgentMiddleware`.

``(a)wrap_tool_call`` passes tool results through unchanged and queues (or,
opt-in, immediately writes) one observation per result:

* a normal result is stored as ``source="tool"`` + ``mark:tool``;
* a failed call — a ``ToolMessage`` with ``status="error"`` or an exception
  raised by the tool — is stored with ``mark:tool`` + ``mark:error`` and low
  trust. Errors are evidence that something went wrong, not facts; they are
  never shown as notes and never fed to the ContextWatch (a10 finding N3).

The result's :func:`~mark.context.history.message_key` travels with the
queued observation, so history archival (``history="mark"``) and the run-end
flush never store the same ToolMessage twice.
"""
from __future__ import annotations

from typing import Any, Callable

from mark.adapters.langchain._middleware_provenance import is_tool_error, tool_observe_options
from mark.adapters.langchain._middleware_support import _GraphBubbleUp, _run_sync, content_text
from mark.context.history import message_key

#: Option key carrying the message key through the observation queue.
KEY_OPTION = "_mark_message_key"
_SKIP_IMMEDIATE = ("write", "delete", "remove")


class _ToolWrapperMixin:
    """Tool-result observation; expects the run/queue/archive helpers of the host."""

    def _called_tool_name(self, request: Any) -> str:
        tool_call = getattr(request, "tool_call", None) or {}
        return str(tool_call.get("name", "unknown_tool"))

    def _observe_tool_result(self, request: Any, result: Any) -> Any:
        """Queue (or schedule) observation of a tool result.

        Returns a coroutine for immediate observation, else ``None``.
        """
        if not (self._write_outcomes and self._observe_tool_results):
            return None
        tool_name = self._called_tool_name(request)
        error = is_tool_error(result)
        content = content_text(getattr(result, "content", "") or "")[:900]
        if not content or (len(content) < 20 and not error):
            return None
        key = message_key(result) if getattr(result, "tool_call_id", None) else None
        if key is not None and self._is_archived(key):
            return None                      # history archival already stored it
        verb = "failed" if error else "returned"
        observation = f"Tool [{tool_name}] {verb}: {content}"
        options = tool_observe_options(error=error)
        run = self._run(getattr(request, "state", None))
        if self._immediate_tool_observe and not any(s in tool_name.lower() for s in _SKIP_IMMEDIATE):
            self._record("observe_immediate", tool=tool_name, chars=len(content))
            return self._observe_keyed(observation, key, options)
        self._queue_observation(run, observation, **options, **({KEY_OPTION: key} if key else {}))
        return None

    async def _observe_keyed(self, content: str, key: str | None, options: dict[str, Any]) -> Any:
        """Write one observation now and register its message key on success."""
        result = await self._backend_call("observe", content, agent_id=self._agent_id, **options)
        if result.ok and key is not None:
            self._mark_archived(key)
        return result

    def _tool_exception_observation(self, request: Any, exc: BaseException) -> Any:
        """Coroutine storing a raised tool exception (or ``None`` when not stored).

        Written immediately: the run's state is released right after, so a
        queued observation would be lost. Interrupts are control flow, not errors.
        """
        if (isinstance(exc, _GraphBubbleUp) or not isinstance(exc, Exception)
                or not (self._write_outcomes and self._observe_tool_results)):
            return None
        text = " ".join(f"{type(exc).__name__}: {exc}".split())[:900]
        return self._observe_keyed(f"Tool [{self._called_tool_name(request)}] failed: {text}", None,
                                   tool_observe_options(error=True))

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Sync: execute the tool and queue its result for run-end archival."""
        try:
            result = handler(request)
        except BaseException as exc:
            self._store_tool_exception_sync(request, exc)
            self._on_run_error(getattr(request, "state", None), exc)
            raise
        try:
            pending = self._observe_tool_result(request, result)
            if pending is not None:
                _run_sync(pending)
        except Exception as exc:
            self._hook_failed("wrap_tool_call", exc)
        return result

    def _store_tool_exception_sync(self, request: Any, exc: BaseException) -> None:
        try:
            pending = self._tool_exception_observation(request, exc)
            if pending is not None:
                _run_sync(pending)
        except Exception as error:
            self._hook_failed("wrap_tool_call", error)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """Async: execute the tool and archive its result into MARK.

        The result is returned unchanged — MARK is completely transparent.
        With ``immediate_tool_observe=True`` the observation runs as a tracked
        background task; otherwise it is queued until the run ends.
        """
        try:
            result = await handler(request)
        except BaseException as exc:
            try:
                pending = self._tool_exception_observation(request, exc)
                if pending is not None:
                    await pending
            except Exception as error:
                self._hook_failed("wrap_tool_call", error)
            self._on_run_error(getattr(request, "state", None), exc)
            raise
        try:
            pending = self._observe_tool_result(request, result)
            if pending is not None:
                await self._spawn(self._run(getattr(request, "state", None)), pending)
        except Exception as exc:
            self._hook_failed("wrap_tool_call", exc)
        return result
