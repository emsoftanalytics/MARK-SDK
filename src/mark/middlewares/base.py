"""Framework-neutral middleware primitives for MARK runtime operations."""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareContext:
    """Mutable operation envelope passed through MARK middleware."""

    operation: str
    runtime: Any
    agent_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stopped: bool = False
    error: BaseException | None = None

    @property
    def rejected(self) -> bool:
        """Return True when a governance gate rejected this write operation."""
        governance = (self.payload.get("metadata") or {}).get("mark_governance")
        if isinstance(governance, dict) and governance.get("passed") is False:
            return True
        state = self.payload.get("state")
        return getattr(state, "value", state) == "quarantined"

    @property
    def succeeded(self) -> bool:
        """Return True when the handler ran to completion and was not rejected."""
        return not self.stopped and self.error is None and not self.rejected

    def stop(self, result: Any = None) -> None:
        """Stop the operation and use result as the final value."""
        self.result = result
        self.stopped = True


class MarkMiddleware(Protocol):
    """Protocol implemented by core and adapter middleware."""

    name: str

    def bind(self, runtime: Any) -> None:
        """Optionally configure a runtime when the middleware is registered."""

    def before(self, context: MiddlewareContext) -> None:
        """Run before the operation handler."""

    def after(self, context: MiddlewareContext) -> None:
        """Run after the operation handler."""


class BaseMiddleware:
    """No-op base class for middleware implementations."""

    name = "base"

    def bind(self, runtime: Any) -> None:
        """Configure a runtime when the middleware is registered."""

    def before(self, context: MiddlewareContext) -> None:
        """Run before the operation handler."""

    def after(self, context: MiddlewareContext) -> None:
        """Run after the operation handler."""


class MiddlewareStack:
    """Ordered middleware stack for local MARK runtime operations."""

    def __init__(self, middleware: Iterable[MarkMiddleware] | None = None) -> None:
        self._middleware: list[MarkMiddleware] = list(middleware or [])
        self._runtime: Any = None

    def bind(self, runtime: Any) -> None:
        """Bind all middleware to a runtime."""
        self._runtime = runtime
        for middleware in self._middleware:
            middleware.bind(runtime)

    def add(self, middleware: MarkMiddleware) -> MarkMiddleware:
        """Append middleware and bind it when the stack already has a runtime."""
        self._middleware.append(middleware)
        if self._runtime is not None:
            middleware.bind(self._runtime)
        return middleware

    def extend(self, middleware: Iterable[MarkMiddleware]) -> None:
        """Append a sequence of middleware."""
        for item in middleware:
            self.add(item)

    def list(self) -> list[MarkMiddleware]:
        """Return middleware in execution order."""
        return list(self._middleware)

    @classmethod
    def from_bound(cls, middleware: Iterable[MarkMiddleware], runtime: Any) -> "MiddlewareStack":
        """Return a stack for *runtime* holding middleware that is already bound.

        Use this to derive an independent stack from an existing one (e.g. a
        per-agent copy of the runtime stack) without re-binding its items.
        """
        stack = cls()
        stack._runtime = runtime
        stack._middleware = list(middleware)
        return stack

    def run(
        self,
        operation: str,
        *,
        runtime: Any,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
        handler: Callable[[MiddlewareContext], Any],
    ) -> Any:
        """Run middleware around an operation handler."""
        context = MiddlewareContext(
            operation=operation,
            runtime=runtime,
            agent_id=agent_id,
            payload=dict(payload or {}),
        )

        executed: list[MarkMiddleware] = []
        try:
            for middleware in self._middleware:
                middleware.before(context)
                executed.append(middleware)
                if context.stopped:
                    break

            if not context.stopped:
                context.result = handler(context)
        except BaseException as exc:
            context.error = exc

        # After-hooks of every middleware whose before-hook completed always
        # run, even when a later middleware or the handler raised. Hooks can
        # inspect ``context.error`` / ``context.succeeded`` to skip side effects.
        for middleware in reversed(executed):
            try:
                middleware.after(context)
            except Exception as after_exc:
                if context.error is None:
                    context.error = after_exc
                else:
                    logger.warning(
                        "middleware %r after-hook failed while unwinding %s: %s: %s",
                        getattr(middleware, "name", middleware),
                        type(context.error).__name__,
                        type(after_exc).__name__,
                        after_exc,
                    )

        if context.error is not None:
            raise context.error
        return context.result


__all__ = [
    "BaseMiddleware",
    "MarkMiddleware",
    "MiddlewareContext",
    "MiddlewareStack",
]
