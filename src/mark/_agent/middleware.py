# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Middleware handling for MarkAgent.

Agents share the runtime-wide middleware stack until they need their own:
``use()`` (or passing ``middleware=`` at construction) copies the inherited,
already-bound middleware into a private stack so other agents and the
runtime's memory operations are never affected.
"""
from __future__ import annotations

from collections.abc import Iterable

from mark._agent.run_types import AgentResult, AgentRun
from mark.middlewares.base import MarkMiddleware, MiddlewareContext, MiddlewareStack


class AgentMiddlewareMixin:
    """``agent.run`` middleware routing; relies on MarkAgent's attributes."""

    _middleware: MiddlewareStack | None
    _shares_runtime_stack: bool
    _runtime: object
    _agent_id: str

    def use(self, *middleware: MarkMiddleware) -> "MarkAgent":
        """Append agent-level middleware and return this agent for chaining.

        Only this agent is affected: when the agent was built on the
        runtime-wide middleware stack, that stack is copied first so other
        agents (and the runtime's own memory operations) are left untouched.
        """
        if self._middleware is None:
            self._middleware = self._new_stack([], self._runtime)
        elif self._shares_runtime_stack:
            self._middleware = self._new_stack(self._middleware.list(), self._runtime)
            self._shares_runtime_stack = False
        self._middleware.extend(middleware)
        return self

    async def _run_with_middleware(self, request: AgentRun) -> AgentResult:
        """Run ``before`` hooks, the agent (unless stopped), then ``after`` hooks.

        A middleware may stop the run and supply a result (an AgentResult or
        anything string-like, which is wrapped), or replace
        ``payload["request"]`` with another AgentRun. ``after`` hooks run in
        reverse order for every middleware whose ``before`` ran, even when
        the run raises.
        """
        if self._middleware is None:
            return await self._execute(request)

        context = MiddlewareContext(
            operation="agent.run",
            runtime=self._runtime,
            agent_id=self._agent_id,
            payload={"request": request, "prompt": request.prompt, "metadata": dict(request.metadata)},
        )
        executed: list[MarkMiddleware] = []
        try:
            for middleware in self._middleware.list():
                middleware.before(context)
                executed.append(middleware)
                if context.stopped:
                    break

            if context.stopped:
                result = context.result
                if isinstance(result, AgentResult):
                    context.result = result
                else:
                    context.result = AgentResult(
                        output="" if result is None else str(result),
                        context=self._empty_context(request.prompt),
                        events=[{"type": "middleware.stop"}],
                        metadata={"agent_id": self._agent_id},
                    )
            else:
                current = context.payload.get("request", request)
                if not isinstance(current, AgentRun):
                    raise TypeError("agent.run middleware payload['request'] must be an AgentRun")
                context.result = await self._execute(current)
        except Exception as exc:
            context.metadata["error"] = repr(exc)
            raise
        finally:
            for middleware in reversed(executed):
                middleware.after(context)

        if not isinstance(context.result, AgentResult):
            raise TypeError("agent.run middleware must return an AgentResult or a string-like stop result")
        return context.result

    def _build_middleware_stack(
        self,
        runtime: object | None,
        middleware: Iterable[MarkMiddleware] | None,
    ) -> MiddlewareStack | None:
        """Pick the stack for a new agent.

        No agent-level middleware: share the runtime stack (or none).
        Otherwise build a private stack: runtime middleware first, then ours.
        """
        runtime_stack = getattr(runtime, "middleware", None)
        if middleware is None and isinstance(runtime_stack, MiddlewareStack):
            return runtime_stack
        if middleware is None:
            return None

        # Runtime middleware is already bound to the runtime; bind only the
        # agent-level additions so no middleware is bound twice.
        inherited = runtime_stack.list() if isinstance(runtime_stack, MiddlewareStack) else []
        stack = self._new_stack(inherited, runtime or self)
        stack.extend(list(middleware))
        return stack

    @staticmethod
    def _new_stack(inherited: list[MarkMiddleware], runtime: object) -> MiddlewareStack:
        """Return a stack bound to *runtime* holding already-bound *inherited* items."""
        return MiddlewareStack.from_bound(inherited, runtime)
