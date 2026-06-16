"""Sandbox middleware for explicit local execution operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class SandboxMiddleware(BaseMiddleware):
    """Local sandbox execution wrapper for explicit sandbox operations."""

    sandbox: Any = None
    enabled: bool = False
    name: str = "sandbox"
    last_result: Any = None

    def bind(self, runtime: Any) -> None:
        if self.sandbox is None:
            try:
                from mark.middlewares.sandbox import DockerSandbox

                self.sandbox = DockerSandbox()
            except Exception:
                self.sandbox = None

    def before(self, context: MiddlewareContext) -> None:
        if context.operation != "sandbox_execute":
            return
        if not self.enabled or self.sandbox is None:
            context.stop(
                {
                    "ok": False,
                    "sandbox_mode": "blocked/local-disabled",
                    "stderr": "Local sandbox middleware is disabled.",
                }
            )
            return

        code = str(context.payload.get("code") or "")
        timeout = context.payload.get("timeout")
        workspace = context.payload.get("workspace")
        env = context.payload.get("env")
        self.last_result = self.sandbox.execute(
            code,
            timeout=timeout,
            workspace=workspace,
            env=env,
        )
        context.stop(self.last_result)


__all__ = ["SandboxMiddleware"]
