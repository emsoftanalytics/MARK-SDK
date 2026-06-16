"""Query expansion middleware."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class QueryExpansionMiddleware(BaseMiddleware):
    """Configure a runtime query expander and optionally enable expansion."""

    expander: Any
    enabled_by_default: bool = True
    name: str = "query_expansion"

    def bind(self, runtime: Any) -> None:
        if hasattr(runtime, "configure_query_expander"):
            runtime.configure_query_expander(self.expander)

    def before(self, context: MiddlewareContext) -> None:
        if self.enabled_by_default and context.operation == "retrieve":
            context.payload["expand"] = True


__all__ = ["QueryExpansionMiddleware"]
