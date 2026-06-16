"""Compression middleware."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mark.middlewares.base import BaseMiddleware, MiddlewareContext


@dataclass
class CompressionMiddleware(BaseMiddleware):
    """Configure a runtime compressor and optionally enable compression."""

    compressor: Any
    enabled_by_default: bool = True
    name: str = "compression"

    def bind(self, runtime: Any) -> None:
        if hasattr(runtime, "configure_compressor"):
            runtime.configure_compressor(self.compressor)

    def before(self, context: MiddlewareContext) -> None:
        if self.enabled_by_default and context.operation == "retrieve":
            context.payload["compress"] = True


__all__ = ["CompressionMiddleware"]
