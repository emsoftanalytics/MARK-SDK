"""Pluggable middleware for MARK runtime operations."""

from mark.middlewares.base import (
    BaseMiddleware,
    MarkMiddleware,
    MiddlewareContext,
    MiddlewareStack,
)

_LAZY_EXPORTS = {
    "CompressionMiddleware": "mark.middlewares.compression",
    "GapHealingMiddleware": "mark.middlewares.gap_healing",
    "GovernanceMiddleware": "mark.middlewares.governance",
    "LifecycleMiddleware": "mark.middlewares.lifecycle",
    "MediaContinuityMiddleware": "mark.middlewares.media_continuity",
    "ObservabilityMiddleware": "mark.middlewares.observability",
    "ObserveMiddleware": "mark.middlewares.observe",
    "QueryExpansionMiddleware": "mark.middlewares.query_expansion",
    "RecallMiddleware": "mark.middlewares.recall",
    "SandboxMiddleware": "mark.middlewares.sandbox",
    "SkillMiddleware": "mark.middlewares.skills",
    "SyncMiddleware": "mark.middlewares.sync",
    "TrustBusMiddleware": "mark.middlewares.trust_bus",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'mark.middlewares' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "BaseMiddleware",
    "CompressionMiddleware",
    "GapHealingMiddleware",
    "GovernanceMiddleware",
    "LifecycleMiddleware",
    "MarkMiddleware",
    "MediaContinuityMiddleware",
    "MiddlewareContext",
    "MiddlewareStack",
    "ObservabilityMiddleware",
    "ObserveMiddleware",
    "QueryExpansionMiddleware",
    "RecallMiddleware",
    "SandboxMiddleware",
    "SkillMiddleware",
    "SyncMiddleware",
    "TrustBusMiddleware",
]
