"""Optional framework adapters bundled with mark-sdk.

The adapter namespace is MIT framework glue. It must call public MARK APIs and
must not implement proprietary MARK Cloud intelligence.
"""

from mark.adapters.backend import BackendResult, LocalMarkBackend, MarkBackend

__all__ = ["BackendResult", "LocalMarkBackend", "MarkBackend"]
