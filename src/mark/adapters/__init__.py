"""Optional framework adapters bundled with mark-sdk.

The adapter namespace is Apache-2.0 framework glue. It must call public MARK APIs and
must not implement private runtime intelligence.
"""

from mark.adapters.backend import BackendResult, LocalMarkBackend, MarkBackend

__all__ = ["BackendResult", "LocalMarkBackend", "MarkBackend"]
