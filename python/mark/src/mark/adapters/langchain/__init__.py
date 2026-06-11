"""LangChain adapters for MARK.

Install optional dependencies with ``pip install "mark-sdk[langchain]"``.
"""

from mark.adapters.langchain.middleware import MarkAgentMiddleware, MarkMemoryMiddleware

__all__ = ["MarkAgentMiddleware", "MarkMemoryMiddleware", "create_mark_tools"]


def create_mark_tools(*args, **kwargs):
    """Create LangChain tools backed by a MARK backend."""
    from mark.adapters.langchain.tools import create_mark_tools as _create_mark_tools

    return _create_mark_tools(*args, **kwargs)
