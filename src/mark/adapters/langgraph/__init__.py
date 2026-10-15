"""LangGraph adapter namespace for MARK.

``MarkState`` is a plain ``TypedDict`` and has no runtime dependency. To use it
inside a LangGraph graph, install LangGraph alongside the adapters, e.g.
``pip install "mark-sdk[adapters]" langgraph`` (or ``"mark-sdk[tutorial]"``,
which bundles LangGraph).
"""

from mark.adapters.langgraph.state import MarkState

__all__ = ["MarkState"]
