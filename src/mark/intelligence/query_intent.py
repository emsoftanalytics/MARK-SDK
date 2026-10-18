"""What a question wants from retired memories: the current state, or the story.

Memories retired through the API (superseded decisions, resolved bugs and
TODOs) are demoted so that "what is it now?" questions see the live value
first. That demotion is wrong for questions about the past or about a
problem: "tomllib TypeError binary mode", "why did old databases fail?",
"how did we fix the collision bug?" are answered *by* the resolved bug and
its fix, so those rank by relevance like any other memory.

:func:`retirement_applies` encodes that split:

- **no demotion** for history questions (``ORIGINAL`` intent: "originally",
  "before", "previously"), for problem questions (error / exception text,
  "why did/does ...", "fail", "crash", "bug", "broke") and for fix questions
  ("how did we fix", "what fixed", "workaround");
- **demotion** for everything else: explicit current-state questions ("now",
  "current", "still open") and default lookups of a value ("How are short
  codes generated?", "What is the code length?").

A problem question wins over "still": "Why did expired links still resolve?"
asks about a bug, not about the present.
"""
from __future__ import annotations

import re
from functools import cache

from mark.intelligence.query_understanding import TemporalIntent, detect_temporal_intent


@cache
def _problem_re() -> re.Pattern[str]:
    # Compiled on first use: keeps the CLI's cold import cheap.
    return re.compile(
        r"(?-i:\b\w*(?:Error|Exception|Warning|Fault)\b)"  # TypeError, OperationalError (case matters)
        r"|\bTraceback\b|\berrno\b|\bexit\s+code\s+\d"
        r"|\b(?:why|how\s+come)\b"
        r"|\b(?:error|errors|exception|fail\w*|crash\w*|broke\w*|broken|bug\w*|regression|traceback"
        r"|raise[sd]?|threw|throws?|panic\w*|segfault\w*|stack\s*trace|timeouts?|timed\s+out"
        r"|outage|incident)\b"
        r"|\b(?:fix\w*|workaround\w*|patch\w*|resolv\w*|solv\w*|repair\w*)\b(?!\s+(?:is|are)\s+(?:open|pending))"
        r"|['\"`].{0,80}?(?:failed|error|denied|not\s+found|missing|unique|constraint).{0,80}?['\"`]",
        re.I,
    )


@cache
def _open_state_re() -> re.Pattern[str]:
    # "Which bugs are still open?", "open TODOs": a current-state ask about problems.
    return re.compile(r"\b(?:still\s+open|are\s+open|open\s+(?:bugs?|todos?|items?|issues?|tasks?)"
                      r"|unresolved|outstanding|pending)\b", re.I)


def is_problem_query(query: str) -> bool:
    """True when the question is about an error, a failure or how one was fixed."""
    return _problem_re().search(query) is not None and _open_state_re().search(query) is None


def retirement_applies(query: str) -> bool:
    """Whether superseded / resolved memories should be demoted for *query* (see module docstring)."""
    if is_problem_query(query):
        return False
    return detect_temporal_intent(query) != TemporalIntent.ORIGINAL
