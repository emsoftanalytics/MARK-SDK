"""Briefing presets by model size: how much memory a model can actually use.

Measured with real local models on the 25-probe personal-assistant scenario
(history/usage-test-0.2.0a9/small, re-run for 0.2.0a10): bigger budgets do
not help a small model, they bury the one line it needs. The numbers below
are the values those runs chose (see the a10 context report).

=========  =====================  =======  ================================
preset     models                 budget   shape
=========  =====================  =======  ================================
default    7B+ / hosted           400*     every section, no assistant lines
small      ~1-3B (llama3.2:1b-3b) 200      dense: 4 request lines, 1 per
                                           subject, no assistant lines
tiny       <=1B (qwen3:0.6b)      160      facts and the best 3 lines only
=========  =====================  =======  ================================

``*`` the default ContextWatch has no token budget (``budget_chars`` 1800);
MarkChat uses 400 tokens.

No preset adds a skill. The compact skill (``mark-memory-compact``, ~170
tokens) is shipped for hosts that want one, but on llama3.2:1b it lowered
accuracy from 18/25 to 10/25 even with the full 200-token memory beside it
(5/25 when charged to the same 200 tokens): the briefing headers carry the
guidance instead. A host that adds a skill counts its tokens inside the same
budget (:func:`split_budget`), so "budget 300" means 300 tokens of MARK text
in the prompt in total, never 300 plus a 750-token skill (M7).
"""
from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "small": {"budget_chars": 1000, "max_anchors": 3, "per_anchor": 1, "line_chars": 160,
              "budget_tokens": 200, "relevant_lines": 4, "assistant_lines": 0,
              "change_lines": 5, "follow_links": 1, "outdated": "drop", "relevance_floor": 0.85,
              "fact_dates": False, "ttl_turns": 0, "profile_facts": 8},
    "tiny": {"budget_chars": 600, "max_anchors": 2, "per_anchor": 1, "line_chars": 120,
             "budget_tokens": 160, "relevant_lines": 3, "assistant_lines": 0,
             "change_lines": 3, "follow_links": 0, "outdated": "drop", "relevance_floor": 0.85,
             "fact_dates": False, "ttl_turns": 0, "profile_facts": 4},
}

def preset_settings(name: str | None, **overrides: Any) -> dict[str, Any]:
    """ContextWatch keyword arguments for preset *name* with *overrides* applied."""
    key = name or "default"
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    settings = dict(PRESETS[key])
    settings.update(overrides)
    return settings


def split_budget(budget_tokens: int, fixed_text: str, counter: Any = None) -> int:
    """Tokens left for the briefing after *fixed_text* (a skill) is charged to the budget.

    Never negative; a skill larger than the budget leaves 0 for memory, which
    is why hosts pick the compact skill for small budgets.
    """
    from mark.context.budget import estimate_tokens

    used = (counter or estimate_tokens)(fixed_text) if fixed_text else 0
    return max(0, int(budget_tokens) - int(used))
