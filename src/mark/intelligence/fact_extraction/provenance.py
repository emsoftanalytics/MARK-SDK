"""Who said it: provenance prefixes on stored chat turns and agent notes.

MarkChat and the agent middleware store turns with a prefix that says who
spoke: "User said: ...", "Assistant replied: ...", "Tool [search]
returned: ...", "Agent reasoning: ...". Facts come from what the *user*
states. An assistant's reply restates, guesses or role-plays ("so you live
in Porto now"), and tool output is untrusted data, so neither is ever a
source of facts.

:func:`user_text` keeps the user's lines (prefix removed) and drops lines
spoken by anyone else. A line without a prefix belongs to the speaker of
the last prefixed line (a reply over several lines); text with no prefix at
all is returned unchanged.
"""
from __future__ import annotations

import re

# "User said:", "User:", "Human:", "Me:" -> the user's own words.
_USER = re.compile(r"^\s*(?:\[[^\]\n]*\]\s*)?(?:user|human|me)(?:\s+(?:said|says|wrote|asked))?\s*:\s*",
                   re.IGNORECASE)
# Assistant / agent / tool turns: never extracted. ("System:" is left alone: ops
# logs use it as an ordinary label, "[2026-03-01] SYSTEM: atlas-core moved ...".)
_OTHER = re.compile(
    r"^\s*(?:\[[^\]\n]*\]\s*)?(?:"
    r"(?:assistant|ai|bot|claude|gpt)(?:\s+(?:said|says|replied|answered|wrote|responded))?"
    r"|agent(?:\s+(?:reasoning|said|replied|completed\s+task\.\s*response|response|note|thought|plan))?"
    r"|tool(?:\s*\[[^\]\n]*\])?(?:\s+(?:returned|result|output|said|error))?|\[called\s+tool\]"
    r"|observation|function(?:\s+result)?)\s*:\s*",
    re.IGNORECASE)
_CALLED_TOOL = re.compile(r"^\s*\[called tool\]", re.IGNORECASE)


def user_text(text: str) -> str:
    """*text* with non-user turns removed and user prefixes stripped (see module docstring)."""
    if not text or (":" not in text and "[called tool]" not in text.lower()):
        return text or ""
    kept: list[str] = []
    speaker: str | None = None  # "user" | "other" | None (no prefix seen yet)
    for line in text.splitlines():
        if _OTHER.match(line) is not None or _CALLED_TOOL.match(line):
            speaker = "other"
            continue
        if (user := _USER.match(line)) is not None:
            speaker = "user"
            kept.append(line[user.end():])
            continue
        if speaker != "other":
            kept.append(line)
    return "\n".join(kept)
