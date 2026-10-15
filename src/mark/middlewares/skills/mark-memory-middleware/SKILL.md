---
name: mark-memory-middleware
description: Use when a LangChain agent is wrapped with MARK middleware and should treat MARK as fast, transparent project memory for recall, grounding, and durable observation.
---

# MARK Memory Middleware

Use this skill when MARK middleware adds a `[MARK project memory]` block to
your system message. The block holds memory relevant to the task and, when
anticipation is on, a `<mark_briefing>` prepared from the conversation.

## Reading The Memory Block

- **Memories** are records relevant to the current request, most relevant
  first.
- **Facts** (`subject attribute = value (since date)`) are the current values
  of things that change over time. Prefer them over older free-text records.
- **Corrections** ("X is V now; W is outdated") mean a value you stated is
  superseded. Use the current value from now on.
- **Not in memory** lists subjects MARK has no record of. Say you do not know
  rather than guessing.

## Operating Loop

1. Recall: read the memory block before planning.
2. Ground: compare memory with the user request and live tool results.
3. Act: use current files and tool outputs as the source of truth.
4. React: when search, subagent, or lookup results arrive, the middleware
   refreshes memory before your next model call.
5. Reflect: summarise only durable decisions, fixes, conventions, and
   completed work.
6. Remember: the middleware records useful model and tool output for you.

## Rules

- Treat MARK as background memory, not a replacement for reading files.
- Prefer live repository evidence over stale memory.
- If memory conflicts with current files or user instructions, follow the
  current evidence and state the correction in your summary.
- Do not store secrets, credentials, tokens, private keys, hidden reasoning,
  temporary guesses, or large copied source files.
- Keep final summaries concise and useful for future agents.

## Middleware Contract

The middleware is fast, bounded, and fails safe:

- it retrieves before model calls within a time limit, and a memory failure
  never stops the agent,
- it injects at most `max_context_chars` of memory,
- it refreshes on richer triggers such as search, retrieval, or subagent
  output rather than on every file read,
- it records tool results and responses, and writes durable observations
  when the run ends,
- it logs its MARK calls so traces show what happened.
