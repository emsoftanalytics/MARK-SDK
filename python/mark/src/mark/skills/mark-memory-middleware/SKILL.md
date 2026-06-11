---
name: mark-memory-middleware
description: Use when a LangChain agent is wrapped with MARK middleware and should treat MARK as fast, transparent project memory for recall, grounding, and durable observation.
---

# MARK Memory Middleware

Use this skill when MARK middleware injects a `[MARK project memory]` block or records middleware diagnostics.

## Operating Loop

1. Recall: read `[MARK project memory]` before planning.
2. Ground: compare memory with the user request and live tool results.
3. Act: use current files and tool outputs as the source of truth.
4. React: if search, subagent, or external lookup results appear, let middleware refresh memory context.
5. Reflect: summarize only durable decisions, fixes, conventions, and completed work.
6. Remember: middleware observes useful outcomes automatically.

## Rules

- Treat MARK as background memory, not a replacement for file reads.
- Prefer live repository evidence over stale memory.
- If memory conflicts with current files or user instructions, follow current evidence and produce a correction-worthy summary.
- Do not store secrets, credentials, tokens, private keys, hidden reasoning, temporary guesses, or large copied source files.
- Keep final summaries concise and useful for future agents.

## Middleware Contract

The middleware should be optimistic, fast, and reactive:

- retrieve before the first meaningful model decision when cached or fast enough,
- inject bounded memory context,
- avoid retrieving from ordinary file reads already present in the context window,
- refresh on richer triggers such as search, grep, web lookup, retrieval, or subagent output,
- queue tool and response observations, then flush durable observations at session end,
- log MARK endpoint calls so benchmark traces can show what actually happened.
