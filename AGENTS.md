# MARK Python SDK Agent Guide

This repository is the local-first MARK SDK: agent memory, recall, facts,
briefings, and the integrations that put them in an agent loop. Read
[ARCHITECTURE.md](ARCHITECTURE.md) for the module map and data flows, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the house rules.

## Product boundary

- `mark` must work without network, Docker, accounts, or remote APIs.
- `mark.adapters` holds optional glue for LangChain, MCP, and LangGraph. It
  uses public APIs only.
- `mark.middlewares.sync.CloudSync` prepares redacted envelopes locally;
  callers supply any remote client.
- Hosted features are not part of this package (MARK Cloud — coming soon). Do
  not describe or implement them here.
- Do not turn `mark` into a presentation app or product UI.

## Agent usage pattern

When MARK is active, treat it as persistent working memory, not an optional
lookup step:

1. **Recall** before planning: retrieve, or read the injected briefing.
2. **Ground** memory against current evidence (files, tool output, the user).
3. **Act** with the remembered context.
4. **Reflect** on what changed.
5. **Remember** verified facts, decisions, and procedures for future agents.
   Record values that change over time as facts (`memory.facts.set`).

When memory reports a subject as not in memory, say so instead of guessing.
When a briefing corrects a value, use the current value.

When demonstrating MARK, show the same task without MARK and with MARK.

## SDK integration modes

| Mode | Entry point | Use when |
| --- | --- | --- |
| Direct runtime | `Mark.local()`, `mark.memory`, `mark.runtime.memory(agent_id)`, `mark.wrap_agent(...)` | Application code owns the agent loop. |
| Briefings | `ContextWatch(memory)`: `observe(...)`, then `briefing().text` before each model call | A plain loop should get memory without issuing searches. |
| Middleware | `MarkAgentMiddleware` (LangChain, `anticipate=True` by default); `mark.middlewares` for runtime operations | Memory should be injected and recorded automatically. |
| Tools | `create_mark_tools(...)`: `mark_retrieve`, `mark_write`, `mark_observe` | The model should decide when to read or write memory. |
| MCP | `create_mark_mcp_server(...)`, same three tools over stdio | External MCP clients need memory. |
| CLI | `mark ingest`, `store`, `recall`, `recall-many`, `fact set/get/history/retract`, `stats`, `serve` | Scripts, shells, and agents that run commands. |
| Skill | `src/mark/middlewares/skills/mark-usage/SKILL.md` | An agent loads portable `SKILL.md` instructions. |

`mark.memory`, the LangChain and MCP adapters, and the CLI all default to the
agent namespace `"__mark__"`, so memory written through one is visible to the
others. Prefer middleware or briefings for
automatic context, and the direct runtime when the application controls each
step.

## Development rules

- Keep core dependencies minimal; use optional extras for anything heavy.
- Every behaviour change needs a test. Tests run offline.
- Python files stay at or below 300 lines, with a module docstring.
- Every module must import on its own (`tests/test_cold_imports.py`).
- Do not require LangChain, MCP, or remote-service packages in the core.
- Do not store secrets in examples or fixtures, except fake values used by
  redaction tests.
- Ship non-Python skill assets as package data so installed users can load
  them through `importlib.resources`.
- Changelog entries describe user-facing features and fixes only, in plain,
  professional language.

## Verification

From the repository root:

```bash
uv run --extra dev pytest
uv build
```

Check that the package ships the MARK usage skill:

```bash
uv run python -B -c "from importlib.resources import files; print(files('mark.middlewares.skills').joinpath('mark-usage/SKILL.md').read_text().splitlines()[1])"
```
