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

1. **Brief** at the start of a session: `memory.brief()`, `mark brief`, or
   the MCP `mark_brief` tool (current facts, open TODOs, decisions, bugs and
   fixes, recent changes in one call).
2. **Recall** before planning: `memory.recall(query)`, or read the injected
   briefing.
3. **Ground** memory against current evidence (files, tool output, the user).
   Live evidence wins; then fix memory.
4. **Act** with the remembered context.
5. **Remember** verified outcomes with a kind:
   `remember(text, kind="decision" | "todo" | "bug" | "fix" | "preference" | "snippet" | "api" | "note")`,
   or start the text with a label (`TODO:`, `DECISION:`, `BUG:`, `FIX:`, ...),
   which sets the same kind in Python, the CLI and MCP.
6. **Close the loop.** `resolve(id)` a TODO or bug when it is done;
   `supersede(id, new_text)` a note that changed instead of adding a second
   version; record values that change (ports, owners, versions, test
   commands) as facts (`memory.facts.set(subject, attribute, value)`) or let
   extraction do it (`fact_extractor="patterns"`, `extract_facts=True`,
   `mark ingest --extract-facts`).

Refer to code as `path/file.py::Symbol`, which survives edits, rather than
`file.py:123`. When memory reports a subject as not in memory, say so instead
of guessing. When a briefing corrects a value, use the current value. For
"how did X change" or "what happened to X" questions, read the subject's
timeline (`memory.timeline(subject, attribute)` or `mark timeline`) instead
of a top-k recall. Never store secrets (MARK redacts common ones, but do not
rely on it) or hidden reasoning; `forget(id, secure=True)` removes a mistake.

When demonstrating MARK, show the same task without MARK and with MARK.

## SDK integration modes

| Mode | Entry point | Use when |
| --- | --- | --- |
| Direct runtime | `Mark.local()`, `mark.memory` or `mark.runtime.memory(agent_id)`: `remember`, `recall`, `brief`, `resolve`, `supersede`, `forget`, `facts`, `timeline` (async twins `aremember`, `arecall`, ...); `mark.wrap_agent(...)` | Application code owns the agent loop. |
| Briefings | `ContextWatch(memory, budget_tokens=...)` or `ContextWatch.for_preset(memory, "small")`: `observe(...)` or `observe_messages(messages)`, then `briefing().as_context()` or `inject(messages, briefing)` before each model call | A plain loop should get memory without issuing searches. |
| Context window | `MarkChat(memory, model=..., base_url=..., keep_last=..., preset="small")` | MARK should hold the conversation and the model see only recent messages plus a briefing. |
| Middleware | `MarkAgentMiddleware` (LangChain, `anticipate=True` by default; `budget_tokens=`, `history="mark"`, `preset=`); `mark.middlewares` for runtime operations | Memory should be injected and recorded automatically. |
| Tools | `create_mark_tools(...)`: `mark_retrieve`, `mark_write`, `mark_observe` | The model should decide when to read or write memory. |
| MCP | `mark mcp` or `python -m mark.adapters.mcp` (same options: `--db`, `--agent`, `--briefing`, `--preset`, `--tools` (may name `mark_briefing`), `--allow-agent`, `--user-subject`, `--extract-facts`); `create_mark_mcp_server(...)`: `mark_brief`, `mark_retrieve`, `mark_write` (`kind`), `mark_observe`, `mark_fact`, `mark_timeline`, `mark_list`, `mark_resolve`, `mark_supersede`, `mark_forget` (id, or whole-word `match` with `confirm=true` for several; `secure`); `--briefing` adds `mark_briefing`; unknown arguments are errors | External MCP clients need memory. |
| CLI | `mark remember --kind`, `recall [--kind K [--all]] [--open]`, `brief`, `resolve`, `supersede`, `forget ID [--secure]` / `forget --match TEXT [--dry-run] [--yes] [--all]`, `list --kind --open`, `fact set/get/history/list/retract`, `timeline`, `ingest [--extract-facts]`, `export`/`import`, `stats`, `config show` / `config user-subject NAME`, `serve --detach` / `serve status` / `serve stop` (agents: always `--detach`), `mcp` | Scripts, shells, and agents that run commands. |
| Skill | `src/mark/middlewares/skills/mark-usage/SKILL.md` (opt-in `mark-memory-compact` for small models) | An agent loads portable `SKILL.md` instructions. |

`mark.memory`, the LangChain and MCP adapters, and the CLI all default to the
agent namespace `"__mark__"`, so memory written through one is visible to the
others. They also find the store by one rule: a project folder keeps it in
`<folder>/.mark/memory.db`, a `.mark` folder is the store folder and a `.db`
file is the database (`Mark.local("./.mark")` and `mark --db .mark` open the
same file). The user subject (who "I" is) is stored with the store
(`Mark.local(user_subject=...)`, `mark config user-subject NAME`). IDs in CLI
and MCP output are 8-character short ids, accepted back as input. Prefer middleware or briefings for
automatic context, and the direct runtime when the application controls each
step.

## Development rules

- Keep core dependencies minimal; use optional extras for anything heavy.
- Every behaviour change needs a test. Tests run offline.
- Python files stay at or below 300 lines, with a module docstring; keep
  import paths stable when splitting a module.
- Public verbs are plain and synchronous; async twins take an `a` prefix
  (`recall` / `arecall`). `mark.memory` and `runtime.memory(id)` share verbs
  and defaults. Keep old names working as aliases.
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

When other pytest sessions run on the same machine, isolate yours with
`-p no:cacheprovider --basetemp=<a private temporary directory>`. Changes to
the store, index or retrieval pipeline also run
`uv run --extra fast python benchmarks/scale/bench.py --n 20000 --check`.

Check that the package ships the MARK usage skill:

```bash
uv run python -B -c "from importlib.resources import files; print(files('mark.middlewares.skills').joinpath('mark-usage/SKILL.md').read_text().splitlines()[1])"
```
