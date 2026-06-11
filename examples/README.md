# MARK SDK Examples

These examples are developer usage tests and public proof points for MARK as
an agent memory runtime. They show the SDK doing more than storing facts:
injecting context into work, preserving session continuity, preparing safe sync
envelopes, and exposing the same local memory through agent integrations.

- `01_local_memory_live.py` — stores project memory and retrieves it through
  the public `Mark.local()` API.
- `02_agent_ab_live.py` — runs the same callable without MARK and with MARK
  context injection.
- `03_sessions_and_observe_live.py` — records session continuity facts through
  `mark.observe()` and retrieves them by session prefix.
- `04_sync_boundary_live.py` — prepares a redacted sync envelope without
  importing or calling any cloud service.
- `run_live_examples.py` — runs all live examples as developer usage tests.
- [getting_started_with_mark.ipynb](getting_started_with_mark.ipynb) — the full
  walkthrough: local memory basics, seeding project conventions, running the
  same coding agent with and without MARK (middleware, tools, and MCP modes),
  inspecting learned memory, and sealing memory blocks into a provenance chain.

## What these examples demonstrate

- **Memory changes behavior:** `02_agent_ab_live.py` shows the same agent answer
  differently when MARK injects relevant project context.
- **Creation and workflow state can persist:** session-aware observation keeps
  continuity facts retrievable across later work.
- **Memory remains inspectable:** the notebook walks through stored memories,
  adapter usage, MCP exposure, and provenance sealing.
- **The SDK boundary is local:** sync examples prepare redacted envelopes but do
  not import or call hosted cloud service code.

## Running live examples

```bash
python examples/run_live_examples.py
```

The live examples use temporary local project directories and require no API
keys, cloud account, Docker daemon, or network access.

## Running the notebook

```bash
pip install "mark-sdk[tutorial]" jupyter
jupyter lab examples/getting_started_with_mark.ipynb
```

The agent sections need a running [Ollama](https://ollama.com) or any Provider compatible with langchain, with any
tool-calling chat model — set `MODEL` in the first code cell to one you have
installed. Everything else (memory, blocks, retrieval, MCP server creation)
runs fully offline.

## Upcoming proof demos

The next documentation update should add two reproducible examples that make
MARK's differentiation visible at a glance:

- `character-consistency/`: a creative workflow where memory preserves a
  character, object, style, or scene across repeated generations.
- `multi-agent-coding/`: a workflow where one agent plans, another implements,
  and another tests while sharing decisions through MARK.
