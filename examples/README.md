# MARK SDK Examples

These examples are developer usage tests and public proof points for MARK as
an agent memory runtime. They show the SDK doing more than storing facts:
injecting context into work, preserving session continuity, preparing safe sync
envelopes, and exposing the same memory through agent integrations.

- `01_local_memory_live.py` — stores project memory and retrieves it through
  the public `Mark.local()` API.
- `02_agent_ab_live.py` — runs the same callable without MARK and with MARK
  context injection.
- `03_sessions_and_observe_live.py` — records session continuity facts through
  `mark.observe()` and retrieves them by session prefix.
- `04_sync_envelope_live.py` — prepares a redacted sync envelope.
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
- **Sync remains explicit:** sync examples prepare redacted envelopes but do not
  perform uploads.

## Running live examples

```bash
python examples/run_live_examples.py
```

The first four live examples use temporary local project directories and
require no API keys, Docker daemon, or network access.

Generated media demo outputs and heavyweight provider clients are not included
in `mark-sdk`.

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

The next packaged proof demo should stay offline: a workflow where one agent
plans, another implements, and another tests while sharing decisions through
MARK.
