# MARK SDK Examples

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
