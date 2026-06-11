# MARK Python SDK

`mark-sdk` is a local-first cognitive memory runtime for agent applications.
It installs as `mark-sdk` from PyPI and imports as `mark` in Python.

MARK gives agents durable local memory: session-aware retrieval, graph-linked
facts, memory blocks with tamper-evident provenance, working memory,
consolidation, and pruning — all on your machine, no account or network
required.

```python
from mark import Mark

with Mark.local(project_path=".") as mark:
    mark.memory.block("project").write(
        "The API framework is FastAPI.",
        importance=0.9,
    )

    context = mark.memory.retrieve("Which API framework does this project use?")
    print(context.as_text())
```

## Install

MARK supports Python 3.10 and newer.

```bash
pip install mark-sdk
```

Optional framework adapters are bundled under `mark.adapters` and installed
through extras:

```bash
pip install "mark-sdk[langchain]"   # LangChain tools + agent middleware
pip install "mark-sdk[mcp]"        # MCP server over MARK memory
pip install "mark-sdk[adapters]"   # both of the above
```

Verify the installed package:

```bash
python -c "from mark import Mark; print(Mark.local('.').memory.list_blocks())"
```

## Quickstart

### 1. Store and recall project memory

```python
from mark import Mark

with Mark.local(".") as mark:
    mark.memory.block("architecture").write(
        "Authentication uses FastAPI dependencies and JWT.",
        importance=0.8,
    )

    result = mark.memory.retrieve("How is authentication implemented?")
    print(result.as_text())
```

### 2. Use MARK with an agent

```python
import asyncio

from mark import Mark

def coding_agent(prompt: str) -> str:
    if "FastAPI" in prompt:
        return "Use FastAPI dependency injection for this endpoint."
    return "I need more project context."

with Mark.local(".") as mark:
    mark.memory.block("project").write("This backend uses FastAPI.")
    agent = mark.wrap_agent(coding_agent, blocks=["project"])

    result = asyncio.run(agent.run("Add a health-check endpoint."))
    print(result.output)
```

### 3. Use sessions for long-running work

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("video-agent")

    memory.observe(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="season-01/episode-01/scene-04",
        memory_type="scene",
    )

    scene_context = memory.retrieve_sync(
        "What must stay consistent for Elena?",
        session_prefix="season-01/",
    )
    print(scene_context.as_context())
```

### 4. Group memory into blocks with provenance

Memory blocks bundle related fragments, nodes, and edges into one unit per
topic, session, or world-bible scope. Blocks link to each other forward and
backward, can be sealed into a tamper-evident hash chain, and can be
quarantined — isolating bad memory without affecting the rest of the agent.

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("video-agent")
    graph  = memory.blocks()
    chain  = memory.chain()

    scene = graph.create_block("ep01-scene04", session_id="season-01/episode-01")
    graph.add_fragment(scene.id, memory.store_sync("Elena hides the key in the rafters."))

    # Retrieval scoped to one block
    result = memory.retrieve_sync("Where is the key?", block_id=scene.id)

    # Seal the block into the agent's provenance chain
    sealed = chain.seal(scene.id)
    print(sealed.content_hash)

    # Verify integrity later — pinpoints any corrupted member
    print(chain.verify(scene.id).valid)
    print(chain.verify_chain().valid)

    # Isolate a bad block without touching anything else
    chain.quarantine(scene.id)
```

### 5. Plug MARK into a LangChain agent (adapters)

`MarkAgentMiddleware` turns MARK into a transparent context-window manager for
any LangChain v1 agent: it retrieves relevant memory before each model call,
injects it into the system message, and archives the agent's reasoning and
tool results in the background — the agent code stays unchanged.

```python
from langchain.agents import create_agent

from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware

with Mark.local(".") as mark:
    middleware = MarkAgentMiddleware(
        backend=LocalMarkBackend(mark, default_agent_id="coder"),
        agent_id="coder",
        max_context_chars=1400,     # context budget injected per model call
        write_outcomes=True,        # archive AI reasoning steps
        observe_tool_results=True,  # archive tool outputs as they happen
    )

    agent = create_agent(model, tools, middleware=[middleware])
    result = agent.invoke({"messages": [("user", "Fix the failing build.")]})
```

Prefer explicit control? `create_mark_tools` exposes memory as ordinary
LangChain tools the model calls itself:

```python
from mark.adapters.langchain import create_mark_tools

tools = create_mark_tools(LocalMarkBackend(mark, default_agent_id="coder"))
agent = create_agent(model, [*tools, *my_other_tools])
```

And `mark.adapters.mcp` serves the same memory to any MCP-compatible client:

```python
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import create_mark_mcp_server

server = create_mark_mcp_server(LocalMarkBackend(mark))
server.run()  # stdio MCP server
```

The full middleware walkthrough — the same agent run with and without MARK,
side by side — lives in the tutorial notebook under
[examples/](examples/).

### 6. Keep canonical facts in a world bible

```python
from mark import Mark

with Mark.local(".") as mark:
    mark.world_bible.remember(
        "Elena is left-handed and always wears the red scarf in episode 01.",
        tags=["character:elena", "wardrobe"],
    )

    facts = mark.world_bible.check("Elena wardrobe and physical traits")
    for fact in facts:
        print(fact.content)
```

## Features

- SQLite-backed local memory at `.mark/memory.db`
- memory fragments, sessions, graph nodes, graph edges, and world-bible facts
- graph-scoped memory blocks with forward/backward links, SHA-256 seal +
  verify, and quarantine isolation
- local vector retrieval with graph expansion
- session-aware retrieval by exact session, session prefix, tags, tier, scope,
  and block
- deterministic local entity extraction; optionally bring your own LLM
- optional contextual compression and query expansion
- working memory with TTL expiry; consolidation, deduplication, and pruning
- trust-aware local bus for multi-agent sharing
- local governance heuristics and audit records
- local observability with JSONL trace/replay support
- optional framework adapters under `mark.adapters.*` (LangChain tools and
  middleware, MCP server)

Everything in this package runs locally under the MIT license.

## MARK Cloud — coming soon

A hosted MARK Cloud is in development: managed memory for teams, cross-device
sync, shared agent memory, and a dashboard to inspect what your agents know.
It will connect through the same public hook interfaces this SDK already
ships — code written against local MARK will work unchanged. Watch the
repository for the announcement.

## Examples

See [examples/](examples/) for runnable developer usage tests and walkthroughs:

```bash
python examples/run_live_examples.py
```

The live examples cover:

- local project memory storage and retrieval,
- A/B agent usage with and without MARK context injection,
- session-aware `observe()` continuity retrieval,
- redacted sync envelope preparation without cloud transport.

The notebook in [examples/](examples/) provides a longer step-by-step
walkthrough for memory, retrieval, sessions, adapters, and agent integration.

## Development

```bash
python examples/run_live_examples.py
uv run --extra dev pytest
uv build
```

## Contributing & support

- **Found a bug? Have an idea?** [Open an issue](https://github.com/emsoftanalytics/MARK-SDK/issues/new/choose) —
  the templates take two minutes, and "this surprised me" reports are welcome too.
- **Want to contribute?** Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [`good first issue`](https://github.com/emsoftanalytics/MARK-SDK/labels/good%20first%20issue) label.
  Draft PRs and questions are encouraged.
- **Using MARK in a project?** Tell us in
  [Discussions](https://github.com/emsoftanalytics/MARK-SDK/discussions) — real
  workloads drive the roadmap.
- **Security issues:** see [SECURITY.md](SECURITY.md) — please report privately.

## License

MIT — see [LICENSE](LICENSE).
