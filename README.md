# MARK Python SDK

`mark-sdk` is a local-first agent memory runtime. It installs as `mark-sdk`
from PyPI and imports as `mark` in Python.

MARK helps agents remember what they create, decide, observe, and learn across
long-running workflows. It is built for more than chat history: stories,
characters, images, videos, software projects, plans, design decisions, tool
results, and multi-agent workflow state all need continuity.

The SDK combines local memory storage, retrieval, context injection,
observability, graph-scoped memory blocks, and tamper-evident provenance into a
small developer API. Everything runs locally by default: no account, network,
cloud service, or API key is required.

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

## Why MARK

Most memory integrations start and end with `store()` and `retrieve()`. MARK is
designed as an agent memory runtime: it participates in the agent loop, injects
useful context before work, observes outcomes after work, and keeps memory
inspectable as connected evidence rather than flat chunks.

Use MARK when an agent needs to:

- remember creative continuity: characters, objects, scenes, styles, plot
  points, and generated artifacts;
- remember project decisions: architecture, conventions, constraints,
  implementation plans, and test results;
- coordinate across agents: one agent plans, another implements, another tests,
  while all share durable local context;
- preserve trust and provenance: group related memory into blocks, seal them,
  verify them later, and quarantine bad memory without breaking the rest of the
  agent;
- adopt memory incrementally: add middleware or tools to an existing agent
  rather than rebuilding the application around a database.

MARK is not a hosted memory database. The open SDK is the MIT-licensed local
runtime and integration layer. Hosted services, dashboards, and commercial
cloud infrastructure are separate from this package.

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

### 2. Run the same agent with memory

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

### 3. Track continuity across creative sessions

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

### 4. Group creations and decisions into provenance blocks

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

### 5. Add memory to a LangChain agent as middleware

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

### 6. Keep canonical creative facts in a world bible

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

- **Agent-loop integration:** wrap simple callables, attach LangChain
  middleware, expose MARK as tools, or serve memory over MCP.
- **Pluggable middleware:** compose framework-neutral middleware for recall
  defaults, observation metadata, compression, query expansion, gap healing,
  lifecycle maintenance, observability, and sync envelope preparation.
- **Structured memory:** fragments, sessions, graph nodes, graph edges,
  world-bible facts, and graph-scoped blocks.
- **Retrieval pipeline:** local vector retrieval with graph expansion, scoring,
  gap reporting, optional query expansion, and optional contextual compression.
- **Creative continuity:** session prefixes, tags, scopes, world-bible facts,
  and blocks make it natural to track characters, objects, locations, scenes,
  styles, and generated artifacts.
- **Workflow memory:** store plans, conventions, decisions, tool outputs, test
  results, and implementation state for coding or autonomous agents.
- **Provenance and integrity:** SHA-256 block sealing, whole-chain verification,
  pinpointed corruption reports, and quarantine isolation.
- **Memory lifecycle:** working memory with TTL expiry, consolidation,
  deduplication, pruning, reinforcement, and local governance gates.
- **Multi-agent sharing:** trust-aware local bus for publisher trust,
  subscriptions, snapshots, and trust-filtered retrieval.
- **Local observability:** JSONL traces, replay support, session activity logs,
  and middleware observation of reasoning/tool outcomes.
- **MIT local boundary:** everything in this package runs locally; cloud
  clients are caller-supplied and hosted services are outside the SDK.

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
walkthrough for memory, retrieval, sessions, middleware, tools, MCP exposure,
memory inspection, and provenance sealing.

Planned proof-oriented examples:

- `examples/character-consistency/`: demonstrate a character, object, or style
  staying consistent across repeated creative generations.
- `examples/multi-agent-coding/`: demonstrate planner, implementer, and tester
  agents sharing decisions and workflow state through MARK.

## Development

```bash
python examples/run_live_examples.py
uv run --extra dev pytest
uv build
```

## Middleware

Core middleware lives under `mark.middleware` and wraps existing public runtime
operations. It is framework-neutral, so LangChain, MCP, and future adapters can
share the same behavior instead of reimplementing recall, observation,
compression, tracing, and sync logic.

```python
from mark import Mark, ObserveMiddleware, ObservabilityMiddleware, RecallMiddleware

with Mark.local(
    ".",
    middleware=[
        RecallMiddleware(compress=True, expand=True),
        ObserveMiddleware(source="agent", tags=["workflow"]),
        ObservabilityMiddleware(),
    ],
) as mark:
    memory = mark.runtime.memory("coder")
    memory.observe("Use FastAPI dependency injection for routes.")
    result = memory.retrieve_sync("How should routes be implemented?")
```

Middleware is local and MIT-safe. Hosted retrieval, governed compression,
browser healing, and cloud observability remain cloud/client responsibilities
behind explicit hooks or caller-supplied clients.

Available local middleware includes recall, observe/writeback, compression,
query expansion, governance, lifecycle, observability, sync, trust bus,
sandbox, gap healing, and media continuity wrappers.

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
