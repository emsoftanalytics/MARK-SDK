# MARK (Memory-Augmented Agent Recall Kit) Python SDK

MARK is local-first memory for AI agents. It installs as `mark-sdk` from PyPI
and imports as `mark`.

An agent stores what it learns, decides, and produces. MARK recalls the
relevant parts later, tracks values that change over time, and can prepare a
briefing before the agent asks. Everything runs locally in a SQLite file: no
account, network, or API key is required.

```python
from mark import Mark

with Mark.local(".") as mark:
    mark.memory.block("project").write("The API framework is FastAPI.", importance=0.9)

    context = mark.memory.retrieve("Which API framework does this project use?")
    print(context.as_text())
```

## What MARK provides

| Capability | What it does |
| --- | --- |
| Memory and recall | Hybrid keyword (BM25) and vector search with rank fusion. No LLM calls on store or recall. |
| Sessions and blocks | Partition memory by session, group it into blocks, seal blocks into a tamper-evident chain. |
| Facts | Values that change over time, with full history, corrections, and point-in-time lookups. |
| Time-aware recall | Questions about the current or original state return the newest or earliest record first. |
| Gap reports | Recall reports names and identifiers that are not in memory, so an agent can say it does not know. |
| Briefings | `ContextWatch` follows a conversation and prepares relevant memory before each model call. |
| Integrations | LangChain middleware and tools, an MCP server, and the `mark` command line. |
| Local embeddings | Ollama, OpenAI-compatible servers, FastEmbed, Model2Vec, or the zero-dependency default. |

## Install

MARK supports Python 3.10 and newer. The core package depends only on
`pydantic`.

```bash
pip install mark-sdk
```

| Extra | Adds |
| --- | --- |
| `mark-sdk[fast]` | numpy-vectorised vector search (recommended) |
| `mark-sdk[langchain]` | LangChain tools and agent middleware |
| `mark-sdk[mcp]` | MCP server over MARK memory |
| `mark-sdk[adapters]` | both LangChain and MCP |
| `mark-sdk[crypto]` | encryption at rest (`FernetEncryptionProvider`) |
| `mark-sdk[ollama]`, `[fastembed]`, `[model2vec]`, `[sentence-transformers]` | local embedding providers |
| `mark-sdk[middleware]` | all local middleware batteries |

Check the installation:

```bash
python -c "import mark; print(mark.__version__)"
```

## Quickstart

The snippets below run offline, except sections 7 and 10, which need a local
model server. Each opens a store under the current directory
(`./.mark/memory.db`).

### 1. Memory blocks and recall

`mark.memory` is the simple API. Blocks are named groups of related memory.

```python
from mark import Mark

with Mark.local(".") as mark:
    decisions = mark.memory.block("decisions", kind="procedures")
    decisions.write("Authentication uses FastAPI dependencies and JWT.", importance=0.8)

    result = mark.memory.retrieve("How is authentication implemented?", blocks=["decisions"])
    print(result.as_text())
```

`mark.memory` writes to the default agent namespace, `"__mark__"`. For a
separate namespace with the full API, use `mark.runtime.memory("<agent id>")`.
The LangChain, MCP, and CLI integrations all default to `"__mark__"`.

### 2. Agent namespaces and sessions

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("video-agent")
    memory.store_sync(
        "Elena enters the North Warehouse wearing the red scarf.",
        session_id="season-01/episode-01",
    )

    result = memory.retrieve_sync("What does Elena wear?", session_prefix="season-01/")
    print(result.as_context())
```

`session_id` matches one session; `session_prefix` matches a group of them.

### 3. Facts that change over time

A fact is the value of an attribute of a subject, such as a service's port.
Later values supersede earlier ones by when they became true (`valid_from`),
not by when they were recorded, and nothing is deleted.

```python
from mark import Mark

with Mark.local(".") as mark:
    facts = mark.runtime.memory("ops").facts
    facts.set("atlas-core", "port", "6430", valid_from="2026-03-01")
    facts.set("atlas-core", "port", "4444", valid_from="2026-03-30")
    facts.set("atlas-core", "port", "5000", valid_from="2026-03-15")  # arrives late

    print(facts.get("atlas-core", "port").value)                    # 4444
    print([f.value for f in facts.history("atlas-core", "port")])   # ['6430', '5000', '4444']
    print(facts.as_of("atlas-core", "port", "2026-03-20").value)    # 5000

    # Fix a wrong value instead of adding a new one:
    facts.set("atlas-core", "port", "4445", valid_from="2026-03-30", correction=True)
    # Withdraw a fact recorded by mistake (kept for audit):
    facts.retract("atlas-core", "port", value="5000")
```

Several values recorded for the same time are kept as a sequence, and the last
one is current. Subjects and attributes match regardless of case and
punctuation (`Atlas-Core` is `atlas core`).

### 4. Time-aware recall and gap reports

Free-text memories can carry an event time with `occurred_at=` or a leading
`[YYYY-MM-DD]` stamp. Questions about the current or original state use it.

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("ops")
    memory.store_sync("atlas-core moved to region eu-west.", occurred_at="2026-03-01")
    memory.store_sync("atlas-core moved to region us-east.", occurred_at="2026-04-02")

    now = memory.retrieve_sync("Which region does atlas-core use now?")
    print(now.temporal_intent, now.top().content)        # current ... us-east.

    first = memory.retrieve_sync("Which region did atlas-core use originally?")
    print(first.temporal_intent, first.top().content)    # original ... eu-west.

    unknown = memory.retrieve_sync("Who owns zephyr-9?")
    print(unknown.missing_anchors, unknown.gap_detected)  # ['zephyr 9'] True
```

`RetrievalResult.facts` holds the facts about the subjects in the query, and
`missing_anchors` lists names and identifiers that occur nowhere in memory.

### 5. Batch store and recall

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("ops")
    ids = memory.store_many_sync(
        ["HAL-4433: checkout returns 502 errors.", "dana owns checkout-api."],
        occurred_at=["2026-03-12", "2026-03-01"],
    )
    results = memory.retrieve_many_sync(["What is HAL-4433?", "Who owns checkout-api?"])
    print([r.top().content for r in results])
```

`store_many_sync` embeds items in batches and writes each batch in one
transaction. When middleware is registered, each item goes through the full
middleware path instead.

### 6. Briefings before each model call

`ContextWatch` follows the conversation and keeps relevant memory ready
without the agent issuing a search. A briefing lists current facts and
memories for the subjects in play, corrects values the agent states that
memory knows are outdated, and names subjects that are not in memory.

```python
from mark import ContextWatch, Mark

with Mark.local(".") as mark:
    ops = mark.runtime.memory("ops")
    ops.facts.set("checkout-api", "port", "8080", valid_from="2026-03-01")
    ops.facts.set("checkout-api", "port", "9090", valid_from="2026-03-20")

    with ContextWatch(ops) as watch:
        watch.observe("Is checkout-api healthy? Also check zephyr-9.", role="user")
        briefing = watch.briefing(wait=1.0)
        system_prompt = "You are an on-call engineer.\n" + briefing.text
        print(briefing.text)        # facts for checkout-api; zephyr-9 not in memory

        watch.observe("I'll restart checkout-api on port 8080.", role="assistant")
        print(watch.briefing(wait=1.0).corrections)   # 9090 is current; 8080 is outdated
```

Call `watch.observe(...)` for user messages, model replies (`role="assistant"`),
and tool results (`role="tool"`), and add `watch.briefing().text` to the
system prompt before each model call.

### 7. LangChain agents

`MarkAgentMiddleware` adds memory to any LangChain v1 agent. Before each model
call it injects relevant memory and, with `anticipate=True` (the default), the
`ContextWatch` briefing. After the call it records useful model and tool
output. The example below needs a chat model; it uses a local Ollama model.

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain.middleware import MarkAgentMiddleware

model = init_chat_model("ollama:llama3.2:3b")   # any LangChain chat model

with Mark.local(".") as mark:
    mark.memory.block("project").write("Builds run with `uv run --extra dev pytest`.")
    middleware = MarkAgentMiddleware(
        backend=LocalMarkBackend(mark),
        max_context_chars=1400,     # memory budget per model call
    )
    agent = create_agent(model, tools=[], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "How do I run the tests?")]})
    print(result["messages"][-1].content)
```

The middleware supports `invoke`, `ainvoke`, and `astream`, isolates state
between concurrent runs, and degrades gracefully if memory fails. Gap healing
(`auto_heal=True`) is off by default because it can reach the network.

To let the model decide when to use memory, expose it as tools instead:

```python
from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.langchain import create_mark_tools

with Mark.local(".") as mark:
    tools = create_mark_tools(LocalMarkBackend(mark))   # mark_retrieve, mark_write, mark_observe
    print([tool.name for tool in tools])
```

Use one surface by default: middleware for automatic context, or tools for
deliberate reads and writes.

### 8. MCP server

`mark.adapters.mcp` serves memory to any MCP client over stdio. It works with
`mcp` 1.x and 2.x and exposes `mark_retrieve`, `mark_write`, and
`mark_observe`.

```python
from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import create_mark_mcp_server

with Mark.local(".") as mark:
    server = create_mark_mcp_server(
        LocalMarkBackend(mark),
        allowed_agent_ids={"__mark__"},   # restrict which namespaces clients may use
    )
    server.run()                          # blocks; speaks MCP over stdio
```

### 9. Command line

The `mark` command works on a store given by `--db`, `$MARK_DB`, or
`./.mark/memory.db`, in the agent namespace given by `--agent`, `$MARK_AGENT`,
or `__mark__`.

```bash
printf '[2026-03-01] atlas-core runs on port 6430.\n[2026-03-02] HAL-4433: checkout returns 502 errors.\n' > ops.log
mark ingest ops.log --session day-1            # one memory per line
mark store "dana owns atlas-core" --occurred-at 2026-03-05
mark fact set atlas-core port 4444 --valid-from 2026-03-30
mark fact get atlas-core
mark fact history atlas-core port
mark recall "Which port does atlas-core use now?"
printf 'Who owns atlas-core?\nWhat is HAL-4433 about?\n' | mark recall-many -
mark stats
```

`mark fact set --correction` replaces a wrong value, and `mark fact retract`
withdraws one. `recall` and `recall-many` accept `-k N` and `--json`.

`mark serve` keeps the store open so each command answers in about 0.1 s. It
binds to loopback only, requires a token kept in a private state file next to
the database, and allows one server per database. Other `mark` commands use it
automatically; `--no-server` runs in-process. See [SECURITY.md](SECURITY.md).

The CLI uses the default embedder and does not support encryption; use the
Python API for those.

### 10. Local embedding models

The default `HashEmbeddingProvider` needs no model download. For semantic
recall, pass any local embedding model:

```python
from mark import Mark
from mark.embeddings import (
    FastEmbedProvider,                  # ONNX, no torch:  mark-sdk[fastembed]
    Model2VecProvider,                  # static, numpy:   mark-sdk[model2vec]
    OllamaEmbeddingProvider,            # Ollama:          mark-sdk[ollama]
    OpenAICompatibleEmbeddingProvider,  # llama.cpp server, LM Studio, vLLM, LocalAI
)

embedder = OllamaEmbeddingProvider(model="nomic-embed-text")
# or: OpenAICompatibleEmbeddingProvider(base_url="http://localhost:1234/v1", model="nomic-embed-text-v1.5")

with Mark.local(".", embedder=embedder) as mark:
    mark.memory.block("project").write("The API framework is FastAPI.")
```

HTTP providers batch requests, apply timeouts, and retry transient failures.
MARK records each embedder's fingerprint, so switching models re-embeds
existing memory instead of mixing vector spaces.

### 11. Provenance blocks

Blocks can be sealed into a per-agent SHA-256 hash chain, verified later, and
quarantined without affecting other memory.

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("video-agent")
    graph, chain = memory.blocks(), memory.chain()

    shot = graph.create_block("ep01-scene04", session_id="season-01/episode-01")
    graph.add_fragment(shot.id, memory.store_sync("Elena hides the key in the rafters."))
    print(memory.retrieve_sync("Where is the key?", block_id=shot.id).top().content)

    chain.seal(shot.id)
    print(chain.verify(shot.id).valid, chain.verify_chain().valid)   # True True
    chain.quarantine(shot.id)      # removes the block's members from recall
```

### 12. Wrap a plain agent function

```python
import asyncio

from mark import Mark

def coding_agent(prompt: str) -> str:
    return "Use FastAPI dependency injection." if "FastAPI" in prompt else "I need more context."

with Mark.local(".") as mark:
    mark.memory.block("project").write("This backend uses FastAPI.")
    agent = mark.wrap_agent(coding_agent, blocks=["project"])
    print(asyncio.run(agent.run("Add a health-check endpoint.")).output)
```

## Middleware

Framework-neutral middleware in `mark.middlewares` wraps the runtime's
`store`, `observe`, and `retrieve` operations. None of it is active unless you
pass it in:

```python
from mark import Mark, ObservabilityMiddleware, ObserveMiddleware, RecallMiddleware

with Mark.local(".", middleware=[
    RecallMiddleware(expand=True),
    ObserveMiddleware(source="agent", tags=["workflow"]),
    ObservabilityMiddleware(),
]) as mark:
    memory = mark.runtime.memory("coder")
    memory.observe("Use FastAPI dependency injection for routes.")
    print(memory.retrieve_sync("How should routes be implemented?").top().content)
```

Available middleware: recall, observe, compression, query expansion, gap
healing, governance, lifecycle, observability, sync, trust bus, sandbox, and
media continuity. See [ARCHITECTURE.md](ARCHITECTURE.md) for what each does.

## Encryption

With `mark-sdk[crypto]`, `Mark.local(".", encryption=FernetEncryptionProvider(passphrase=...))`
encrypts memory content and fact values at rest. Tags, metadata, session ids,
fact subjects and attributes, graph labels, and embeddings stay in plaintext so
they can be filtered. Encrypted stores use vector search only. See
[SECURITY.md](SECURITY.md).

## Performance

Measured with `benchmarks/bench_runtime.py` (offline, default configuration,
SQLite on disk, numpy installed):

| Memories | Store | Retrieve p50 | Retrieve p95 | hit@1 / hit@5 |
| --- | --- | --- | --- | --- |
| 2,000 | 0.6 ms | 12.5 ms | 23 ms | 14/14 / 14/14 |
| 10,000 | 0.7 ms | 32 ms | 81 ms | 14/14 / 14/14 |

Reproduce with `uv run --extra fast python benchmarks/bench_runtime.py`.
`benchmarks/retention_ab/` compares a relay of agents with and without MARK.

## Examples

[examples/](examples/) holds short scripts that run offline: recall, an LLM
with and without memory, sessions, facts over time, `ContextWatch`, and
redacted sync. A notebook walks through middleware, tools, and MCP.

```bash
python examples/run_live_examples.py
```

## MARK Cloud — coming soon

Watch out for MARK Cloud updates.

## Contributing and support

- Bugs and ideas: [open an issue](https://github.com/emsoftanalytics/MARK-SDK/issues/new/choose).
- Contributing: read [CONTRIBUTING.md](CONTRIBUTING.md). Work happens on
  branches and lands on `main` through reviewed pull requests.
- Design and code layout: [ARCHITECTURE.md](ARCHITECTURE.md).
- Plans: [ROADMAP.md](ROADMAP.md).
- Questions and use cases: [Discussions](https://github.com/emsoftanalytics/MARK-SDK/discussions).
- Security issues: report privately, see [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
