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
| Fact extraction | Facts recorded automatically from stored text, by built-in rule packs or a local LLM. |
| Timelines | Everything memory knows about a subject, oldest first, with no top-k cap. |
| Time-aware recall | Questions about the current or original state return the newest or earliest record first. |
| Gap reports | Recall reports names and identifiers that are not in memory, so an agent can say it does not know. |
| Briefings | `ContextWatch` prepares relevant memory before each model call, in any loop, within a token budget. |
| Context replacement | MARK holds the conversation; the model sees the last few messages plus a briefing. |
| Image memory | Images remembered by path and hash, found by caption, and checked against facts. |
| Integrations | LangChain middleware and tools, an MCP server, and the `mark` command line. |
| Local embeddings | `"auto"` or a spec string: Model2Vec, FastEmbed, Ollama, OpenAI-compatible, or the zero-dependency default. |

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
| `mark-sdk[vision]` | CLIP image embeddings for image memory |
| `mark-sdk[middleware]` | all local middleware batteries |

Check the installation:

```bash
python -c "import mark; print(mark.__version__)"
```

## Quickstart

The snippets below run offline, except sections 7, 8, and 11, which need a
local model server. Each opens a store under the current directory
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
`mark.wrap_agent(fn, blocks=[...])` wraps a plain agent function so each call
recalls memory first.

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

### 4. Automatic facts and timelines

With `fact_extractor="patterns"`, every write is scanned for stated changes
and descriptions ("moved from port A to port B", "X is owned by Y", "X runs
on port P", "X runs in eu-west-1", "Y owns X", "A is based in Berlin", "A
reports to B", "status: resolved") and matching facts are recorded, dated by
the event time. A line listing several facts about one subject ("X is owned
by Y, listens on port P and uses PostgreSQL 15") yields one fact per clause. A timeline lists every
fact and memory about a subject, oldest first.

```python
from mark import Mark

with Mark.local(".", fact_extractor="patterns") as mark:
    ops = mark.runtime.memory("ops")
    ops.store_many_sync([
        "[2026-03-01] atlas-core port is 6430.",
        "[2026-03-30] CHANGE: atlas-core moved from port 6430 to port 4444.",
        "[2026-04-02] Ownership of atlas-core was transferred to dana.",
    ])
    print(ops.facts.get("atlas-core", "port").value)    # 4444
    print(ops.facts.get("atlas-core", "owner").value)   # dana
    print(ops.timeline("atlas-core", "port").as_text())  # both port events, oldest first
```

Without a configured extractor, pass `extract_facts=True` to one write. For
phrasing the rule packs miss, `LLMFactExtractor(model=..., base_url=...)` asks
a local OpenAI-compatible model; its jobs run on a background worker from a
durable queue, so writes never wait for the model. `retrieve_sync(...,
order="chronological")` returns ordinary recall results oldest first.

### 5. Time-aware recall and gap reports

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

`RetrievalResult.facts` holds the facts about the subjects in the query that
match the attribute asked for ("Which port ...") or all of them for an open
question ("Tell me about atlas-core"), and `missing_anchors` lists names and
identifiers that occur nowhere in memory.

`store_many_sync` embeds items in batches and writes each batch in one
transaction (when middleware is registered, each item takes the full
middleware path instead). `retrieve_many_sync` and the async `retrieve_many`
answer several queries with shared filtering.

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

    with ContextWatch(ops, budget_tokens=300) as watch:    # hard cap on the briefing
        watch.observe("Is checkout-api healthy? Also check zephyr-9.", role="user")
        briefing = watch.briefing(wait=1.0)
        system_prompt = "You are an on-call engineer.\n" + briefing.text
        print(briefing.text)        # facts for checkout-api; zephyr-9 not in memory

        watch.observe("I'll restart checkout-api on port 8080.", role="assistant")
        print(watch.briefing(wait=1.0).corrections)   # 9090 is current; 8080 is outdated
```

Call `watch.observe(...)` for user messages, model replies (`role="assistant"`),
and tool results (`role="tool"`), and add `watch.briefing().text` to the
system prompt before each model call. In any loop that keeps an OpenAI-style
message list, `watch.observe_messages(messages)` feeds only the messages it
has not seen, and `inject(messages, briefing)` (from `mark.context.watch`)
returns a new list with the briefing in the system message.

`budget_tokens` caps the briefing. When lines must go, corrections are kept
first, then "Not in memory" notices, facts, records, and linked subjects;
`briefing.usage` reports what was used. Tokens are estimated as characters / 4
unless you pass `token_counter=` (any `str -> int` tokenizer).
`ContextWatch.for_small_models(memory)` is a tighter preset (175 tokens).

### 7. MARK as the context window

Instead of resending the whole conversation, let MARK hold it. `MarkChat`
talks to any OpenAI-compatible endpoint with the standard library only. Each
call sends the system prompt with a budgeted briefing and the last `keep_last`
messages; every exchange is stored before it leaves the window.

```python
from mark import Mark, MarkChat

with Mark.local(".") as mark:
    with MarkChat(mark.runtime.memory("chat"), model="llama3.2:3b",
                  base_url="http://localhost:11434/v1",   # any OpenAI-compatible server
                  keep_last=2, budget_tokens=200) as chat:
        chat.send("My hive is called Colmeia Azul.")
        chat.send("Name one spring flower bees like. One word.")
        chat.send("And one summer flower? One word.")
        print(chat.send("What is my hive called?"))   # recalled from MARK, not the window
```

`chat.prepare(text)` and `chat.commit(text, reply)` split `send` for loops
that call the model themselves. With LangChain, pass `history="mark"` and
`keep_last=` to `MarkAgentMiddleware` (section 8) for the same behaviour.

### 8. LangChain agents

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

`budget_tokens=` puts one token cap over everything the middleware injects.
`history="mark", keep_last=6` sends the model only the system message, the
last six messages, and MARK's context: dropped turns are stored first, a tool
call is never separated from its result, and if storing fails the full
history is sent. The graph state keeps the full transcript.

To let the model decide when to use memory, expose it as tools instead:
`create_mark_tools(LocalMarkBackend(mark))` from `mark.adapters.langchain`
returns `mark_retrieve`, `mark_write`, and `mark_observe`. Use one surface by default: middleware for automatic context, or tools for
deliberate reads and writes.

### 9. MCP server

`mark.adapters.mcp` serves memory to any MCP client over stdio. It works with
`mcp` 1.x and 2.x and exposes `mark_retrieve`, `mark_write`, and
`mark_observe`. With `briefing=True` it also exposes `mark_briefing`: the
client passes the latest message (or recent messages) and a `session_id`, and
gets back a ready-made briefing, optionally capped by `budget_tokens`. Each
session has its own watch; `mark://briefing/{session_id}` reads the current
briefing without observing anything.

```python
from mark import Mark
from mark.adapters.backend import LocalMarkBackend
from mark.adapters.mcp import create_mark_mcp_server

with Mark.local(".") as mark:
    server = create_mark_mcp_server(
        LocalMarkBackend(mark),
        allowed_agent_ids={"__mark__"},   # restrict which namespaces clients may use
        briefing=True,                    # add the mark_briefing tool
    )
    server.run()                          # blocks; speaks MCP over stdio
```

### 10. Command line

The `mark` command works on a store given by `--db`, `$MARK_DB`, or
`./.mark/memory.db`, in the agent namespace given by `--agent`, `$MARK_AGENT`,
or `__mark__`.

```bash
printf '[2026-03-01] atlas-core port is 6430.\n[2026-03-02] HAL-4433: checkout returns 502 errors.\n' > ops.log
mark ingest ops.log --session day-1 --extract-facts   # one memory per line, facts from stated values
mark store "dana owns atlas-core" --occurred-at 2026-03-05
mark fact set atlas-core port 4444 --valid-from 2026-03-30
mark fact get atlas-core
mark fact history atlas-core port
mark recall "Which port does atlas-core use now?"
mark timeline atlas-core port                  # every port event, oldest first
printf 'Who owns atlas-core?\nWhat is HAL-4433 about?\n' | mark recall-many -
mark stats
```

`mark fact set --correction` replaces a wrong value, and `mark fact retract`
withdraws one. `recall` and `recall-many` accept `-k N` and `--json`;
`recall --chronological` lists results oldest first. `timeline` accepts
`--limit N`, `--since`, `--until`, and `--json`.

`mark serve` keeps the store open so each command answers in about 0.1 s. It
binds to loopback only, requires a token kept in a private state file next to
the database, and allows one server per database. Other `mark` commands use it
automatically; `--no-server` runs in-process. See [SECURITY.md](SECURITY.md).

The CLI uses the default embedder (or `$MARK_EMBEDDER`) and refuses to open
encrypted stores; use the Python API for those.

### 11. Local embedding models

The default `HashEmbeddingProvider` is deterministic and needs no model
download. For semantic recall, pass a provider or a spec string:

```python
from mark import Mark

# "auto": the best model already on disk (Model2Vec, then FastEmbed), else hash.
# Explicit specs: "hash[:dim]", "model2vec[:model]", "fastembed[:model]",
# "ollama[:model][|url]", "openai:<base_url>|<model>".
with Mark.local(".", embedder="ollama:nomic-embed-text-v2-moe") as mark:
    mark.memory.block("project").write("The API framework is FastAPI.")
    print(mark.memory.retrieve("Which web framework do we use?").as_text())
```

`"auto"` never downloads a model or contacts a server; Ollama and
OpenAI-compatible servers are used only when named. `MARK_EMBEDDER` sets the
spec for every store opened without `embedder=` (including the CLI), and
`MARK_EMBEDDER_API_KEY` supplies an OpenAI-compatible key. Provider classes
(`OllamaEmbeddingProvider`, `FastEmbedProvider`, `Model2VecProvider`,
`OpenAICompatibleEmbeddingProvider`) are in `mark.embeddings`.

Hybrid fusion adapts to the embedder: lexical embedders keep rank fusion, and
semantic models use score fusion so the vector channel's best paraphrase match
is not outranked by a stray keyword. MARK records each embedder's fingerprint,
so switching models re-embeds existing memory once instead of mixing vector
spaces.

### 12. Image memory

`memory.images` remembers image files by path and SHA-256 (never their bytes),
links them to the current facts of their subjects, finds them by caption, and
checks a frame against those facts with a describer you supply.

```python
from pathlib import Path

from mark import Mark

Path("scene5.png").write_bytes(b"stand-in for a generated frame")

with Mark.local(".") as mark:
    story = mark.runtime.memory("story")
    story.facts.set("Elena", "scarf", "red")
    frame = story.images.remember("scene5.png", caption="Elena at the station", subjects=["Elena"])
    print(story.images.find("Elena station", k=1)[0].record.caption)

    describe = lambda path: {"scarf": "blue"}     # a vision model in real use
    report = story.images.check_consistency(frame, "Elena", describe=describe)
    print(report.ok, report.mismatches)           # False [{'attribute': 'scarf', 'expected': 'red', 'observed': 'blue', ...}]
    print(story.images.verify(frame).ok)          # True: the file still matches its hash
```

`mark.media.OpenAIVisionDescriber` uses a vision model on any
OpenAI-compatible server, and `ClipImageEmbeddingProvider` (`mark-sdk[vision]`)
adds search by image similarity.

### 13. Provenance blocks

`memory.blocks()` groups memories into blocks, and `memory.chain()` seals
them into a per-agent SHA-256 hash chain that can be verified later
(`verify`, `verify_chain`). `quarantine` removes a block's members from
recall without deleting them. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Middleware

Framework-neutral middleware in `mark.middlewares` wraps the runtime's
`store`, `observe`, and `retrieve` operations. None of it is active unless you
pass it in, for example
`Mark.local(".", middleware=[RecallMiddleware(expand=True), ObservabilityMiddleware()])`.

Available middleware: recall, observe, compression, query expansion, gap
healing, governance, lifecycle, observability, sync, trust bus, sandbox, and
media continuity. See [ARCHITECTURE.md](ARCHITECTURE.md) for what each does.

## Encryption

With `mark-sdk[crypto]`, `Mark.local(".", encryption=FernetEncryptionProvider(passphrase=...))`
encrypts memory content and fact values at rest. Tags, metadata, session ids,
fact subjects and attributes, graph labels, and embeddings stay in plaintext so
they can be filtered. Encrypted stores use vector search only. An encrypted
store opened without its key (or with the wrong one) raises an error instead
of being read or written. On POSIX, the `.mark/` directory MARK creates is
private (`0700`) and the database file is `0600`. See [SECURITY.md](SECURITY.md).

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
with and without memory, sessions, facts over time, `ContextWatch`, redacted
sync, briefings in a plain loop, visual continuity, and MARK as the context
window. A notebook walks through middleware, tools, and MCP.

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
