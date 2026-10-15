# MARK Python SDK Architecture

This document explains how the MARK SDK is organised and how data moves
through it. It describes version 0.2.0a8. For usage, start with the
[README](README.md).

MARK is a local-first memory runtime for agents. Everything described here
runs in-process against a local SQLite file. Remote behaviour enters only
through hooks and clients that the developer supplies.

## Design rules

- **Local by default.** The core needs no network, account, Docker, or API
  key. Its only required dependency is `pydantic`.
- **No LLM calls on the hot path.** Store and recall are deterministic. LLM
  extraction, query expansion, and compression are opt-in.
- **Filter before scoring.** Agent, session, tag, block, and state filters are
  applied in SQL before any ranking, so other namespaces never take result
  slots.
- **Nothing is silently lost.** Superseded facts stay in history, retracted
  facts and rejected writes are kept for audit, and quarantine hides memory
  without deleting it.
- **Small modules.** Each module has one responsibility, a module docstring,
  and at most 300 lines. Large classes are composed from mixins.

## Module map

| Package | Responsibility | Start reading at |
| --- | --- | --- |
| `mark` (`__init__.py`) | Public names, loaded lazily on first access so `import mark` stays fast | `__init__.py` |
| `mark.runtime` | `Mark`: the high-level entry point (`Mark.local()`) | `runtime.py` |
| `mark.memory` | `MarkRuntime` (wiring), `MarkMemory` (per-agent facade), `SimpleMemory`, facts, blocks, lifecycle | `memory/runtime.py`, `memory/mark_memory.py` |
| `mark.memory.facade` | Mixins that make up `MarkMemory`: `write`, `write_guard`, `retrieve`, `observing`, `graph_ops`, `lifecycle`, `healing`, `views` | `memory/facade/__init__.py` |
| `mark.store` | `LocalMemoryStore` (SQLite), `JsonMemoryStore`, session activity log | `store/sqlite.py` |
| `mark.store._sqlite` | Store mixins: `schema`, `ddl`, `connection`, `fragments`, `fragment_reads`, `search`, `facts`, `blocks`, `graph`, `events`, `rows`, `codec` | `store/_sqlite/__init__.py` |
| `mark.intelligence` | Retrieval pipeline, query understanding, temporal ordering, reranking, extraction, compression, query expansion | `intelligence/retrieval_pipeline.py` |
| `mark.index` | In-memory cosine `VectorIndex` with pure-Python and numpy backends | `index/vector.py` |
| `mark.embeddings` | Embedding providers and the embedding cache | `embeddings/__init__.py` |
| `mark.context` | `ContextWatch` and `Briefing` (anticipatory context), `ContextBuilder` | `context/watch.py` |
| `mark.cli`, `mark._cli` | The `mark` command, `mark serve`, and its client | `cli.py`, `_cli/server.py` |
| `mark.adapters` | `LocalMarkBackend`, LangChain middleware and tools, MCP server, LangGraph state | `adapters/backend.py` |
| `mark.middlewares` | Opt-in middleware batteries and packaged agent skills | `middlewares/base.py` |
| `mark.agent`, `mark._agent` | `MarkAgent`: wraps a callable or LLM with memory, policies, and skills | `agent.py` |
| `mark.plasticity` | Decay, Hebbian reinforcement, pruning | `plasticity/__init__.py` |
| `mark.security` | Encryption providers, content hashing, redaction | `security/encryption.py` |
| `mark.types` | Pydantic data types: fragments, blocks, graph, gap reports | `types/memory.py` |
| `mark.policies`, `mark.plugins`, `mark.rl` | Policy registry, plugin hooks, reinforcement-learning data collection | each `__init__.py` |

## Main objects

```text
Mark.local(project_path)                      mark.runtime.Mark
 ├─ .memory   SimpleMemory                    blocks + retrieve, agent "__mark__"
 ├─ .runtime  MarkRuntime                     wiring, lifecycle, buses, tracer
 │    ├─ store      LocalMemoryStore          SQLite, schema v4
 │    ├─ embedder   EmbeddingProvider         default HashEmbeddingProvider
 │    ├─ pipeline   RetrievalPipeline         hybrid recall
 │    └─ .memory(agent_id) -> MarkMemory      per-agent facade
 │          ├─ store_sync / store_many_sync / observe
 │          ├─ retrieve_sync / retrieve_many_sync -> RetrievalResult
 │          ├─ .facts  FactsView              bi-temporal facts
 │          └─ .blocks() / .chain()           BlockGraph / BlockChain
 ├─ .wrap_agent(fn) -> MarkAgent
 └─ .policies, .skills, .world_bible
```

`MarkRuntime.local(store_path=...)` can be used without `Mark` when an
application only needs the engine. The CLI does this.

## Memory model

```text
fragment → one memory: content, agent, session, tags, importance, confidence, state
node     → a named concept (character, service, location) linked to fragments
edge     → a typed, weighted relation between two nodes
block    → a bounded group of fragments, nodes, and edges
chain    → sealed blocks linked by hash, per agent
fact     → (subject, attribute) → value, with a validity interval and history
```

- **Agent namespace.** Every fragment and fact belongs to one `agent_id`.
  `mark.memory` and all adapters default to `"__mark__"`.
- **Sessions** partition a namespace (`session_id`, or `session_prefix` for a
  group such as `"season-01/"`).
- **States.** Fragments move through `MemoryState` (unverified, promoted,
  quarantined, and others). Only retrievable states are returned by recall.
- **Event time.** `occurred_at` (or a leading `[YYYY-MM-DD]` stamp in the
  content) records when an event happened, separately from when it was stored.

## Storage

`LocalMemoryStore` (`store/sqlite.py`) is one class composed from the mixins
in `store/_sqlite/`. Tables: `fragments`, `documents`, `nodes`, `edges`,
`blocks`, `block_nodes`, `block_links`, `facts`, `session_events`, and
`schema_info`. An FTS5 table indexes fragment content for keyword search.

- **Schema version 4** added the `facts` table. Stores upgrade in place at
  open: `ddl.py` holds per-version migrations (`_MIGRATIONS`,
  `_COLUMN_MIGRATIONS`) and `schema.py` applies them in one transaction.
- **Concurrency.** File databases use one connection per thread in WAL mode.
  Every write runs in `_writing`: one locked transaction that rolls back on
  error. The in-memory store serialises all access.
- **Integrity.** Each fragment stores a content hash. A fragment that fails
  verification on read is not returned.
- **Encryption.** With an encryption provider, fragment content and fact
  values are encrypted. Content hashes become keyed HMACs, and the FTS5 index
  is not built, so encrypted stores use vector search only. The store records
  whether it is encrypted and refuses to open with a missing or wrong key.
- **Vectors.** Embeddings are stored with each fragment. `VectorIndex` is an
  in-memory index rebuilt lazily from the store. The default backend uses pure
  Python and switches to numpy once an index is large enough, if numpy is
  installed.

## Write path

```text
mark.memory.block("b").write(text)  or  memory.store_sync(text, occurred_at=...)
  │
  ├─ middleware stack, if any ("store" operation: governance, redaction, sync ...)
  ▼
facade/write_guard.py   build the MemoryFragment, normalise occurred_at,
                        run the consolidation gate; rejected writes are stored
                        as QUARANTINED for audit and never indexed
  ▼
RetrievalPipeline.index_fragment(s)   embed (store_many_sync: batches of 64)
  ▼
LocalMemoryStore.store / store_many   fragment row + FTS5 row, one transaction
  ▼
VectorIndex.add   the fragment is retrievable from the next recall
```

`observe()` follows the same path and can add graph nodes and edges through
deterministic or LLM extraction when `auto_structure=True` is enabled by
middleware. Fact writes (`memory.facts.set`) go straight to the `facts` table
and do not change which fragments are retrievable.

## Recall path

`RetrievalPipeline` (`intelligence/retrieval_pipeline.py`) runs these stages:

| Stage | Module | What happens |
| --- | --- | --- |
| Scope | `retrieval_index.py`, `store/_sqlite/search.py` | Compute the allowed fragment ids in SQL (agent, session, tags, block, state, TTL). Cached by `write_version`. |
| Index sync | `retrieval_index.py` | Embed fragments missing from the vector index. Re-embed if the embedder fingerprint changed. |
| Channels | `retrieval_fusion.py` | Vector search and FTS5 (BM25) keyword search over the allowed ids, fused with Reciprocal Rank Fusion. |
| Expansion and rerank | `graph_expansion.py`, `reranker.py` | Pull in graph neighbours with damped scores, penalise contradicted fragments, apply bounded priors (importance, confidence). |
| Temporal order | `temporal_order.py`, `query_understanding.py` | For "now" or "original" questions, reorder records about the asked subject by event time. Same-day events keep their order. |
| Feedback | `retrieval_feedback.py` | Record access. Reinforce importance only for results with real query evidence (`ReinforceMode`). |
| Reporting | `retrieval_reporting.py` | Attach facts for the subjects in the query, list `missing_anchors`, and build the gap report. |
| Compression | `retrieval_pipeline.py` | Optional, when a compressor is configured and requested. |

The result is a `RetrievalResult` (`intelligence/retrieval_result.py`) with
`fragments`, `scores`, calibrated `relevance`, per-channel hit counts,
`temporal_intent`, `facts`, `missing_anchors`, and `gap_report`.
`as_context()` renders it for a model and escapes memory content so it cannot
break out of the context block.

**Gap reports** come from calibrated query evidence, not from ranking scores.
A query that names a distinctive identifier (a service, ticket, or person)
that occurs nowhere in memory gets a high-severity gap and appears in
`missing_anchors`.

`retrieve_many_sync` runs several queries with shared scope filtering and
index sync.

## Facts

`FactsView` (`memory/facts.py`) is the API; `store/_sqlite/facts.py` is the
storage. Each fact row has two time axes:

- **Valid time** (`valid_from`, `valid_to`): when the value was true.
  `valid_to` is derived from the next value on the same timeline, so a
  late-arriving update is placed correctly in history.
- **Transaction time** (`recorded_at`, `retracted_at`): when the store learned
  or withdrew the value. Retracted rows are kept for audit and ignored by
  queries.

Several values with the same `valid_from` form a sequence; the last is
current. `correction=True` replaces the value(s) recorded for that time.
Subjects and attributes are matched by normalised keys, so `Atlas-Core` and
`atlas core` are the same subject.

## Briefing path

`ContextWatch` (`context/watch.py`, `watch_prefetch.py`, `watch_support.py`)
prepares memory before the agent asks:

```text
watch.observe(text, role)          user / assistant / tool / system
  │  extract anchors: identifiers, names, known fact subjects
  ▼
background prefetch (one worker thread)
  │  retrieve memories and facts for each new anchor and for the task
  │  follow links: subjects named in an anchor's memories are prefetched too
  │  check assistant claims against fact history → corrections
  ▼
watch.briefing(wait=...) -> Briefing
     text         <mark_briefing> block grouped by subject, within budget_chars
     corrections  "X is V now; W is outdated"
     unknown      anchors not in memory ("say so rather than guessing")
```

Anchors decay after `ttl_turns` turns without a mention. A failed prefetch is
logged and dropped; it never surfaces in the agent's turn. `briefing()` does
not block on search unless `wait` is given.

`MarkAgentMiddleware` creates one watch per agent run when `anticipate=True`
and the backend exposes `memory_for()` (as `LocalMarkBackend` does). Before
each model call it feeds the watch the new messages and appends the briefing
to the `[MARK project memory]` block it injects
(`adapters/langchain/_middleware_watch.py`).

## Integrations

| Surface | Module | Notes |
| --- | --- | --- |
| LangChain middleware | `adapters/langchain/middleware.py` and `_middleware_*.py` mixins | Retrieval before model calls, briefing, observation of model and tool output. Sync and async. Per-run state. |
| LangChain tools | `adapters/langchain/tools.py` | `mark_retrieve`, `mark_write`, `mark_observe`. |
| MCP server | `adapters/mcp/__init__.py` | Same three tools over stdio. `allowed_agent_ids` restricts namespaces. Supports `mcp` 1.x and 2.x. |
| Backend protocol | `adapters/backend.py` | `MarkBackend` protocol; `LocalMarkBackend` runs blocking work off the event loop. |
| Command line | `cli.py`, `_cli/` | See below. |
| Agent wrapper | `agent.py`, `_agent/` | `Mark.wrap_agent()` for plain callables and LLM functions. |

### Command line and `mark serve`

`mark.cli` parses arguments and prints results. `mark._cli` holds the rest:

- `config.py`: database path (`--db`, `$MARK_DB`, `./.mark/memory.db`) and the
  sibling files `<db>.server.json` (state) and `<db>.server.lock`.
- `engine.py`: `_Engine` runs commands against one database. It is the only
  place the CLI imports MARK.
- `server.py`: `mark serve`, a loopback-only HTTP server that keeps an
  `_Engine` open.
- `client.py`: finds a live server, authenticates it, and forwards the
  command; otherwise runs the command in-process.

When a server answers, the client never imports MARK, so a command costs one
small Python start-up plus the search. The security model is described in
[SECURITY.md](SECURITY.md).

## Lifecycle and plasticity

| `MarkRuntime` method | Role |
| --- | --- |
| `working_memory(agent_id)` | TTL-bound short-term memory |
| `consolidate(agent_id)` | Governed promotion of working memory |
| `deduplicate(agent_id)` | Merge near-duplicate fragments, keeping provenance |
| `prune(agent_id)` | Decay, fragment pruning, edge dissolution |
| `run_cycle(agent_id)` | Consolidate, deduplicate, and prune |
| `integrity_check()` | SQLite health check |

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("coder")
    memory.store_sync("FastAPI handles routing.")
    print(mark.runtime.run_cycle("coder"))
    print(mark.runtime.integrity_check())    # [] when healthy
```

Maintenance never modifies fragments that belong to sealed blocks.
Promoted memories are exempt from decay.

## Blocks and provenance

- A block groups fragments, nodes, and edges. Node membership is many-to-many,
  so a canonical entity is never duplicated across blocks.
- Blocks link to other blocks (`BlockLink`) and can be traversed in both
  directions.
- `BlockChain.seal` hashes a block's immutable member fields with SHA-256 and
  links it to the agent's previous sealed block. Mutable fields such as
  importance and access counts are excluded, so reinforcement never breaks a
  seal.
- `verify()` reports corrupted, missing, or injected members;
  `verify_chain()` checks the whole chain.
- `quarantine()` removes a block's members from recall and traversal without
  deleting them.

## Middleware

`mark.middlewares` wraps the public `store`, `observe`, and `retrieve`
operations of `MarkMemory`. Nothing is active until it is passed to
`Mark.local(..., middleware=[...])` or `runtime.use(...)`.

```text
agent or framework → adapter → MiddlewareStack → MarkMemory → store / pipeline
```

| Middleware | Purpose |
| --- | --- |
| `RecallMiddleware` | Default retrieval options: compression, expansion, escalation |
| `ObserveMiddleware` | Default observation source, tags, and metadata |
| `CompressionMiddleware` | Bind a compressor and compress by default |
| `QueryExpansionMiddleware` | Bind a query expander and expand by default |
| `GapHealingMiddleware` | Opt-in gap healing through a web lookup skill |
| `GovernanceMiddleware` | Governance gates; quarantine rejected writes |
| `LifecycleMiddleware` | Run maintenance after writes |
| `ObservabilityMiddleware` | Trace events around middleware operations |
| `SyncMiddleware` | Prepare redacted sync envelopes; upload only through a caller-supplied client |
| `TrustBusMiddleware` | Publish writes to the trust-aware bus |
| `SandboxMiddleware` | Sandbox execution; blocked unless a backend is supplied |
| `MediaContinuityMiddleware` | Defaults for creative continuity (characters, locations, objects) |

When middleware is registered, batch operations fall back to one call per
item so no item skips governance or redaction.

## Sync, observability, and the trust bus

- `CloudSync.prepare_delta()` builds a `SyncDelta` of redacted fragments and
  events. `CloudSync.sync()` forwards it only to a client object the caller
  supplies. The SDK implements no remote transport.
- `LocalTracer` (`MarkRuntime.tracer()`) records local events; a session view
  is available through `MarkRuntime.session_trace(session_id)`.
- `TrustAwareGlobalMemoryBus` shares messages between agents with publisher
  trust levels, subscriptions, snapshots, and optional signing. Messages are
  stored as global fragments and reload when the store is reopened.

## Package boundary

This package contains the local runtime, storage, retrieval, facts, context
watch, CLI, middleware, and optional framework adapters. Hosted
infrastructure is not part of it (MARK Cloud — coming soon). Adapters are thin
glue over public APIs and must not move transport or account logic into the
core.
