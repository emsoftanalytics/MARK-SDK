# MARK Python SDK Architecture

This document explains how the MARK SDK is organised and how data moves
through it. It describes version 0.2.0a9. For usage, start with the
[README](README.md).

MARK is a local-first memory runtime for agents. Everything described here
runs in-process against a local SQLite file. Remote behaviour enters only
through hooks and clients that the developer supplies.

## Design rules

- **Local by default.** The core needs no network, account, Docker, or API
  key. Its only required dependency is `pydantic`.
- **No LLM calls on the hot path.** Store and recall are deterministic. LLM
  extraction, query expansion, and compression are opt-in, and an LLM fact
  extractor runs on a background worker, never inside a write.
- **No implicit network.** Model servers (embedders, fact extractors,
  describers, chat endpoints) are contacted only when the caller names them.
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
| `mark.memory` | `MarkRuntime` (wiring; maintenance in `runtime_maintenance.py`), `MarkMemory` (per-agent facade), `SimpleMemory`, facts, blocks, lifecycle | `memory/runtime.py`, `memory/mark_memory.py` |
| `mark.memory.facade` | Mixins that make up `MarkMemory`: `write`, `write_guard`, `fact_extraction`, `retrieve`, `retrieve_async`, `timeline`, `observing`, `graph_ops`, `lifecycle`, `healing`, `views` | `memory/facade/__init__.py` |
| `mark.memory.timeline*` | Subject timelines: `Timeline`/`TimelineEvent` (`timeline.py`), collection (`timeline_build.py`), the two-clock ordering shared with chronological recall (`timeline_order.py`) | `memory/timeline.py` |
| `mark.store` | `LocalMemoryStore` (SQLite), `JsonMemoryStore`, session activity log | `store/sqlite.py` |
| `mark.store._sqlite` | Store mixins: `schema`, `ddl`, `connection`, `encryption_guard`, `fragments`, `fragment_reads`, `mentions`, `search`, `facts`, `fact_jobs`, `blocks`, `graph`, `events`, `rows`, `codec` | `store/_sqlite/__init__.py` |
| `mark.intelligence` | Retrieval pipeline, query understanding, temporal ordering, fusion (`retrieval_fusion.py`, `fusion_scoring.py`), fact relevance, reranking, extraction, compression, query expansion | `intelligence/retrieval_pipeline.py` |
| `mark.intelligence.fact_extraction` | `FactCandidate`, the `FactExtractor` protocol, `RulePack` and `BUILTIN_PACKS`, `PatternFactExtractor`, `LLMFactExtractor`, `FactExtractionService` and its background worker | `intelligence/fact_extraction/__init__.py` |
| `mark.index` | In-memory cosine `VectorIndex` with pure-Python and numpy backends | `index/vector.py` |
| `mark.embeddings` | Embedding providers, the embedding cache, and `auto_embedder` / `resolve_embedder` for spec strings (`auto.py`) | `embeddings/__init__.py` |
| `mark.context` | `ContextWatch` and `Briefing` (`watch*.py`; rendering in `watch_render.py`, message lists and `inject` in `watch_messages.py`), `ContextBudget` (`budget.py`), `split_history` (`history.py`), `MarkChat` (`loop.py`), `ContextBuilder` | `context/watch.py` |
| `mark.media` | Image memory: `ImageMemory`, records, caption and CLIP search, a separate image-vector table, consistency checks, `OpenAIVisionDescriber` | `media/memory.py` |
| `mark.cli`, `mark._cli` | The `mark` command, `mark serve`, and its client | `cli.py`, `_cli/server.py` |
| `mark.adapters` | `LocalMarkBackend`, LangChain middleware and tools, MCP server, LangGraph state | `adapters/backend.py` |
| `mark.middlewares` | Opt-in middleware batteries and packaged agent skills | `middlewares/base.py` |
| `mark.agent`, `mark._agent` | `MarkAgent`: wraps a callable or LLM with memory, policies, and skills | `agent.py` |
| `mark.plasticity` | Decay, Hebbian reinforcement, pruning | `plasticity/__init__.py` |
| `mark.security` | Encryption providers, content hashing, redaction | `security/encryption.py` |
| `mark.errors` | Exception hierarchy, including the store-encryption errors | `errors.py` |
| `mark._private_fs` | Creates `.mark/` (0700) and database files (0600) on POSIX | `_private_fs.py` |
| `mark.types` | Pydantic data types: fragments, blocks, graph, gap reports | `types/memory.py` |
| `mark.policies`, `mark.plugins`, `mark.rl` | Policy registry, plugin hooks, reinforcement-learning data collection | each `__init__.py` |

## Main objects

```text
Mark.local(project_path)                      mark.runtime.Mark
 ├─ .memory   SimpleMemory                    blocks + retrieve, agent "__mark__"
 ├─ .runtime  MarkRuntime                     wiring, lifecycle, buses, tracer
 │    ├─ store      LocalMemoryStore          SQLite, schema v5
 │    ├─ embedder   EmbeddingProvider         default HashEmbeddingProvider
 │    ├─ pipeline   RetrievalPipeline         hybrid recall, embedder-aware fusion
 │    ├─ fact_extraction  FactExtractionService   when fact_extractor= is set
 │    └─ .memory(agent_id) -> MarkMemory      per-agent facade
 │          ├─ store_sync / store_many_sync / observe   (extract_facts=)
 │          ├─ retrieve_sync / retrieve_many_sync / retrieve_many -> RetrievalResult
 │          ├─ .timeline(subject, attribute) -> Timeline
 │          ├─ .facts  FactsView              bi-temporal facts
 │          ├─ .images ImageMemory            image memory (path + sha256)
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
`blocks`, `block_nodes`, `block_links`, `facts`, `fact_jobs`,
`fact_extractions`, `session_events`, and `schema_info`. An FTS5 table indexes
fragment content for keyword search. Image memory adds `mark_image_vectors`
on first use.

- **Schema version 5** added `fact_jobs` (the durable extraction queue) and
  `fact_extractions` (the ledger that makes re-extraction idempotent);
  version 4 added `facts`. Stores upgrade in place at open: `ddl.py` holds
  per-version migrations (`_MIGRATIONS`, `_COLUMN_MIGRATIONS`) and
  `schema.py` applies them in one transaction.
- **Concurrency.** File databases use one connection per thread in WAL mode;
  switching a new database to WAL is retried when several processes open it
  at once. Every write runs in `_writing`: one locked transaction that rolls
  back on error. The in-memory store serialises all access.
- **Permissions.** `_private_fs.py` creates the `.mark/` directory 0700 and
  the database file 0600 on POSIX (SQLite's `-wal` and `-shm` files inherit
  that). Existing paths keep their permissions.
- **Integrity.** Each fragment stores a content hash. A fragment that fails
  verification on read is not returned.
- **Encryption.** With an encryption provider, fragment content and fact
  values are encrypted. Content hashes become keyed HMACs, and the FTS5 index
  is not built, so encrypted stores use vector search only.
  `encryption_guard.py` records in `schema_info` whether the store is
  encrypted, plus an encrypted canary. Opening an encrypted store without a
  key raises `EncryptionRequiredError`, a wrong key `EncryptionKeyError`, and
  a key for a non-empty plaintext store `PlaintextStoreError`; an empty
  plaintext store may become encrypted.
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
  ▼
facade/fact_extraction.py   if extraction is on: re-read the stored fragments,
                            then extract inline (patterns) or enqueue a job (LLM)
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
| Index sync | `retrieval_index.py` | Embed fragments missing from the vector index. Re-embed once if the embedder fingerprint changed. |
| Channels | `retrieval_fusion.py`, `fusion_scoring.py` | Vector search and FTS5 (BM25) keyword search over the allowed ids, fused by Reciprocal Rank Fusion for lexical embedders or by score fusion for semantic ones (`FusionConfig`). |
| Expansion and rerank | `graph_expansion.py`, `reranker.py` | Pull in graph neighbours with damped scores, penalise contradicted fragments, apply bounded priors (importance, confidence). |
| Temporal order | `temporal_order.py`, `query_understanding.py` | For "now" or "original" questions, reorder records about the asked subject by event time. Same-day events keep their order. |
| Feedback | `retrieval_feedback.py` | Record access. Reinforce importance only for results with real query evidence (`ReinforceMode`). |
| Reporting | `retrieval_reporting.py`, `fact_relevance.py` | Attach the facts for the subjects in the query that answer it (the attribute asked about, the role a who-question names, or all for an open question), list `missing_anchors`, and build the gap report. |
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

`retrieve_many_sync` (and the async `retrieve_many`) runs several queries with
shared scope filtering and index sync. `order="chronological"` re-sorts the
same results oldest first with the two-clock rule described under
[Timelines](#timelines).

**Fusion.** `FusionConfig(mode="auto", lexical_weight=0.1,
anchor_lexical_weight=0.7)` (`retrieval_policy.py`) chooses the strategy from
the embedder's `semantic` flag. The hash embedder is lexical, so its vector
channel and BM25 vote by rank (RRF). With a semantic model, RRF flattened the
vector channel's margin and a candidate sharing one stray word with the query
could beat the true paraphrase, so score fusion combines each candidate's
cosine (relative to the best hit) with query-term coverage. Coverage gets the
larger share only when the query names an identifier or name found in the
candidate. Set it with `RetrievalPipeline(fusion=...)` or
`runtime.pipeline.configure_fusion(...)`. `benchmarks/embeddings/` holds the
evaluation and tuning scripts. On its 60-query set, hybrid hit@1 rose from
44 to 51 (potion-base-8M), 45 to 54 (potion-retrieval-32M), 48 to 59
(bge-small), and 49 to 58 (MiniLM); the hash embedder is unchanged at 36.

**Embedder selection.** `resolve_embedder` (`embeddings/auto.py`) turns the
`embedder=` argument into a provider: an instance is used as is, a spec
string (`"auto"`, `"hash[:dim]"`, `"model2vec[:model]"`, `"fastembed[:model]"`,
`"ollama[:model][|url]"`, `"openai:<url>|<model>"`) is built, and `None` means
the hash embedder unless `MARK_EMBEDDER` is set. `"auto"` picks a Model2Vec or
FastEmbed model that is already cached and otherwise falls back to hash; it
never downloads a model or probes a server.

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

## Fact extraction

`intelligence/fact_extraction/` turns stored text into facts. It runs only when
the runtime has a `fact_extractor` (`"patterns"` or any object with
`extract(text, *, occurred_at=None) -> list[FactCandidate]`) or a write passes
`extract_facts=True`.

```text
store_sync / store_many_sync completes (middleware, governance, redaction ran)
  ▼
FactExtractionMixin._extract_stored   re-read fragments by id, skip quarantined ones
  ▼
FactExtractionService.submit
  ├─ inline extractor (PatternFactExtractor)   extract now → facts table
  └─ background extractor (LLMFactExtractor)   enqueue fact_jobs row → worker thread
                                               claims with a lease, extracts, settles
```

- **Pattern packs.** A `RulePack` is data: regular expressions with
  `{S}`/`{V}`/`{A}` placeholders, plus attribute mappings. The built-in
  packs cover generic English for status, ownership ("owned by", "owns"),
  assignment, membership, changes ("moved from A to B"), present-tense
  descriptions (port, cloud region, placement, location, reporting line,
  datastore), and key-value lines. Vague verbs fire only when the value says
  what it is: "uses" needs a known database product, "runs in" a cloud-region
  id or an attribute noun ("runs in production" yields nothing). Text is split
  into clauses (`;`, `, `, and "and" before a verb); a clause that starts with
  a verb borrows the previous clause's subject within the same sentence.
  Questions, conditionals, negations, and intentions are skipped,
  and a leading `[YYYY-MM-DD]` stamp or `occurred_at` dates the fact.
- **Queue.** `fact_jobs` has one row per (fragment, extractor). Failed jobs are
  retried up to `max_attempts`. A job whose lease expired (the process died)
  is claimed again, and jobs left at shutdown run on the next start.
  `memory.facts.extract_pending()` drains the queue in the calling thread.
- **Idempotence.** `fact_extractions` records a hash of (text, occurrence,
  subject, attribute, value, valid_from), so re-ingesting a log never
  appends duplicate values. Extracted facts carry `source_fragment_id` and
  `metadata` (`extracted_by`, `rule`, `confidence`).
- **Never fails a write.** Extraction errors are logged and the write stands.

On the retention benchmark's large profile with the generic packs, precision
was 1.0 for every target attribute. Recall was 0.86 to 1.0 for owner, port,
region, database, team, city, and status, and 0.63 for ticket assignee,
because lines that state several changes in one clause are missed.

## Timelines

`memory.timeline(subject, attribute=None, limit=, since=, until=)`
(`memory/timeline_build.py`) returns every event about one subject, with no
top-k cap:

- **Collect.** `store/_sqlite/mentions.py` pages through FTS5 phrase
  candidates (all retrievable rows for encrypted stores). The builder keeps
  the memories that contain the subject as a whole anchor, so `titan` never
  matches `titanium`, and adds the subject's fact history. With an
  attribute, a memory must also mention the attribute.
- **Deduplicate.** A fact extracted from a collected memory is folded into
  that memory's event (`fact_id`).
- **Order** (`timeline_order.py`, shared with `order="chronological"`). Dated
  events are ordered by declared time (`occurred_at`, a date stamp, or
  `valid_from`), then storage order. An undated record stored after every
  dated one goes last. Other undated records cannot be placed: they follow
  the dated ones, labelled, and never count as latest. Storage time is never
  compared with declared time.

`Timeline.latest()`, `as_text()`, `as_context()` (escaped for a prompt) and
`to_dict()` render it; `mark timeline` uses the same builder.

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
                  (or budget_tokens, see below)
     corrections  "X is V now; W is outdated"
     unknown      anchors not in memory ("say so rather than guessing")
     usage        BudgetUsage when a token budget applies
```

A subject only the model named (not the user or a tool) is not reported as
"Not in memory"; it appears once memory confirms it. Loops that store each
turn call `watch.refresh()` so subjects first seen before they were stored
are fetched again.

**Any loop.** `observe_messages(messages)` (`watch_messages.py`) takes an
OpenAI-style or LangChain message list and feeds only the messages not seen
before, routing assistant tool calls and tool results. `inject(messages,
briefing, budget_tokens=None)` returns a new list with the briefing in the
system message and replaces an earlier `<mark_briefing>` block.
`ContextWatch.for_small_models()` uses smaller limits (`line_chars` 120, three
subjects, one record each) and a 175-token budget.

**Token budget.** `ContextBudget(max_tokens, fill=False, token_counter=None)`
(`budget.py`) is a hard cap. `watch_render.py` turns a snapshot into
`BudgetItem` lines tagged with a section, and the budget keeps whole lines by
section priority (corrections > unknown > facts > records > linked >
related) until the rendered text, wrapper included, fits. Only the first line
of the top section may be cut short. Kept lines stay in their original order.
With `fill=True`, producers lift their count limits and fill up to the cap.
The default counter, `ceil(chars / 4)`, is an estimate; pass a tokenizer
(`str -> int`) for an exact cap.

Anchors decay after `ttl_turns` turns without a mention. A failed prefetch is
logged and dropped; it never surfaces in the agent's turn. `briefing()` does
not block on search unless `wait` is given.

`MarkAgentMiddleware` creates one watch per agent run when `anticipate=True`
and the backend exposes `memory_for()` (as `LocalMarkBackend` does). Before
each model call it feeds the watch the new messages and appends the briefing
to the `[MARK project memory]` block it injects
(`adapters/langchain/_middleware_watch.py`). With `budget_tokens=`, the
briefing, retrieved memories, and any reasoning plan become one list of
budget items under a single cap (`_middleware_context.py`).

## Context replacement

With `MarkAgentMiddleware(history="mark", keep_last=6)` or `MarkChat`, MARK
replaces the conversation history instead of adding to it:

```text
messages ─ split_history(keep_last) ─┬─ kept: system + last keep_last messages
                                     └─ dropped: stored in MARK first
model sees: system (+ budgeted briefing) + kept
```

`split_history` (`context/history.py`) always keeps system messages and the
current user turn, never starts a window between a tool call and its results,
and drops orphaned tool results. The middleware stores dropped messages before
the model call and drops a message only after its write succeeded; if
writing fails it sends the full history for that call. Graph state is not
changed, so the checkpointer keeps the full transcript. `MarkChat`
(`context/loop.py`) is the same loop without a framework: it calls any
OpenAI-compatible `/chat/completions` endpoint with the standard library and
stores each exchange before the window slides.

## Image memory

`memory.images` (`media/memory.py`) stores each image as a fragment tagged
`mark:image` whose metadata holds the resolved path, SHA-256, size, subjects,
and the ids of the subjects' current facts. Image bytes are never stored.
`find` uses CLIP vectors from an `ImageEmbeddingProvider` when one is set,
kept in `mark_image_vectors` apart from text embeddings and keyed by
fingerprint and file hash. Otherwise it runs normal recall over captions.
`verify` re-hashes files. `check_consistency` compares what a describer (for
example `OpenAIVisionDescriber`) reports with the expected facts.

## Integrations

| Surface | Module | Notes |
| --- | --- | --- |
| LangChain middleware | `adapters/langchain/middleware.py` and `_middleware_*.py` mixins | Retrieval before model calls, briefing, observation of model and tool output. Sync and async. Per-run state. |
| LangChain tools | `adapters/langchain/tools.py` | `mark_retrieve`, `mark_write`, `mark_observe`. |
| MCP server | `adapters/mcp/__init__.py`, `_briefing.py` | Same three tools over stdio. `allowed_agent_ids` restricts namespaces. `briefing=True` adds `mark_briefing` and `mark://briefing/{session_id}`, one watch per (agent, session) in a bounded LRU. Supports `mcp` 1.x and 2.x. |
| Reference chat loop | `context/loop.py` | `MarkChat` for any OpenAI-compatible endpoint. |
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

This package contains the local runtime, storage, retrieval, facts, fact
extraction, timelines, context watch, image memory, CLI, middleware, and
optional framework adapters. Hosted
infrastructure is not part of it (MARK Cloud — coming soon). Adapters are thin
glue over public APIs and must not move transport or account logic into the
core.
