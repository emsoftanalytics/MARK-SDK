# MARK Python SDK Architecture

This document explains how the MARK SDK is organised and how data moves
through it. It describes version 0.2.0a10. For usage, start with the
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
- **Nothing is silently lost.** Superseded facts and memories stay in
  history, retracted facts and rejected writes are kept for audit, and
  quarantine hides memory without deleting it. Deletion happens only when
  asked for (`forget`, `facts.delete`), and then it cascades to everything
  derived from the deleted memory.
- **Safe by default.** Every write is validated, redacted, and screened for
  prompt injection before it is stored.
- **Small modules.** Each module has one responsibility, a module docstring,
  and at most 300 lines. Large classes are composed from mixins.

## Module map

| Package | Responsibility | Start reading at |
| --- | --- | --- |
| `mark` (`__init__.py`) | The curated public names (25), loaded lazily so `import mark` stays fast; former names import with a `DeprecationWarning` | `__init__.py` |
| `mark.runtime` | `Mark`: the high-level entry point (`Mark.local()`); store-wide data operations (`agents`, `export_jsonl`, `import_jsonl`, `vacuum`) in `runtime_data.py`, late-binding `configure_*` hooks in `runtime_config.py` | `runtime.py` |
| `mark.kinds` | Memory kinds (`decision`, `todo`, `bug`, `fix`, `preference`, `snippet`, `api`, `note`) and lifecycle tags; standard library only, so the CLI can use it | `kinds.py` |
| `mark.memory` | `MarkRuntime` (wiring, `shutdown`, `configure_user_subject`; maintenance in `runtime_maintenance.py`), `MarkMemory` (per-agent facade), `SimpleMemory` (`mark.memory`, verbs in `simple_verbs.py`), facts, blocks, timelines | `memory/runtime.py`, `memory/mark_memory.py` |
| `mark.memory.facade` | Mixins that make up `MarkMemory`: `verbs` (remember/recall and async twins), `write`, `write_batch` (`StoreReport`), `write_validation`, `write_guard`, `fact_extraction`, `retrieve`, `retrieve_async`, `revisions` (supersede/update/resolve), `lifecycle` (forget, get, list, count), `portability` (JSONL), `documents`, `timeline`, `observing`, `graph_ops`, `healing`, `views` | `memory/facade/__init__.py` |
| `mark.memory.timeline*` | Subject timelines: `Timeline`/`TimelineEvent` (`timeline.py`), collection (`timeline_build.py`), the two-clock ordering shared with chronological recall (`timeline_order.py`) | `memory/timeline.py` |
| `mark.documents` | `store_document` / `remember_file`: loaders (text, Markdown, code, HTML, PDF), structure-aware chunking, `DocumentRecord`, the `memory.documents` view | `documents/__init__.py` |
| `mark.store` | `LocalMemoryStore` (SQLite), `JsonMemoryStore`, session activity log | `store/sqlite.py` |
| `mark.store._sqlite` | Store mixins: `schema`, `ddl`, `connection`, `encryption_guard`, `fragments`, `fragment_reads`, `lifecycle` (cascading forget, secure erase, vacuum), `document_index`, `mentions`, `search`, `fts_text`/`fts_query` (two FTS5 tables, bounded keyword queries), `vector_snapshot`, `sketches`/`sketch_store` (encrypted timelines), `facts`, `facts_multi`, `fact_codec` (typed values), `fact_jobs`, `job_owner`, `blocks`, `graph`, `events`, `rows`, `codec` | `store/_sqlite/__init__.py` |
| `mark.intelligence` | Retrieval pipeline and its stages: `retrieval_index`, `index_bulk_load`, `retrieval_time` + `date_parsing`, `retrieval_fusion` + `fusion_scoring`, `graph_expansion`, `reranker`, `multi_hop`, `supersession`, `retrieval_ordering` + `temporal_order`, `retrieval_reporting` + `fact_relevance` + `query_anchors`, `passage_window`; plus extraction, compression and query expansion | `intelligence/retrieval_pipeline.py` |
| `mark.intelligence.fact_extraction` | `FactCandidate`, the `FactExtractor` protocol, `RulePack` and `BUILTIN_PACKS`, `PatternFactExtractor`, `LLMFactExtractor`, `FactExtractionService` and its background worker | `intelligence/fact_extraction/__init__.py` |
| `mark.index` | In-memory cosine `VectorIndex` with pure-Python and numpy backends, bulk loading (`vector_bulk.py`) | `index/vector.py` |
| `mark.embeddings` | Embedding providers (hash v3 by default), the embedding cache, Unicode-aware segmentation (`segments.py`), `auto_embedder` / `resolve_embedder` (`auto.py`), `embedder_info()` (`info.py`) | `embeddings/__init__.py` |
| `mark.context` | `ContextWatch` and `Briefing` (`watch*.py`), presets by model size (`presets.py`), provenance and trust of conversation memories (`provenance.py`), the session brief (`session_brief.py`), `ContextBudget` (`budget.py`), `split_history` (`history.py`), `MarkChat` (`loop.py`), `ContextBuilder` | `context/watch.py` |
| `mark.media` | Image memory (`ImageMemory`, records, caption and CLIP search, consistency checks, `OpenAIVisionDescriber`), file sniffing (`files.py`), transcribers for audio and video (`transcribers.py`) | `media/memory.py` |
| `mark.cli`, `mark._cli` | The `mark` command (`parser.py`, `output.py`, `engine.py`, `engine_ops.py`), `mark serve` and its client, `mark mcp` and `mark serve stop` (`extras.py`) | `cli.py`, `_cli/server.py` |
| `mark.adapters` | `LocalMarkBackend`, LangChain middleware and tools, the MCP server and its launcher (`python -m mark.adapters.mcp`), LangGraph state | `adapters/backend.py` |
| `mark.middlewares` | Opt-in middleware batteries and packaged agent skills (`mark-usage`, `mark-memory-middleware`, opt-in `mark-memory-compact`) | `middlewares/base.py` |
| `mark.agent`, `mark._agent` | `MarkAgent`: wraps a callable or LLM with memory, policies, and skills | `agent.py` |
| `mark.plasticity` | Decay, Hebbian reinforcement, pruning | `plasticity/__init__.py` |
| `mark.security` | The write policy (`write_policy.py`): secret and PII redaction (`redaction.py`, `secret_rules.py`, `pii_rules.py`) and prompt-injection screening (`injection.py`); encryption providers; content hashing | `security/write_policy.py` |
| `mark.errors` | Exception hierarchy: `MarkError`, store encryption and version errors, `InvalidInputError` / `InvalidMemoryError` / `MemoryTooLargeError`, `UnknownBlockError`, `MemoryNotFoundError`, `StoreClosedError`, `EmbedderUnavailableError` | `errors.py` |
| `mark._private_fs` | Creates `.mark/` (0700) and database and export files (0600) on POSIX | `_private_fs.py` |
| `mark.types` | Pydantic data types: fragments, blocks, graph, gap reports | `types/memory.py` |
| `mark.policies`, `mark.plugins`, `mark.rl` | Policy registry, plugin hooks, reinforcement-learning data collection | each `__init__.py` |

## Main objects

```text
Mark.local(folder or file.db, redact=, injection=, max_memory_bytes=, fact_extractor=, embedder=, encryption=)
 ├─ .memory   SimpleMemory                    the memory verbs + blocks, agent "__mark__"
 ├─ .runtime  MarkRuntime                     wiring, lifecycle, write policy, buses, tracer
 │    ├─ store      LocalMemoryStore          SQLite, schema v6
 │    ├─ embedder   EmbeddingProvider         default HashEmbeddingProvider (v3)
 │    ├─ pipeline   RetrievalPipeline         hybrid recall, embedder-aware fusion
 │    ├─ write_policy WritePolicy             redaction + injection screening
 │    ├─ fact_extraction  FactExtractionService   when fact_extractor= is set
 │    └─ .memory(agent_id) -> MarkMemory      per-agent facade
 │          ├─ remember / aremember, store_many_sync -> StoreReport, observe
 │          ├─ recall / arecall, retrieve_many_sync -> RetrievalResult
 │          ├─ supersede / update / resolve / forget / get / list / count
 │          ├─ brief() -> Briefing           session-start brief
 │          ├─ .timeline(subject, attribute) -> Timeline
 │          ├─ .facts  FactsView              bi-temporal, typed, multi-valued facts
 │          ├─ store_document / remember_file, .documents  DocumentsView
 │          ├─ .images ImageMemory            image memory (path + sha256)
 │          ├─ export_jsonl / import_jsonl / vacuum / embedder_info
 │          └─ .blocks() / .chain()           BlockGraph / BlockChain
 ├─ .agents(), .export_jsonl(), .import_jsonl(), .vacuum(), .embedder_info()
 ├─ .wrap_agent(fn) -> MarkAgent              callable, invoke, ainvoke
 └─ .policies, .skills, .world_bible
```

**Naming.** Plain verbs are synchronous; an `a` prefix marks the async twin
(`recall` / `arecall`). `mark.memory` and `runtime.memory(id)` share the
verbs, arguments and defaults (importance 0.5, confidence 1.0). The 0.2.0a9
names (`store_sync`, `retrieve_sync`, `retrieve`) remain as aliases.

**Rendering family.** Every result type a model may read (`RetrievalResult`,
`Timeline`, `Briefing`, `ContextBundle`) offers `as_context()` (escaped, for
a prompt), `as_text()` (for people and logs) and `to_dict()` (for JSON).

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
- **Kinds** are tags (`kind:todo`), so every tag filter scopes by kind with
  no schema change. **Lifecycle** is tags plus `metadata["lifecycle"]`:
  `supersede` and `update` write a new memory and tag the old one
  `mark:superseded`; `resolve` tags a TODO or bug `mark:resolved` (and
  superseded). Retrieval demotes both but keeps them findable for history
  questions; `list(open=True)` leaves them out.
- **Provenance.** `metadata["source"]` records where a memory came from
  (`user`, `assistant`, `tool`, `system`, `document`, `extracted`). Model
  replies stored by `MarkChat` and the LangChain middleware are low trust
  (importance 0.2, confidence 0.3, tag `mark:assistant`); questions are
  tagged `mark:question`. The reranker scales both by 0.6, and briefings never
  anchor on them.
- **Security findings** live in `metadata["mark_security"]` (`redacted`,
  `untrusted`, `injection`), with the tag `mark:untrusted` for memories that
  look like instructions.
- **Documents.** A document is a row in `documents` (title, source, SHA-256,
  encrypted text when encryption is on) plus ordinary chunk memories tagged
  `mark:document` that carry `document_id` and their heading path.

## Storage

`LocalMemoryStore` (`store/sqlite.py`) is one class composed from the mixins
in `store/_sqlite/`. Tables: `fragments`, `documents`, `nodes`, `edges`,
`blocks`, `block_nodes`, `block_links`, `facts`, `fact_jobs`,
`fact_job_owners`, `fact_extractions`, `session_events`, `vector_snapshots`,
`vector_snapshot_meta`, `vector_dirty`, and `schema_info`. Two FTS5 tables
index fragment content: an English one with the porter stemmer and a
multilingual one with no stemmer, CJK text segmented into bigrams, and emoji
and symbols kept as tokens (`fts_text.py`). Image memory adds
`mark_image_vectors` on first use; encrypted stores may add
`fragment_sketches` (opt-in).

- **Schema version 6** added the multilingual keyword table, vector
  snapshots and a partial index for TTL expiry; version 5 added `fact_jobs`
  and `fact_extractions`; version 4 added `facts`. Stores upgrade in place at
  open: `ddl.py` holds per-version migrations (`_MIGRATIONS`,
  `_COLUMN_MIGRATIONS`) and `schema.py` applies them in one transaction. A
  store stamped with a newer version than this build supports is refused
  before any write (`StoreVersionError`).
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
  in-memory index loaded lazily from the store, in bulk (`vector_bulk.py`,
  `index_bulk_load.py`). The default backend uses pure Python and switches to
  numpy once an index is large enough, if numpy is installed. Agents with at
  least 50,000 memories get a **vector snapshot** (`vector_snapshot.py`): the
  normalised vectors in a few large blobs, keyed by embedder fingerprint, so
  a large store reopens in about a second and a half at 100k memories
  instead of eleven. Writes record changed ids in `vector_dirty`, so a
  snapshot never serves a stale vector. Encrypted stores never write one.
- **Encrypted timelines.** Without an FTS index, an encrypted store finds a
  subject's mentions through keyed token sketches (`sketches.py`): a 512-bit
  keyed Bloom filter per memory, kept in process memory, so only candidate
  memories are decrypted. `persist_token_sketches()` keeps them on disk
  (with the leakage trade-off described in [SECURITY.md](SECURITY.md)).
- **Lifecycle** (`lifecycle.py`). `forget_fragment` deletes the row, its FTS
  rows and extraction jobs, and retracts the facts extracted from it;
  `secure=True` overwrites the content first, hard-deletes those facts and
  enables `secure_delete`; `vacuum()` merges the FTS index, checkpoints and
  truncates the WAL, and runs `VACUUM`.

## Write path

```text
memory.remember(text, kind=, occurred_at=, source=)   (also store_sync, observe, block writes)
  │
  ├─ middleware stack, if any ("store" operation: governance, sync ...)
  ▼
facade/write_validation.py   reject bad input up front: InvalidMemoryError, or
                             MemoryTooLargeError above max_memory_bytes (1 MB)
  ▼
mark.security.WritePolicy    redact secrets ([SECRET:<kind>]); screen for prompt
                             injection: tag mark:untrusted, or quarantine
  ▼
facade/write_guard.py        build the MemoryFragment (kind tag, provenance,
                             occurred_at), run the consolidation gate; rejected
                             writes are stored as QUARANTINED for audit, never indexed
  ▼
RetrievalPipeline.index_fragment(s)   embed (store_many_sync: every item before writing)
  ▼
LocalMemoryStore.store / store_many   fragment row + FTS5 rows, one transaction
  ▼
VectorIndex.add   the fragment is retrievable from the next recall
  ▼
facade/fact_extraction.py   if extraction is on: re-read the stored fragments,
                            then extract inline (patterns) or enqueue a job (LLM)
```

`store_many_sync` validates and embeds every item before it writes, commits
all of them in one transaction, and returns a `StoreReport`: the ids in input
order with `None` for skipped items, plus `skipped` (index and reason),
`stored`, `existing` and `ids`. `strict=True` raises instead of skipping, and
`idempotency_key=` makes a retried batch store nothing twice (`mark ingest`
sets one automatically). Documents (`store_document`, `remember_file`) are
chunked by `mark.documents` and written through the same path, then recorded
in `documents`.

`observe()` follows the same path and can add graph nodes and edges through
deterministic or LLM extraction when `auto_structure=True` is enabled by
middleware. Fact writes (`memory.facts.set`) go straight to the `facts` table
and do not change which fragments are retrievable.

## Recall path

`RetrievalPipeline` (`intelligence/retrieval_pipeline.py`) runs these stages:

| Stage | Module | What happens |
| --- | --- | --- |
| Scope | `retrieval_index.py`, `store/_sqlite/search.py` | Compute the allowed fragment ids in SQL (agent, session, tags and kinds, block, state, TTL). Cached by `write_version`. |
| Index sync | `retrieval_index.py`, `index_bulk_load.py` | Bulk-load stored vectors (or the agent's snapshot) and embed fragments missing from the index. Re-embed once if the embedder fingerprint changed. |
| Time | `retrieval_time.py`, `date_parsing.py` | Apply `since`/`until` and dates named in the query ("3rd of May", "last Tuesday", "Q3") to event time. |
| Channels | `retrieval_fusion.py`, `fusion_scoring.py`, `store/_sqlite/fts_query.py` | Vector search and keyword search over both FTS5 tables, fused by Reciprocal Rank Fusion for lexical embedders or by score fusion for semantic ones (`FusionConfig`). Above about 5,000 rows a keyword query requires its rarest terms and only ranks by the rest, and id lists are never sent to SQLite, so cost follows the rarest term rather than the store size. |
| Expansion and rerank | `graph_expansion.py`, `reranker.py` | Pull in graph neighbours with damped scores (any relation name), penalise contradicted fragments, scale low-trust assistant replies and questions by 0.6, apply bounded priors (importance, confidence). |
| Multi-hop | `multi_hop.py` | Names and identifiers in the best results that the query did not mention lead to a bounded second search, so 2-3 step chains ("who manages the owner of X?") reach the top results. |
| Supersession and order | `supersession.py`, `retrieval_ordering.py`, `temporal_order.py` | Free-text updates ("moved from", "no longer", "Correction:", "switched to", "used to") rank ahead of what they replace; `mark:superseded` and `mark:resolved` memories are demoted; "now" questions put the newest record about the subject first and "originally" questions the earliest. |
| Feedback | `retrieval_feedback.py` | Record access. Reinforce importance only for results with real query evidence (`ReinforceMode`). |
| Reporting | `retrieval_reporting.py`, `fact_relevance.py`, `query_anchors.py` | Attach the facts that answer the question (the attribute asked about, at most three per subject; "I/my" questions use the configured user subject; facts reached through matched memories when no subject is named), list `missing_anchors` (names and code identifiers in CamelCase, snake_case or dotted form that occur nowhere), and build the gap report. |
| Compression | `retrieval_ordering.py` | Optional, when a compressor is configured and requested. |

The result is a `RetrievalResult` (`intelligence/retrieval_result.py`) with
`fragments`, `scores`, calibrated `relevance` (the score shown in rendered
output and used by the gap report), per-channel hit counts,
`temporal_intent`, `facts`, `missing_anchors`, and `gap_report`.
`as_context()` renders it for a model, escapes memory content so it cannot
break out of the context block, labels untrusted memories, caps the echoed
query, and shows the best passage of a long memory instead of dropping it
(`passage_window.py`). A blank query returns nothing.

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

- **Typed values** (`fact_codec.py`). Text is stored as text, so older rows
  read back unchanged; numbers, booleans, lists and objects are stored as
  JSON and read back with their Python type. `Fact.text` is the rendering
  every briefing and recall result uses.
- **Multi-valued attributes** (`facts_multi.py`). `facts.add()` records one
  more current value (phone numbers, languages, team members) and
  `facts.remove()` ends one.
- **Scheduled values.** A value whose `valid_from` is in the future is not
  current until then; `facts.scheduled()` lists them.
- **Deletion.** `facts.retract()` withdraws a value and keeps it for audit;
  `facts.delete()` removes rows for good.
- **The user subject.** First-person statements ("I live in Porto") are facts
  about the user subject, `"user"` unless `runtime.configure_user_subject()`
  names it, and "I/my" questions look facts up under the same subject.

## Fact extraction

`intelligence/fact_extraction/` turns stored text into facts. It runs only when
the runtime has a `fact_extractor` (`"patterns"` or any object with
`extract(text, *, occurred_at=None) -> list[FactCandidate]`) or a write passes
`extract_facts=True`.

```text
remember / store_many_sync completes (validation, write policy, middleware ran)
  ▼
FactExtractionMixin._extract_stored   re-read fragments by id, skip quarantined ones
  ▼
FactExtractionService.submit
  ├─ inline extractor (PatternFactExtractor)   extract now → facts table
  └─ background extractor (LLMFactExtractor)   enqueue fact_jobs row → worker thread
                                               claims with a lease, extracts, settles
```

- **Coverage.** Besides changes and descriptions, the built-in packs read
  first-person statements, personal details and preferences, prices and
  measurements with units, `key: value`, `k=v`, JSON and YAML records,
  Markdown tables, numbered steps (including number words), active voice
  ("Priya manages Tomasz"), and developer notes (`DECISION:`, `TODO:`,
  `BUG:`, `FIX:`, "the test command is ...", "switched from A to B"). "No
  longer" retracts the current value and "Correction:" replaces a wrong one
  (`extract_changes()`). Writes report how many facts they produced
  (`memory.last_extracted`, `DocumentRecord.facts_extracted`).
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
- **Queue and job lifecycle.** `fact_jobs` has one row per (fragment,
  extractor). Workers claim jobs four at a time under a lease
  (`fact_job_lease=`, default 300 s) and record the claiming process in
  `fact_job_owners` (`job_owner.py`: host, boot id, pid and process start
  time). A process that opens the store requeues jobs whose owner is
  certainly gone (same host, process dead) at once instead of waiting for the
  lease; owners on other hosts fall back to the lease. Failed jobs are retried
  up to `max_attempts`. `shutdown(timeout=5.0)` lets the worker finish its
  current job within the timeout and returns unfinished claims to the queue.
  `memory.facts.extract_pending()` drains the queue in the calling thread and
  waits at most 30 s by default for jobs another worker already claimed.
- **Idempotence.** `fact_extractions` records a hash of (text, occurrence,
  subject, attribute, value, valid_from), so re-ingesting a log never
  appends duplicate values. Extracted facts carry `source_fragment_id` and
  `metadata` (`extracted_by`, `rule`, `confidence`).
- **Never fails a write.** Extraction errors are logged and the write stands.

On a held-out set of everyday and developer statements, the packs reached
precision 1.0 and recall 0.944; on the retention benchmark precision stays
1.0 for every target attribute.

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
**Presets** (`presets.py`) size a briefing for the model that reads it,
using values measured with real local models: `"small"` (~1-3B: 200 tokens,
the request's best four lines first, one record per subject, 160-character
lines) and `"tiny"` (under 1B: 160 tokens, facts and the best three lines).
`ContextWatch.for_preset(memory, name)` (and `for_small_models()`,
`for_tiny_models()`) build a watch from one; `MarkChat` defaults to
`"small"`, and `MarkAgentMiddleware(preset=)` and `mark mcp --preset` accept
them. No preset adds a skill: on llama3.2:1b any added instruction block
lowered accuracy, so the briefing's headers ("Known facts (current):",
"Corrections:", "Related notes:", "Not in memory:") carry the guidance. A
host that adds a skill charges its tokens to the same budget
(`split_budget`).

**Provenance in briefings** (`provenance.py`). User statements, tool output
and the model's own replies are kept apart: assistant replies never anchor a
briefing, are never shown as something the user said, and are left out by
the presets; questions and the current turn are not echoed back as memory.

**Session brief** (`session_brief.py`). `memory.brief(budget_tokens=400,
focus=None)`, `mark brief` and the MCP `mark_brief` tool return one budgeted
`<mark_briefing>` block for the start of a session: current facts, open TODOs
(with short ids), recent decisions, bugs and fixes, and recent fact changes,
newest first, deduplicated. A TODO closed by `resolve()`, a status, or a
later DONE/FIX note naming it is left out. Kinds come from `kind:` tags or a
content prefix such as `TODO:`.

**Token budget.** `ContextBudget(max_tokens, fill=False, token_counter=None)`
(`budget.py`) is a hard cap. `watch_render.py` turns a snapshot into
`BudgetItem` lines tagged with a section, and the budget keeps whole lines by
section priority (corrections > unknown > relevant > facts > records >
linked > related > assistant) until the rendered text, wrapper included, fits. Only the first line
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
OpenAI-compatible `/chat/completions` endpoint with the standard library. It
stores one dense memory per user message ("User said: ...") and the model's
reply as a separate low-trust memory (`store_assistant=False` drops it), both
before the window slides. The middleware archives turns the same way
(`store_assistant=`, `preset=`) and counts its injected skill inside
`budget_tokens`.

## Image memory

`memory.images` (`media/memory.py`) stores each image as a fragment tagged
`mark:image` whose metadata holds the resolved path, SHA-256, size, subjects,
and the ids of the subjects' current facts. Image bytes are never stored.
`find` uses CLIP vectors from an `ImageEmbeddingProvider` when one is set,
kept in `mark_image_vectors` apart from text embeddings and keyed by
fingerprint and file hash. Otherwise it runs normal recall over captions.
`verify` re-hashes files. `check_consistency` compares what a describer (for
example `OpenAIVisionDescriber`) reports with the expected facts.
`images.remember` checks the file's content and raises `NotAnImageError` for
anything that is not an image.

`memory.remember_file(path)` sniffs the content (`media/files.py`) and routes
it: text, Markdown, code, HTML and PDF go to `store_document`, images to
`images.remember`, and audio or video to a transcriber (anything with
`transcribe(path) -> str`, such as `OpenAITranscriber`) whose text is then
stored as a document. MARK never decodes audio itself.

## Integrations

| Surface | Module | Notes |
| --- | --- | --- |
| LangChain middleware | `adapters/langchain/middleware.py` and `_middleware_*.py` mixins | Retrieval before model calls, briefing, observation of model and tool output. Sync and async. Per-run state. |
| LangChain tools | `adapters/langchain/tools.py` | `mark_retrieve`, `mark_write`, `mark_observe`. |
| MCP server | `adapters/mcp/__init__.py`, `_tools.py`, `_tools_more.py`, `_briefing.py`, `launcher.py` | `mark_retrieve`, `mark_write`, `mark_observe`, `mark_fact`, `mark_timeline`, `mark_brief`, `mark_forget` over stdio, with plain-text output and domain errors returned as `isError` results. `tools=` exposes a subset and `allowed_agent_ids` restricts namespaces. `briefing=True` adds `mark_briefing` and `mark://briefing/{session_id}`, one watch per (agent, session) in a bounded LRU. `python -m mark.adapters.mcp --db PATH` and `mark mcp` start it on a store file or `.mark` folder. Supports `mcp` 1.x and 2.x. |
| Reference chat loop | `context/loop.py` | `MarkChat` for any OpenAI-compatible endpoint. |
| Backend protocol | `adapters/backend.py` | `MarkBackend` protocol; `LocalMarkBackend` runs blocking work off the event loop. |
| Command line | `cli.py`, `_cli/` | See below. |
| Agent wrapper | `agent.py`, `_agent/` | `Mark.wrap_agent()` for plain callables and LLM functions; the wrapped agent is callable and offers `invoke` and `ainvoke`. |

### Command line and `mark serve`

`mark.cli` parses arguments and prints results. `mark._cli` holds the rest:

- `config.py`: database path (`--db`, `$MARK_DB`, `./.mark/memory.db`) and the
  sibling files `<db>.server.json` (state) and `<db>.server.lock`.
- `engine.py`: `_Engine` runs commands against one database. It is the only
  place the CLI imports MARK.
- `engine_ops.py`: the lifecycle and data commands (`forget`, `resolve`,
  `supersede`, `list`, `export`, `import`, `brief`, `agents`).
- `output.py`: human-readable output: 8-character short ids (accepted by
  every command that takes an id), dates with a `#k` suffix only when dates
  collide, kinds, and relevance scores.
- `server.py`: `mark serve`, a loopback-only HTTP server that keeps an
  `_Engine` open. One server per database is enforced with an OS file lock;
  the server warms its index in the background on start and accepts an
  authenticated stop request (`mark serve stop`).
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
| `warm_up()` | Load the vector index ahead of the first recall (used by `mark serve`) |
| `shutdown(timeout=5.0)` | Stop the fact worker and worker pools within `timeout`, return unfinished jobs to the queue, and close the store; later calls raise `StoreClosedError` |

```python
from mark import Mark

with Mark.local(".") as mark:
    memory = mark.runtime.memory("coder")
    memory.remember("FastAPI handles routing.")
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
| `GovernanceMiddleware` | Governance gates; quarantine rejected writes (`FailureFilter`, which keeps error output out of retrieval, is opt-in on `ConsolidationGate`) |
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
