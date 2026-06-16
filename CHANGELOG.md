# Changelog

All notable changes to the MARK Python SDK will be documented here.

This project follows semantic versioning for public package releases.

## 0.2.0a6 - 2026-06-16

### Added

- Added `AgentRun`, a reusable per-run lifecycle object for `MarkAgent`
  executions, including prompts, session data, metadata, retrieved memories,
  policy decisions, skill outputs, stored output IDs, and middleware mutations.
- Added production-oriented `MarkAgent` support for optional middleware,
  policy actions, skill execution, output persistence, per-run metadata, and
  runtime-backed recall while preserving a no-LLM fallback path.
- Added `mark.middlewares`, a framework-neutral middleware layer for wrapping
  local `store`, `observe`, and `retrieve` operations through public runtime
  APIs.
- Added local middleware for recall defaults, observation source/tags/metadata,
  compressor binding, query-expander binding, explicit gap healing, lifecycle
  run-cycle triggers, local tracing, sync envelope preparation, governance,
  trust-bus publishing, blocked-by-default sandbox execution, and local media
  continuity memory.
- Added middleware extras so developers can install the full middleware bundle
  with `mark-sdk[middleware]` or choose specific battery targets such as
  `middleware-governance`, `middleware-trust-bus`, `middleware-sandbox`, and
  `middleware-media-continuity`.
- Added `MarkRuntime.add_middleware()`, `MarkRuntime.use()`, and
  `Mark.local(..., middleware=[...])` for ergonomic middleware composition.
- Added broader MCP adapter tool options for scoped retrieval, observation,
  canonical writes, tags, session IDs, agent IDs, and block filters.
- Added packaged MARK usage skills under `mark.middlewares.skills` so installed
  users can load middleware guidance through `importlib.resources`.
- Added regression coverage for LangChain tool sync calls inside running event
  loops, native async `ainvoke()` usage, middleware context injection, MCP tool
  options, and `MarkAgent` lifecycle behavior.
- Live developer usage examples under `examples/`:
  - `01_local_memory_live.py` for local `Mark.local()` storage/retrieval.
  - `02_agent_ab_live.py` for with/without MARK agent context comparison.
  - `03_sessions_and_observe_live.py` for session-aware `observe()` continuity.
  - `04_sync_envelope_live.py` for redacted sync envelope preparation.
  - `run_live_examples.py` to run the live examples as MVP smoke tests.
- Python 3.10+ release coverage in CI and release workflows
  (`3.10`, `3.11`, `3.12`, `3.13`, `3.14`).
- Graph-scoped memory blocks: blocks now group nodes and the edges between
  them per topic, session, or world-bible scope (schema v2 with in-place
  migration of existing stores).
  - `BlockGraph` (`memory.blocks()`): many-to-many node membership,
    block-scoped edges, forward/backward `BlockLink`s, and BFS traversal.
  - `BlockChain` (`memory.chain()`): `seal()` computes a deterministic SHA-256
    content hash over immutable member fields and chains sealed blocks through
    `prev_block_hash`; `verify()` pinpoints corrupted, missing, or injected
    members; `verify_chain()` checks whole-chain integrity.
  - Block quarantine: a quarantined block's fragments leave retrieval and
    traversal without affecting other blocks or agent function; `release()`
    restores it.
  - Retrieval accepts `block_id` / `block_ids` filters
    (`retrieve_sync(..., block_id=...)`), and quarantined-block members are
    always excluded from results.
- Functional test suites for the MCP server adapter and LangChain tools
  adapter (round-trip write/retrieve/observe against the local backend).
- `examples/getting_started_with_mark.ipynb`: runnable notebook walkthrough —
  the same coding agent without MARK, with `MarkAgentMiddleware`, with MARK
  LangChain tools, and over MCP, plus memory inspection and block provenance.
- URL safety validation for the web lookup skill: only public http/https
  hosts are fetchable (file://, loopback, private, and link-local targets are
  rejected).
- Docstrings across the entire public API surface (modules, classes, and
  functions).

### Changed

- Changed the SDK license from MIT to Apache-2.0 for explicit patent protection.
- Split middleware into one package per battery under `mark.middlewares`.
- Reworked LangChain tool creation to expose `StructuredTool` instances with
  both sync and async implementations backed by the same MARK backend.
- Clarified LangChain guidance: middleware is the automatic context-loading
  path, tools are the deliberate/canonical memory-write path, and combining both
  is an advanced mode that should avoid duplicate context.
- Updated the LangChain extra to install the agent middleware dependency surface
  required by the current adapter tests.
- Reframed `MarkAgentMiddleware` as a message-context memory layer and made the
  optional `thinking_model` documentation explicitly experimental and opt-in.
- Repositioned public documentation around MARK as an agent memory
  runtime for creative continuity, workflow state, middleware integration,
  structured recall, and provenance rather than a generic memory store.
- Expanded architecture and examples documentation to explain agent-loop
  integration, retrieval/context injection, creation memory, and planned
  proof-oriented demos for character consistency and multi-agent coding.
- Updated README and examples documentation so media, video, music, and
  generated demo outputs are not described as current package examples.
- Core install now depends only on `pydantic`; unused `openai` and
  `langchain-openai` core dependencies were removed.
- The `adapters` extra now installs LangChain and MCP dependencies only.
- README rewritten as a concise install/usage guide with adapter examples;
  ARCHITECTURE.md now documents the local runtime including memory blocks.
- Sync redaction now recursively redacts metadata, source fields, tags, and
  optional event payloads before a sync envelope leaves local code.
- The explicit web fetch skill now rejects hostnames that resolve to private,
  loopback, link-local, reserved, multicast, or unspecified addresses.

### Fixed

- Fixed `MarkAgent` so configured policies, skills, middleware, and output
  storage participate in agent runs instead of being isolated constructor data.
- Fixed LangChain MARK tools raising inside notebook/async environments when a
  sync `invoke()` happened while an event loop was already running.
- Fixed LangChain middleware context handling for missing system messages and
  string/list message content.
- Fixed canonical `mark_write(..., canonical=True)` memories to store with high
  importance and canonical tags so retrieval ranks them above episodic
  observations.
- Fixed MCP and backend retrieval wiring so block filters are passed through the
  public runtime/backend API consistently.

### Removed

- Removed packaged media/image/video/music demo files and their test claims from
  the current SDK package.
- Removed heavy provider-demo dependencies such as `gradio-client` and `pillow`
  from SDK extras and lockfile state.
- The empty LlamaIndex adapter namespace and its extra.
- Tests and tutorial scripts are no longer packaged in the sdist; the wheel
  ships only the `mark` package.

## 0.2.0a4 - 2026-06-07

### Changed

- Renamed the PyPI distribution from `mark` to `mark-sdk` while preserving the
  Python import name `mark`, because the public PyPI name `mark` is already
  occupied by another project.
- Added PyPI project URLs pointing at the public SDK repository:
  `emsoftanalytics/MARK-SDK`.
- Reworked the README into a release-ready quickstart with install, tutorial,
  build, publish, and repository strategy sections.
- Added a runnable `tutorial.py` coding-task walkthrough using the installed
  `mark` package import.
- Added a GitHub Actions release workflow for tests, build artifacts, and PyPI
  Trusted Publishing.

## 0.2.0a3 - 2026-06-07

### Added

- `LocalMemoryStore` lifecycle methods: `update_importance`, `touch` (records
  `last_accessed_at` and increments `access_count` in fragment metadata),
  `list_edges`, `delete_edge`, `update_edge_weight`, `edges_to_node`,
  `integrity_check`, `export`, and `schema_version`.
- Schema versioning via `schema_info` table and `_MIGRATIONS` dict. Pre-versioned
  databases (created before `schema_info` existed) are detected from fragment
  presence and migrated forward automatically.
- `MarkRuntime` hippocampus lifecycle: `working_memory()`, `consolidate()`,
  `prune()`, `run_cycle()`, and `integrity_check()`.
- Hebbian reinforcement wired through `RetrievalPipeline.retrieve()`: each
  retrieved fragment is touched (access count incremented) and its importance
  boosted via dampened LTP. Co-retrieved fragment pairs have shared graph edges
  strengthened via `EdgeCoActivation`. Each edge is strengthened at most once
  per retrieval (dedup by edge ID). Both outgoing and incoming edges are
  considered (undirected strengthening).
- `touch()` now runs before `on_access()` in the reinforcement loop so that
  `access_count` is current when the Hebbian boost is computed — first-access
  reinforcement now takes effect immediately.
- 20 new tests in `test_hippocampus.py` covering store methods, lifecycle,
  reinforcement order, co-activation (outgoing and incoming), schema versioning,
  export, and pre-versioned DB migration.
- Session-aware retrieval with exact-session, session-prefix, tags, tier, and
  scope filters.
- Graph API for nodes, edges, neighborhoods, attached fragments, graph context,
  and graph-expanded retrieval.
- Local media memory helpers for character, object, location, scene/session, and
  world-bible continuity.
- Contextual compression with no-op, window, and developer-provided LLM modes.
- Structured extraction with deterministic extraction, optional developer LLM
  extraction, and merge/dedup logic.
- Query expansion with no-op, keyword, and developer-provided LLM variants.
- Embedding cache provider for lightweight local reuse.
- JSON-to-SQLite migration for legacy `memory.json` stores.
- Deduplication consolidation with clustering, centroids, provenance, graph-node
  reattachment, and `run_cycle()` integration.
- Trust-aware global bus with publisher trust, subscriptions, snapshots,
  persisted rehydration, and trust-filtered retrieval.
- Local governance heuristics: sanitizer, validation gate, duplicate hash gate,
  failure filter, consolidation gate, and in-process audit log.
- Local observability: tracer, session trace, JSONL event replay, and runtime
  maintenance events.
- Sync preparation with `SyncDelta`, block filtering, local redaction, optional
  event inclusion, and optional caller-supplied upload forwarding.
- Local RL/plasticity data structures and tests kept Apache-2.0-safe.

### Changed

- `Mark.local()` is now the single public entrypoint. It delegates to
  `MarkRuntime.local()` (SQLite at `.mark/memory.db`), wraps it in
  `SimpleMemory`, and exposes the full runtime via `mark.runtime`. The
  `Mark.local()` / `MarkRuntime.local()` split is resolved: `MarkRuntime`
  remains importable for explicit use, but developers should start with
  `Mark.local()`.
- README and ARCHITECTURE updated to reflect the unified entrypoint and SQLite
  default. JSON references removed.
- `ConsolidationManager` now runs local governance before promotion and records
  audit entries through `MarkRuntime.governance_audit_log()`.
- `MarkRuntime.run_cycle()` now performs consolidation, deduplication, and
  pruning, returning a richer maintenance summary.
- `MarkRuntime.trust_bus()` and `global_bus()` now return stable runtime-scoped
  instances instead of fresh wrappers on every call.

### Fixed

- Co-activation double-strengthening: the same edge could previously be
  strengthened once as an outgoing edge from fragment A and again as an incoming
  edge from fragment B in a single retrieval. A `seen_edges` set now ensures each
  edge is updated at most once per call.
- Reinforcement order: `touch()` now precedes `on_access()`, and the fragment is
  re-fetched so the boost formula sees the correct `access_count`.
- Pre-versioned SQLite DBs (no `schema_info` table) are now detected by checking
  for existing fragment rows; they are migrated forward and stamped with the
  current schema version rather than being silently treated as fresh databases.
- Secret redaction now catches `API key: ...` and `sk-...` token patterns in
  local sync envelopes.
- Trust-bus messages can be rehydrated from persisted global-bus fragments.
- Deduplication now reattaches nodes from duplicate fragments to the kept
  canonical fragment before deleting the duplicate.

## 0.2.0a2 - Unreleased

### Added

- SDK architecture document covering `Mark.local()`, `MarkRuntime.local()`,
  runtime extension points, plugin placement, and MCP placement.
- Coding-task tutorial showing MARK as a tool and middleware with a simple
  with/without MARK benchmark.
- Repo-level example relocation for generated notebook documentation assets.

### Changed

- README now documents the difference between `Mark.local()` and
  `MarkRuntime.local()`.

## 0.2.0a1 - Previous alpha checkpoint

### Added

- Lab-derived local hippocampus/cortex runtime through `MarkRuntime.local()`.
- SQLite local memory store, local vector index, graph-aware retrieval pipeline,
  plugin hooks, and stronger memory lifecycle types.
- Conversation memory, persona memory, global memory bus, working memory,
  consolidation, contradiction detection, local plasticity, RL signal schemas,
  policy safety constraints, optional embedding adapters, and Docker sandbox
  support.

### Changed

- Built-in skills keep the async skill contract and support synchronous usage
  through `run_sync()`.
- The package version now uses PyPI-compatible pre-release numbering for SDK
  phase checkpoints.

## 0.1.0 - Previous foundation

### Added

- Local `Mark.local()` runtime with JSON-backed project memory.
- Lab-derived `MarkRuntime.local()` runtime with SQLite storage, local vector
  index, graph expansion, retrieval pipeline, and plugin hooks.
- Memory types for fragments, blocks, documents, nodes, edges, scopes, tiers,
  lifecycle states, gap reports, and attribution.
- Agent wrapper, policy registry, skill registry, built-in MARK memory skill,
  and packaged `mark-usage` skill instructions.
- Conversation memory, persona memory, global memory bus, working memory,
  consolidation, contradiction detection, local plasticity, and safety schemas.
- Optional local embedding adapters for Ollama and sentence-transformers.
- Local sandbox support for Docker-backed code execution.
- Coding-task tutorial showing MARK as a tool and middleware with simple
  benchmarks.

### Changed

- Built-in skills keep the async skill contract and support synchronous usage
  through `run_sync()`.
- Public SDK comments describe extension points by capability only.

### Security

- Added content hashing, optional local encryption provider interfaces, redaction
  helpers, and raw-data opt-in boundaries for local signal collection.
