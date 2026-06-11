# MARK Python SDK Architecture

The MARK Python SDK is a local-first runtime for agent memory. It is the open,
MIT-licensed layer that helps agents recall project facts, organize working
context, and expose memory to tools, middleware, and framework adapters.

Everything documented here runs locally. Hosted MARK services are separate
products that connect through the SDK's public plugin hooks; their design is
out of scope for this document.

## Entrypoints

### `Mark.local()`

`Mark.local()` is the public entrypoint for local use cases.

It is designed for:

- tutorials and quick scripts,
- lightweight coding agents,
- local project memory,
- SQLite persistence in `.mark/memory.db`,
- direct use as a tool, middleware helper, or agent wrapper.

It wires:

- `MarkRuntime.local()` as the runtime engine,
- `SimpleMemory` as the ergonomic block/retrieve surface,
- `ContextBuilder`,
- `PolicyRegistry`,
- `SkillRegistry`,
- `MarkAgent`.

```python
from mark import Mark

with Mark.local(".") as mark:
    mark.memory.block("project").write("The API framework is FastAPI.")
    print(mark.memory.retrieve("Which API framework?").as_text())
```

Advanced path:

```python
with Mark.local(".") as mark:
    agent = mark.runtime.memory("coder")
    agent.store_sync("FastAPI handles routing.")
    mark.runtime.run_cycle("coder")          # consolidate + deduplicate + prune
    mark.runtime.integrity_check()           # []
```

### `MarkRuntime.local()`

`MarkRuntime.local()` is the explicit runtime engine. Most developers should
start with `Mark.local()`, but framework integrations and advanced
applications can use `MarkRuntime` directly.

It wires:

- `LocalMemoryStore` with SQLite,
- `HashEmbeddingProvider` or developer-provided `EmbeddingProvider`,
- `VectorIndex`,
- `RetrievalPipeline`,
- `PluginRegistry`,
- `MarkMemory`,
- `GlobalMemoryBus`,
- `TrustAwareGlobalMemoryBus`,
- `LocalTracer`,
- `GovernanceAuditLog`.

## Memory Model

```text
fragment  → atomic evidence (content, session, tags, importance, confidence)
node      → semantic anchor built from fragments (character, fact, location…)
edge      → directed, typed, weighted relation between nodes
block     → bounded set of fragments, nodes, and the edges between them
chain     → blocks sealed and linked forward/backward per agent
```

An agent owns a namespace. Sessions partition it (`session_id`,
`session_prefix`), tags narrow it, and blocks group it into inspectable units.

### Memory Blocks And Provenance

Blocks are graph containers: a block can represent a topic within a session,
an entire session, or a world-bible scope.

- Node membership is many-to-many (`block_nodes`), so canonical entities are
  never duplicated across blocks.
- Edges may be scoped to a block; scoped edges only connect member nodes.
- Blocks connect to other blocks through directed `BlockLink`s and are
  traversable forward and backward, like nodes through edges.
- Sealing (`BlockChain.seal`) computes a deterministic SHA-256 content hash
  over immutable member fields and links the block to the previously sealed
  block (`prev_block_hash`), forming a local, tamper-evident hash chain.
- `verify()` pinpoints corrupted, missing, or injected members;
  `verify_chain()` checks the whole chain.
- Quarantining a block removes its members from retrieval and traversal
  without affecting any other block or the agent's function.
- Mutable plasticity fields (importance, access counts, edge weights) are
  excluded from hashes so reinforcement never breaks a seal.

## Lifecycle

| Method | Role |
|---|---|
| `working_memory(agent_id)` | TTL-bound short-term memory |
| `consolidate(agent_id)` | Governed working-memory promotion |
| `deduplicate(agent_id)` | Embedding cluster merge and provenance |
| `prune(agent_id)` | Decay, fragment pruning, edge dissolution |
| `run_cycle(agent_id)` | Consolidate, deduplicate, prune |
| `integrity_check()` | SQLite health check |

Retrieval is live. Each retrieved fragment is touched, access count increases,
importance can strengthen through Hebbian reinforcement, and co-retrieved graph
edges strengthen through co-activation. Promoted/permanent memories are exempt
from decay.

Consolidation is governed locally before promotion. `ConsolidationGate` redacts
common secrets/PII, blocks empty or low-confidence content, rejects failure
artifacts, and can reject exact duplicate hashes. `MarkRuntime` keeps an
in-process `GovernanceAuditLog`.

Deduplication is transparent and local. The SDK clusters stored embeddings,
keeps the highest-importance canonical fragment, records `merged_from_ids`,
and reattaches graph nodes from deleted duplicates to the kept fragment.

## Architecture Diagram

```mermaid
flowchart TD
    Dev[Developer Agent] --> Simple["Mark.local()"]

    Simple --> SM[SimpleMemory]
    Simple --> Skills[SkillRegistry]
    Simple --> Policies[PolicyRegistry]
    Simple --> Agent[MarkAgent]
    Simple --> RT[MarkRuntime]

    SM --> RT
    RT --> SQLite[(LocalMemoryStore SQLite)]
    RT --> Embeddings[EmbeddingProvider]
    RT --> Vector[VectorIndex]
    RT --> Retrieval[RetrievalPipeline]
    RT --> Blocks[BlockGraph / BlockChain]
    RT --> Bus[GlobalMemoryBus]
    RT --> TrustBus[TrustAwareGlobalMemoryBus]
    RT --> Obs[LocalTracer]
    RT --> Governance[Local Governance]
    RT --> Plugins[PluginRegistry]
    RT --> Lifecycle[Lifecycle]

    Retrieval --> Hebbian[Hebbian Reinforcement]
    Lifecycle --> Consolidation[ConsolidationManager]
    Lifecycle --> Dedup[DeduplicationConsolidator]
    Lifecycle --> Pruner[MemoryPruner]
    Consolidation --> Governance
```

## Package Boundary

This package includes:

- memory data contracts,
- SQLite storage with schema versioning and in-place migrations,
- local retrieval, graph expansion, and gap reporting,
- graph-scoped memory blocks, block links, seal/verify, and quarantine,
- working memory, governed consolidation, deduplication, and pruning,
- graph nodes, edges, neighborhoods, and graph-connected context,
- session-aware retrieval and session activity logs,
- local governance heuristics and in-process audit logs,
- local observability events, session traces, and JSONL replay,
- local sync delta preparation with redaction,
- trust-aware global memory bus,
- plugin interfaces and hook constants,
- optional framework adapters and skills.

Hosted services, managed infrastructure, and commercial features are not part
of this package. The only paths from the SDK outward are `PluginRegistry`
hooks and clients that the developer explicitly supplies.

## Tool, Middleware, And Adapters

MARK can be used in three common ways:

- As a tool: the agent explicitly calls MARK retrieval.
- As middleware: MARK injects relevant memory before agent planning and records
  observations after tool/action results.
- As an MCP server: `mark.adapters.mcp` exposes local memory operations to any
  MCP-compatible client.

Framework adapters ship inside `mark-sdk` under `mark.adapters` with optional
dependency extras:

```text
mark/
  adapters/
    backend.py    MarkBackend protocol + LocalMarkBackend
    langchain/    tools + AgentMiddleware
    mcp/          MCP server factory
```

```bash
pip install "mark-sdk[langchain]"
pip install "mark-sdk[mcp]"
pip install "mark-sdk[adapters]"
```

Adapter modules are thin framework glue over public MARK APIs. Adapter
namespaces without complete implementations are excluded from distributions
until they have real code and tests.

## Sync And Observability

The SDK can prepare local sync envelopes but does not implement any remote
transport. `CloudSync.prepare_delta()` returns a `SyncDelta` containing
redacted fragments, optional local events, options, and stats.
`CloudSync.sync()` forwards that envelope only when a caller supplies a client
object with `upload_delta(delta)`.

```mermaid
flowchart LR
    Runtime[MarkRuntime] --> Store[(SQLite)]
    Runtime --> Tracer[LocalTracer]
    Store --> Prepare[CloudSync.prepare_delta]
    Tracer --> Prepare
    Prepare --> Delta[SyncDelta]
    Delta --> OptionalClient[Developer-supplied client]
```

`LocalTracer` is available through `MarkRuntime.tracer()`, and
`MarkRuntime.session_trace(session_id)` provides a session-scoped view of
local events.

## Trust-Aware Bus

`TrustAwareGlobalMemoryBus` wraps `GlobalMemoryBus` with publisher trust,
subscriptions, snapshots, and trust-filtered retrieval. Messages persist as
global-bus fragments with trust tags and can be rehydrated when a runtime
reopens the same SQLite store.

```mermaid
flowchart TB
    Publisher[Agent Publisher] --> TrustBus[TrustAwareGlobalMemoryBus]
    TrustBus --> Store[(SQLite Global Fragments)]
    TrustBus --> Snapshot[BusSnapshot]
    TrustBus --> Subscription[BusSubscription]
    Subscription --> Consumer[Consumer Agent]
```
