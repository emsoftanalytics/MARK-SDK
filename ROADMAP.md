# MARK SDK Roadmap

This roadmap covers the open Python SDK: work that can be implemented,
tested, documented, and shipped in the `mark-sdk` package. Plans change as we
learn from real workloads; open a discussion if something here matters to you.

## Current release: 0.2.0a9

Everything from 0.2.0a8 (local memory, hybrid recall, time-aware recall,
bi-temporal facts, gap reports, batch operations, `ContextWatch`, LangChain
middleware and tools, the MCP server, the `mark` command line, local
embedders, provenance blocks, and encryption at rest), plus:

- **Automatic fact extraction.** Built-in rule packs record facts from stored
  text (`fact_extractor="patterns"` or `extract_facts=True`). A local LLM
  extractor runs from a durable background queue. `mark ingest
  --extract-facts` does the same from the command line.
- **Subject timelines.** `memory.timeline(subject, attribute)` and
  `mark timeline` list everything memory knows about a subject, oldest first.
  `order="chronological"` and `mark recall --chronological` sort recall
  results the same way.
- **Proactive context beyond LangChain.** `observe_messages()` and `inject()`
  bring briefings to any agent loop, MCP clients get an opt-in
  `mark_briefing` tool and resource, and `ContextWatch.for_small_models()`
  is a preset for small models.
- **Token-budgeted context.** `ContextBudget` puts a hard token cap, filled by
  priority, on briefings and on everything the middleware injects.
- **Context replacement.** `MarkAgentMiddleware(history="mark")` and the
  `MarkChat` reference loop keep only recent messages in the prompt and let
  MARK supply the rest.
- **Better embeddings.** `embedder="auto"` and spec strings, with fusion that
  adapts to the embedder (hybrid hit@1 up 7 to 11 queries out of 60 for
  semantic models). The deterministic hash embedder stays the default.
- **Image memory.** `memory.images` remembers images by path and hash, finds
  them by caption or CLIP similarity, and checks them against facts.
- **Hardening.** Encrypted stores refuse to open without the right key, and
  new store directories and files are private on POSIX.

## Next: 0.2.0a10

- **Fast cold start and backup.** Faster first use of a large store, and
  simple backup and restore.
- **A simpler API.** Fewer entry points for common tasks.
- **Windows and macOS CI, and fuzzing.** Test every release on all three
  platforms, and fuzz the parsers and the store.
- **The retention A/B benchmark as a regression suite.** Run it on every
  release so recall quality cannot quietly regress.
- **Fact extraction for multi-clause lines.** Lines that state several
  changes at once, the main gap in the current rule packs.
- **Alias resolution in timelines.** Gather a subject's events under all of
  its names.
- **Encryption support in the command line.**

## Later

Under consideration: sealing fact timelines into the block chain, an entity
channel as a third retrieval signal, an optional local reranker, and a
reproducible long-memory evaluation harness.

## Good areas for contributors

- **Examples and recipes** for coding, research, and creative-continuity
  agents, each comparing behaviour with and without MARK.
- **Adapter hardening**: LangChain edge cases, MCP client setup notes, tests.
- **Retrieval diagnostics**: explain why a memory was returned.
- **Migrations**: fixtures for older alpha stores.
- **Redaction coverage** for sync envelopes and logs.
- **Documentation** that describes what MARK does today, without claims the
  SDK does not support.

## Not in scope for this package

- Account-based or hosted infrastructure (MARK Cloud — coming soon).
- Required network calls during normal local use.
- Heavy media-generation dependencies in the install path.
- Generated media in the repository or distribution.

## How to help

Start with a small issue, work on a branch, and open a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md).
