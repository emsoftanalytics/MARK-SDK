# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
#
# Mark — the single public entrypoint for the local MARK SDK.
#
# Mark.local() composes the full SQLite-backed MarkRuntime engine and
# exposes two surfaces:
#
#   Simple path (small projects, notebooks, scripts):
#       mark.memory.block("project").write("FastAPI is used.", importance=0.9)
#       bundle = mark.memory.retrieve("Which framework?")
#
#   Advanced path (multi-agent, full pipeline, explicit agent IDs):
#       agent  = mark.runtime.memory("coding-agent")
#       bus    = mark.runtime.global_bus()
#       store  = mark.runtime.store
#       plugins = mark.runtime.plugins
#
# MarkRuntime is the engine. Mark is the brand and the developer API.
"""Mark: the public local entrypoint wiring runtime, memory, skills, and agents.

``MarkConfig`` and the late-binding ``configure_*`` hooks live in
:mod:`mark.runtime_config`; both are available from this module.
"""
from __future__ import annotations

from dataclasses import dataclass  # noqa: F401  (kept for backward compatibility)
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from mark.agent import MarkAgent
from mark.context import ContextBuilder
from mark.memory.simple import SimpleMemory
from mark.policies import PolicyRegistry
from mark.middlewares.base import MarkMiddleware
from mark.middlewares.skills import SkillRegistry
from mark.runtime_config import MarkConfig, MarkConfigureMixin

if TYPE_CHECKING:
    from mark.middlewares.media_continuity.world_bible import WorldBibleMemory
    from mark.memory.observe import ObserveResult
    from mark.middlewares import MarkMiddleware


class Mark(MarkConfigureMixin):
    """
    Primary local MARK runtime entrypoint.

    Use Mark.local() to create an instance. All data stays local unless
    you explicitly register a HOOK_SYNC plugin.

    Simple API (mark.memory):
        mark.memory.block("project").write("...")
        mark.memory.retrieve("query").as_text()
        mark.memory.list_blocks()

    Advanced API (mark.runtime):
        mark.runtime.memory("agent-id")   → MarkMemory (full pipeline)
        mark.runtime.global_bus()         → GlobalMemoryBus
        mark.runtime.store                → LocalMemoryStore
        mark.runtime.plugins              → PluginRegistry
        mark.runtime.pipeline             → RetrievalPipeline
    """

    def __init__(
        self,
        *,
        config:          MarkConfig,
        memory:          SimpleMemory,
        policies:        PolicyRegistry,
        skills:          SkillRegistry,
        context_builder: ContextBuilder,
        _runtime: object,
    ) -> None:
        self.config          = config
        self.memory          = memory
        self.policies        = policies
        self.skills          = skills
        self.context_builder = context_builder
        self._runtime_engine = _runtime
        self._world_bible: "WorldBibleMemory | None" = None

    @property
    def runtime(self) -> object:
        """
        The underlying MarkRuntime engine.

        Exposes the full pipeline for advanced use cases:
            mark.runtime.memory("my-agent")
            mark.runtime.global_bus()
            mark.runtime.store
            mark.runtime.plugins
        """
        return self._runtime_engine

    @classmethod
    def local(
        cls,
        project_path: str | Path = ".",
        store_path:   str | Path | None = None,
        *,
        embedder:        object | None = None,
        encryption:      object | None = None,
        workers:         int = 4,
        llm:             object | None = None,
        compressor:      object | None = None,
        query_expander:  object | None = None,
        enable_activity_log: bool = False,
        retrieval_top_k: int = 20,
        middleware:      list["MarkMiddleware"] | None = None,
        fact_extractor:  object | None = None,
    ) -> "Mark":
        """
        Create a local Mark instance backed by SQLite.

        Args:
            project_path: Root directory of your project (default: cwd).
            store_path:   Override the .mark/ directory location.
            embedder:     EmbeddingProvider, or a spec string: "auto" (best local
                          model, else hash), "hash", "ollama:<model>", ...
                          Default: HashEmbeddingProvider, or MARK_EMBEDDER if set.
            encryption:   EncryptionProvider (default: no encryption).
            workers:      Thread-pool size for async retrieval (default: 4).
            llm:          LLMProvider for local entity extraction in observe().
                          Optional — observe() works without it (stores only).
            compressor:      ContextualCompressor for post-retrieval compression.
                             Optional — NoopCompressor is used when not provided.
                             Use LLMContextualCompressor(llm=...) for LLM-based
                             query-relevant compression before context injection.
            retrieval_top_k: Candidate pool size fed to the reranker when an LLM
                             compressor is configured (default 20).  Without a
                             compressor the built-in policy's top_k is used
                             (fast=5, balanced=10, deep=15).
            fact_extractor:  Extract facts from every write: None (off, default),
                             "patterns" (built-in rule packs, inline) or any
                             FactExtractor (e.g. LLMFactExtractor, queued).

        The .mark/ directory is created under project_path if it does not
        exist (mode 0700 on POSIX). SQLite database is at .mark/memory.db
        (created 0600).
        """
        from mark._private_fs import ensure_private_dir
        from mark.memory.runtime import MarkRuntime

        project  = Path(project_path).expanduser().resolve()
        mark_dir = (Path(store_path).expanduser().resolve()
                    if store_path else project / ".mark")
        ensure_private_dir(mark_dir)  # 0700 when MARK creates it; existing dirs untouched
        db_path = mark_dir / "memory.db"

        runtime = MarkRuntime.local(
            store_path           = db_path,
            embedder             = embedder,         # type: ignore[arg-type]
            encryption           = encryption,       # type: ignore[arg-type]
            workers              = workers,
            llm                  = llm,              # type: ignore[arg-type]
            compressor           = compressor,       # type: ignore[arg-type]
            query_expander       = query_expander,   # type: ignore[arg-type]
            enable_activity_log  = enable_activity_log,
            retrieval_top_k      = retrieval_top_k,
            middleware           = middleware,
            fact_extractor       = fact_extractor,
        )

        # One-time migration: if a legacy memory.json exists, import it into SQLite
        json_path = mark_dir / "memory.json"
        if json_path.exists():
            from mark.memory.migration import migrate_json_to_sqlite
            migrate_json_to_sqlite(json_path, runtime.store)

        memory = SimpleMemory(runtime)

        return cls(
            config          = MarkConfig(project_path=project, store_path=mark_dir),
            memory          = memory,
            policies        = PolicyRegistry.with_builtins(),
            skills          = SkillRegistry.with_builtins(memory=memory),
            context_builder = ContextBuilder(),
            _runtime        = runtime,
        )

    @property
    def world_bible(self) -> "WorldBibleMemory":
        """Canonical facts store — writes are immediately PROMOTED (never decay).

        Use this for any ground-truth facts your agent should never forget or
        contradict: character traits, physical laws, world history, system
        invariants, domain axioms. Works for any long-running project.

        Example::

            mark.world_bible.remember("Elena is left-handed.", tags=["character:elena"])
            mark.world_bible.remember("The warehouse was destroyed in Phase 2.")

            # Local continuity pre-check:
            conflicts = mark.world_bible.check("Elena uses her right hand.")
        """
        if self._world_bible is None:
            from mark.middlewares.media_continuity.world_bible import WorldBibleMemory
            self._world_bible = WorldBibleMemory(
                self._runtime_engine.memory("__world_bible__")  # type: ignore[union-attr]
            )
        return self._world_bible

    def observe(
        self,
        text:        str,
        *,
        agent_id:    str = "__mark__",
        session_id:  str | None = None,
        importance:  float = 0.5,
        tags:        list[str] | None = None,
        source:      str | None = None,
        metadata:    dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> "ObserveResult":
        """Top-level observe — routes to the named agent's memory.

        Equivalent to mark.runtime.memory(agent_id).observe(...). Use this
        when working with multiple agents from the same Mark instance.

        Example::

            mark.observe(
                "Elena enters the North Warehouse wearing the red scarf.",
                agent_id    = "video-agent",
                session_id  = "season-01/ep-02/scene-04",
                memory_type = "scene",
            )
        """
        mem = self._runtime_engine.memory(agent_id)  # type: ignore[union-attr]
        return mem.observe(
            text,
            session_id  = session_id,
            importance  = importance,
            tags        = tags,
            source      = source,
            metadata    = metadata,
            memory_type = memory_type,
        )

    def wrap_agent(
        self,
        llm:    Callable[..., object],
        *,
        blocks: list[str] | None = None,
        policy: str = "default",
        skills: list[str] | None = None,
        middleware: list[MarkMiddleware] | None = None,
        agent_id: str = "mark-agent",
        max_context_chars: int = 6000,
    ) -> MarkAgent:
        """Wrap an LLM callable with MARK memory, policies, and skills.

        Note: ``agent_id`` defaults to ``"mark-agent"``, unlike the
        ``"__mark__"`` default used elsewhere (``observe``, the CLI). It
        labels the agent in middleware and skill contexts and in the
        ``agent:<id>`` tag on stored outputs; memory is still read from and
        written to ``self.memory``. Pass ``agent_id="__mark__"`` to match.
        """
        skill_prompt = ""
        runtime_skill_prompt = getattr(self.runtime, "skill_prompt", None)
        if callable(runtime_skill_prompt):
            try:
                skill_prompt = runtime_skill_prompt()
            except Exception:
                skill_prompt = ""
        return MarkAgent(
            llm          = llm,
            memory       = self.memory,
            policies     = self.policies,
            skills       = self.skills,
            block_labels = blocks or [],
            policy_name  = policy,
            skill_names  = skills or [],
            skill_prompt = skill_prompt,
            runtime      = self.runtime,
            middleware   = middleware,
            agent_id     = agent_id,
            default_max_context_chars = max_context_chars,
        )

    def shutdown(self) -> None:
        """Shut down the runtime engine and release resources."""
        if hasattr(self._runtime_engine, "shutdown"):
            self._runtime_engine.shutdown()

    def __enter__(self) -> "Mark":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
