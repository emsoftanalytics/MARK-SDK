"""Local multi-agent shared memory bus."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from mark.embeddings import EmbeddingProvider
from mark.intelligence import RetrievalPipeline, RetrievalPolicy, RetrievalResult
from mark.store import LocalMemoryStore
from mark.types import MemoryFragment, MemoryScope, MemoryState

from mark.memory.mark_memory import MarkMemory


class GlobalMemoryBus:
    """Shared local memory bus for multi-agent workflows."""

    _BUS_AGENT = "__global_bus__"

    def __init__(
        self,
        *,
        store: LocalMemoryStore,
        pipeline: RetrievalPipeline,
        embedder: EmbeddingProvider,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._memory = MarkMemory(
            agent_id=self._BUS_AGENT,
            store=store,
            pipeline=pipeline,
            embedder=embedder,
            executor=executor,
        )

    async def publish(
        self,
        content: str,
        *,
        publisher: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
    ) -> str:
        """Publish content onto the bus."""
        return await self._memory.store(
            content,
            importance=importance,
            scope=MemoryScope.GLOBAL,
            tags=["global_bus", *(tags or [])],
            source=publisher,
            state=MemoryState.UNVERIFIED,
        )

    def publish_sync(
        self,
        content: str,
        *,
        publisher: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
    ) -> str:
        """Synchronous variant of publish()."""
        return self._memory.store_sync(
            content,
            importance=importance,
            scope=MemoryScope.GLOBAL,
            tags=["global_bus", *(tags or [])],
            source=publisher,
            state=MemoryState.UNVERIFIED,
        )

    async def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
    ) -> RetrievalResult:
        """Retrieve memory relevant to the query."""
        return await self._memory.retrieve(query, policy=policy)

    def retrieve_sync(
        self,
        query: str,
        policy: RetrievalPolicy = RetrievalPolicy.BALANCED,
    ) -> RetrievalResult:
        """Synchronous variant of retrieve()."""
        return self._memory.retrieve_sync(query, policy=policy)

    def all_facts(self, limit: int = 200) -> list[MemoryFragment]:
        """Return all bus facts up to the limit."""
        return self._memory.list(limit=limit)

    def snapshot(self) -> dict[str, str]:
        """Return a serializable snapshot of current state."""
        return {fragment.id: fragment.content for fragment in self.all_facts()}
