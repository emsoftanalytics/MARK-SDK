from __future__ import annotations

from pathlib import Path

from mark import Mark, MarkRuntime
from mark.embeddings import HashEmbeddingProvider
from mark.index import VectorIndex
from mark.intelligence import RetrievalPolicy
from mark.plugins import HOOK_SCORING, MarkCorePlugin, PluginRegistry
from mark.sandbox import DockerSandbox
from mark.security import ContentHasher, NoOpEncryptionProvider
from mark.skills import AgentPersona
from mark.store import LocalMemoryStore
from mark.types import MemoryEdge, MemoryFragment, MemoryNode, MemoryScope, MemoryState
from mark.types.graph import EdgeRelation


def test_mark_local_uses_sqlite_backend(tmp_path: Path) -> None:
    mark = Mark.local(project_path=tmp_path)
    mark.memory.block("project").write("Current SDK SQLite backend works.")

    bundle = mark.memory.retrieve("SQLite backend")

    assert "SQLite backend works" in bundle.as_text()
    assert (tmp_path / ".mark" / "memory.db").exists()
    mark.shutdown()


def test_sqlite_store_hashes_and_round_trips_fragment() -> None:
    store = LocalMemoryStore(encryption=NoOpEncryptionProvider())
    fragment = MemoryFragment(
        content="MARK stores durable local fragments.",
        agent_id="agent-a",
        state=MemoryState.UNVERIFIED,
    )

    fragment_id = store.store(fragment)
    loaded = store.get(fragment_id)

    assert loaded is not None
    assert loaded.content == "MARK stores durable local fragments."
    assert ContentHasher.METADATA_KEY in loaded.metadata
    assert ContentHasher.check(loaded)


def test_vector_index_and_mark_runtime_retrieve_memory() -> None:
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider(dim=64))
    memory = runtime.memory("coder")

    memory.store_sync("The project uses FastAPI for the cloud API.", importance=0.9)
    memory.store_sync("The local SDK starts from a lightweight hippocampus.", importance=0.8)
    result = memory.retrieve_sync("Which framework powers the cloud API?", policy=RetrievalPolicy.BALANCED)

    assert result.fragments
    assert any("FastAPI" in fragment.content for fragment in result.fragments)
    assert result.scores[0] > 0
    runtime.shutdown()


def test_graph_expansion_recovers_related_fragment() -> None:
    runtime = MarkRuntime.local(embedder=HashEmbeddingProvider(dim=64))
    first_id = runtime.memory("agent").store_sync("Alpha module owns authentication.")
    second_id = runtime.memory("agent").store_sync("Beta module validates sessions.")
    source = MemoryNode(label="auth", agent_id="agent", fragment_id=first_id)
    target = MemoryNode(label="sessions", agent_id="agent", fragment_id=second_id)
    runtime.store.store_node(source)
    runtime.store.store_node(target)
    runtime.store.store_edge(
        MemoryEdge(
            source_id=source.id,
            target_id=target.id,
            relation=EdgeRelation.RELATED_TO,
            agent_id="agent",
            weight=1.0,
        )
    )

    result = runtime.memory("agent").retrieve_sync("authentication", policy=RetrievalPolicy.DEEP)

    assert {fragment.id for fragment in result.fragments} >= {first_id, second_id}
    runtime.shutdown()


def test_persona_global_bus_plugin_registry_and_sandbox_boundaries() -> None:
    runtime = MarkRuntime.local()
    persona = AgentPersona("agent", runtime)
    persona.learn_sync("I prefer concise, tested Python code.")
    bus = runtime.global_bus()
    bus.publish_sync("Shared milestone is SDK hippocampus MVP.", publisher="agent")

    assert "tested Python code" in persona.as_context_sync()
    assert "SDK hippocampus MVP" in " ".join(bus.snapshot().values())

    class DummyPlugin(MarkCorePlugin):
        name = "dummy"
        version = "0.1.0"
        provides = {HOOK_SCORING}

        def get_hook(self, hook_name: str):
            return lambda *args, **kwargs: []

    registry = PluginRegistry()
    registry.register(DummyPlugin())
    assert registry.has(HOOK_SCORING)

    sandbox = DockerSandbox(runner=_fake_runner)
    result = sandbox.execute("print('blocked')")
    assert result.sandbox_mode == "blocked/docker-unavailable"
    runtime.shutdown()


def test_vector_index_persistence(tmp_path: Path) -> None:
    index = VectorIndex()
    index.add("frag-1", [1.0, 0.0], "alpha")
    index.add("frag-2", [0.0, 1.0], "beta")
    path = tmp_path / "index.json"

    index.save(path)
    loaded = VectorIndex.load(path)

    assert loaded.search([1.0, 0.0], top_k=1)[0].fragment_id == "frag-1"


class _Completed:
    returncode = 1
    stdout = ""
    stderr = "docker unavailable"


def _fake_runner(*args, **kwargs):
    return _Completed()
