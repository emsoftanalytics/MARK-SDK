"""Runtime configuration of a RetrievalPipeline (late-binding setters).

Kept apart from the read path so the orchestrator stays readable; every
setter only swaps one attribute that later retrievals read.
"""
from __future__ import annotations

from mark.intelligence.compressor import ContextualCompressor
from mark.intelligence.query_expander import QueryExpander
from mark.intelligence.retrieval_policy import FusionConfig, ReinforceMode


class _PipelineConfigMixin:
    """Setters for compressor, query expander, LLM pool size, reinforcement, fusion, user subject."""

    def configure_compressor(self, compressor: ContextualCompressor) -> None:
        """Replace the active compressor at runtime (late-binding)."""
        self._compressor = compressor

    def configure_query_expander(self, expander: QueryExpander) -> None:
        """Replace the active query expander at runtime (late-binding)."""
        self._query_expander = expander

    def configure_top_k_with_llm(self, top_k: int) -> None:
        """Set the candidate pool size used when an LLM compressor is active."""
        self._top_k_with_llm = max(1, top_k)

    def configure_reinforce(self, mode: ReinforceMode | str) -> None:
        """Set how retrieval feeds back into plasticity (relevant | all | none)."""
        self._reinforce = ReinforceMode(mode)

    def configure_fusion(self, fusion: FusionConfig) -> None:
        """Set how vector and lexical evidence are fused (see FusionConfig)."""
        self._fusion = fusion

    def configure_user_subject(self, subject: str) -> None:
        """Set the fact subject that first-person questions ("Where do I live?") map to."""
        self._user_subject = subject


    def _without_tags(self, agent_id: str, allowed_ids: set[str], tags: list[str]) -> set[str]:
        """*allowed_ids* minus memories carrying any of *tags* (``SessionFilter.exclude_tags``).

        Applied before scoring, so excluded memories (resolved TODOs for an
        "open only" recall) never take a top-k slot.
        """
        id_set = getattr(self._store, "retrievable_id_set", None) or getattr(self._store, "retrievable_ids", None)
        excluded: set[str] = set()
        for tag in tags:
            if id_set is not None:
                excluded |= set(id_set(agent_id, tags=[tag]))
            else:  # custom stores: filter the fetched fragments
                excluded |= {fid for fid, f in self._get_many(allowed_ids).items() if tag in (f.tags or ())}
        return allowed_ids - excluded if excluded else allowed_ids
