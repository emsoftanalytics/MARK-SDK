# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Graph types: nodes, edges, relations, and block links."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mark.types.memory import MemoryDocument, MemoryFragment, MemoryScope


class NodeType(str, Enum):
    """
    What kind of concept a MemoryNode represents.

    General:
    CONCEPT    abstract idea or category        ("authentication", "rate limiting")
    ENTITY     named person, place, or thing    ("FastAPI", "PostgreSQL")
    FACT       asserted statement of truth      ("API runs on port 8054")
    EVENT      something that happened          ("deploy on 2026-05-01")
    DOCUMENT   reference to a MemoryDocument   (the document itself as a node)
    FRAGMENT   reference to a MemoryFragment   (fine-grained fact node)
    TOPIC      higher-level subject grouping    ("security", "performance")
    AGENT      an agent identity node           (for multi-agent graphs)

    Media / narrative:
    CHARACTER  a named person or role           ("Elena", "Detective Ray")
    OBJECT     a named physical object          ("Red Scarf", "Gun", "Key")
    LOCATION   a named place or setting         ("North Warehouse", "Safe House")
    SCENE      a discrete scene or episode unit ("Episode-2/Scene-14")
    """
    CONCEPT   = "concept"
    ENTITY    = "entity"
    FACT      = "fact"
    EVENT     = "event"
    DOCUMENT  = "document"
    FRAGMENT  = "fragment"
    TOPIC     = "topic"
    AGENT     = "agent"
    CHARACTER = "character"
    OBJECT    = "object"
    LOCATION  = "location"
    SCENE     = "scene"


class EdgeRelation(str, Enum):
    """
    Directed semantic relationship between two MemoryNodes.

    General:
    RELATED_TO   general association; used for graph expansion during retrieval
    SUPPORTS     source provides evidence for / confirms target
    CONTRADICTS  source conflicts with target (triggers contradiction flag)
    CAUSES       source causally led to target (event → event chains)
    PART_OF      source is a component of target (hierarchy)
    DERIVED_FROM source was inferred or extracted from target
    REFERENCES   source cites or links to target (document → document)
    MENTIONS     source casually refers to target (weak link)

    Media / narrative (expansive — graph-expander traverses these):
    OWNS         character/agent owns or carries an object
    APPEARS_IN   character/object appears in a scene or episode
    LOCATED_AT   character/object is present at a location
    SYMBOLIZES   object or event carries symbolic/thematic meaning
    """
    RELATED_TO   = "related_to"
    SUPPORTS     = "supports"
    CONTRADICTS  = "contradicts"
    CAUSES       = "causes"
    PART_OF      = "part_of"
    DERIVED_FROM = "derived_from"
    REFERENCES   = "references"
    MENTIONS     = "mentions"
    OWNS         = "owns"
    APPEARS_IN   = "appears_in"
    LOCATED_AT   = "located_at"
    SYMBOLIZES   = "symbolizes"

    EXPANSIVE: ClassVar[Set["EdgeRelation"]]

    @property
    def is_expansive(self) -> bool:
        """True if the GraphExpander should traverse this edge to find related context."""
        return self in EdgeRelation.EXPANSIVE


EdgeRelation.EXPANSIVE = {
    EdgeRelation.RELATED_TO,
    EdgeRelation.SUPPORTS,
    EdgeRelation.PART_OF,
    EdgeRelation.DERIVED_FROM,
    EdgeRelation.OWNS,
    EdgeRelation.APPEARS_IN,
    EdgeRelation.LOCATED_AT,
}


class MemoryNode(BaseModel):
    """
    A vertex in the MARK knowledge graph.

    A MemoryNode wraps either a MemoryFragment (for fine-grained facts) or a
    MemoryDocument (for the document as a whole). It carries a weight that
    represents how central / important this node is — used to boost retrieval
    score when the node is reached via graph expansion.

    Retrieval pipeline:
      1. Vector search → top Fragment ids
      2. Look up MemoryNodes whose fragment_id is in that set
      3. Traverse outgoing EXPANSIVE edges
      4. Collect target nodes' fragment_ids → expand the retrieved set
    """
    model_config = ConfigDict(frozen=True)

    label:       str
    id:          str            = Field(default_factory=lambda: str(uuid4()))
    node_type:   NodeType       = NodeType.FACT
    ref_id:      str            = Field(default_factory=lambda: str(uuid4()))
    agent_id:    Optional[str]  = None
    fragment_id: Optional[str]  = None
    document_id: Optional[str]  = None
    weight:      float          = Field(default=1.0, ge=0.0, le=1.0)
    metadata:    Dict[str, Any] = Field(default_factory=dict)
    created_at:  datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("label")
    @classmethod
    def label_not_empty(cls, v: str) -> str:
        """Validate that the label is not blank."""
        v = v.strip()
        if not v:
            raise ValueError("label must not be blank")
        return v

    def summary(self) -> str:
        """Return a compact human-readable description."""
        ref = f"frag={self.fragment_id}" if self.fragment_id else f"doc={self.document_id}"
        return f"[NODE|{self.node_type.value}|w={self.weight:.2f}] {self.label!r} ({ref})"


class MemoryEdge(BaseModel):
    """
    A directed, typed, weighted connection between two MemoryNodes.

    source → relation → target

    weight reflects the strength of the relationship (1.0 = very strong,
    0.0 = very weak). The GraphExpander multiplies this weight by the
    source node's retrieval score to compute the expanded candidate's score,
    so weak edges don't flood the context with distantly related fragments.

    CONTRADICTS edges are stored here so the retrieval pipeline can detect
    when a retrieved node has a known contradiction and apply a penalty.
    """
    model_config = ConfigDict(frozen=True)

    source_id:  str
    target_id:  str
    relation:   EdgeRelation
    agent_id:   str
    id:         str            = Field(default_factory=lambda: str(uuid4()))
    weight:     float          = Field(default=1.0, ge=0.0, le=1.0)
    block_id:   Optional[str]  = None
    metadata:   Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expansive(self) -> bool:
        """Return True when graph expansion traverses this edge."""
        return self.relation.is_expansive

    def is_contradiction(self) -> bool:
        """Return True for CONTRADICTS edges."""
        return self.relation == EdgeRelation.CONTRADICTS

    def summary(self) -> str:
        """Return a compact human-readable description."""
        return (
            f"[EDGE] {self.source_id[:8]}… "
            f"─[{self.relation.value}|w={self.weight:.2f}]→ "
            f"{self.target_id[:8]}…"
        )


class BlockLink(BaseModel):
    """
    A directed, weighted connection between two MemoryBlocks.

    source_block → relation → target_block

    Block links let blocks be traversed forward and backward exactly like
    nodes are traversed through edges. Typical relations:

      "follows"     target continues from source (scene 2 follows scene 1)
      "references"  source cites facts that live in target
      "part_of"     source is a sub-topic of target
      "related_to"  general association

    The relation is a free string so domains can define their own vocabulary;
    the values above are conventions, not an enum.
    """
    model_config = ConfigDict(frozen=True)

    source_block_id: str
    target_block_id: str
    agent_id:        str
    relation:        str            = "follows"
    id:              str            = Field(default_factory=lambda: str(uuid4()))
    weight:          float          = Field(default=1.0, ge=0.0, le=1.0)
    metadata:        Dict[str, Any] = Field(default_factory=dict)
    created_at:      datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        """Return a compact human-readable description."""
        return (
            f"[BLOCK-LINK] {self.source_block_id[:8]}… "
            f"─[{self.relation}|w={self.weight:.2f}]→ "
            f"{self.target_block_id[:8]}…"
        )


def chunk_document(document: MemoryDocument, *, max_chars: int = 1000) -> List[str]:
    """Split a MemoryDocument's content into non-empty string chunks."""
    content = document.content.strip()
    return [
        content[i : i + max_chars].strip()
        for i in range(0, len(content), max_chars)
        if content[i : i + max_chars].strip()
    ]
