# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Shared fixtures and fragment builders for the maintenance-integrity test modules."""
from __future__ import annotations

import pytest

from mark.memory.block_chain import BlockChain
from mark.memory.block_graph import BlockGraph
from mark.store import LocalMemoryStore
from mark.types import MemoryBlock, MemoryFragment


AGENT = "maint-agent"


@pytest.fixture()
def store():
    s = LocalMemoryStore(":memory:")
    yield s
    s.close()


def _frag(content: str, **kw) -> MemoryFragment:
    return MemoryFragment(content=content, agent_id=AGENT, **kw)


def _sealed_block_with(store, *fragments: MemoryFragment, name: str = "sealed") -> MemoryBlock:
    graph = BlockGraph(store, AGENT)
    block = graph.create_block(name)
    for fragment in fragments:
        store.store(fragment)
        graph.add_fragment(block.id, fragment.id)
    return BlockChain(store, AGENT).seal(block.id)
