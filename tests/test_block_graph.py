# SPDX-License-Identifier: MIT
# Copyright (c) 2025 MARK Contributors
"""Graph-scoped memory blocks: membership, block-scoped edges, links, traversal."""

import pytest

from mark.memory.block_graph import BlockGraph
from mark.store import LocalMemoryStore
from mark.types import BlockStatus, MemoryFragment, MemoryNode
from mark.types.graph import EdgeRelation, NodeType

AGENT = "block-test-agent"


@pytest.fixture()
def store():
    s = LocalMemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def graph(store):
    return BlockGraph(store, AGENT)


def _node(store, label, fragment_id=None):
    node = MemoryNode(label=label, node_type=NodeType.CHARACTER, agent_id=AGENT, fragment_id=fragment_id)
    store.store_node(node)
    return node


def test_create_block_defaults(graph):
    block = graph.create_block("ep01", session_id="s01/ep01", block_type="world-bible")
    assert block.status == BlockStatus.OPEN
    assert block.session_id == "s01/ep01"
    assert block.block_type == "world-bible"
    fetched = graph.get(block.id)
    assert fetched is not None
    assert fetched.name == "ep01"
    assert graph.by_name("ep01").id == block.id


def test_node_membership_is_many_to_many(graph, store):
    b1 = graph.create_block("scene-1")
    b2 = graph.create_block("scene-2")
    elena = _node(store, "Elena")

    graph.add_node(b1.id, elena.id)
    graph.add_node(b2.id, elena.id)
    graph.add_node(b1.id, elena.id)  # idempotent

    assert [n.id for n in graph.nodes(b1.id)] == [elena.id]
    assert [n.id for n in graph.nodes(b2.id)] == [elena.id]
    assert {b.id for b in graph.blocks_for_node(elena.id)} == {b1.id, b2.id}

    assert graph.remove_node(b1.id, elena.id) is True
    assert graph.nodes(b1.id) == []
    assert {b.id for b in graph.blocks_for_node(elena.id)} == {b2.id}


def test_add_fragment_assigns_block(graph, store):
    block = graph.create_block("topic")
    fragment = MemoryFragment(content="Elena wears a red scarf.", agent_id=AGENT)
    store.store(fragment)
    graph.add_fragment(block.id, fragment.id)
    fragments = store.fragments_for_block(block.id)
    assert [f.id for f in fragments] == [fragment.id]


def test_block_scoped_edge_requires_membership(graph, store):
    block = graph.create_block("scene")
    a = _node(store, "Elena")
    b = _node(store, "Red Scarf")
    graph.add_node(block.id, a.id)

    with pytest.raises(ValueError, match="members"):
        graph.add_edge(block.id, a.id, b.id, EdgeRelation.OWNS)

    graph.add_node(block.id, b.id)
    edge = graph.add_edge(block.id, a.id, b.id, "owns")
    assert edge.block_id == block.id
    assert [e.id for e in graph.edges(block.id)] == [edge.id]


def test_block_links_forward_and_backward(graph):
    b1 = graph.create_block("ep1")
    b2 = graph.create_block("ep2")
    link = graph.link(b1.id, b2.id, relation="follows")

    assert [l.id for l in graph.links_forward(b1.id)] == [link.id]
    assert [l.id for l in graph.links_backward(b2.id)] == [link.id]
    assert graph.links_forward(b2.id) == []

    assert graph.unlink(link.id) is True
    assert graph.links_forward(b1.id) == []


def test_link_validation(graph):
    b1 = graph.create_block("solo")
    with pytest.raises(ValueError):
        graph.link(b1.id, b1.id)
    with pytest.raises(KeyError):
        graph.link(b1.id, "missing-block")


def test_traverse_forward_backward_both(graph):
    b1 = graph.create_block("b1")
    b2 = graph.create_block("b2")
    b3 = graph.create_block("b3")
    graph.link(b1.id, b2.id)
    graph.link(b2.id, b3.id)

    forward = graph.traverse(b1.id, direction="forward", depth=3)
    assert [b.id for b in forward] == [b2.id, b3.id]

    backward = graph.traverse(b3.id, direction="backward", depth=3)
    assert [b.id for b in backward] == [b2.id, b1.id]

    both = graph.traverse(b2.id, direction="both", depth=1)
    assert {b.id for b in both} == {b1.id, b3.id}

    shallow = graph.traverse(b1.id, direction="forward", depth=1)
    assert [b.id for b in shallow] == [b2.id]

    with pytest.raises(ValueError):
        graph.traverse(b1.id, direction="sideways")


def test_traverse_skips_quarantined_blocks(graph, store):
    from mark.memory.block_chain import BlockChain
    b1 = graph.create_block("good-1")
    b2 = graph.create_block("bad")
    b3 = graph.create_block("good-2")
    graph.link(b1.id, b2.id)
    graph.link(b2.id, b3.id)

    BlockChain(store, AGENT).quarantine(b2.id)

    reached = graph.traverse(b1.id, direction="forward", depth=3)
    assert [b.id for b in reached] == []  # path is cut at the quarantined block

    including = graph.traverse(b1.id, direction="forward", depth=3, include_quarantined=True)
    assert [b.id for b in including] == [b2.id, b3.id]


def test_writes_rejected_on_non_open_blocks(graph, store):
    from mark.memory.block_chain import BlockChain
    block = graph.create_block("sealed-soon")
    node = _node(store, "Elena")
    graph.add_node(block.id, node.id)

    BlockChain(store, AGENT).seal(block.id)
    with pytest.raises(ValueError, match="OPEN"):
        graph.add_node(block.id, node.id)
    with pytest.raises(ValueError, match="OPEN"):
        graph.add_fragment(block.id, "any-fragment")


def test_list_by_status(graph, store):
    from mark.memory.block_chain import BlockChain
    open_block = graph.create_block("open")
    sealed_block = graph.create_block("done")
    BlockChain(store, AGENT).seal(sealed_block.id)

    assert {b.id for b in graph.list()} == {open_block.id, sealed_block.id}
    assert [b.id for b in graph.list(status=BlockStatus.OPEN)] == [open_block.id]
    assert [b.id for b in graph.list(status=BlockStatus.SEALED)] == [sealed_block.id]
