# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Somtochukwu Owen Ikeanyi
"""Focused mixins that together make up :class:`mark.memory.mark_memory.MarkMemory`.

Each module owns one responsibility of the per-agent memory facade:

* ``core``        shared attribute contract every mixin relies on
* ``write_guard`` fragment construction, consolidation gate, quarantine
* ``write``       store / store_sync / store_many_sync (middleware-routed)
* ``fact_extraction`` automatic fact extraction after writes
* ``retrieve``    retrieve / retrieve_sync / retrieve_many_sync, escalation
* ``retrieve_async`` retrieve_many (async batch retrieval)
* ``timeline``    whole-history subject timelines (no top-k cap)
* ``healing``     explicit gap healing through a web lookup skill
* ``observing``   observe() and optional structured extraction into the graph
* ``graph_ops``   concept nodes, edges, and neighbourhood traversal
* ``lifecycle``   listing, forgetting, promoting, and clearing fragments
* ``views``       scoped helper views (characters, sessions, blocks, chain)

Import :class:`MarkMemory` from :mod:`mark.memory.mark_memory`; these modules
are implementation detail and deliberately export nothing at package level
(keeping this ``__init__`` empty avoids import cycles).
"""
