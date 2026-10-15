"""Building blocks of :class:`mark.store.sqlite.LocalMemoryStore`.

The store is one class composed from focused mixins, one per concern:

- ``codec``      — shared constants and pure row/value helpers
- ``ddl``        — table, index and migration definitions (data only)
- ``schema``     — schema creation, migrations, FTS5 setup, meta, maintenance
- ``connection`` — per-thread connections, locking, transactions, listeners
- ``fragments``  — fragment writes, batch writes, access/importance updates
- ``fragment_reads`` — fragment lookups, filtered listings, embeddings
- ``search``     — cached retrievable id sets, lexical (FTS5) search
- ``blocks``     — memory blocks, block membership and block links
- ``graph``      — documents, nodes and edges
- ``facts``      — bi-temporal (subject, attribute) -> value facts
- ``events``     — session event log
- ``rows``       — decoding SQLite rows into domain models

Import the public class from :mod:`mark.store.sqlite`, not from here.
"""
