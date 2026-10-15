"""Secrets are redacted before memory is written, and sync envelopes stay redacted.

Run:  python examples/06_safe_sync.py
"""
from tempfile import TemporaryDirectory

from mark import Mark
from mark.middlewares.sync import CloudSync

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    note = mark.memory.remember("Deploys use staging. API key: sk-exampleSecret123456")

    # Redaction is on by default: the key never reaches the database.
    print(mark.memory.get(note).content)

    # A sync envelope is prepared locally; nothing is uploaded.
    envelope = CloudSync().prepare_delta(mark.runtime)
    print(envelope.fragments[0]["content"])

    assert "sk-exampleSecret123456" not in mark.memory.get(note).content
    assert "sk-exampleSecret123456" not in str(envelope.fragments)
