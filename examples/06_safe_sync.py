"""Prepare memory for sync with secrets redacted (nothing is uploaded).

Run:  python examples/06_safe_sync.py
"""
from tempfile import TemporaryDirectory

from mark import CloudSync, Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    mark.memory.block("project").write("Deploys use staging. API key: sk-exampleSecret123456")

    envelope = CloudSync().prepare_delta(mark.runtime)
    print(envelope.fragments[0]["content"])

    assert "sk-exampleSecret123456" not in str(envelope.fragments)
