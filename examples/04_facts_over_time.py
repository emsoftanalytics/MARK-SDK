"""Track values that change over time and ask what is true now.

Run:  python examples/04_facts_over_time.py
"""
from tempfile import TemporaryDirectory

from mark import Mark

with TemporaryDirectory() as folder, Mark.local(folder) as mark:
    ops = mark.runtime.memory("ops")

    # Record a value each time it changes (in any order).
    ops.facts.set("checkout-api", "port", "8080", valid_from="2026-03-01")
    ops.facts.set("checkout-api", "port", "9090", valid_from="2026-03-20")
    ops.facts.set("checkout-api", "port", "8443", valid_from="2026-03-10")

    print("Now:     ", ops.facts.get("checkout-api", "port").value)
    print("History: ", [f.value for f in ops.facts.history("checkout-api", "port")])
    print("Mar 15:  ", ops.facts.as_of("checkout-api", "port", "2026-03-15").value)

    # Recall understands "now" and attaches the current fact.
    result = ops.retrieve_sync("Which port does checkout-api use now?")
    print("Recall:  ", [f.as_text() for f in result.facts])

    # And says so when it has never heard of something.
    unknown = ops.retrieve_sync("Which port does billing-api use?")
    print("Unknown: ", unknown.missing_anchors)

    assert ops.facts.get("checkout-api", "port").value == "9090"
    assert unknown.missing_anchors == ["billing api"]
