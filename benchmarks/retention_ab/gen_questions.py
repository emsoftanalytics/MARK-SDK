"""Question/answer construction for the retention benchmark generator (gen.py).

``build_questions`` is called by gen.py after the log has been generated, with
the same ``random.Random`` instance, so the rng call order (and therefore the
output) is identical to the original single-file generator.
"""
from __future__ import annotations

import random


def build_questions(rng: random.Random, state: dict, history: dict, services: list[str],
                    ticket_ids: list[str], large: bool) -> list[dict]:
    """Return the gold question list (id, category, question, answer, aliases)."""
    questions: list[dict] = []
    # Per-category question counts.
    CU_N, HIST_N, LONG_N, MH_N, ID_N, ST_N = (15, 8, 8, 5, 6, 5) if large else (10, 5, 5, 4, 4, 3)

    def changes(kind: str, name: str, attr: str) -> int:
        """Number of updates after the initial value."""
        return len(history[(kind, name, attr)]) - 1

    def ask(qid: str, cat: str, text: str, answer, aliases: list[str] | None = None) -> None:
        questions.append({"id": qid, "category": cat, "question": text,
                          "answer": str(answer), "aliases": aliases or []})

    # Current value of a service attribute that changed at least twice.
    updated = [(s, a) for s in services for a in ("port", "owner", "region", "db") if changes("svc", s, a) >= 2]
    rng.shuffle(updated)
    for n, (s, a) in enumerate(updated[:CU_N], 1):
        phrase = {"port": f"Which port does {s} listen on now?", "owner": f"Who currently owns {s}?",
                  "region": f"Which region does {s} run in now?", "db": f"Which datastore does {s} use now?"}[a]
        ask(f"CU{n}", "current-after-multiple-updates", phrase, state["svc"][s][a])

    # Original value of an attribute that later changed.
    hist_pool = [(s, a) for s in services for a in ("port", "region", "db")
                 if changes("svc", s, a) >= 1 and (s, a) not in updated[:CU_N]]
    rng.shuffle(hist_pool)
    for n, (s, a) in enumerate(hist_pool[:HIST_N], 1):
        first = history[("svc", s, a)][0][1]
        label = {"port": "port", "region": "region", "db": "datastore"}[a]
        ask(f"HIST{n}", "original-value-before-changes",
            f"What was the original {label} of {s} when it was first inventoried?", first)

    # Early facts that were never updated (plus three fixed ones).
    long_pool = [(s, a) for s in services for a in ("port", "region", "db", "owner") if changes("svc", s, a) == 0]
    rng.shuffle(long_pool)
    for n, (s, a) in enumerate(long_pool[:LONG_N], 1):
        phrase = {"port": f"Which port does {s} listen on?", "owner": f"Who owns {s}?",
                  "region": f"Which region does {s} run in?", "db": f"Which datastore does {s} use?"}[a]
        ask(f"LONG{n}", "early-fact-never-updated", phrase, state["svc"][s][a])
    ask(f"LONG{LONG_N + 1}", "early-fact-never-updated",
        "Per ADR-019, what percentage of production traffic does the canary stage receive?",
        "7 percent", ["7%", "7 percent", "seven percent"])
    ask(f"LONG{LONG_N + 2}", "early-fact-never-updated", "What is the retention period for audit logs?", "400 days")
    ask(f"LONG{LONG_N + 3}", "early-fact-never-updated", "What is the guest Wi-Fi network name?", "halcyon-guest-7")

    # Multi-hop: ticket -> service -> current owner, and ticket -> assignee -> team.
    mh = [t for t in ticket_ids]
    rng.shuffle(mh)
    for n, tid in enumerate(mh[:MH_N], 1):
        svc = state["ticket"][tid]["service"]
        ask(f"MH{n}", "multi-hop-ticket-to-current-owner",
            f"Who currently owns the service affected by ticket {tid}?", state["svc"][svc]["owner"])
    for n, tid in enumerate(mh[MH_N:2 * MH_N], MH_N + 1):
        assignee = state["ticket"][tid]["assignee"]
        ask(f"MH{n}", "multi-hop-ticket-assignee-team",
            f"Which team is the current assignee of ticket {tid} on now?", state["person"][assignee]["team"])

    # Exact ticket-ID lookup by topic, and ticket status after several updates.
    id_pool = [t for t in ticket_ids if t not in mh[:2 * MH_N]]
    rng.shuffle(id_pool)
    for n, tid in enumerate(id_pool[:ID_N], 1):
        ask(f"ID{n}", "exact-id-lookup",
            f"Which ticket ID tracks the issue '{state['ticket'][tid]['topic']}'?", tid)
    status_pool = [t for t in ticket_ids if changes("ticket", t, "status") >= 2]
    rng.shuffle(status_pool)
    for n, tid in enumerate(status_pool[:ST_N], 1):
        ask(f"ST{n}", "ticket-status-after-updates",
            f"What is the current status of ticket {tid}?", state["ticket"][tid]["status"])

    # Scripted misc facts that were updated on fixed days.
    ask("MISC1", "current-after-multiple-updates", "Who is the current primary on-call?",
        state["misc"]["oncall-primary"]["value"])
    ask("MISC2", "current-after-multiple-updates", "What is the current approved Q2 infrastructure budget?",
        state["misc"]["q2-infra-budget"]["value"], ["455,000", "455000", "455k"])
    ask("MISC3", "current-after-multiple-updates", "When is the disaster-recovery drill scheduled now?",
        state["misc"]["dr-drill"]["value"], ["April 16", "16 April"])

    # Abstention: entities that never appear in the log.
    ask("ABS1", "abstention-never-mentioned", "Which port does the titan-billing service listen on?", "unknown")
    ask("ABS2", "abstention-never-mentioned", "Which team is Marcus Webb on?", "unknown")
    ask("ABS3", "abstention-never-mentioned", "What is the status of ticket HAL-0042?", "unknown")

    assert len({q["id"] for q in questions}) == len(questions), "duplicate question ids"
    return questions
