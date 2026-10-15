# Run log — retention A/B run 2 (large profile), 2026-10-15

Subagents: general-purpose, model inherited from the orchestrating session (Claude Opus 5.5).
MARK: v0.2.0a8 @ release/v0.2.0a8 a981601, SDK `mark` CLI with `mark serve`,
default HashEmbeddingProvider (hash-v2:256), numpy backend, SQLite file store.
Arm C: harness ingested chunks 1-10 (4,494 lines) with `mark ingest`; no curation agents.

| Session | Arm | Agent's one-line summary | Tokens | Tool calls | Duration |
|---|---|---|---|---|---|
| 1 | A (notes) | Wrote 64 people, 156 services, 12 tickets, on-call, ADR-007, Wi-Fi; changelog section for later agents | 64,996 | 6 | 87 s |
| final | C (MARK raw) | Answered 61, 3 unknown (ABS1-3). DEVIATION: read mark.db directly (SQL) to break same-day ties for MH10, ST1, CU11; briefly wrote a temp file in run/ | 94,424 | 20 | 132 s |
| 1 | B (MARK) | Ingested 239 lines + 1,003 dated facts (services, people, tickets, misc) + 1 handover note | 58,473 | 8 | 183 s |
| 2 | A (notes) | Merged 03-05..03-08: 16 new tickets + all reassignments/status changes, ADR-011/013/019/023, DR drill, budget, ~70 service changes, 16 people changes, changelog | 70,921 | 4 | 114 s |
| 2 | B (MARK) | Ingested 155 lines + 136 fact updates + ADRs/drill/budget facts + 1 note on same-day changes (found: fact history keeps only the last value per date) | 50,713 | 5 | 75 s |
| 3 | A (notes) | Merged 03-09..03-12: 16 new tickets, ADR-029, ~200 service changes, ~75 people changes, all ticket changes as inline original->new[date] chains | 105,993 | 6 | 231 s |
| 3 | B (MARK) | Ingested 524 lines + 562 fact updates + 2 notes (same-day sequences; ticket->service->owner->team chains). Hit client/server version skew: `fact set` via server raised TypeError (client code updated mid-run, server still old) -> agent used `--no-server` (in-process, newer code) | 63,119 | 23 | 463 s |
| — | B, C | Servers restarted on a8@071269e (handshake fingerprint fix). Sessions 4-10 and finals use this build; sessions 1-2 used a8@a981601 via server; session 3 used in-process a8@6278455 for fact writes. | | | |
| 4 | B (MARK) | Ingested 525 lines + 526 dated fact updates + 2 notes (new-ticket chains; same-day repeats) | 70,551 | 11 | 222 s |
| 4 | A (notes) | Merged 03-13..03-16: 13 new tickets, ~250 service, ~45 HR, ~70 ticket changes as dated old->new chains; ownership notes; "no change" markers | 115,311 | 5 | 244 s |
| 5 | B (MARK) | Ingested 528 lines + 540 fact updates (incl. on-call handover) + summary note; left a stray test fact it could not remove (no retract API — product gap) | 53,781 | 10 | 145 s |
| 5 | A (notes) | Merged 03-17..03-20: ~270 service, ~60 HR, ~75 ticket changes, 12 new tickets, on-call handover; did not re-read file after writing | 137,593 | 5 | 339 s |
| 6 | B (MARK) | Ingested 499 lines + 524 dated fact updates + summary note (same-day steps; 12 new ticket chains) | 52,610 | 9 | 371 s |
| 6 | A (notes) | Merged 03-21..03-24: ~330 service, ~56 people changes, 12 new tickets + all ticket changes, changelog with same-day chains | 151,677 | 7 | 385 s |
| 7 | B (MARK) | Ingested 542 lines + ~510 dated facts (incl. Q2 budget revision) + 2 notes (same-day steps; new ticket chains) | 58,288 | 13 | 161 s |
| 7 | A (notes) | Merged 03-25..03-28: 12 new tickets, ~290 service, ~70 people, ~75 ticket changes, budget revision; changelog + ownership notes | 162,720 | 5 | 403 s |
| 8 | B (MARK) | Ingested 508 lines + 521 fact updates + 3 notes (new ticket chains, same-day steps, no-change markers) | 54,946 | 10 | 138 s |
| 8 | A (notes) | Merged 03-29..04-01: ~280 service, ~55 people, ~65 ticket updates, 12 new tickets, unchanged-markers, ownership notes, changelog | 241,498 | 244 | 775 s |
| 9 | B (MARK) | Ingested 465 lines + 475 facts (incl. DR drill postponement) + summary note + postponement memory | 48,260 | 8 | 109 s |
| 9 | A (notes) | Merged 04-02..04-05: ~255 service, ~65 HR, ~80 ticket changes, 12 new tickets, DR drill postponement; per-entity counts checked | 265,444 | 294 | 822 s |
| 10 | B (MARK) | Ingested 509 lines + 456 fact updates + 2 notes (on-call handover to Rosa Delgado; 3 new tickets) | 47,714 | 8 | 135 s |
| — | C | First C answers (build a8@a981601) moved to results_v1/; C final re-run on a8@071269e | | | |
| final | B (MARK) | Answered 61, 3 unknown (ABS1-3) | 67,317 | 10 | 73 s |
| final (re-run) | C (MARK raw) | Answered 61, 3 unknown; flagged MH10 as a coin flip (same-day transfers, recall output shows no order) | 81,852 | 15 | 113 s |
| 10 | A (notes) | Merged 04-06..04-09: 437 change lines (count-checked), 3 new tickets, on-call handover; changelog | 258,562 | 253 | 851 s |
| final | A (notes) | Answered 61, 3 unknown (ABS1-3) | 116,085 | 4 | 29 s |
