# Run log — retention A/B, 2026-10-15

Subagents: general-purpose, model inherited from the orchestrating session (Claude Opus 5.5).
MARK: v0.2.0a7 @ release/v0.2.0a7 (repo HEAD at run time recorded below), default
HashEmbeddingProvider (hash-v2:256), numpy backend, SQLite file store.

| Session | Arm | Agent's one-line summary | Tokens | Tool calls | Duration |
|---|---|---|---|---|---|
96c8681
| 1 | B (MARK) | Ingested all 124 log lines (session "day1-log") + 7 summary notes (state changes, team rosters, ticket→service→owner→team chains, ADRs/misc); 131 memories | 49,675 | 3 | 43 s |
| 1 | A (notes) | Hand-wrote notes.md: 32 people, 65 services (owner/port/region/db/release + originals), 7 tickets, 5 ADRs, misc | 51,405 | 5 | 51 s |
| 2 | A (notes) | Merged 03-09..03-16: 11 people moves, all service changes w/ history, 8 new tickets + status/assignee changes, ADR-029 | 57,563 | 4 | 73 s |
| 2 | B (MARK) | Ingested 101 lines (session day2) + 48 own notes (10 updated summaries, 38 per-service state snapshots); 280 memories | 71,390 | 8 | 175 s |
| 3 | A (notes) | Merged 03-17..03-24 in place with history trails: 11 people moves, ~38 service changes, 8 new tickets + 7 updates, on-call handover | 62,093 | 4 | 86 s |
| 3 | B (MARK) | Ingested 103 lines + 49 own notes (ticket chains, HR history, change logs, 40 current-state lines, owner→services); 432 memories | 82,605 | 15 | 265 s |
| 4 | A (notes) | Merged 03-25..04-01: 13 people moves, service changes w/ history, 8 new tickets + 8 updates, budget revision (kept original) | 67,296 | 4 | 101 s |
| 4 | B (MARK) | Ingested 115 lines + 56 own notes (HR/ownership as of 04-01, all ticket statuses, change logs, 44 current-state lines, budget); 603 memories | 92,105 | 12 | 273 s |
| 5 | A (notes) | Merged 04-02..04-09: 10 people moves, ~50 service changes, 5 new tickets + 8 updates, on-call Rosa Delgado, DR drill 04-16 | 70,572 | 4 | 111 s |
| 5 | B (MARK) | Ingested 100 lines + 10 summary notes (final team/city for everyone, on-call, change logs, new tickets, DR drill); 713 memories | 75,776 | 7 | 120 s |
| final | A (notes) | Answered 44, 3 unknown (ABS1-3) | 54,915 | 3 | 17 s |
| final | B (MARK) | Answered 44, 3 unknown (ABS1-3) | 93,080 | 9 | 88 s |
