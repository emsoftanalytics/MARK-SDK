# Run 1 results — 2026-10-15

Setup: 5 ingest sessions + 1 question session per arm, a fresh general-purpose
subagent per session (model: Claude Opus 5.5, inherited), MARK v0.2.0a7 at
commit 96c8681 (default hash-v2 embedder, numpy backend). Protocol and prompts:
`../README.md`, `../prompts.md`. Per-session detail: `runlog.md`.

## Accuracy

| Category | A — notes file | B — MARK |
| --- | --- | --- |
| current-after-multiple-updates | 13/13 | 13/13 |
| early-fact-never-updated | 8/8 | 8/8 |
| original-value-before-changes | 5/5 | 5/5 |
| multi-hop-ticket-to-current-owner | 4/4 | 4/4 |
| multi-hop-ticket-assignee-team | 4/4 | 4/4 |
| exact-id-lookup | 4/4 | 4/4 |
| ticket-status-after-updates | 3/3 | 3/3 |
| abstention-never-mentioned | 3/3 | 3/3 |
| **Total** | **44/44** | **44/44** |

## Cost

| | A — notes file | B — MARK |
| --- | --- | --- |
| Subagent tokens (6 sessions) | 363,844 | 464,631 |
| Wall time (6 sessions) | 439 s | 964 s |
| Final memory | `notes.md`, 25 KB | 713 memories (543 raw log lines + 170 agent notes), 2.1 MB SQLite |
| Final-session tokens | 54,915 | 93,080 |

## Interpretation

- **Tie at the ceiling.** Both arms answered every question correctly, so this
  run does not separate them. The task was not hard enough for this model.
- **Why the notes arm kept up.** The agents maintained a compact "current state
  plus history trail" document. At 25 KB (about 6k tokens) the whole memory
  fits easily into every session's context, so each agent could re-read and
  rewrite it. A diligent frontier model loses nothing at this scale.
- **MARK cost more here.** The MARK agents both ingested raw lines and wrote
  their own curated summaries (170 notes), and the final agent spent tokens on
  recall queries. That's roughly 28% more tokens and 2.2× the wall time for the
  same accuracy.
- **Scaling signal.** Arm A's ingest-session cost grew every session
  (51k → 58k → 62k → 67k → 71k tokens) because each agent re-reads the whole,
  growing notes file. Arm B's cost depends on how much each agent chose to
  curate, not on the size of memory. This suggests the crossover point lies at
  a memory size well beyond this run, but the run itself does not show it.

## What this run does and does not show

It shows that MARK v0.2.0a7 retrieval is reliable enough for a relay of agents
to lose nothing over 543 changing facts. That covers exact IDs, near-duplicate
service names, multi-hop chains, original-vs-current values and abstention.

It does not show an accuracy advantage for MARK. Experiments that could:

1. **Scale:** 10× the log (about 5,000 lines over 20 sessions), so the notes
   file can no longer be re-read and rewritten each session.
2. **Retrieval-only ablation:** MARK with raw ingestion only (no agent-written
   summaries), to measure MARK's recall on its own.
3. **Weaker or cheaper models:** repeat both arms with a smaller model, where
   disciplined note-keeping is less reliable.
4. **Context budget:** cap the final agent's readable memory (e.g. 4k tokens),
   the realistic constraint for long-running agents.
