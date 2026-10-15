# Run 2 results — large profile, 2026-10-15

Setup: `gen.py --profile large` (4,494-line log, 160 services with
near-duplicate names, 64 people, ~120 tickets, 10 sessions, 61 questions).
A fresh general-purpose subagent per session (Claude Opus 5.5). MARK v0.2.0a8
through the SDK's `mark` command with a warm `mark serve`. Per-session detail
and build changes during the run: `runlog.md`.

## Arms

- **A — without MARK:** agents keep a hand-written `notes.md`.
- **B — with MARK:** agents ingest the log and record facts and notes in MARK.
- **C — MARK, raw log only:** the harness ingests the log mechanically; only
  the final question agent runs.

## Accuracy

| Category | A — notes | B — MARK, curated | C — MARK, raw |
| --- | --- | --- | --- |
| current-after-multiple-updates | 18/18 | 18/18 | 18/18 |
| early-fact-never-updated | 11/11 | 11/11 | 11/11 |
| original-value-before-changes | 8/8 | 8/8 | 8/8 |
| exact-id-lookup | 6/6 | 6/6 | 6/6 |
| multi-hop-ticket-to-current-owner | 5/5 | 5/5 | 5/5 |
| multi-hop-ticket-assignee-team | 5/5 | 5/5 | 4/5 |
| ticket-status-after-updates | 5/5 | 5/5 | 5/5 |
| abstention-never-mentioned | 3/3 | 3/3 | 3/3 |
| **Total** | **61/61** | **61/61** | **60/61** |

## Cost

| | A — notes | B — MARK, curated | C — MARK, raw |
| --- | --- | --- | --- |
| Agent tokens, total | 1,690,800 | 625,772 | 81,852 |
| Relative to A | 1× | 2.7× fewer | 20.7× fewer |
| Session 1 → session 10 | 65k → 259k | 58k → 48k | — |
| Tool calls in session 10 | 253 | 8 | — |
| Wall time | 71 min | 35 min | 2 min |
| Final memory | 118 KB notes file | 4.5k memories + ~5k facts | 4.5k memories |

## Findings

1. **Accuracy held for all three arms** on a 10× harder task. A careful
   frontier model keeping notes still answered everything; MARK matched it.
2. **Cost is where they diverge.** Without MARK, every agent must re-read and
   rewrite the entire notes file, so per-session cost grew 4× as memory grew
   and the agent fell back to hundreds of small edits once the file was large.
   With MARK, per-session cost stayed flat because agents only pull what they
   need. MARK-B used 2.7× fewer tokens and half the time.
3. **Raw MARK is the cheapest by far.** Storing the log verbatim and asking
   questions cost 82k tokens for the whole job (21× less than notes) for 60/61.
4. **The one miss is a real MARK limitation.** MH10 needed the later of two
   same-day transfers for one person. MARK stores them in order and orders
   them for "now" queries about that subject, but the follow-up recall the
   agent made (about the person, not "now") listed both without showing
   which came last. Follow-up: expose same-day order in recall output.

## Integrity notes

- The first arm-C answer set (build a981601) read `mark.db` directly to break
  three same-day ties, against the protocol. It is kept as
  `armC/answers_first_attempt_protocol_deviation.json` and not counted. The
  counted arm-C answers are a clean re-run on build 071269e.
- During sessions 1–3 the build changed under arm B (a client/server version
  skew made `fact set` fail through the server; the agent ran in-process).
  This led to the handshake fingerprint fix. Servers were restarted on
  071269e for sessions 4–10 and the finals. See `runlog.md`.
- Single run per arm, one model, synthetic task: treat the numbers as
  directional.

## Artefacts

`armA/notes.md`, `armB/mark.db.gz`, `armC/mark.db.gz` (gunzip to inspect with
`mark --db ... recall`), every arm's `answers.json`, `graded.json`, `runlog.md`.
