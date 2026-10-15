# Retention A/B: agent relay with and without MARK

A controlled experiment that measures how well knowledge survives across agent
sessions when an agent has MARK as its memory versus its own hand-written notes.

## Task

A fictional company, Halcyon Labs, produces a 40-day operations log (543 lines):
64 services with near-duplicate names (`atlas-gate` vs `atlas-gateway`), 32
people, 36 tickets, architecture decisions, and 10–15 state changes per day
(ports, owners, regions, datastores, releases, team transfers, relocations,
ticket status and assignee changes, on-call handovers, budget and plan
revisions), plus office chatter as noise.

The log is split into 5 consecutive chunks. Each chunk is handled by a **fresh
agent with no context** (a relay). The chunk is deleted after its session. A
sixth fresh agent then answers 44 questions using only the memory system.

| Category | Count | What it tests |
| --- | --- | --- |
| current-after-multiple-updates | 13 | latest value after several overwrites |
| early-fact-never-updated | 8 | long-range retention of day-1 facts |
| original-value-before-changes | 5 | history that later updates overwrote |
| multi-hop-ticket-to-current-owner | 4 | ticket → service → *current* owner |
| multi-hop-ticket-assignee-team | 4 | ticket → *current* assignee → *current* team |
| exact-id-lookup | 4 | find a ticket id from its description |
| ticket-status-after-updates | 3 | latest status after transitions |
| abstention-never-mentioned | 3 | answer "unknown" for unseen entities |

## Arms

Both arms use the same model and identical prompts except for the memory
paragraph (see `prompts.md`).

- **A — without MARK:** a `notes.md` file the agents read and edit with their
  file tools. Notes must be written by the agent (no shell copying of the log).
- **B — with MARK:** a `mark` command over MARK. Agents may ingest raw log
  lines, store their own notes and, in run 2, record facts. No files on disk.
  Run 1 used `markcli.py` (MARK v0.2.0a7: `ingest`, `store`, `recall`,
  `stats`); run 2 used the SDK's own `mark` command (v0.2.0a8) with
  `mark serve`.

The asymmetry is deliberate and is the thing being measured: MARK's design is
to observe everything cheaply and retrieve relevant evidence on demand, while
an agent without a memory system must decide what to write down and carry it
forward itself.

## Files

| File | Purpose |
| --- | --- |
| `gen.py` | Deterministic generator (seed 20261015): writes `chunks/chunk_{1..5}.log`, `questions.json`, and private `answers.json` |
| `gen_data.py`, `gen_questions.py` | Generator modules used by `gen.py`: static vocabulary, and question/answer construction |
| `markcli.py` | The minimal `mark` CLI used by arm B in run 1 (run 2 used the SDK's `mark` command) |
| `grade.py` | Strict, LLM-free grader (exact value or alias; hedges listing competing values are wrong) |
| `prompts.md` | Verbatim session prompt templates |
| `results/` | Run 1: run log, both arms' final memory and answers, graded results, `RESULTS.md` |
| `results/run2/` | Run 2 (large profile, arms A, B and C): run log, final memory, graded results, `RESULTS.md` |

## Profiles

| Profile | Log | Sessions | Questions | Used in |
| --- | --- | --- | --- | --- |
| `small` (default) | 543 lines, 64 services, 32 people | 5 | 44 | run 1 (v0.2.0a7) |
| `large` | 4,494 lines, 160 services, 64 people, ~120 tickets | 10 | 61 | run 2 (v0.2.0a8) |

Run 2 adds **arm C**: the raw log is ingested into MARK mechanically and only
the final question agent runs. It measures MARK's retrieval on its own, with
no agent curation.

## Reproduce

```bash
cd benchmarks/retention_ab
python gen.py [--profile large] [--out DIR]     # chunks/, questions.json, answers.json
mkdir -p run/armA run/armB
printf '# Halcyon Labs operational notes (relay)\n' > run/armA/notes.md
cat > run/armB/mark <<EOF
#!/bin/sh
MARK_DB="$PWD/run/armB/mark.db" exec mark "\$@"   # SDK command; run 1 used: python "$PWD/markcli.py"
EOF
chmod +x run/armB/mark
run/armB/mark serve &          # optional: keeps the store warm between calls

for k in 1 2 3 4 5; do        # large profile: 1 .. 10
  cp chunks/chunk_$k.log run/armA/today.log
  cp chunks/chunk_$k.log run/armB/today.log
  # Launch one fresh agent per arm with the session-K prompt from prompts.md,
  # wait for both to finish, then:
  rm run/armA/today.log run/armB/today.log
done

cp questions.json run/armA/ && cp questions.json run/armB/
# Launch the final-session prompt for each arm, then:
python grade.py --key answers.json run/armA/answers.json run/armB/answers.json --out run/graded.json
kill %1                        # stop mark serve, if started
```

Agents must not be able to read `gen*.py`, `answers.json` or `chunks/` during
the run. Keep them outside the arm workspaces, as the original runs did.

For arm C, ingest every chunk into a third database with `mark ingest` and run
only the final-session prompt. Session prompts for both runs are in
`prompts.md`. The sessions need an agent runtime and a model; the generator
and grader run offline.
