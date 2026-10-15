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
- **B — with MARK:** a `mark` CLI over MARK v0.2.0a7 (`ingest`, `store`,
  `recall`, `stats`). Agents may ingest raw log lines and store their own
  notes. No files on disk.

The asymmetry is deliberate and is the thing being measured: MARK's design is
to observe everything cheaply and retrieve relevant evidence on demand, while
an agent without a memory system must decide what to write down and carry it
forward itself.

## Files

| File | Purpose |
| --- | --- |
| `gen.py` | Deterministic generator (seed 20261015): writes `chunks/chunk_{1..5}.log`, `questions.json`, and private `answers.json` |
| `markcli.py` | The `mark` CLI used by arm B |
| `grade.py` | Strict, LLM-free grader (exact value or alias; hedges listing competing values are wrong) |
| `prompts.md` | Verbatim session prompt templates |
| `results/` | Run 1: run log, both arms' final memory and answers, graded results, `RESULTS.md` |

## Reproduce

```bash
cd benchmarks/retention_ab
python gen.py                                   # chunks/, questions.json, answers.json
mkdir -p run/armA run/armB
printf '# Halcyon Labs operational notes (relay)\n' > run/armA/notes.md
cat > run/armB/mark <<EOF
#!/bin/sh
MARK_DB="$PWD/run/armB/mark.db" exec python "$PWD/markcli.py" "\$@"   # python with mark-sdk[fast] installed
EOF
chmod +x run/armB/mark

for k in 1 2 3 4 5; do
  cp chunks/chunk_$k.log run/armA/today.log
  cp chunks/chunk_$k.log run/armB/today.log
  # Launch one fresh agent per arm with the session-K prompt from prompts.md,
  # wait for both to finish, then:
  rm run/armA/today.log run/armB/today.log
done

cp questions.json run/armA/ && cp questions.json run/armB/
# Launch the final-session prompt for each arm, then:
python grade.py run/armA/answers.json run/armB/answers.json --out results/graded.json
```

Agents must not be able to read `gen.py`, `answers.json` or `chunks/` during
the run. Keep them outside the arm workspaces, as the original run did.
