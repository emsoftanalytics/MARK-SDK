"""Scale benchmark: store throughput, recall latency and cold start at N memories.

Runs one synthetic workload end to end and prints a JSON report:

  store     store_many_sync in batches; µs per memory
  recall    p50 / p95 over needle and generic queries in the writing process
  cold      in a fresh process: open the store + first recall, then recall again
  timeline  one entity timeline
  write+recall  alternate remember() and recall(): the recall right after a
            write (an agent's usual loop), in the writer and after a reopen
  quality   20 planted "needle" memories must be found (hit@1 / hit@5)

CI (small size, under a minute on a laptop; fails loudly on a > 2x regression
against the committed budgets in ``thresholds.json``)::

    python benchmarks/scale/bench.py --n 20000 --check

Full size (about 15-20 minutes and ~2 GB of disk for 1M; put the store on a
disk with room, not tmpfs, and keep it with --keep to rerun only the cold
phase later with --reuse)::

    python benchmarks/scale/bench.py --n 1000000 --dir /data/mark-1m --keep

Encrypted stores (needs the ``cryptography`` extra) add ``--encrypted``; the
report then also covers timelines (first and warm) and a keyword probe.
Timings are wall-clock: on a shared or busy machine compare runs made under
similar load (the report records the CPU count).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBJ = [f"svc-{i:04d}" for i in range(2000)]
VERB = ["restarted", "deployed", "scaled", "patched", "rolled back", "audited", "paged", "migrated"]
WHY = ["after a memory spike", "for the quarterly audit", "due to a bad config", "on customer request",
       "during the freeze", "because of a CVE", "after latency alarms", "to test failover"]
NEEDLES = 20
# Metrics where lower is better and that the budget file may bound.
CHECKED = ("store_us_per_item", "recall_p50_ms", "recall_p95_ms", "cold_start_ms", "timeline_warm_ms",
           "write_recall_p50_ms", "write_recall_p95_ms")


def _rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def _corpus(n: int, seed: int = 1):
    """Yield (batch of texts, needle offset or None) with needles planted mid-way."""
    rnd = random.Random(seed)
    batch = 10_000 if n >= 100_000 else 2_000
    needle_start = (n // 2 // batch) * batch
    for start in range(0, n, batch):
        items = [f"{rnd.choice(SUBJ)} was {rnd.choice(VERB)} {rnd.choice(WHY)} (ticket T-{i})."
                 for i in range(start, min(n, start + batch))]
        planted = start == needle_start
        if planted:
            for j in range(NEEDLES):
                items[j] = f"NEEDLE-{j}: the vault code for site {j} is {7000 + j}."
        yield items, planted


def _open(path: Path, args):
    from mark import Mark

    encryption = None
    if args.encrypted:
        from mark.security import FernetEncryptionProvider

        encryption = FernetEncryptionProvider(passphrase="bench-passphrase-not-secret")
    return Mark.local(str(path), embedder=args.embedder, encryption=encryption)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _recall_round(memory, needles: dict[int, str], rnd: random.Random, generic: int) -> dict:
    """Needle + generic queries; the first query is reported separately (it pays warm-up)."""
    latencies: list[float] = []
    hit1 = hit5 = 0
    first = None
    for j in range(NEEDLES):
        for query in (f"vault code for site {j}", f"NEEDLE-{j}"):
            t0 = time.perf_counter()
            ids = [f.id for f in memory.retrieve_sync(query).fragments]
            elapsed = time.perf_counter() - t0
            if first is None:
                first = elapsed
            else:
                latencies.append(elapsed)
            if needles:
                hit1 += ids[:1] == [needles[j]]
                hit5 += needles[j] in ids[:5]
    for _ in range(generic):
        t0 = time.perf_counter()
        memory.retrieve_sync(f"why was {rnd.choice(SUBJ)} {rnd.choice(VERB)}?")
        latencies.append(time.perf_counter() - t0)
    return {
        "first_ms": round(1000 * (first or 0.0), 1),
        "p50_ms": round(1000 * statistics.median(latencies), 1),
        "p95_ms": round(1000 * _percentile(latencies, 0.95), 1),
        "hit1": hit1,
        "hit5": hit5,
    }


def _write_recall_round(memory, rounds: int, rnd: random.Random, prefix: str) -> dict:
    """Alternate one remember() and one recall(); time the recall that follows each write."""
    recalls: list[float] = []
    cpu: list[float] = []
    for i in range(rounds):
        subject = rnd.choice(SUBJ)
        memory.store_sync(f"{subject} was {rnd.choice(VERB)} by the on-call bot (probe {prefix}-{i}).")
        # Thread CPU as well: on a loaded machine or a slow disk wall time
        # mostly measures the neighbours and fsync (BLAS threads would inflate process time).
        t0, c0 = time.perf_counter(), time.thread_time()
        memory.retrieve_sync(f"what happened to {subject}?")
        recalls.append(time.perf_counter() - t0)
        cpu.append(time.thread_time() - c0)
    return {
        "p50_ms": round(1000 * statistics.median(recalls), 1),
        "p95_ms": round(1000 * _percentile(recalls, 0.95), 1),
        "cpu_p50_ms": round(1000 * statistics.median(cpu), 1),
    }


def _store_phase(path: Path, args, out: dict) -> dict[int, str]:
    needles: dict[int, str] = {}
    t0 = time.perf_counter()
    with _open(path, args) as mark:
        memory = mark.runtime.memory(args.agent)
        for items, planted in _corpus(args.n):
            ids = memory.store_many_sync(items)
            if planted:
                needles = {j: ids[j] for j in range(NEEDLES)}
        out["store_total_s"] = round(time.perf_counter() - t0, 2)
        out["store_us_per_item"] = round(1e6 * out["store_total_s"] / args.n, 1)
        warm = _recall_round(memory, needles, random.Random(2), args.generic)
        out["warm_first_ms"] = warm["first_ms"]
        out["recall_p50_ms"], out["recall_p95_ms"] = warm["p50_ms"], warm["p95_ms"]
        out["needle_hit@1"], out["needle_hit@5"] = f"{warm['hit1']}/{2 * NEEDLES}", f"{warm['hit5']}/{2 * NEEDLES}"
        mixed = _write_recall_round(memory, args.write_recall, random.Random(4), "warm")
        out["write_recall_p50_ms"], out["write_recall_p95_ms"] = mixed["p50_ms"], mixed["p95_ms"]
        out["write_recall_cpu_p50_ms"] = mixed["cpu_p50_ms"]
    (path / "bench_needles.json").write_text(json.dumps(needles))
    return needles


def _cold_phase(path: Path, args, out: dict, needles: dict[int, str]) -> None:
    """Reopen (as a new process would) and time open + first recall, then steady state."""
    t0, c0 = time.perf_counter(), time.process_time()
    with _open(path, args) as mark:
        memory = mark.runtime.memory(args.agent)
        out["open_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        memory.retrieve_sync("vault code for site 3")
        out["cold_start_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        # CPU time too: on a busy machine wall time mostly measures the neighbours.
        out["cold_start_cpu_ms"] = round(1000 * (time.process_time() - c0), 1)
        again = _recall_round(memory, needles, random.Random(3), args.generic)
        out["reopen_p50_ms"], out["reopen_p95_ms"] = again["p50_ms"], again["p95_ms"]
        out["reopen_needle_hit@1"] = f"{again['hit1']}/{2 * NEEDLES}"
        t1 = time.perf_counter()
        timeline = memory.timeline("svc-0042")
        out["timeline_ms"] = round(1000 * (time.perf_counter() - t1), 1)
        out["timeline_events"] = len(getattr(timeline, "events", None) or list(timeline))
        # A second subject: encrypted stores pay a one-time sketch pass on the first.
        t1 = time.perf_counter()
        memory.timeline("svc-0100")
        out["timeline_warm_ms"] = round(1000 * (time.perf_counter() - t1), 1)
        if args.encrypted:
            t1 = time.perf_counter()
            memory.store_sync("encrypted keyword probe: zebra-7731 handled the outage")
            hits = memory.retrieve_sync("zebra-7731").fragments
            out["encrypted_keyword_ms"] = round(1000 * (time.perf_counter() - t1), 1)
            out["encrypted_keyword_hit"] = bool(hits) and "zebra-7731" in hits[0].content
        # Last: it adds memories (kept stores grow by --write-recall per rerun).
        mixed = _write_recall_round(memory, args.write_recall, random.Random(5), f"reopen-{os.getpid()}")
        out["reopen_write_recall_p50_ms"], out["reopen_write_recall_p95_ms"] = mixed["p50_ms"], mixed["p95_ms"]


def _cold_in_child(path: Path, argv: list[str]) -> dict:
    """Run the cold phase in a fresh interpreter, as a restarted application would."""
    report = path / "bench_cold.json"
    keep = [a for a in argv if a not in ("--check", "--keep")]
    for flag in ("--dir", "--json", "--thresholds"):
        if flag in keep:
            i = keep.index(flag)
            del keep[i:i + 2]
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *keep, "--reuse", "--dir", str(path), "--json", str(report)],
        check=True, stdout=subprocess.DEVNULL,
    )
    child = json.loads(report.read_text())
    return {k: v for k, v in child.items() if k not in ("n", "embedder", "encrypted", "cpus", "db_mb")}


def _check(out: dict, thresholds: Path) -> list[str]:
    """Return a failure line per metric that exceeds its budget (2x the reference run)."""
    budgets = json.loads(thresholds.read_text()).get(str(out["n"]), {})
    failures = []
    for metric in CHECKED:
        limit = budgets.get(metric)
        if limit is not None and out.get(metric) is not None and out[metric] > limit:
            failures.append(f"{metric}={out[metric]} exceeds budget {limit} (2x reference)")
    if out.get("needle_hit@5", "0/0").split("/")[0] != str(2 * NEEDLES):
        failures.append(f"needle_hit@5={out.get('needle_hit@5')} (every needle must be found)")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--agent", default="scale")
    parser.add_argument("--generic", type=int, default=40, help="generic queries per round")
    parser.add_argument("--write-recall", type=int, default=30, help="remember/recall pairs per write+recall round")
    parser.add_argument("--dir", help="store directory (default: a temp dir, removed afterwards)")
    parser.add_argument("--keep", action="store_true", help="keep the store directory")
    parser.add_argument("--reuse", action="store_true", help="skip the store phase; reuse --dir")
    parser.add_argument("--encrypted", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail on > 2x regression vs thresholds.json")
    parser.add_argument("--thresholds", default=str(HERE / "thresholds.json"))
    parser.add_argument("--json", help="also write the report to this file")
    args = parser.parse_args(argv)

    path = Path(args.dir) if args.dir else Path(tempfile.mkdtemp(prefix=f"mark_scale_{args.n}_"))
    path.mkdir(parents=True, exist_ok=True)
    out: dict = {"n": args.n, "embedder": args.embedder, "encrypted": args.encrypted, "cpus": os.cpu_count()}
    try:
        if args.reuse:
            needles = {int(k): v for k, v in json.loads((path / "bench_needles.json").read_text()).items()}
            _cold_phase(path, args, out, needles)
            out["rss_peak_mb"] = _rss_mb()
        else:
            _store_phase(path, args, out)
            out["rss_store_mb"] = _rss_mb()
            out.update(_cold_in_child(path, argv if argv is not None else sys.argv[1:]))
        out["db_mb"] = round(sum(f.stat().st_size for f in (path / ".mark").iterdir()) / 1e6, 1)
    finally:
        if not (args.keep or args.reuse):
            shutil.rmtree(path, ignore_errors=True)
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    if args.check:
        failures = _check(out, Path(args.thresholds))
        for line in failures:
            print(f"SCALE REGRESSION: {line}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
