"""Grade answers.json files against the private gold answers.

    python grade.py armA/answers.json armB/answers.json [--key answers.json] [--out results.json]

Scoring is strict and LLM-free: an answer is correct when, after normalising
case/punctuation/whitespace, it equals the gold value or an alias, or contains
it with no other competing value of the same kind (a hedge listing several
candidate ports/names is wrong). Abstention questions accept "unknown" and
equivalents ("not mentioned", "no record", ...).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
GOLD: dict[str, dict] = {}


def load_key(path: Path) -> None:
    """Load the private gold answers written by gen.py."""
    GOLD.clear()
    GOLD.update({q["id"]: q for q in json.loads(path.read_text())})
UNKNOWN = ("unknown", "not mentioned", "no record", "not found", "no information", "not in memory", "n/a")
VALUE_PATTERNS = [r"\b\d{4}\b", r"HAL-\d{4}", r"[a-z]{2}-[a-z]+-\d"]


def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s'.:-]", " ", text)
    return " ".join(text.split())


def correct(qid: str, answer: str) -> bool:
    gold = GOLD[qid]
    ans = norm(str(answer or ""))
    if gold["answer"] == "unknown":
        return any(token in ans for token in UNKNOWN)
    if any(token == ans for token in UNKNOWN):
        return False
    targets = [norm(gold["answer"]), *(norm(a) for a in gold["aliases"])]
    if ans in targets:
        return True
    if not any(t and t in ans for t in targets):
        return False
    # Contains the gold value: reject hedges listing other values of the same kind.
    for pattern in VALUE_PATTERNS:
        found = set(re.findall(pattern, ans, flags=re.I))
        gold_found = set(re.findall(pattern, norm(gold["answer"]), flags=re.I))
        if gold_found and found - gold_found:
            return False
    return True


def grade(path: Path) -> dict:
    answers = json.loads(path.read_text())
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows = []
    for qid, gold in GOLD.items():
        ok = correct(qid, answers.get(qid, ""))
        per_cat[gold["category"]][0] += ok
        per_cat[gold["category"]][1] += 1
        rows.append({"id": qid, "category": gold["category"], "gold": gold["answer"],
                     "answer": answers.get(qid, ""), "correct": ok})
    total = sum(r["correct"] for r in rows)
    return {"file": str(path), "score": total, "total": len(rows),
            "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_cat.items())}, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--key", default=str(HERE / "answers.json"), help="gold answers from gen.py")
    parser.add_argument("--out")
    args = parser.parse_args()
    load_key(Path(args.key))
    results = [grade(Path(f)) for f in args.files]
    for res in results:
        print(f"{res['file']}: {res['score']}/{res['total']}")
        for cat, frac in res["by_category"].items():
            print(f"   {cat:38s} {frac}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
