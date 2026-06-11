"""Run all live MARK examples as developer usage tests.

Run:
    python examples/run_live_examples.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


EXAMPLES = [
    "01_local_memory_live.py",
    "02_agent_ab_live.py",
    "03_sessions_and_observe_live.py",
    "04_sync_boundary_live.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in EXAMPLES:
        print(f"\n== {script} ==")
        runpy.run_path(str(root / script), run_name="__main__")
    print("\nAll live examples passed")


if __name__ == "__main__":
    main()
