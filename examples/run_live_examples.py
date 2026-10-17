"""Run every example as a quick usage test.

Run:  python examples/run_live_examples.py
"""
import runpy
import sys
from pathlib import Path

EXAMPLES = [
    "01_remember_and_recall.py",
    "02_agent_with_memory.py",
    "03_sessions.py",
    "04_facts_over_time.py",
    "05_context_watch.py",
    "06_safe_sync.py",
    "07_briefing_any_loop.py",
    "08_visual_continuity.py",
    "09_context_replacement.py",
    "10_coding_agent_memory.py",
    "11_documents_and_files.py",
]

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder))  # lets examples import the shared _llm helper
    for script in EXAMPLES:
        print(f"\n== {script} ==")
        runpy.run_path(str(folder / script), run_name="__main__")
    print("\nAll examples passed.")
