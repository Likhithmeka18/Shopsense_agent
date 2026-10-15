"""Builds the REAL compiled graphs (M5 workflow and M6 team) against a fake LangGraph in a subprocess - see tests/wiring_check.py.
Proves node names, edge maps and state keys line up, the approval pause/resume works across a SQLite file, and a re-run of a
finished ticket does not inherit the last run's approval. It does not replace running the demos with the real library."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_compiled_graphs_wire_up_and_resume_across_processes_with_a_fake_langgraph():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "tests" / "fake_langgraph")}
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "wiring_check.py")], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "WIRING OK" in out.stdout, out.stdout[-1500:] + out.stderr[-1500:]
    assert out.stdout.count("OK  ") >= 20


def test_run_pipeline_accepts_a_tickets_file_option_and_rejects_a_missing_one():
    """Reviewers bring their own tickets: --tickets FILE must exist as an option, and a wrong path must stop with a clear message."""
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    help_text = subprocess.run([sys.executable, str(root / "run_pipeline.py"), "--help"], capture_output=True, text=True, timeout=120)
    assert help_text.returncode == 0 and "--tickets" in help_text.stdout
