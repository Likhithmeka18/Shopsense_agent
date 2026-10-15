"""
Runs a shipped demo/pipeline script OFFLINE: fake LangGraph on the path, fake litellm, and the live extractor / policy answerer
replaced by the keyword extractor and a retrieval-only knowledge base. Proves the script's own logic (assertions, resume flow,
argument handling) works; it says nothing about the real model's behaviour. Usage: python tests/demo_harness.py <script.py> [args...]
"""
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
fake = types.ModuleType("litellm"); fake.completion = lambda **k: (_ for _ in ()).throw(RuntimeError("no LLM offline")); sys.modules["litellm"] = fake

import shopsense.runtime as runtime                      # noqa: E402
from ticket_sim import RetrievalOnlyKB, keyword_extract  # noqa: E402

_real_build_deps = runtime.build_deps
_kb = RetrievalOnlyKB()


def offline_build_deps(**kw):
    kw.setdefault("extract", keyword_extract)
    if kw.get("kb") is None:
        kw["kb"] = _kb
    return _real_build_deps(**kw)


runtime.build_deps = offline_build_deps
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(str(ROOT / script), run_name="__main__")
