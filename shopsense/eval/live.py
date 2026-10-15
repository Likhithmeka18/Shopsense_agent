"""
ShopSense Eval: run the golden set END TO END through the real ticket graph with the real LLM (needs GROQ_API_KEY or
whatever SHOPSENSE_LLM_MODEL points at, plus `langgraph`).

    python run_golden_eval.py --live

Each case gets a fresh MockAPI, ledger and contact history (so one case's refunds can never leak into the next), but the
policy index is built once. A case that raises -- a rate limit, a network error -- is recorded as FAILED with the error
text instead of aborting the run, and a per-case pause (`SHOPSENSE_EVAL_PAUSE_S`, default 2) keeps free-tier token budgets alive.
"""
from __future__ import annotations

import os
import time

from ..tools.order_tools import MockAPI
from ..tools.policy_tools import PolicyKnowledgeBase
from ..workflows import ticket_resolution_graph as g
from .golden import score_case


def outcome_for_case(graph, deps_factory, case: dict) -> dict:
    t = case["ticket"]
    deps = deps_factory()
    graph = graph(deps)
    state = g.seed_state(ticket_id=t["ticket_id"], customer_id=t["customer_id"], text=t["text"], channel=t.get("channel", "chat"),
                         history=case.get("history"))
    result, _config = g.run_ticket(graph, state)
    return g.outcome_of(result)


def run_cases_live(cases: list[dict], *, kb=None, graph_builder=None, extract=None, pause_s: float | None = None, verbose: bool = True) -> list[dict]:
    if graph_builder is None:
        if not g.LANGGRAPH_AVAILABLE:
            raise RuntimeError("--live needs langgraph: pip install langgraph langgraph-checkpoint-sqlite")
        graph_builder = g.build_ticket_graph
    kb = kb or PolicyKnowledgeBase()
    pause = float(os.environ.get("SHOPSENSE_EVAL_PAUSE_S", "2")) if pause_s is None else pause_s
    scores = []
    for case in cases:
        try:
            out = outcome_for_case(graph_builder, lambda: g.Deps(api=MockAPI(), kb=kb, **({'extract': extract} if extract else {})), case)
            s = score_case(case, out)
            s["answer"] = out["answer"]
        except Exception as e:                                   # noqa: BLE001 -- one bad call must not sink the run
            s = {"id": case["id"], "category": case["category"], "added_at": case["added_at"], "route_ok": False, "codes_ok": False,
                 "contains_all_ok": False, "contains_any_ok": False, "must_not_ok": True, "must_not_hits": [], "unsafe": False,
                 "passed": False, "error": f"{type(e).__name__}: {str(e)[:160]}", "answer": ""}
        scores.append(s)
        if verbose:
            bad = [k for k, v in s.items() if k.endswith("_ok") and not v]
            print(f"  {s['id']}: {'PASS' if s['passed'] else 'FAIL ' + (s.get('error') or str(bad))}")
        if pause:
            time.sleep(pause)
    return scores
