"""
Run with the fake LangGraph on PYTHONPATH (tests/test_wiring.py does that in a subprocess): builds the REAL compiled
graphs (build_ticket_graph, build_support_team) and drives them end to end, including an approval pause that is resumed
from a second graph instance over the same SQLite file ("another process"). Prints OK lines; exits non-zero on any failure.
"""
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
fake = types.ModuleType("litellm"); fake.completion = lambda **k: (_ for _ in ()).throw(RuntimeError("no LLM offline")); sys.modules["litellm"] = fake

from shopsense.teams import support_team as st            # noqa: E402
from shopsense.tools.order_tools import MockAPI           # noqa: E402
from shopsense.workflows import ticket_resolution_graph as g   # noqa: E402
from ticket_sim import keyword_extract                    # noqa: E402

assert g.LANGGRAPH_AVAILABLE and st.LANGGRAPH_AVAILABLE, "fake langgraph was not picked up"


class KB:
    def answer(self, q, **k):
        return {"answer": "See policy.", "sources": [{"id": 0, "source": "faq", "section": "x"}]}


def deps():
    return g.Deps(api=MockAPI(), kb=KB(), extract=keyword_extract)


def check(cond, msg):
    if not cond:
        print("FAIL", msg); sys.exit(1)
    print("OK  ", msg)


MUG = ("TCK-W1", "CUST-0066", "The ceramic mug set from ORD-4589 arrived smashed 5 days ago, there are pieces everywhere.")
JACKET = ("TCK-W2", "CUST-0082", "I'd like to return the denim jacket from ORD-4503, it doesn't fit.")
LEGAL = ("TCK-W3", "CUST-0070", "My order ORD-4382 never came and I have spoken to my lawyer about this.")

for label, build in (("M5 workflow", g.build_ticket_graph), ("M6 team", st.build_support_team)):
    d = deps()
    graph = build(d)
    res, _ = g.run_ticket(graph, g.seed_state(ticket_id=JACKET[0], customer_id=JACKET[1], text=JACKET[2]))
    check(res.get("status") == "resolved" and d.actions.ledger.records[0]["approver_role"] == "system", f"{label}: low-value refund resolves automatically")
    d = deps()
    graph = build(d)
    res, cfg = g.run_ticket(graph, g.seed_state(ticket_id=LEGAL[0], customer_id=LEGAL[1], text=LEGAL[2]))
    check(res.get("status") == "escalated_to_human" and res["route"] == "human_review" and not d.actions.ledger.records, f"{label}: legal threat is handed to a human")

    d = deps()
    path = Path(tempfile.mkdtemp()) / "cp.db"
    graph = build(d, g.build_sqlite_checkpointer(str(path)))
    res, cfg = g.run_ticket(graph, g.seed_state(ticket_id=MUG[0], customer_id=MUG[1], text=MUG[2]))
    check("__interrupt__" in res and not d.actions.ledger.records, f"{label}: high-value refund pauses before any side effect")
    payload = res["__interrupt__"][0].value
    check(payload["kind"] == "ticket_approval" and payload["escalation_codes"] == ["ESC-L1"], f"{label}: pause payload names ESC-L1")

    # "another process": a brand-new graph + connection over the same file
    d2 = deps()
    graph2 = build(d2, g.build_sqlite_checkpointer(str(path)))
    # a second paused ticket on the same file, resumed with a role that lacks the authority
    res_b, _ = g.run_ticket(graph2, g.seed_state(ticket_id="TCK-W1b", customer_id=MUG[1], text=MUG[2]))
    check("__interrupt__" in res_b, f"{label}: a second ticket pauses independently (own thread)")
    refused = g.resume_ticket(graph2, g.thread_config("TCK-W1b"), decision="approve", approver_role="system")
    check(refused["status"] == "needs_higher_approval" and not d2.actions.ledger.records, f"{label}: a system-role 'approval' is refused by the commit")
    done = g.resume_ticket(graph2, g.thread_config(MUG[0]), decision="approve", approver_role="manager")
    check(done["status"] == "resolved_after_approval" and d2.actions.ledger.records[0]["approver_role"] == "manager", f"{label}: resumed from SQLite in a fresh graph, manager approval commits")
    # Re-running the SAME ticket on the SAME checkpointed thread (a new process with fresh tables and an empty action ledger) must
    # not inherit the previous run's decision/route/action: it has to pause again and wait for a NEW decision.
    d3 = deps()
    graph3 = build(d3, g.build_sqlite_checkpointer(str(path)))
    res_again, _ = g.run_ticket(graph3, g.seed_state(ticket_id=MUG[0], customer_id=MUG[1], text=MUG[2]))
    check("__interrupt__" in res_again and not d3.actions.ledger.records, f"{label}: a re-run of a completed ticket pauses again instead of reusing last time's approval")
    done_again = g.resume_ticket(graph3, g.thread_config(MUG[0]), decision="deny")
    check(done_again["status"] == "denied_by_reviewer" and not d3.actions.ledger.records, f"{label}: ...and the new decision (deny) is the one that counts")
    j1, _ = g.run_ticket(graph3, g.seed_state(ticket_id=JACKET[0], customer_id=JACKET[1], text=JACKET[2]))
    j2, _ = g.run_ticket(graph3, g.seed_state(ticket_id=JACKET[0], customer_id=JACKET[1], text=JACKET[2]))
    check(j1["route"] == j2["route"] == "auto" and len(d3.actions.ledger.records) == 1, f"{label}: re-running an auto-resolved ticket does not pay twice")
print("WIRING OK")
