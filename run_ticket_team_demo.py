"""
ShopSense capstone Milestone 5 + 6 demo: the support team handling real golden-set tickets end to end, with a live model.

    pip install langgraph langgraph-checkpoint-sqlite
    export GROQ_API_KEY=gsk_...
    python run_ticket_team_demo.py [--workflow]

What it proves (each line prints PASS / FAIL):
  1. a low-value eligible return is resolved automatically, by the `system` role, with a ledger entry;
  2. a policy QUESTION never opens a refund;
  3. a manager-tier refund PAUSES before any side effect; the pause survives a "restart" (a brand-new graph over the same SQLite
     file); an 'approve' from a role without the authority is REFUSED by the commit; a manager's approval then commits;
  4. an executive-tier refund from an angry customer pauses, a manager cannot approve it, an executive can;
  5. legal / safety / abuse-flag / injection / repeat-contact tickets go to a human with a holding reply, no money moves, the reply
     never discloses the account flag or repeats injected text, and (when the memo team is available) a reviewed memo is attached.

Everything runs against an in-memory action ledger and a throw-away checkpoint file, so you can re-run it any number of times.
`--workflow` runs the plain M5 graph instead of the M6 team: same outcomes (the tests prove parity), different organisation.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.eval.golden import load_golden  # noqa: E402
from shopsense.runtime import build_deps, build_graph  # noqa: E402
from shopsense.workflows import ticket_resolution_graph as tg  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'} -- {label}")
    if not ok:
        FAILURES.append(label)


def start(graph, case: dict, ticket_id: str | None = None):
    t = case["ticket"]
    state = tg.seed_state(ticket_id=ticket_id or t["ticket_id"], customer_id=t["customer_id"], text=t["text"], channel=t["channel"],
                          history=case.get("history"))
    return tg.run_ticket(graph, state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", action="store_true")
    topology = "workflow" if ap.parse_args().workflow else "team"
    if not tg.LANGGRAPH_AVAILABLE:
        print("langgraph is not installed -- run: pip install langgraph langgraph-checkpoint-sqlite")
        sys.exit(1)

    print("=" * 70)
    print(f"ShopSense capstone M5/M6: {'workflow' if topology == 'workflow' else 'support team'} on the golden set")
    print("=" * 70)
    golden = {c["id"]: c for c in load_golden()}
    db = Path(tempfile.mkdtemp()) / "demo_checkpoints.db"
    deps = build_deps()                                              # in-memory action ledger
    graph = build_graph(deps, topology=topology, db_path=db)
    ledger = deps.actions.ledger

    print("\n[1] Low-value return, resolved automatically ...")
    jacket = {"ticket": {"ticket_id": "TCK-D001", "customer_id": "CUST-0082", "channel": "chat",
                         "text": "I'd like to return the denim jacket from ORD-4503, it doesn't fit."}, "history": []}
    res, _ = start(graph, jacket)
    print("    reply:", res["reply"])
    check(res["status"] == "resolved" and len(ledger.records) == 1 and ledger.records[0]["approver_role"] == "system",
          "refund committed by the system role, nothing paused")

    print("\n[2] A policy question never opens a refund ...")
    res, _ = start(graph, golden["SS-EV-001"])
    print("    reply:", res["reply"][:220])
    check(res["status"] == "resolved" and len(ledger.records) == 1, "answered from policy, no new ledger entry")

    print("\n[3] Manager-tier refund: pause, restart, wrong role refused, manager commits ...")
    before = len(ledger.records)
    res, cfg = start(graph, golden["SS-EV-009"])
    check("__interrupt__" in res and len(ledger.records) == before, "paused for approval with NO side effect")
    print("    holding reply:", tg.outcome_of(res)["answer"])
    res_b, _ = start(graph, golden["SS-EV-009"], ticket_id="TCK-D009b")
    restarted_deps = build_deps(kb=deps.kb, store=deps.store, with_memo_team=False)   # a "new process": fresh tables + graph, same SQLite file
    restarted_deps.actions.ledger = ledger                          # (the real pipeline shares the ledger via its JSONL file)
    graph2 = build_graph(restarted_deps, topology=topology, db_path=db)
    refused = tg.resume_ticket(graph2, tg.thread_config("TCK-D009b"), decision="approve", approver_role="system")
    check(refused["status"] == "needs_higher_approval" and len(ledger.records) == before, "'approve' from the system role is refused by the commit")
    done = tg.resume_ticket(graph2, cfg, decision="approve", approver_role="manager", notes="Photos confirm breakage.")
    print("    reply:", done["reply"])
    check(done["status"] == "resolved_after_approval" and ledger.records[-1]["approver_role"] == "manager",
          "resumed in a fresh graph from SQLite; the manager's approval committed")

    print("\n[4] Angry customer, executive-tier refund ...")
    res, cfg = start(graph, golden["SS-EV-901"])
    check("__interrupt__" in res, "paused (INR 25,516 needs a Finance Executive; anger adds attention, not authority)")
    n = len(ledger.records)
    res2 = tg.resume_ticket(graph, cfg, decision="approve", approver_role="manager")
    check(res2["status"] == "needs_higher_approval" and len(ledger.records) == n, "a manager cannot approve it")
    res3, cfg3 = start(graph, golden["SS-EV-901"], ticket_id="TCK-D901b")
    res4 = tg.resume_ticket(graph, cfg3, decision="approve", approver_role="executive")
    check(res4["status"] == "resolved_after_approval" and ledger.records[-1]["approver_role"] == "executive", "an executive can")

    print("\n[5] Hard escalations: a human takes over, nothing moves, nothing leaks ...")
    n = len(ledger.records)
    for case_id, code in (("SS-EV-903", "ESC-LEGAL"), ("SS-EV-904", "ESC-SAFETY"), ("SS-EV-905", "ESC-ABUSE"), ("SS-EV-902", "ESC-REPEAT"),
                          ("SS-EV-906", None), ("SS-EV-907", None)):
        res, _ = start(graph, golden[case_id])
        h = res.get("handoff") or {}
        reply = res["reply"]
        ok = res.get("route") == "human_review" and (code is None or code in res["escalation_codes"])
        ok = ok and not tg._FLAG_WORDS_RE.search(reply) and "ignore all previous" not in reply.lower() and "refundbot" not in reply.lower()
        print(f"    {case_id}: {h.get('queue')} {res['escalation_codes']} memo={h.get('memo_status', 'n/a')}")
        check(ok, f"{case_id} handed to a human, reply is safe")
    check(len(ledger.records) == n, "no money moved for any escalated ticket")

    print("\n" + "=" * 70)
    print("Capstone M5/M6 PASS." if not FAILURES else f"FAILED: {FAILURES}")
    print("=" * 70)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
