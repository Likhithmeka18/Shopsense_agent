"""
ShopSense: resume a paused ticket (capstone M5/M6 human-approval step).

run_pipeline.py pauses a ticket when a refund needs a person (INR 2,000+) or the customer is angry. The approver runs, later and
in a BRAND NEW process:

    python resume_ticket_approval.py <ticket_id> <approve|deny|modify> [--role manager|executive] [--amount N] [--notes "..."]
                                     [--workflow]

It reopens the SAME durable SQLite checkpoint, so the pause survives a restart. The decision then flows into `commit`, which
enforces the role limit ITSELF: an 'approve' from the wrong role is refused, not honoured. `modify --amount N` can only lower
the refund. Use --workflow if the ticket was run with `run_pipeline.py --workflow`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.observability import set_tracer, tracer_from_env  # noqa: E402
from shopsense.runtime import ACTION_LEDGER, TICKET_DB, TRACE_FILE, build_deps, build_graph, pending_node  # noqa: E402
from shopsense.workflows import ticket_resolution_graph as tg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticket_id")
    ap.add_argument("decision", choices=["approve", "deny", "modify"])
    ap.add_argument("--role", default="manager", choices=["manager", "executive", "trust_safety"])
    ap.add_argument("--amount", type=float, default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--workflow", action="store_true")
    args = ap.parse_args()

    if not tg.LANGGRAPH_AVAILABLE:
        print("langgraph is not installed -- run: pip install langgraph langgraph-checkpoint-sqlite")
        sys.exit(1)
    if not TICKET_DB.exists():
        print(f"No checkpoint file at {TICKET_DB} -- has run_pipeline.py paused a ticket yet?")
        sys.exit(1)

    # Resume only needs the approval -> commit -> reply tail, so the policy index and memo team are not built here.
    from shopsense.memory.memory_store import MemoryStore
    deps = build_deps(kb=_NoKB(), store=MemoryStore(), ledger_path=ACTION_LEDGER, with_memo_team=False)
    graph = build_graph(deps, topology="workflow" if args.workflow else "team", db_path=TICKET_DB)
    if not any("approval" in n for n in pending_node(graph, args.ticket_id)):
        print(f"No paused approval for {args.ticket_id} (never started, already decided, or run with the other topology).")
        sys.exit(1)

    tracer = tracer_from_env(trace_file=TRACE_FILE)
    set_tracer(tracer)
    result = tg.resume_ticket(graph, tg.thread_config(args.ticket_id), decision=args.decision, notes=args.notes,
                              approver_role=args.role, amount_inr=args.amount)
    tracer.flush()
    print(f"status: {result['status']}")
    print(f"action: {result.get('action_result')}")
    if result.get("paused"):
        print(f"\nThe {args.role} role does not have the authority for this amount; nothing was changed and the ticket is STILL "
              "waiting for approval.\nAsk a higher role to run this command with --role.")
        return
    print(f"reply to the customer: {result.get('reply')}")
    if (result.get("action_result") or {}).get("unavailable"):
        print("\nThe order system could not be reached, so NOTHING was changed and the customer was told a team member will confirm.\n"
              "The ticket is now in the human queue with your decision recorded on its hand-off packet (decision_so_far). This run is\n"
              "finished, so re-running this command will not retry it: apply the decision by hand or through the Order-Actions tools\n"
              "once the order system is back. Actions are idempotent per ticket, so applying it twice cannot pay twice.")
    if result["status"] == "needs_higher_approval":
        print(f"\nThe {args.role} role does not have the authority for this amount; nothing was changed. "
              "Ask a higher role to run this command with --role.")


class _NoKB:
    def answer(self, *a, **k):
        raise RuntimeError("policy search is not available while resuming")


if __name__ == "__main__":
    main()
