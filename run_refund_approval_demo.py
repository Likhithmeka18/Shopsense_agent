"""
ShopSense: Milestone 5 demo -- a durable, resumable, human-gated refund
approval workflow (shopsense/workflows/refund_approval_graph.py).

Requires `langgraph` and `langgraph-checkpoint-sqlite` (both in
requirements.txt), plus a live LLM behind SHOPSENSE_LLM_MODEL for the
drafting node. Run with:

    pip install langgraph langgraph-checkpoint-sqlite
    export GROQ_API_KEY=gsk_...
    python run_refund_approval_demo.py

What this proves, in order:

  1. A real PENDING refund request from the mock data (executive_review
     tier, chosen by decide_refund_review from the refund records) starts
     the workflow, and the graph drafts
     a compliant recommendation, passes its own deterministic checks, and
     PAUSES at human_approval_node -- durably, via a SqliteSaver pointed at
     a real file on disk.
  2. The Python process-level graph object is thrown away and a BRAND NEW
     one is built from the same db file -- simulating a
     process restart -- and
     the review resumes from where it paused, with no state lost.
  3. finalize_node fires exactly once (not twice) despite the resume,
     because it lives downstream of interrupt(). A counting MemoryStore
     proves it: exactly ONE episodic memory is written by the graph itself
     (not by this script), carrying the human's decision.
  4. A second review, seeded with a deliberately bad drafting node that can
     never produce a compliant recommendation, proves the revision cap:
     it reaches escalate_unresolved_node within MAX_REVISIONS attempts and
     never reaches a human at all -- a bounded cost, not an infinite loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.memory.memory_store import MemoryStore  # noqa: E402
from shopsense.tools.order_tools import MockAPI  # noqa: E402
from shopsense.workflows.refund_approval_graph import (  # noqa: E402
    LANGGRAPH_AVAILABLE,
    MAX_REVISIONS,
    build_refund_approval_graph,
    build_sqlite_checkpointer,
    find_pending_refunds,
    resume_refund_review,
    start_refund_review,
)

DB_PATH = Path(__file__).resolve().parent / "data" / "shopsense" / "workflow_state" / "refund_reviews.db"


class CountingStore(MemoryStore):
    """A real MemoryStore that remembers every write it receives, so the
    demo can PROVE finalize ran exactly once."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.writes: list[tuple[str, str, str]] = []

    def write(self, customer_id, kind, text, **metadata):
        self.writes.append((customer_id, kind, text))
        return super().write(customer_id, kind, text, **metadata)


def main() -> None:
    print("=" * 70)
    print("ShopSense Milestone 5: refund-approval LangGraph workflow")
    print("=" * 70)

    if not LANGGRAPH_AVAILABLE:
        print(
            "\nlanggraph is not installed here -- run:\n"
            "    pip install langgraph langgraph-checkpoint-sqlite\n"
            "then re-run this script. (The decision logic this graph relies "
            "on is already verified without the package -- see "
            "dev_smoke_check.py / tests/test_workflows.py.)"
        )
        sys.exit(1)

    api = MockAPI()
    pending = find_pending_refunds(api)
    wanted = [d for d in pending if d["approval_tier"] == "executive_review"] or pending
    if not wanted:
        print("No pending manager/executive refund in the mock data -- nothing to demo.")
        sys.exit(1)
    case = wanted[0]
    order_id, customer_id = case["order_id"], case["customer_id"]
    ticket_id = f"TCK-REFUND-{order_id}"
    store = CountingStore()
    # A stale paused/finished thread from an earlier run would make this a
    # resume of the old state, not a fresh review -- use a per-run ticket id.
    import time
    ticket_id += f"-{int(time.time())}"

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[1] Starting review for {order_id} (customer {customer_id}), refund {case['refund_id']}, "
          f"INR {case['refund_amount_inr']:.0f}, tier={case['approval_tier']}")
    print(f"    durable checkpoint file: {DB_PATH}")

    checkpointer = build_sqlite_checkpointer(str(DB_PATH))
    graph = build_refund_approval_graph(checkpointer=checkpointer, store=store)
    result, config = start_refund_review(
        graph,
        ticket_id=ticket_id,
        customer_id=customer_id,
        order_id=order_id,
        refund_amount_inr=case["refund_amount_inr"],
        approval_tier=case["approval_tier"],
        case_facts=case["case_facts"],
    )
    assert store.writes == [], "nothing may be recorded before the human decides"

    if "__interrupt__" not in result:
        print(f"    did not pause -- ended with status={result.get('status')!r}. "
              f"(Expected if the AI never produced a compliant draft within "
              f"{MAX_REVISIONS} revisions -- see escalate_unresolved below.)")
        sys.exit(0)

    payload = result["__interrupt__"][0].value
    print("\n[2] PAUSED for human approval. A real approver would see:")
    for k, v in payload.items():
        print(f"      {k}: {v}")

    print("\n[3] Simulating a process restart: discarding this graph object, "
          "opening a NEW checkpointer against the SAME db file ...")
    del graph, checkpointer
    fresh_checkpointer = build_sqlite_checkpointer(str(DB_PATH))
    fresh_graph = build_refund_approval_graph(checkpointer=fresh_checkpointer, store=store)

    print("[4] Resuming as the human approver, with decision='approve' ...")
    final = resume_refund_review(
        fresh_graph, config, decision="approve", notes="Verified against ESC-L2 SLA, sign-off granted."
    )
    print(f"    status: {final['status']}")
    print(f"    final_outcome: {final['final_outcome']}")

    import json

    record = json.loads(final["final_outcome"])
    assert final["status"] == "approve" and record["status"] == "approve", "status must reflect the human decision"
    assert len(store.writes) == 1, f"finalize must write exactly once, wrote {len(store.writes)}"
    assert store.writes[0][0] == customer_id and "approve" in store.writes[0][2]
    print("    finalize ran exactly once and recorded the decision into the customer's memory:")
    print(f"      {store.writes[0][2]}")

    print("\n[5] Proving the revision cap: a review that can NEVER produce a "
          "compliant draft must still terminate.")
    from shopsense.workflows import refund_approval_graph as mod

    original_draft_node = mod._draft_recommendation_node

    def _stubborn_draft_node(state):
        # Always omits the escalation code -- checks_node can never approve
        # this, no matter how many times it's asked to "fix" it.
        out = original_draft_node(state)
        out["recommendation"] = "This refund needs sign-off. (missing code and amount on purpose)"
        return out

    mod._draft_recommendation_node = _stubborn_draft_node
    try:
        stubborn_checkpointer = build_sqlite_checkpointer(str(DB_PATH))
        stubborn_graph = build_refund_approval_graph(checkpointer=stubborn_checkpointer, store=store)
        other = ([d for d in pending if d["approval_tier"] == "manager_review"] or [case])[0]
        writes_before = len(store.writes)
        stubborn_result, _ = start_refund_review(
            stubborn_graph,
            ticket_id=f"TCK-STUBBORN-{other['order_id']}-{int(time.time())}",
            customer_id=other["customer_id"],
            order_id=other["order_id"],
            refund_amount_inr=other["refund_amount_inr"],
            approval_tier=other["approval_tier"],
            case_facts=other["case_facts"],
        )
        assert len(store.writes) == writes_before, "an unresolved review must not record a human decision"
        assert stubborn_result["status"] == "escalated_no_ai_draft", stubborn_result["status"]
        assert stubborn_result["revision_count"] >= MAX_REVISIONS
        print(f"    status: {stubborn_result['status']} after "
              f"{stubborn_result['revision_count']} revision(s) -- budget enforced, "
              f"never reached a human.")
    finally:
        mod._draft_recommendation_node = original_draft_node

    print("\n" + "=" * 70)
    print("Milestone 5 PASS: drafted, checked, paused durably, resumed after a")
    print("simulated restart, finalized exactly once, and the revision cap")
    print("terminates an unfixable case without ever reaching a human.")
    print("=" * 70)


if __name__ == "__main__":
    main()
