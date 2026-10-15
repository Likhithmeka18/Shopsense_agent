"""
ShopSense: resume a paused Milestone 5 refund-approval review.

run_pipeline.py starts reviews but never resumes them -- a human approver
does that, whenever they get to it, by running:

    python resume_refund_review.py <ticket_id> <approve|deny|modify> ["notes"]

This opens the SAME durable SqliteSaver file run_pipeline.py wrote to, in
a BRAND NEW process. That is the real "restart and resume" pattern that
LangGraph checkpointing makes possible -- here it's just two different Python invocations, which is the
more realistic shape for an actual human-in-the-loop approval step anyway.

Requires `langgraph` and `langgraph-checkpoint-sqlite` -- see README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.memory.memory_store import MemoryStore  # noqa: E402
from shopsense.workflows.refund_approval_graph import (  # noqa: E402
    LANGGRAPH_AVAILABLE,
    build_refund_approval_graph,
    build_sqlite_checkpointer,
    resume_refund_review,
    thread_config,
)

DB_PATH = Path(__file__).resolve().parent / "data" / "shopsense" / "workflow_state" / "refund_reviews.db"


def main() -> None:
    if not LANGGRAPH_AVAILABLE:
        print("langgraph is not installed -- run: pip install langgraph langgraph-checkpoint-sqlite")
        sys.exit(1)
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ticket_id, decision = sys.argv[1], sys.argv[2]
    notes = sys.argv[3] if len(sys.argv) > 3 else ""
    if decision not in ("approve", "deny", "modify"):
        print("decision must be one of: approve, deny, modify")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"No durable checkpoint file at {DB_PATH} -- has run_pipeline.py started a review yet?")
        sys.exit(1)

    checkpointer = build_sqlite_checkpointer(str(DB_PATH))
    graph = build_refund_approval_graph(checkpointer=checkpointer, store=MemoryStore())
    config = thread_config(ticket_id)
    if "human_approval" not in tuple(graph.get_state(config).next):
        print(f"No paused refund review for {ticket_id} (never started, or already decided).")
        sys.exit(1)
    result = resume_refund_review(graph, config, decision=decision, notes=notes)

    print(f"status: {result['status']}")
    print(f"final_outcome: {result['final_outcome']}")
    print("(the decision was recorded into the customer's memory by the workflow's finalize node.)")


if __name__ == "__main__":
    main()
