"""
ShopSense: Milestone 6 demo (local researcher) -- a supervised multi-agent
team drafts an escalation-review memo for a ticket flagged
`requires_human_review` or `contains_injection_attempt` at Milestone 1
intake (shopsense/teams/escalation_review_team.py).

Requires `langgraph` (in requirements.txt) and a live LLM behind
SHOPSENSE_LLM_MODEL. Run with:

    pip install langgraph
    export GROQ_API_KEY=gsk_...
    python run_escalation_team_demo.py

Finds the first ESC-2 ticket in the sample data (via Milestone 1's own
extraction), runs the five-agent team on it with the LOCAL (no-MCP)
researcher, and proves two invariants: every planned topic actually got researched, and every
citation in the shipped memo resolves to a real finding. It then runs the
team again with a deliberately hallucinating writer to prove the
fact-checker catches and removes a fabricated citation, and a third time
with a stubborn writer to prove the revision cap still terminates into
`escalate_node` when the team genuinely cannot converge.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_common import assert_memo_invariants, find_escalated_ticket  # noqa: E402
from shopsense.memory.memory_store import MemoryStore  # noqa: E402
from shopsense.teams.escalation_review_team import (  # noqa: E402
    LANGGRAPH_AVAILABLE,
    MAX_REVISIONS,
    build_team,
    make_local_researcher_node,
    seed_state,
    writer_node,
)
from shopsense.tools.order_tools import MockAPI  # noqa: E402
from shopsense.tools.policy_tools import PolicyKnowledgeBase  # noqa: E402


def main() -> None:
    print("=" * 70)
    print("ShopSense Milestone 6: escalation-review supervisor team (local)")
    print("=" * 70)

    if not LANGGRAPH_AVAILABLE:
        print(
            "\nlanggraph is not installed here -- run:\n    pip install langgraph\n"
            "then re-run this script. (The team's decision logic is already "
            "verified without the package -- see dev_smoke_check.py / "
            "tests/test_teams.py.)"
        )
        sys.exit(1)

    api = MockAPI()
    store = MemoryStore()
    kb = PolicyKnowledgeBase()
    researcher_node = make_local_researcher_node(kb, api, store)
    team = build_team(researcher_node)

    ticket, order_id, reasons = find_escalated_ticket(api, prefer="safety_net:ESC-SAFETY")
    if ticket is None:
        print("No ESC-2 escalated ticket found in the sample data -- nothing to demo.")
        sys.exit(1)

    print(f"\n[1] Reviewing {ticket.ticket_id} (escalated at intake: {reasons})")
    seed = seed_state(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id or f"GUEST-{ticket.ticket_id}",
        ticket_text=ticket.text,
        order_id=order_id,
        escalation_reasons=reasons,
        api=api,
    )
    final = team.invoke(seed, {"recursion_limit": 50})

    print("\nTRAJECTORY")
    print("----------")
    for line in final["log"]:
        print(" ", line)

    supported_ids = {f["id"] for f in final["findings"]}
    cited = set(re.findall(r"\[(F\d+)\]", final["draft"]))
    assert set(final["plan"]) <= {f["topic"] for f in final["findings"]}, "a planned topic went unresearched"
    assert final["revision_count"] <= MAX_REVISIONS, "the revise loop ran past its budget"

    if final["status"] == "":
        assert final["fact_check"].get("ok") and final["review"].get("approved")
        assert cited <= supported_ids, f"unsupported citation survived: {sorted(cited - supported_ids)}"
        outcome = f"converged in {final['revision_count']} revision(s)"
    else:
        assert final["status"] == "escalated_to_human_no_memo"
        outcome = "escalated to a human with no memo (team could not converge in budget)"

    assert_memo_invariants(final)
    print(f"\nPASS -- {outcome}. Citations: {sorted(cited)}, all backed by retrieved findings.")
    print("\n--- escalation-review memo ---\n")
    print(final["draft"])

    print("\n[2a] Proving an UNCITED claim in the VERIFIED section is caught ...")

    def uncited_writer(state):
        out = writer_node(state)
        if not state["draft"]:
            out = {**out, "draft": out["draft"].replace(
                "VERIFIED (system findings):",
                "VERIFIED (system findings):\n- This customer has never contacted us before.", 1)}
        return out

    import shopsense.teams.escalation_review_team as team_mod

    original_writer = team_mod.writer_node
    team_mod.writer_node = uncited_writer
    try:
        uncited_final = build_team(make_local_researcher_node(kb, api, store)).invoke(seed, {"recursion_limit": 50})
    finally:
        team_mod.writer_node = original_writer
    assert any("uncited" in line and "1 uncited" in line for line in uncited_final["log"]), "fact-checker must flag the uncited claim"
    assert "never contacted us before" not in uncited_final["draft"], "the uncited claim must not survive"
    assert uncited_final["status"] == "" and uncited_final["review"].get("approved"), \
        "after the fix the team must CONVERGE on a revised draft (re-fact-checked), not just keep rewriting"
    print("PASS -- an uncited factual claim was planted, flagged by the fact-checker, and removed on revision.")

    print("\n[2b] Proving the fact-checker catches a fabricated citation ...")

    def hallucinating_writer(state):
        out = writer_node(state)
        if not state["draft"]:
            out = {**out, "draft": out["draft"] + "\nThis customer has never contacted us before. [F9]"}
            out["log"] = ["writer[hallucinating]: draft with a fabricated [F9] citation"]
        return out

    # Swapping a node on an already-compiled LangGraph isn't possible, so
    # rebuild with the module-level writer_node patched for this run only.
    team_mod.writer_node = hallucinating_writer
    try:
        halluc_team = build_team(make_local_researcher_node(kb, api, store))
        halluc_final = halluc_team.invoke(seed, {"recursion_limit": 50})
    finally:
        team_mod.writer_node = original_writer

    assert any("F9" in line for line in halluc_final["log"]), "the fact-checker must have named F9"
    assert halluc_final["revision_count"] >= 1, "the loop-back edge must have fired"
    assert "[F9]" not in halluc_final["draft"], "the fabricated citation must not survive into the memo"
    assert halluc_final["status"] == "" and halluc_final["fact_check"].get("ok"), \
        "the revised draft must be re-fact-checked and pass -- a stale verdict must not trap the loop"
    print("PASS -- a fabricated citation was planted, caught by a critic that never saw the")
    print("ticket text or the plan, and removed by a revision the supervisor routed.")

    print("\n[3] Proving the revision cap terminates an unfixable case ...")

    def stubborn_writer(state):
        return {
            "draft": "This memo cites nothing real. [F99]",
            "revision_count": state["revision_count"] + (1 if state["draft"] else 0),
            "log": [f"writer[stubborn]: rewrite #{state['revision_count']} (fixed nothing)"],
        }

    team_mod.writer_node = stubborn_writer
    try:
        stuck_team = build_team(make_local_researcher_node(kb, api, store))
        stuck_final = stuck_team.invoke(seed, {"recursion_limit": 50})
    finally:
        team_mod.writer_node = original_writer

    assert stuck_final["status"] == "escalated_to_human_no_memo"
    assert stuck_final["revision_count"] == MAX_REVISIONS
    print(f"PASS -- status={stuck_final['status']} after {stuck_final['revision_count']} "
          f"revision(s); the supervisor never had 'writer' as a legal route once the budget "
          f"was spent, so neither the policy nor the model could pick it again.")

    print("\n" + "=" * 70)
    print("Milestone 6 (local researcher) PASS.")
    print("=" * 70)


if __name__ == "__main__":
    main()
