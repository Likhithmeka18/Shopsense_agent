"""
Capstone M6 tests: the Triage / Policy-RAG / Order-Actions / Escalation-Reviewer team
(shopsense/teams/support_team.py), run through tests/support_team_sim.py. Parity with the M5 workflow is the main property:
organising the same decisions as a supervised team must not change a single outcome. Compiled-graph wiring needs langgraph
(skipped when it is not installed); `build_support_team` is only checked for its clear error here.
"""
import pytest

from shopsense.eval.golden import load_golden
from shopsense.teams import support_team as st
from shopsense.tools.order_tools import MockAPI
from shopsense.workflows import ticket_resolution_graph as g
from support_team_sim import resume_team, run_team
from test_mcp_ops import mcp_deps
from test_ticket_graph import GOLDEN, ScriptedKB, keyword_extract, make_deps
from ticket_sim import Paused, resume, run

CASES = sorted(GOLDEN)


def seed(case):
    t = case["ticket"]
    return g.seed_state(ticket_id=t["ticket_id"], customer_id=t["customer_id"], text=t["text"], channel=t["channel"], history=case["history"])


def view(r):
    if isinstance(r, Paused):
        return ("paused", tuple(r.state["escalation_codes"]), r.interrupt["holding_reply"], r.state["proposal"].get("amount_inr"))
    return (r["status"], tuple(r.get("escalation_codes") or ()), r["reply"], (r.get("proposal") or {}).get("amount_inr"))


@pytest.mark.parametrize("case_id", CASES)
def test_team_gives_exactly_the_same_outcome_as_the_m5_workflow(case_id):
    case = GOLDEN[case_id]
    a = run(make_deps(), seed(case))
    b = run_team(make_deps(), seed(case))
    assert view(a) == view(b)


def test_supervisor_policy_has_one_legal_next_step_per_state():
    P = st.supervisor_policy
    assert P({}) == "triage"
    base = {"triage": {"x": 1}, "reasons": []}
    assert P(base) == "policy_rag"
    assert P({**base, "policy": {}}) == "order_actions"
    assert P({**base, "policy": {}, "proposal": {}}) == "escalation_reviewer"
    assert P({**base, "reasons": ["safety_net:ESC-LEGAL"]}) == "escalation_reviewer"            # hard fact at intake: skip the middle
    assert P({**base, "reasons": ["requires_human_review"]}) == "policy_rag"                      # a model hint is not a hard fact
    assert P({**base, "policy": {}, "proposal": {}, "route": "approval"}) == "human_approval"
    assert P({**base, "policy": {}, "proposal": {}, "route": "approval", "human_decision": "approve"}) == "order_actions"
    assert P({**base, "policy": {}, "proposal": {}, "route": "auto"}) == "order_actions"
    assert P({**base, "policy": {}, "proposal": {}, "route": "auto", "action_result": {}}) == "escalation_reviewer"
    assert P({**base, "route": "human_review"}) == "escalation_reviewer"
    assert P({**base, "reply": "done"}) == "done"
    assert P({**base, "steps": st.MAX_STEPS}) == "escalate"


def test_scopes_are_enforced_at_runtime():
    @st.scoped("policy_rag")
    def sneaky(state):
        return {"policy": {}, "action_result": {"committed": True}}

    with pytest.raises(PermissionError):
        sneaky({})

    @st.scoped("escalation_reviewer")
    def meddling(state):
        return {"route": "auto", "facts": {"order": "forged"}}

    with pytest.raises(PermissionError):
        meddling({})


def test_nobody_but_order_actions_can_commit_and_nobody_but_the_reviewer_can_route():
    writers = lambda key: {a for a, keys in st.AGENT_SCOPES.items() if key in keys}
    assert writers("action_result") == {"order_actions"}
    assert writers("route") == {"escalation_reviewer", "escalate"}
    assert writers("proposal") == {"order_actions"}
    assert writers("human_decision") == {"human_approval"}
    assert writers("policy") == {"policy_rag"}
    assert writers("handoff") == {"escalation_reviewer"}


def test_every_scope_key_is_a_real_state_key():
    keys = set(st.TeamState.__annotations__)
    for role, scope in st.AGENT_SCOPES.items():
        assert scope <= keys, (role, scope - keys)


def test_hard_escalation_never_wakes_policy_rag_or_order_actions():
    t = GOLDEN["SS-EV-903"]["ticket"]
    r = run_team(make_deps(), g.seed_state(ticket_id="T", customer_id=t["customer_id"], text=t["text"]))
    agents_seen = [line.split(" ")[2] for line in r["log"] if line.startswith("supervisor: ->")]
    assert agents_seen == ["triage", "escalation_reviewer", "done"], agents_seen
    assert r["status"] == "escalated_to_human"


def test_approval_flow_through_the_team_matches_the_workflow_including_role_enforcement():
    t = GOLDEN["SS-EV-901"]["ticket"]
    deps = make_deps()
    paused = run_team(deps, seed(GOLDEN["SS-EV-901"]))
    assert isinstance(paused, Paused) and deps.actions.ledger.records == []
    refused = resume_team(deps, paused, decision="approve", approver_role="manager")
    assert refused["status"] == "needs_higher_approval" and deps.actions.ledger.records == []
    done = resume_team(deps, paused, decision="approve", approver_role="executive")
    assert done["status"] == "resolved_after_approval" and deps.actions.ledger.records[0]["approver_role"] == "executive"
    supervised = [line for line in done["log"] if line.startswith("supervisor: ->")]
    assert any("human_approval" in line for line in supervised)


def test_step_budget_hands_a_runaway_ticket_to_a_human():
    r = st.escalate_node({})
    assert r["route"] == "human_review" and r["status"] == "escalated_to_human" and r["reply"]
    # a reviewer decision that never arrives (malformed resume payload) cannot loop forever
    deps = make_deps()
    t = GOLDEN["SS-EV-009"]["ticket"]
    paused = run_team(deps, seed(GOLDEN["SS-EV-009"]))
    stuck = run_team(deps, {**paused.state, "human_decision": None})
    assert isinstance(stuck, Paused)                                    # waits for a person; never auto-approves


def test_memo_runner_feeds_the_existing_m6_memo_team_and_survives_it_failing():
    seen = []

    class FakeMemoTeam:
        def invoke(self, seed, cfg):
            seen.append(seed)
            return {"status": "", "draft": "MEMO BODY"}

    deps = make_deps(memo_runner=st.make_memo_runner(FakeMemoTeam(), MockAPI()))
    t = GOLDEN["SS-EV-904"]["ticket"]
    r = run_team(deps, g.seed_state(ticket_id="T7", customer_id=t["customer_id"], text=t["text"]))
    assert r["handoff"]["memo"] == "MEMO BODY" and r["handoff"]["memo_status"] == "converged"
    assert seen[0]["risk_profile"]["esc_codes"] == ["ESC-SAFETY"] and seen[0]["ticket_text"] == t["text"]


def test_repeat_and_lost_codes_reach_the_memo_teams_risk_profile():
    from shopsense.teams.escalation_review_team import build_risk_profile
    p = build_risk_profile(["repeat_contact", "lost_shipment", "safety_net:INJECTION"], "ORD-4382", MockAPI(), "CUST-0070")
    assert p["esc_codes"] == ["ESC-LOST", "ESC-REPEAT"] and p["injection_attempt"]


def test_the_team_runs_unchanged_on_top_of_mcp_tools():
    deps, box, client = mcp_deps()
    r = run_team(deps, g.seed_state(ticket_id="TM", customer_id="CUST-0082", text="I'd like to return the denim jacket from ORD-4503, it doesn't fit."))
    assert r["status"] == "resolved" and box.actions.ledger.records[0]["approver_role"] == "system"
    assert {c[0] for c in client.calls} >= {"order_lookup", "shipping_tracker", "refund_status", "check_return_eligibility", "issue_refund"}


def test_compiled_team_needs_langgraph_and_says_so():
    if st.LANGGRAPH_AVAILABLE:
        return
    with pytest.raises(RuntimeError):
        st.build_support_team(make_deps())
