"""
Milestone 5 tests: the refund-approval LangGraph workflow's decision logic
(shopsense/workflows/refund_approval_graph.py).

Every function imported here works with NO `langgraph` installed -- see
that module's docstring for why. Only the tests under the
`langgraph_graph` class actually build/invoke the compiled graph, and are
skipped (not failed) when `langgraph` isn't present, exactly like
dev_smoke_check.py's README note on this project's network-less build
environment.
"""
import json

import pytest

from shopsense.workflows.refund_approval_graph import (
    LANGGRAPH_AVAILABLE,
    MAX_REVISIONS,
    _checks_node,
    _draft_recommendation_node,
    _escalate_unresolved_node,
    _finalize_node,
    build_finalize_record,
    check_recommendation,
    decide_refund_review,
    escalation_code_for_tier,
    find_pending_refunds,
    make_finalize_node,
    route_after_checks,
    seed_state,
    thread_config,
)
from shopsense.tools.order_tools import MockAPI


def make_state(**overrides):
    state = seed_state(
        ticket_id="TCK-9001",
        customer_id="CUST-1",
        order_id="ORD-9001",
        refund_amount_inr=15000,
        approval_tier="executive_review",
    )
    state.update(overrides)
    return state


# --------------------------------------------------------------------------
# escalation_code_for_tier / seed_state / thread_config
# --------------------------------------------------------------------------

def test_escalation_code_matches_esc1_table():
    assert escalation_code_for_tier("manager_review") == "ESC-L1"
    assert escalation_code_for_tier("executive_review") == "ESC-L2"


def test_escalation_code_defaults_to_stricter_code_for_unknown_tier():
    assert escalation_code_for_tier("something_unexpected") == "ESC-L2"


def test_seed_state_derives_escalation_code():
    state = seed_state(
        ticket_id="TCK-9001", customer_id="CUST-1", order_id="ORD-9001",
        refund_amount_inr=5000, approval_tier="manager_review",
    )
    assert state["escalation_code"] == "ESC-L1"
    assert state["revision_count"] == 0 and state["human_decision"] is None


def test_thread_config_is_stable_per_ticket():
    assert thread_config("TCK-1") == thread_config("TCK-1")
    assert thread_config("TCK-1") != thread_config("TCK-2")


# --------------------------------------------------------------------------
# check_recommendation -- the deterministic critic
# --------------------------------------------------------------------------

def test_check_recommendation_flags_empty_draft():
    state = make_state(recommendation="")
    assert check_recommendation(state) == ["recommendation is empty"]


def test_check_recommendation_passes_a_compliant_draft():
    state = make_state(recommendation="Order ORD-9001 needs ESC-L2 sign-off for a refund of INR 15,000.")
    assert check_recommendation(state) == []


def test_check_recommendation_flags_missing_escalation_code():
    state = make_state(recommendation="Order ORD-9001 needs sign-off for a refund of INR 15,000.")
    issues = check_recommendation(state)
    assert any("escalation code" in i for i in issues)


def test_check_recommendation_flags_missing_order_id():
    state = make_state(recommendation="This refund of INR 15,000 needs ESC-L2 sign-off.")
    issues = check_recommendation(state)
    assert any("ORD-9001" in i for i in issues)


def test_check_recommendation_flags_missing_amount():
    state = make_state(recommendation="Order ORD-9001 needs ESC-L2 sign-off.")
    issues = check_recommendation(state)
    assert any("amount" in i for i in issues)


# --------------------------------------------------------------------------
# route_after_checks -- the pure router, including the loop-cap boundary
# --------------------------------------------------------------------------

def test_route_after_checks_clean_goes_to_human_approval():
    state = make_state(issues=[], revision_count=0)
    assert route_after_checks(state) == "human_approval"


def test_route_after_checks_issues_under_budget_revises():
    state = make_state(issues=["x"], revision_count=0)
    assert route_after_checks(state, max_revisions=MAX_REVISIONS) == "revise"


def test_route_after_checks_issues_at_budget_escalates():
    state = make_state(issues=["x"], revision_count=MAX_REVISIONS)
    assert route_after_checks(state, max_revisions=MAX_REVISIONS) == "escalate_unresolved"


def test_route_after_checks_never_revises_past_the_cap():
    # Regression guard for the exact bug class: a control field that never clears, or a cap that's off by one.
    for revision_count in range(MAX_REVISIONS + 3):
        state = make_state(issues=["still broken"], revision_count=revision_count)
        outcome = route_after_checks(state, max_revisions=MAX_REVISIONS)
        assert outcome in ("revise", "escalate_unresolved")
        if revision_count >= MAX_REVISIONS:
            assert outcome == "escalate_unresolved"


# --------------------------------------------------------------------------
# build_finalize_record -- what finalize_node actually persists
# --------------------------------------------------------------------------

def test_build_finalize_record_shape():
    state = make_state(
        recommendation="Order ORD-9001 needs ESC-L2 sign-off for INR 15,000.",
        human_decision="approve",
        human_notes="looks fine",
        status="approve",
    )
    record = build_finalize_record(state)
    assert record["ticket_id"] == "TCK-9001"
    assert record["order_id"] == "ORD-9001"
    assert record["escalation_code"] == "ESC-L2"
    assert record["decision"] == "approve"
    assert record["notes"] == "looks fine"


def test_build_finalize_record_handles_missing_human_decision():
    # The escalate_unresolved path never reaches human_approval_node, so
    # human_decision/human_notes were never set -- must not KeyError.
    state = make_state()
    record = build_finalize_record(state)
    assert record["decision"] is None
    assert record["notes"] == ""


# --------------------------------------------------------------------------
# Node functions that need an LLM but not langgraph (scripted, no network)
# --------------------------------------------------------------------------

def test_draft_recommendation_node_first_draft_does_not_bump_revision_count(scripted_litellm):
    scripted_litellm.queue_text("Order ORD-9001 needs ESC-L2 sign-off for a refund of INR 15,000.")
    state = make_state(recommendation="", revision_count=0)
    out = _draft_recommendation_node(state)
    assert out["revision_count"] == 0
    assert "ORD-9001" in out["recommendation"]


def test_draft_recommendation_node_redraft_bumps_revision_count(scripted_litellm):
    scripted_litellm.queue_text("Order ORD-9001 needs ESC-L2 sign-off for a refund of INR 15,000.")
    state = make_state(recommendation="previous draft", issues=["missing amount"], revision_count=0)
    out = _draft_recommendation_node(state)
    assert out["revision_count"] == 1


def test_checks_node_records_issue_log_entry():
    state = make_state(recommendation="", revision_count=0)
    out = _checks_node(state)
    assert out["issues"] == ["recommendation is empty"]
    assert len(out["issue_log"]) == 1


def test_escalate_unresolved_node_sets_status_without_a_human_decision():
    state = make_state()
    out = _escalate_unresolved_node(state)
    assert out["status"] == "escalated_no_ai_draft"
    record = json.loads(out["final_outcome"])
    assert record["decision"] is None


def test_finalize_node_uses_human_decision_as_status():
    state = make_state(human_decision="deny", human_notes="insufficient evidence")
    out = _finalize_node(state)
    assert out["status"] == "deny"
    record = json.loads(out["final_outcome"])
    assert record["decision"] == "deny"
    assert record["notes"] == "insufficient evidence"
    assert record["status"] == "deny"  # was "" before: the record was built before status was set


class SpyStore:
    def __init__(self):
        self.writes = []

    def write(self, customer_id, kind, text, **meta):
        self.writes.append((customer_id, kind, text))


def test_finalize_node_with_a_store_writes_exactly_one_episodic_memory():
    spy = SpyStore()
    make_finalize_node(spy)(make_state(human_decision="approve", human_notes="verified"))
    assert len(spy.writes) == 1
    customer_id, kind, text = spy.writes[0]
    assert (customer_id, kind) == ("CUST-1", "episodic")
    assert "approve" in text and "ORD-9001" in text


def test_finalize_node_without_a_store_writes_nothing_and_does_not_crash():
    assert make_finalize_node(None)(make_state(human_decision="deny"))["status"] == "deny"


# --------------------------------------------------------------------------
# decide_refund_review -- the data-driven M5 trigger
# --------------------------------------------------------------------------

def test_pending_executive_refund_starts_a_workflow():
    d = decide_refund_review(MockAPI(), customer_id="CUST-0050", order_id="ORD-4099")
    assert d["start"] and d["approval_tier"] == "executive_review" and d["refund_amount_inr"] == 16597


def test_already_approved_refund_does_not_start_a_workflow():
    d = decide_refund_review(MockAPI(), customer_id="CUST-0050", order_id="ORD-4120")  # REF-0015, approved
    assert not d["start"] and "already approved" in d["reason"]


def test_order_without_a_refund_request_does_not_start_a_workflow():
    assert not decide_refund_review(MockAPI(), customer_id="CUST-0001", order_id="ORD-4001")["start"]


def test_refund_owned_by_another_customer_does_not_start_a_workflow():
    assert not decide_refund_review(MockAPI(), customer_id="CUST-0001", order_id="ORD-4099")["start"]


def test_cancelled_order_never_opens_an_approval_workflow():
    api = MockAPI()
    api.orders["ORD-4099"]["status"] = "cancelled"
    assert not decide_refund_review(api, customer_id="CUST-0050", order_id="ORD-4099")["start"]


def test_abuse_flagged_account_is_routed_away_from_the_workflow():
    api = MockAPI()
    api.customers["CUST-0050"]["abuse_flag"] = "True"
    d = decide_refund_review(api, customer_id="CUST-0050", order_id="ORD-4099")
    assert not d["start"] and "ESC-ABUSE" in d["reason"]


def test_refund_category_ticket_without_order_id_matches_the_single_pending_refund():
    d = decide_refund_review(MockAPI(), customer_id="CUST-0054", order_id=None, category="REFUND")
    assert d["start"] and d["order_id"] == "ORD-4077"


def test_non_refund_ticket_without_order_id_does_not_start_a_workflow():
    assert not decide_refund_review(MockAPI(), customer_id="CUST-0054", order_id=None, category="ORDER")["start"]


def test_find_pending_refunds_lists_every_reviewable_pending_case_in_the_mock_data():
    api = MockAPI()
    found = find_pending_refunds(api)
    ids = {d["order_id"] for d in found}
    assert {"ORD-4064", "ORD-4077", "ORD-4099"} <= ids          # the three original base-sample cases
    assert len(found) > 20                                       # the extension layer adds many more
    for d in found:                                              # none is auto-tier, cancelled, or abuse-flagged
        assert d["approval_tier"] in ("manager_review", "executive_review")
        assert api.orders[d["order_id"]]["status"] != "cancelled"
        assert api.customers[api.orders[d["order_id"]]["customer_id"]]["abuse_flag"] not in ("True", True)


# --------------------------------------------------------------------------
# Full graph wiring -- needs the real `langgraph` package.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph is not installed in this environment")
class TestCompiledGraph:
    def test_build_graph_compiles_with_the_expected_nodes(self):
        from shopsense.workflows.refund_approval_graph import build_refund_approval_graph

        graph = build_refund_approval_graph()
        nodes = set(graph.get_graph().nodes)
        assert {"draft", "checks", "human_approval", "finalize", "escalate_unresolved"} <= nodes


def test_check_recommendation_flags_invented_history():
    state = make_state(case_facts={"refund_id": "REF-1"},
                       recommendation="Order ORD-9001, ESC-L2, INR 15,000. No prior partial refunds or chargebacks recorded.")
    assert sum("unsupported history" in i for i in check_recommendation(state)) >= 2


def test_check_recommendation_allows_a_draft_that_sticks_to_the_facts():
    assert check_recommendation(make_state(recommendation="Order ORD-9001 needs ESC-L2 sign-off for INR 15,000.")) == []


def test_decide_refund_review_supplies_case_facts_for_the_drafter():
    d = decide_refund_review(MockAPI(), customer_id="CUST-0050", order_id="ORD-4099")
    assert d["case_facts"]["refund_id"] == "REF-0014" and d["case_facts"]["refund_covers_full_order_total"] is True


def test_draft_prompt_lists_case_facts_and_forbids_inventing_others(scripted_litellm):
    scripted_litellm.queue_text("Order ORD-9001 needs ESC-L2 sign-off for INR 15,000.")
    _draft_recommendation_node(make_state(case_facts={"refund_reason_on_file": "item_not_as_described"}))
    system, user = (m["content"] for m in scripted_litellm.calls[0]["messages"])
    assert "ONLY the facts" in system and "item_not_as_described" in user
