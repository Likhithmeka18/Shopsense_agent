"""The guarded write tools (shopsense/tools/action_tools.py): the only code that changes an order, refund or cancellation."""
import tempfile
from pathlib import Path

from shopsense.core.safety_net import is_abuse_flagged
from shopsense.tools.action_tools import ActionLedger, ActionTools, limit_for
from shopsense.tools.order_tools import MockAPI


def fresh(path=None):
    api = MockAPI()
    return ActionTools(api, ActionLedger(path)), api


def test_role_limits_follow_the_authorisation_matrix():
    assert limit_for("system") == 2000 and limit_for("manager") == 10000 and limit_for("executive") is None


def test_system_refunds_below_2000_only_and_refusals_change_nothing():
    a, api = fresh()
    assert a.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T1")["ok"]
    big = a.issue_refund("ORD-4445", 13347.0, approver_role="system", ticket_id="T2")
    assert big["refused"] == "above_role_limit" and api.refunds.get("ORD-4445") is None and len(a.ledger.records) == 1
    assert a.issue_refund("ORD-4445", 13347.0, approver_role="manager", ticket_id="T2")["refused"] == "above_role_limit"
    assert a.issue_refund("ORD-4445", 13347.0, approver_role="executive", ticket_id="T2")["ok"]


def test_cumulative_amount_on_one_order_cannot_be_split_under_the_limit():
    a, _ = fresh()
    assert a.issue_refund("ORD-4445", 1500.0, approver_role="system", ticket_id="T1")["ok"]
    assert a.issue_refund("ORD-4445", 1500.0, approver_role="system", ticket_id="T2").get("refused") in ("above_role_limit", "already_refunded")


def test_idempotent_per_action_order_and_ticket():
    a, _ = fresh()
    first = a.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T1")
    again = a.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T1")
    assert again["duplicate"] and again["refund_id"] == first["refund_id"] and len(a.ledger.records) == 1


def test_abuse_flagged_accounts_need_trust_and_safety():
    a, api = fresh()
    oid = next(o for o, row in api.orders.items() if is_abuse_flagged(api.customers.get(row["customer_id"])) and row["status"] == "delivered"
               and float(row["total_amount_inr"]) < 2000)
    assert a.issue_refund(oid, 100.0, approver_role="system", ticket_id="T1")["refused"] == "abuse_flagged_account_requires_trust_and_safety"
    assert a.issue_refund(oid, 100.0, approver_role="executive", ticket_id="T1")["refused"] == "abuse_flagged_account_requires_trust_and_safety"
    assert a.issue_refund(oid, 100.0, approver_role="trust_safety", ticket_id="T1")["ok"]


def test_cannot_refund_more_than_the_order_or_someone_elses_order():
    a, _ = fresh()
    assert a.issue_refund("ORD-4503", 99999.0, approver_role="executive", ticket_id="T1")["refused"] == "amount_out_of_range"
    assert a.issue_refund("ORD-4503", 100.0, approver_role="system", ticket_id="T1", customer_id="CUST-0001")["refused"] == "order_belongs_to_another_customer"


def test_pending_refund_approval_cannot_exceed_what_was_requested():
    a, api = fresh()
    pending = api.refunds["ORD-4553"]
    assert a.approve_pending_refund("ORD-4553", approver_role="executive", amount_inr=float(pending["amount_inr"]) + 1, ticket_id="T1")["refused"] \
        == "cannot_approve_more_than_requested"
    lowered = a.approve_pending_refund("ORD-4553", approver_role="executive", amount_inr=1000.0, ticket_id="T1")
    assert lowered["ok"] and lowered["amount_inr"] == 1000.0


def test_cancel_only_processing_orders():
    a, api = fresh()
    processing = next(o for o, r in api.orders.items() if r["status"] == "processing")
    assert a.cancel_order(processing, ticket_id="T1")["ok"] and api.orders[processing]["status"] == "cancelled"
    assert a.cancel_order("ORD-4445", ticket_id="T1")["refused"] == "only_processing_orders_can_be_cancelled"


def test_a_second_process_sees_what_the_first_one_did():
    path = Path(tempfile.mkdtemp()) / "actions.jsonl"
    a, api = fresh(path)
    processing = next(o for o, r in api.orders.items() if r["status"] == "processing")
    assert a.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T1")["ok"]
    assert a.cancel_order(processing, ticket_id="T1")["ok"]
    b, api2 = fresh(path)                                                      # new tables, same ledger file
    assert api2.refunds["ORD-4503"]["approved"] is True and api2.orders[processing]["status"] == "cancelled"
    assert b.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T9")["refused"] == "already_refunded"
    assert b.issue_refund("ORD-4503", 1623.0, approver_role="system", ticket_id="T1")["duplicate"]
