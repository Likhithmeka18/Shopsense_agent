"""
ShopSense Tools: the ONLY functions that change the world (Milestone 5, capstone: "action execution").

issue_refund / approve_pending_refund / request_replacement / cancel_order are deliberately NOT in the M2 tool schema
that the free-running ShopSenseAgent can call. They are reached only from the ticket graph's commit node, i.e. only
after routing has decided the action is within the caller's authority or a human has approved it. And they enforce
the authority limits THEMSELVES, so even a bug in the graph, a hijacked prompt or a future caller cannot move money
past a limit -- "guardrails against unauthorized high-value refunds" lives at the last line of defence, not only
the first:

    role         may release a refund of ... (cumulative per order)       source
    system/ai    below INR 2,000                                           RF-3, RA-1.1 (AI never approves >= 2,000)
    manager      below INR 10,000                                          RF-3, RA-1
    executive    any amount                                                RF-3, RA-1
    trust_safety (abuse-flagged accounts only; any amount)                 FP-1.2

Refusals are returned, not raised: {"ok": False, "refused": "<reason>"} -- a policy refusal is a normal outcome the
graph has to route on, not an exception. Every successful action is idempotent per (action, order, ticket), because a
LangGraph node can re-run after a crash/resume and a refund must never be issued twice.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from ..core.safety_net import is_abuse_flagged
from .order_tools import MockAPI
from .policy_rules import AUTO_APPROVE_BELOW_INR, EXECUTIVE_FROM_INR

# Exclusive upper bound of what each role may release. None == unlimited.
ROLE_LIMIT_INR: dict[str, Optional[float]] = {
    "system": AUTO_APPROVE_BELOW_INR, "ai_agent": AUTO_APPROVE_BELOW_INR,
    "manager": EXECUTIVE_FROM_INR, "executive": None, "trust_safety": None,
}
HUMAN_ROLES = ("manager", "executive", "trust_safety")


def limit_for(role: str) -> Optional[float]:
    return ROLE_LIMIT_INR.get(role, AUTO_APPROVE_BELOW_INR)      # unknown role == least privilege


class ActionLedger:
    """Append-only record of every committed action. In-memory by default; give it a path to survive restarts, which
    is what makes a resumed-in-a-new-process commit idempotent."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path and self.path.exists():
            self.records = [json.loads(line) for line in open(self.path) if line.strip()]

    def find(self, key: str) -> Optional[dict]:
        return next((r for r in self.records if r["key"] == key), None)

    def add(self, record: dict) -> dict:
        record = {**record, "ts": time.time()}
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def refunded_total(self, order_id: str) -> float:
        return sum(r.get("amount_inr", 0.0) for r in self.records if r["action"] in ("issue_refund", "approve_pending_refund") and r["order_id"] == order_id)


def refused(reason: str, **extra: Any) -> dict:
    return {"ok": False, "refused": reason, **extra}


class ActionTools:
    def __init__(self, api: MockAPI, ledger: Optional[ActionLedger] = None):
        self.api = api
        self.ledger = ledger or ActionLedger()
        self._replay()

    def _replay(self) -> None:
        """MockAPI is rebuilt from the CSVs in every process, but the ledger is on disk. Re-apply what the ledger says already
        happened, so a second process (the approver's resume script, tomorrow's pipeline run) sees an order as refunded or
        cancelled and cannot refund it a second time under a different ticket."""
        for r in self.ledger.records:
            oid = r.get("order_id")
            if r["action"] in ("issue_refund", "approve_pending_refund") and oid in self.api.orders:
                existing = self.api.refunds.get(oid) or {"refund_id": r.get("refund_id", ""), "order_id": oid,
                                                         "customer_id": self.api.orders[oid]["customer_id"], "reason": "ledger_replay",
                                                         "requested_date": ""}
                self.api.refunds[oid] = {**existing, "amount_inr": r["amount_inr"], "approved": True,
                                         "approval_tier": _tier(r["amount_inr"])}
            elif r["action"] == "cancel_order" and oid in self.api.orders:
                self.api.orders[oid] = {**self.api.orders[oid], "status": "cancelled"}

    # ---- guards shared by the money-moving actions ----
    def _check_order(self, order_id: str, customer_id: Optional[str]) -> Optional[dict]:
        order = self.api.orders.get(order_id)
        if order is None:
            return refused("order_not_found")
        if customer_id and not customer_id.startswith("GUEST") and order["customer_id"] != customer_id:
            return refused("order_belongs_to_another_customer")
        if order["status"] == "cancelled":
            return refused("order_is_cancelled")
        return None

    def _check_authority(self, order_id: str, amount: float, role: str, customer: Optional[dict]) -> Optional[dict]:
        if is_abuse_flagged(customer) and role != "trust_safety":
            return refused("abuse_flagged_account_requires_trust_and_safety")
        limit = limit_for(role)
        cumulative = self.ledger.refunded_total(order_id) + amount          # RA-2.1: tier is decided on the order's TOTAL
        if limit is not None and cumulative >= limit:
            return refused("above_role_limit", role=role, limit_inr=limit, cumulative_inr=cumulative)
        return None

    # ---- actions ----
    def issue_refund(self, order_id: str, amount_inr: float, *, approver_role: str = "system", ticket_id: str = "",
                     reason: str = "", customer_id: Optional[str] = None) -> dict:
        key = f"issue_refund:{order_id}:{ticket_id}"
        if (prior := self.ledger.find(key)):
            return {**prior, "ok": True, "duplicate": True}
        if (bad := self._check_order(order_id, customer_id)):
            return bad
        order = self.api.orders[order_id]
        if amount_inr <= 0 or amount_inr > float(order["total_amount_inr"]):
            return refused("amount_out_of_range", order_total_inr=float(order["total_amount_inr"]))
        existing = self.api.refunds.get(order_id)
        if existing and existing["approved"] in ("True", True):
            return refused("already_refunded", refund_id=existing["refund_id"])
        if (bad := self._check_authority(order_id, amount_inr, approver_role, self.api.customers.get(order["customer_id"]))):
            return bad
        refund_id = existing["refund_id"] if existing else f"REF-A{len(self.ledger.records) + 1:04d}"
        record = {"refund_id": refund_id, "order_id": order_id, "customer_id": order["customer_id"], "amount_inr": float(amount_inr),
                  "approval_tier": _tier(amount_inr), "approved": True, "requested_date": "", "reason": reason or "ticket_refund"}
        self.api.refunds[order_id] = {**(existing or {}), **record}
        done = self.ledger.add({"key": key, "action": "issue_refund", "order_id": order_id, "refund_id": refund_id,
                                "amount_inr": float(amount_inr), "approver_role": approver_role, "ticket_id": ticket_id})
        return {**done, "ok": True}

    def approve_pending_refund(self, order_id: str, *, approver_role: str, amount_inr: Optional[float] = None, ticket_id: str = "") -> dict:
        key = f"approve_pending_refund:{order_id}:{ticket_id}"
        if (prior := self.ledger.find(key)):
            return {**prior, "ok": True, "duplicate": True}
        if (bad := self._check_order(order_id, None)):
            return bad
        existing = self.api.refunds.get(order_id)
        if existing is None:
            return refused("no_refund_request_on_file")
        if existing["approved"] in ("True", True):
            return refused("already_refunded", refund_id=existing["refund_id"])
        amount = float(amount_inr if amount_inr is not None else existing["amount_inr"])
        if amount > float(existing["amount_inr"]):
            return refused("cannot_approve_more_than_requested", requested_inr=float(existing["amount_inr"]))
        order = self.api.orders[order_id]
        if (bad := self._check_authority(order_id, amount, approver_role, self.api.customers.get(order["customer_id"]))):
            return bad
        self.api.refunds[order_id] = {**existing, "approved": True, "amount_inr": amount}
        done = self.ledger.add({"key": key, "action": "approve_pending_refund", "order_id": order_id, "refund_id": existing["refund_id"],
                                "amount_inr": amount, "approver_role": approver_role, "ticket_id": ticket_id})
        return {**done, "ok": True}

    def request_replacement(self, order_id: str, *, value_inr: float, approver_role: str = "system", ticket_id: str = "",
                            customer_id: Optional[str] = None) -> dict:
        key = f"request_replacement:{order_id}:{ticket_id}"
        if (prior := self.ledger.find(key)):
            return {**prior, "ok": True, "duplicate": True}
        if (bad := self._check_order(order_id, customer_id)):
            return bad
        order = self.api.orders[order_id]
        if order["status"] != "delivered":
            return refused("order_not_delivered")
        # A replacement ships goods of the same value, so it needs the same authority as a refund of that value.
        if (bad := self._check_authority(order_id, 0.0 + value_inr, approver_role, self.api.customers.get(order["customer_id"]))):
            return bad
        done = self.ledger.add({"key": key, "action": "request_replacement", "order_id": order_id, "value_inr": float(value_inr),
                                "approver_role": approver_role, "ticket_id": ticket_id, "replacement_id": f"RPL-{len(self.ledger.records) + 1:04d}"})
        return {**done, "ok": True}

    def cancel_order(self, order_id: str, *, ticket_id: str = "", customer_id: Optional[str] = None) -> dict:
        key = f"cancel_order:{order_id}:{ticket_id}"
        if (prior := self.ledger.find(key)):
            return {**prior, "ok": True, "duplicate": True}
        order = self.api.orders.get(order_id)
        if order is None:
            return refused("order_not_found")
        if customer_id and not customer_id.startswith("GUEST") and order["customer_id"] != customer_id:
            return refused("order_belongs_to_another_customer")
        if order["status"] != "processing":
            return refused("only_processing_orders_can_be_cancelled", status=order["status"])      # CN-1.1 / CN-2.1
        self.api.orders[order_id] = {**order, "status": "cancelled"}
        prepaid = order["payment_method"] != "Cash on Delivery"
        done = self.ledger.add({"key": key, "action": "cancel_order", "order_id": order_id, "ticket_id": ticket_id,
                                "amount_inr": 0.0, "prepaid": prepaid, "reversal_inr": float(order["total_amount_inr"]) if prepaid else 0.0})
        return {**done, "ok": True}


def _tier(amount: float) -> str:
    from .order_tools import refund_tier_for_amount
    return refund_tier_for_amount(amount)
