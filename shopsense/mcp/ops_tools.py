"""
ShopSense MCP: the order / refund / shipping operations a ticket-handling agent is allowed to perform, as plain Python.

`shopsense_server.py` registers these as MCP tools; keeping the logic here means every rule is unit-testable without the
`fastmcp` package, and the MCP layer stays a thin transport.

THE AUTHORITY RULE (capstone M6: "an MCP server exposing order/refund/shipping mock APIs" - with money behind it):

  * Read tools are open to the agent.
  * Write tools run at the server's own role (default `system`: refunds under INR 2,000 only, RF-3 / RA-1).
  * A caller cannot ask for a higher role. The ONLY way to act as a manager or executive is an `approval_token`: an
    HMAC signed with a secret the model never sees (env SHOPSENSE_APPROVAL_SECRET), minted by the human-approval step
    for exactly one (order, ticket, role, amount). A forged, altered or reused-for-another-order token is ignored and the
    call simply runs as `system` - which refuses anything above its limit.

So even a prompt-injected model with full access to every tool still cannot move more money than a `system` caller may.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

from ..tools import policy_rules as pr
from ..tools.action_tools import ActionLedger, ActionTools
from ..tools.order_tools import MockAPI, build_tools

TRUSTED_ROLES = ("manager", "executive", "trust_safety")
_DEV_SECRET = "shopsense-dev-secret-change-me"


def approval_secret() -> str:
    return os.environ.get("SHOPSENSE_APPROVAL_SECRET", _DEV_SECRET)


def sign_approval(*, order_id: str, ticket_id: str, role: str, amount_inr: float, secret: Optional[str] = None) -> str:
    """Minted by the human-approval step (never by a model). Binds one decision to one order/ticket/role/amount."""
    msg = f"{order_id}|{ticket_id}|{role}|{float(amount_inr):.2f}".encode()
    sig = hmac.new((secret or approval_secret()).encode(), msg, hashlib.sha256).hexdigest()[:32]
    return f"{role}:{sig}"


def verify_approval(token: str, *, order_id: str, ticket_id: str, amount_inr: float, secret: Optional[str] = None) -> Optional[str]:
    """The role the token proves, or None. Constant-time compare."""
    if not token or ":" not in token:
        return None
    role = token.split(":", 1)[0]
    if role not in TRUSTED_ROLES:
        return None
    expected = sign_approval(order_id=order_id, ticket_id=ticket_id, role=role, amount_inr=amount_inr, secret=secret)
    return role if hmac.compare_digest(expected, token) else None


class OpsToolbox:
    """Everything the MCP server exposes, bound to one MockAPI and one ActionLedger. All returns are JSON-safe dicts."""

    def __init__(self, api: Optional[MockAPI] = None, ledger: Optional[ActionLedger] = None, *, role: str = "system",
                 secret: Optional[str] = None):
        self.api = api or MockAPI()
        self.actions = ActionTools(self.api, ledger)
        self.role = role
        self.secret = secret
        self._reads, _ = build_tools(self.api)

    # ---- reads -----------------------------------------------------------------------------------------------------
    def order_lookup(self, order_id: str) -> dict:
        return self._reads["order_lookup"](order_id)

    def shipping_tracker(self, order_id: str) -> dict:
        return self._reads["shipping_tracker"](order_id)

    def refund_status(self, order_id: str) -> dict:
        return self._reads["refund_status"](order_id)

    def list_pending_refunds(self, min_amount_inr: float = 0.0) -> list[dict]:
        out = [{"order_id": oid, "refund_id": r["refund_id"], "amount_inr": float(r["amount_inr"]), "approval_tier": r["approval_tier"]}
               for oid, r in self.api.refunds.items() if r["approved"] not in ("True", True) and float(r["amount_inr"]) >= min_amount_inr]
        return sorted(out, key=lambda r: -r["amount_inr"])

    def check_return_eligibility(self, order_id: str, kind: str = "change_of_mind") -> dict:
        if order_id not in self.api.orders:
            return {"error": f"Order {order_id} not found"}
        if kind not in ("change_of_mind", "defect"):
            return {"error": "kind must be 'change_of_mind' or 'defect'"}
        return pr.return_eligibility(self.api, order_id, kind=kind)

    # ---- guarded writes --------------------------------------------------------------------------------------------
    def _role_for(self, token: str, *, order_id: str, ticket_id: str, amount_inr: float) -> str:
        return verify_approval(token, order_id=order_id, ticket_id=ticket_id, amount_inr=amount_inr, secret=self.secret) or self.role

    def issue_refund(self, order_id: str, amount_inr: float, ticket_id: str = "", customer_id: str = "", reason: str = "",
                     approval_token: str = "") -> dict:
        role = self._role_for(approval_token, order_id=order_id, ticket_id=ticket_id, amount_inr=amount_inr)
        return self.actions.issue_refund(order_id, float(amount_inr), approver_role=role, ticket_id=ticket_id, reason=reason,
                                         customer_id=customer_id or None)

    def approve_pending_refund(self, order_id: str, amount_inr: float = 0.0, ticket_id: str = "", approval_token: str = "") -> dict:
        pending = self.api.refunds.get(order_id)
        amount = float(amount_inr) if amount_inr else (float(pending["amount_inr"]) if pending else 0.0)
        role = self._role_for(approval_token, order_id=order_id, ticket_id=ticket_id, amount_inr=amount)
        return self.actions.approve_pending_refund(order_id, approver_role=role, amount_inr=amount or None, ticket_id=ticket_id)

    def request_replacement(self, order_id: str, value_inr: float, ticket_id: str = "", customer_id: str = "",
                            approval_token: str = "") -> dict:
        role = self._role_for(approval_token, order_id=order_id, ticket_id=ticket_id, amount_inr=value_inr)
        return self.actions.request_replacement(order_id, value_inr=float(value_inr), approver_role=role, ticket_id=ticket_id,
                                                customer_id=customer_id or None)

    def cancel_order(self, order_id: str, ticket_id: str = "", customer_id: str = "") -> dict:
        return self.actions.cancel_order(order_id, ticket_id=ticket_id, customer_id=customer_id or None)

    # ---- registry (name -> callable); the server and the tests both read this -------------------------------------
    READ_TOOLS = ("order_lookup", "shipping_tracker", "refund_status", "list_pending_refunds", "check_return_eligibility")
    WRITE_TOOLS = ("issue_refund", "approve_pending_refund", "request_replacement", "cancel_order")

    def tools(self) -> dict:
        return {name: getattr(self, name) for name in self.READ_TOOLS + self.WRITE_TOOLS}
