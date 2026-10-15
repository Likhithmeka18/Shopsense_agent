"""
ShopSense MCP: the order / refund / shipping operations server (capstone Milestone 6), over stdio.

Run standalone with:  python -m shopsense.mcp.shopsense_server
(it then waits on stdin for JSON-RPC - that silence means it is working; never print to stdout in a stdio server).

Needs an MCP server library (fastmcp, or the official `mcp` SDK -- see compat.py). All logic lives in shopsense/mcp/ops_tools.py (testable without it); this file only registers the
toolbox's methods as MCP tools. Authority is a property of this server's deployment (SHOPSENSE_MCP_ROLE, default
`system`), never of a tool argument - see ops_tools.py for the approval-token rule.

State: one MockAPI plus a JSONL action ledger (data/shopsense/workflow_state/actions.jsonl) so committed actions survive
a restart and a re-run of the same ticket is idempotent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shopsense.mcp.compat import load_server_class  # noqa: E402
from shopsense.mcp.ops_tools import OpsToolbox  # noqa: E402
from shopsense.tools.action_tools import ActionLedger  # noqa: E402

LEDGER_PATH = Path(os.environ.get("SHOPSENSE_ACTION_LEDGER", _REPO_ROOT / "data" / "shopsense" / "workflow_state" / "actions.jsonl"))

FastMCP, _SERVER_LIB = load_server_class()
mcp = FastMCP("shopsense-ops")
_box = OpsToolbox(ledger=ActionLedger(LEDGER_PATH), role=os.environ.get("SHOPSENSE_MCP_ROLE", "system"))


@mcp.tool()
def order_lookup(order_id: str) -> dict:
    """Look up an order's status, total amount, payment method and customer name. Args: order_id like ORD-4445."""
    return _box.order_lookup(order_id)


@mcp.tool()
def shipping_tracker(order_id: str) -> dict:
    """Shipment tracking for an order: carrier, tracking number and scan state."""
    return _box.shipping_tracker(order_id)


@mcp.tool()
def refund_status(order_id: str) -> dict:
    """Whether the order has a refund request, its amount, approval tier and whether it is approved."""
    return _box.refund_status(order_id)


@mcp.tool()
def list_pending_refunds(min_amount_inr: float = 0.0) -> list[dict]:
    """Refund requests still awaiting approval, largest first."""
    return _box.list_pending_refunds(min_amount_inr)


@mcp.tool()
def check_return_eligibility(order_id: str, kind: str = "change_of_mind") -> dict:
    """Which items of an order can still be returned. kind: 'change_of_mind' or 'defect'. Applies category windows,
    festive extensions and the non-returnable rules from the policy documents."""
    return _box.check_return_eligibility(order_id, kind)


@mcp.tool()
def issue_refund(order_id: str, amount_inr: float, ticket_id: str = "", customer_id: str = "", reason: str = "",
                 approval_token: str = "") -> dict:
    """Issue a refund. Runs at the server's role (system: under INR 2,000 only). Larger refunds are refused unless a human
    approval_token minted for exactly this order/ticket/amount is supplied. Idempotent per (order, ticket)."""
    return _box.issue_refund(order_id, amount_inr, ticket_id, customer_id, reason, approval_token)


@mcp.tool()
def approve_pending_refund(order_id: str, amount_inr: float = 0.0, ticket_id: str = "", approval_token: str = "") -> dict:
    """Approve an existing pending refund request. Needs an approval_token for amounts of INR 2,000 or more."""
    return _box.approve_pending_refund(order_id, amount_inr, ticket_id, approval_token)


@mcp.tool()
def request_replacement(order_id: str, value_inr: float, ticket_id: str = "", customer_id: str = "", approval_token: str = "") -> dict:
    """Arrange a replacement for a delivered order. Same authority rule as a refund of the same value."""
    return _box.request_replacement(order_id, value_inr, ticket_id, customer_id, approval_token)


@mcp.tool()
def cancel_order(order_id: str, ticket_id: str = "", customer_id: str = "") -> dict:
    """Cancel an order that is still in processing. Shipped or delivered orders are refused."""
    return _box.cancel_order(order_id, ticket_id, customer_id)


if __name__ == "__main__":
    print(f"[shopsense.mcp.shopsense_server] starting on stdio (role={_box.role}, library={_SERVER_LIB})", file=sys.stderr)
    mcp.run(transport="stdio")
