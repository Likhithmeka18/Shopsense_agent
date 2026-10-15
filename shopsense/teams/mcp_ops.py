"""
ShopSense Teams: the MCP-backed Order-Actions toolset (capstone Milestone 6, part 2).

The Order-Actions agent normally reads orders and acts on them through in-process tools. This module swaps those for the
same operations served over a REAL Model Context Protocol connection (shopsense/mcp/shopsense_server.py, spawned as a
child process over stdio with langchain_mcp_adapters). Nothing else in the team changes - same state, scopes, supervisor
policy and graph - which is the point of MCP: the tools moved, the agent did not.

Two adapters:
  * `McpToolSet`  - a synchronous facade over the async MCP client. `read_tools()` has the same shape as
                    order_tools.build_tools(); `check_return_eligibility` is the `Deps.eligibility` hook.
  * `McpActions`  - the same method names as ActionTools, so commit_node cannot tell the difference. A `system` action is
                    sent as-is; a human-approved action carries an HMAC `approval_token` minted HERE (this process holds
                    the secret, the model never does) - see shopsense/mcp/ops_tools.py.

Needs `fastmcp` + `langchain-mcp-adapters` for the real client. Everything else (this file's logic, the toolbox, the token
rule) is exercised offline by tests/test_mcp_ops.py through a fake client that speaks the same interface.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MCP_CLIENT_AVAILABLE = True
    MCP_IMPORT_ERROR = ""
except ImportError as _e:                 # keep the real reason: "not installed" and "installed but incompatible" look identical otherwise
    MCP_CLIENT_AVAILABLE = False
    MCP_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

from ..mcp.ops_tools import sign_approval
from .mcp_researcher import parse_tool_result

_SERVER_PATH = str(Path(__file__).resolve().parents[2] / "shopsense" / "mcp" / "shopsense_server.py")


def build_ops_mcp_client(env: Optional[dict] = None) -> "MultiServerMCPClient":
    """`env` is merged into the child process's environment (e.g. SHOPSENSE_ACTION_LEDGER, SHOPSENSE_APPROVAL_SECRET)."""
    if not MCP_CLIENT_AVAILABLE:
        raise RuntimeError("langchain-mcp-adapters is not installed. Run:\n    pip install 'mcp<2' langchain-mcp-adapters\nSee README.md.")
    import os
    server = {"transport": "stdio", "command": sys.executable, "args": [_SERVER_PATH]}
    if env:
        server["env"] = {**os.environ, **env}               # a stdio child gets ONLY `env` if given, so carry PATH & friends along
    return MultiServerMCPClient({"shopsense_ops": server})


def _run(coro) -> Any:
    """Run a coroutine from sync code, even if an event loop is already running in this thread (notebooks, async apps)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def target():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:                              # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


class McpToolSet:
    def __init__(self, client: Any):
        self.client = client
        self._tools: Optional[dict] = None

    async def _acall(self, name: str, args: dict) -> Any:
        if self._tools is None:
            self._tools = {t.name: t for t in await self.client.get_tools()}
        if name not in self._tools:
            raise RuntimeError(f"MCP server does not expose a tool named {name!r} (has: {sorted(self._tools)})")
        return parse_tool_result(await self._tools[name].ainvoke(args))

    def call(self, name: str, **args) -> Any:
        result = _run(self._acall(name, args))
        if isinstance(result, str):                             # a tool error comes back as text, not a dict
            return {"error": result}
        return result

    def read_tools(self) -> dict:
        return {"order_lookup": lambda order_id: self.call("order_lookup", order_id=order_id),
                "shipping_tracker": lambda order_id: self.call("shipping_tracker", order_id=order_id),
                "refund_status": lambda order_id: self.call("refund_status", order_id=order_id)}

    def check_return_eligibility(self, order_id: str, kind: str) -> dict:
        return self.call("check_return_eligibility", order_id=order_id, kind=kind)


class McpActions:
    """ActionTools-compatible writes over MCP. `secret=None` uses SHOPSENSE_APPROVAL_SECRET like the server does."""

    def __init__(self, toolset: McpToolSet, *, secret: Optional[str] = None):
        self.toolset, self.secret = toolset, secret

    def _token(self, role: str, order_id: str, ticket_id: str, amount: float) -> str:
        if role == "system":
            return ""
        return sign_approval(order_id=order_id, ticket_id=ticket_id, role=role, amount_inr=amount, secret=self.secret)

    def issue_refund(self, order_id, amount_inr, *, approver_role="system", ticket_id="", reason="", customer_id=None) -> dict:
        return self.toolset.call("issue_refund", order_id=order_id, amount_inr=float(amount_inr), ticket_id=ticket_id,
                                 customer_id=customer_id or "", reason=reason,
                                 approval_token=self._token(approver_role, order_id, ticket_id, float(amount_inr)))

    def approve_pending_refund(self, order_id, *, approver_role, amount_inr=None, ticket_id="") -> dict:
        amount = float(amount_inr or 0.0)
        return self.toolset.call("approve_pending_refund", order_id=order_id, amount_inr=amount, ticket_id=ticket_id,
                                 approval_token=self._token(approver_role, order_id, ticket_id, amount))

    def request_replacement(self, order_id, *, value_inr, approver_role="system", ticket_id="", customer_id=None) -> dict:
        return self.toolset.call("request_replacement", order_id=order_id, value_inr=float(value_inr), ticket_id=ticket_id,
                                 customer_id=customer_id or "",
                                 approval_token=self._token(approver_role, order_id, ticket_id, float(value_inr)))

    def cancel_order(self, order_id, *, ticket_id="", customer_id=None) -> dict:
        return self.toolset.call("cancel_order", order_id=order_id, ticket_id=ticket_id, customer_id=customer_id or "")


def mcp_backed(toolset: McpToolSet, *, secret: Optional[str] = None) -> dict:
    """Keyword arguments for ticket_resolution_graph.Deps that move the Order-Actions agent's tools onto MCP."""
    return {"tools": toolset.read_tools(), "actions": McpActions(toolset, secret=secret), "eligibility": toolset.check_return_eligibility}
