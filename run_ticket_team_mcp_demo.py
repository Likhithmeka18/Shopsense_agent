"""
ShopSense capstone Milestone 6 demo: the same support team with its Order-Actions tools served by a REAL MCP server
(shopsense/mcp/shopsense_server.py, spawned as a child process over stdio).

    pip install langgraph langgraph-checkpoint-sqlite 'mcp<2' langchain-mcp-adapters
    export GROQ_API_KEY=gsk_...
    python run_ticket_team_mcp_demo.py

Proves: (1) the server exposes the order / refund / shipping tools, (2) an automatic refund goes through MCP as the `system`
role, (3) a manager-tier refund pauses and, once a human approves, commits through MCP with a signed approval token, and (4) a
caller cannot talk its way to a higher role: a forged token is ignored and a 25,516 refund is refused at the server.
The server keeps its own tables and a throw-away JSONL ledger; the client process never touches them directly.
"""
from __future__ import annotations

import json
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.runtime import build_deps, build_graph  # noqa: E402
from shopsense.teams import mcp_ops  # noqa: E402
from shopsense.workflows import ticket_resolution_graph as tg  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'} -- {label}")
    if not ok:
        FAILURES.append(label)


def main() -> None:
    if not (tg.LANGGRAPH_AVAILABLE and mcp_ops.MCP_CLIENT_AVAILABLE):
        print("cannot run the MCP demo:")
        if not tg.LANGGRAPH_AVAILABLE:
            print("  langgraph:", tg.LANGGRAPH_IMPORT_ERROR)
        if not mcp_ops.MCP_CLIENT_AVAILABLE:
            print("  MCP client (langchain-mcp-adapters):", mcp_ops.MCP_IMPORT_ERROR)
        print("If that names a missing module, install it:  pip install langgraph langgraph-checkpoint-sqlite 'mcp<2' langchain-mcp-adapters")
        print("If everything is installed, the message above is a version incompatibility -- paste it to get the matching pin.")
        sys.exit(1)
    work = Path(tempfile.mkdtemp())
    ledger_file, secret = work / "server_actions.jsonl", secrets.token_hex(16)
    client = mcp_ops.build_ops_mcp_client(env={"SHOPSENSE_ACTION_LEDGER": str(ledger_file), "SHOPSENSE_APPROVAL_SECRET": secret})
    toolset = mcp_ops.McpToolSet(client)

    print("=" * 70)
    print("ShopSense capstone M6: support team over MCP (order / refund / shipping server)")
    print("=" * 70)

    print("\n[1] The server's tool surface ...")
    try:
        names = sorted(t.name for t in mcp_ops._run(client.get_tools()))
    except BaseException as e:                       # noqa: BLE001 -- an ExceptionGroup traceback hides the one useful fact
        print(f"\n    the MCP server did not start or answer ({type(e).__name__}).")
        print("    Run   python check_mcp_server.py   -- it starts the server alone and prints the server's own error.")
        raise
    print("    tools:", names)
    check({"order_lookup", "shipping_tracker", "refund_status", "issue_refund", "approve_pending_refund", "cancel_order"} <= set(names),
          "order / refund / shipping tools are exposed over MCP")
    check(not any("role" in n for n in names), "no tool takes the caller's role as an argument")
    print("    ORD-4445 via MCP:", toolset.call("order_lookup", order_id="ORD-4445"))

    deps = build_deps(mcp_toolset=toolset, mcp_secret=secret, with_memo_team=False)
    graph = build_graph(deps, topology="team", db_path=work / "cp.db")

    print("\n[2] Automatic refund through MCP ...")
    state = tg.seed_state(ticket_id="TCK-M001", customer_id="CUST-0082", text="I'd like to return the denim jacket from ORD-4503, it doesn't fit.")
    res, _ = tg.run_ticket(graph, state)
    print("    reply:", res["reply"])
    records = [json.loads(line) for line in ledger_file.read_text().splitlines()]
    check(res["status"] == "resolved" and records and records[0]["approver_role"] == "system", "committed by the server as `system`")

    print("\n[3] Manager-tier refund: pause, human approves, commit with a signed token ...")
    state = tg.seed_state(ticket_id="TCK-M009", customer_id="CUST-0066",
                          text="The ceramic mug set from ORD-4589 arrived smashed 5 days ago, there are pieces everywhere.")
    res, cfg = tg.run_ticket(graph, state)
    n = len(ledger_file.read_text().splitlines())
    check("__interrupt__" in res, "paused for a Team Lead")
    done = tg.resume_ticket(graph, cfg, decision="approve", approver_role="manager")
    records = [json.loads(line) for line in ledger_file.read_text().splitlines()]
    print("    reply:", done["reply"])
    check(done["status"] == "resolved_after_approval" and len(records) == n + 1 and records[-1]["approver_role"] == "manager",
          "the server accepted the signed manager token and committed")

    print("\n[4] A caller cannot raise its own role ...")
    forged = toolset.call("approve_pending_refund", order_id="ORD-4553", amount_inr=0.0, ticket_id="TCK-X", approval_token="executive:" + "0" * 32)
    check(forged.get("refused") == "above_role_limit", "a forged executive token is ignored; the call runs as `system` and is refused")
    direct = toolset.call("issue_refund", order_id="ORD-4553", amount_inr=25516.0, ticket_id="TCK-X", customer_id="CUST-0142")
    check(direct.get("refused") == "above_role_limit", "a 25,516 refund without a human approval is refused at the server")

    print("\n" + "=" * 70)
    print("Capstone M6 (MCP) PASS." if not FAILURES else f"FAILED: {FAILURES}")
    print("=" * 70)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
