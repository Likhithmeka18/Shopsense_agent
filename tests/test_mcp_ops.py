"""
Capstone M6 tests: the order/refund/shipping MCP toolbox (shopsense/mcp/ops_tools.py) and the Order-Actions agent's MCP
adapters (shopsense/teams/mcp_ops.py). A FakeMcpClient speaks the same interface as langchain-mcp-adapters (async
get_tools -> tools with async ainvoke returning content blocks), so the JSON round-trip and the sync facade are exercised
with no server process. What it cannot prove - that fastmcp registers and serves the tools over stdio - is run by
run_ticket_team_mcp_demo.py on a machine with the packages.
"""
import json

from shopsense.mcp import ops_tools as ops
from shopsense.teams import mcp_ops
from shopsense.tools.order_tools import MockAPI
from shopsense.workflows import ticket_resolution_graph as g
from ticket_sim import Paused, keyword_extract, resume, run


class FakeTool:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    async def ainvoke(self, args):
        # What a real MCP adapter hands back: JSON inside text content blocks.
        return [{"type": "text", "text": json.dumps(self._fn(**args))}]


class FakeMcpClient:
    def __init__(self, box):
        self.box, self.calls = box, []

    async def get_tools(self):
        def wrap(name, fn):
            def traced(**kw):
                self.calls.append((name, kw))
                return fn(**kw)
            return FakeTool(name, traced)
        return [wrap(n, f) for n, f in self.box.tools().items()]


def mcp_deps(secret="s3cret"):
    box = ops.OpsToolbox(secret=secret)
    client = FakeMcpClient(box)
    toolset = mcp_ops.McpToolSet(client)
    return g.Deps(api=MockAPI(), kb=_KB(), extract=keyword_extract, **mcp_ops.mcp_backed(toolset, secret=secret)), box, client


class _KB:
    def answer(self, query, *, k=3, use_rerank=True, question=None):
        return {"answer": "See policy.", "sources": [{"id": 0, "source": "faq", "section": "x"}]}


def go(deps, text, cust, tid="TCK-M1"):
    return run(deps, g.seed_state(ticket_id=tid, customer_id=cust, text=text))


# ---- the toolbox ------------------------------------------------------------------------------------------------
def test_toolbox_registry_is_exactly_the_documented_surface():
    box = ops.OpsToolbox()
    assert set(box.tools()) == {"order_lookup", "shipping_tracker", "refund_status", "list_pending_refunds", "check_return_eligibility",
                                "issue_refund", "approve_pending_refund", "request_replacement", "cancel_order"}


def test_reads_are_json_safe_and_list_pending_is_sorted_largest_first():
    box = ops.OpsToolbox()
    for name in ("order_lookup", "shipping_tracker", "refund_status", "check_return_eligibility"):
        json.dumps(box.tools()[name]("ORD-4445"))
    pend = box.list_pending_refunds(10_000)
    assert pend and all(p["amount_inr"] >= 10_000 for p in pend) and pend == sorted(pend, key=lambda p: -p["amount_inr"])
    assert box.check_return_eligibility("ORD-9999") == {"error": "Order ORD-9999 not found"}
    assert "error" in box.check_return_eligibility("ORD-4445", "bogus")


def test_a_caller_cannot_raise_its_own_role_only_a_valid_token_can():
    box = ops.OpsToolbox(secret="k")
    big = box.issue_refund("ORD-4553", 25516.0, "T1", "CUST-0142")
    assert big["refused"] == "above_role_limit" and big["role"] == "system"
    for forged in ("executive:deadbeef", "executive", "root:abc", "manager:" + "0" * 32):
        assert box.approve_pending_refund("ORD-4553", 0.0, "T1", forged)["refused"] == "above_role_limit"
    tok = ops.sign_approval(order_id="ORD-4553", ticket_id="T1", role="executive", amount_inr=25516.0, secret="k")
    ok = box.approve_pending_refund("ORD-4553", 0.0, "T1", tok)
    assert ok["ok"] and ok["approver_role"] == "executive"


def test_a_token_is_bound_to_one_order_ticket_role_and_amount():
    tok = ops.sign_approval(order_id="ORD-4553", ticket_id="T1", role="executive", amount_inr=25516.0, secret="k")
    v = lambda **kw: ops.verify_approval(tok, **{"order_id": "ORD-4553", "ticket_id": "T1", "amount_inr": 25516.0, "secret": "k", **kw})
    assert v() == "executive"
    assert v(order_id="ORD-4554") is None and v(ticket_id="T2") is None and v(amount_inr=25517.0) is None and v(secret="other") is None
    box = ops.OpsToolbox(secret="k")
    # a manager token cannot be used for 10,000+ even when the amount matches it
    big = ops.sign_approval(order_id="ORD-4553", ticket_id="T1", role="manager", amount_inr=25516.0, secret="k")
    assert box.approve_pending_refund("ORD-4553", 0.0, "T1", big)["refused"] == "above_role_limit"


def test_writes_are_idempotent_and_cancel_only_touches_processing_orders():
    box = ops.OpsToolbox()
    first = box.issue_refund("ORD-4503", 1623.0, "T9", "CUST-0082")
    again = box.issue_refund("ORD-4503", 1623.0, "T9", "CUST-0082")
    assert first["ok"] and again["duplicate"] and len(box.actions.ledger.records) == 1
    assert box.cancel_order("ORD-4445", "T9", "CUST-0077")["refused"] == "only_processing_orders_can_be_cancelled"


def test_server_file_registers_exactly_the_toolbox_surface():
    text = (ops.__file__.replace("ops_tools.py", "shopsense_server.py"))
    src = open(text).read()
    for name in ops.OpsToolbox.READ_TOOLS + ops.OpsToolbox.WRITE_TOOLS:
        assert f"def {name}(" in src and f"_box.{name}(" in src, name
    assert src.count("@mcp.tool()") == len(ops.OpsToolbox.READ_TOOLS) + len(ops.OpsToolbox.WRITE_TOOLS)
    assert "approver_role" not in src, "the role must never be a tool parameter"


# ---- the adapters --------------------------------------------------------------------------------------------------
def test_toolset_round_trips_content_blocks_and_surfaces_unknown_tools():
    deps, box, client = mcp_deps()
    assert deps.tools["order_lookup"]("ORD-4445")["status"] == box.order_lookup("ORD-4445")["status"]
    assert client.calls[0][0] == "order_lookup"
    try:
        mcp_ops.McpToolSet(client).call("drop_database")
        raise AssertionError("must raise")
    except RuntimeError as e:
        assert "does not expose" in str(e)


def test_whole_ticket_flow_over_mcp_system_refund_then_human_approved_refund():
    deps, box, client = mcp_deps()
    r = go(deps, "I'd like to return the denim jacket from ORD-4503, it doesn't fit.", "CUST-0082", "TCK-M1")
    assert r["route"] == "auto" and r["action_result"]["committed"]
    assert [c for c in client.calls if c[0] == "issue_refund"][0][1]["approval_token"] == "", "system actions carry no token"
    assert box.actions.ledger.records[0]["approver_role"] == "system"

    mug = go(deps, "The ceramic mug set from ORD-4589 arrived smashed 5 days ago.", "CUST-0066", "TCK-M2")
    assert isinstance(mug, Paused)
    assert not [c for c in client.calls if c[0] in ("approve_pending_refund",) or (c[0] == "issue_refund" and c[1]["ticket_id"] == "TCK-M2")]
    done = resume(deps, mug, decision="approve", approver_role="manager")
    assert done["action_result"]["committed"] and box.actions.ledger.records[-1]["approver_role"] == "manager"
    sent = [c for c in client.calls if c[0] == "issue_refund" and c[1]["ticket_id"] == "TCK-M2"][0][1]
    assert sent["approval_token"].startswith("manager:")


def test_a_wrong_secret_on_the_client_means_the_server_refuses_the_human_decision():
    deps, box, _ = mcp_deps()
    deps.actions.secret = "not-the-servers-secret"
    mug = go(deps, "The ceramic mug set from ORD-4589 arrived smashed 5 days ago.", "CUST-0066", "TCK-M3")
    done = resume(deps, mug, decision="approve", approver_role="manager")
    assert done["status"] == "needs_higher_approval" and not box.actions.ledger.records


def test_eligibility_comes_from_the_server_in_mcp_mode():
    deps, box, client = mcp_deps()
    go(deps, "I want to return the denim jacket from ORD-4503.", "CUST-0082", "TCK-M4")
    assert any(c[0] == "check_return_eligibility" for c in client.calls)
