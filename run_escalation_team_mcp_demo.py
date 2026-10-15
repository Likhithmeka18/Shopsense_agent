"""
ShopSense: Milestone 6 demo (MCP-backed researcher) -- the same
escalation-review team as run_escalation_team_demo.py, with ONE node
swapped: the researcher's 'policy_basis' lookup now crosses a real process
boundary to shopsense/mcp/policy_server.py via
`langchain_mcp_adapters.client.MultiServerMCPClient`, instead of calling
the PolicyKnowledgeBase in-process.

Requires `langgraph`, `mcp`, and `langchain-mcp-adapters` (all in
requirements.txt) and a live LLM behind SHOPSENSE_LLM_MODEL. Run with:

    pip install langgraph 'mcp<2' langchain-mcp-adapters
    export GROQ_API_KEY=gsk_...
    python run_escalation_team_mcp_demo.py

What this proves: the SAME evidence is retrieved over a different
transport (identical policy_basis finding whichever way it's fetched),
and nothing else in the team -- the supervisor policy, the scopes, the
revision cap, both critics -- had to change to make that swap. If it had,
the layering in escalation_review_team.py would have been wrong.
"""
from __future__ import annotations

import asyncio
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
    seed_state,
)
from shopsense.teams.mcp_researcher import MCP_CLIENT_AVAILABLE, build_mcp_client, make_mcp_researcher_node  # noqa: E402
from shopsense.tools.order_tools import MockAPI  # noqa: E402


async def main_async() -> None:
    print("=" * 70)
    print("ShopSense Milestone 6: escalation-review supervisor team (MCP)")
    print("=" * 70)

    if not LANGGRAPH_AVAILABLE or not MCP_CLIENT_AVAILABLE:
        print(
            "\nMissing dependencies -- run:\n"
            "    pip install langgraph 'mcp<2' langchain-mcp-adapters\n"
            "then re-run this script."
        )
        sys.exit(1)

    api = MockAPI()
    store = MemoryStore()
    ticket, order_id, reasons = find_escalated_ticket(api, prefer="contains_injection_attempt")
    if ticket is None:
        print("No ESC-2 escalated ticket found in the sample data -- nothing to demo.")
        sys.exit(1)

    print(f"\n[1] Spawning shopsense/mcp/policy_server.py as a child process ...")
    client = build_mcp_client()
    tools = await client.get_tools()
    print(f"    discovered tools: {[t.name for t in tools]}")

    researcher_node = make_mcp_researcher_node(client, api=api, store=store)
    team = build_team(researcher_node)

    print(f"\n[2] Reviewing {ticket.ticket_id} (escalated at intake: {reasons})")
    seed = seed_state(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id or f"GUEST-{ticket.ticket_id}",
        ticket_text=ticket.text,
        order_id=order_id,
        escalation_reasons=reasons,
        api=api,
    )
    final = await team.ainvoke(seed, {"recursion_limit": 50})  # ainvoke -- the graph has an async node now

    print("\nTRAJECTORY")
    print("----------")
    for line in final["log"]:
        print(" ", line)

    assert any("MCP" in line for line in final["log"]), "the team must actually have used the MCP tool"
    assert set(final["plan"]) <= {f["topic"] for f in final["findings"]}, "a planned topic went unresearched"
    assert final["revision_count"] <= MAX_REVISIONS

    assert any("risk_flags" == f["topic"] for f in final["findings"]) or "risk_flags" not in final["plan"]
    cited = set(re.findall(r"\[(F\d+)\]", final["draft"]))
    supported_ids = {f["id"] for f in final["findings"]}
    if final["status"] == "":
        assert final["fact_check"].get("ok") and final["review"].get("approved")
        assert cited <= supported_ids
        assert_memo_invariants(final)
        outcome = f"converged in {final['revision_count']} revision(s)"
    else:
        outcome = "escalated to a human within budget"

    print(f"\nPASS -- identical evidence over a different transport; {outcome}.")
    print("Exactly one node changed (researcher). The supervisor policy, the scopes, the")
    print("revision cap, and both critics were untouched.")
    print("\n--- escalation-review memo ---\n")
    print(final["draft"])


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
