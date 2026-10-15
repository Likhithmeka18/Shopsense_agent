"""
ShopSense Teams: the MCP-backed researcher node (Milestone 6, part 2).

Swaps escalation_review_team.py's LOCAL researcher (a direct in-process
PolicyKnowledgeBase call) for one that retrieves the same 'policy_basis'
finding over a REAL Model Context Protocol server
(shopsense/mcp/policy_server.py), spawned as a child process and talked to
over stdio via `langchain_mcp_adapters.client.MultiServerMCPClient`.

The design goal: swapping
this ONE node changes nothing else in the team -- same state, same scopes,
same supervisor policy, same loop-back cap, same two independent critics.
If swapping a data source forced a change anywhere else, the layering was
wrong.

Needs `mcp` and `langchain-mcp-adapters` (both in requirements.txt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    MCP_CLIENT_AVAILABLE = True
except ImportError:
    # Same reasoning as escalation_review_team.py's langgraph guard: this
    # import failing must not break importing the rest of the module, so
    # dev_smoke_check.py / tests can still exercise parse_tool_result() and
    # the other pure helpers below without the real package installed.
    MCP_CLIENT_AVAILABLE = False

from .escalation_review_team import context_for, safe_ticket_text, scoped, topic_finding_text

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_PATH = str(_REPO_ROOT / "shopsense" / "mcp" / "policy_server.py")


def parse_tool_result(raw: Any) -> Any:
    """MCP tool results arrive as CONTENT BLOCKS, not plain Python values
    -- a tool returning list[dict] comes back as
    [{'type': 'text', 'text': '<json>'}], because the protocol carries
    content, not Python objects. Different adapter/SDK versions also hand
    back a bare string or an already-parsed structure, so this normalizes
    all three. Pure function -- no network, no langgraph -- so it's
    testable on its own."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        raw = "".join(b.get("text", "") for b in raw if b.get("type") == "text")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _require_mcp_client() -> None:
    if not MCP_CLIENT_AVAILABLE:
        raise RuntimeError(
            "langchain-mcp-adapters is not installed. Run:\n"
            "    pip install 'mcp<2' langchain-mcp-adapters\n"
            "See README.md."
        )


def build_mcp_client() -> "MultiServerMCPClient":
    """One MCP client, spawning shopsense/mcp/policy_server.py as a child
    process over stdio. `command=sys.executable` matters: a bare "python"
    resolves against the CHILD process's PATH, which can be a different
    interpreter with none of this project's packages installed -- the
    failure then looks like a protocol error, not an import error."""
    _require_mcp_client()
    return MultiServerMCPClient({
        "shopsense_policy": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [_SERVER_PATH],
        }
    })


def make_mcp_researcher_node(
    client: "MultiServerMCPClient",
    *,
    api: Any = None,
    store: Any = None,
) -> Callable:
    """Build the async, MCP-backed researcher node. Only `policy_basis`
    crosses the MCP boundary; every other topic uses the SAME
    topic_finding_text() the local researcher uses (pass `api` and `store`
    to enable them), so the two researchers cannot drift apart.
    `client.get_tools()` is awaited fresh on every call (fine for a handful
    of calls per review -- a persistent `client.session(...)` is worth it
    only for many more calls)."""
    _require_mcp_client()

    @scoped("researcher")
    async def mcp_researcher_node(state: dict) -> dict:
        ctx = context_for("researcher", state)
        already = {f["topic"] for f in ctx["findings"]}
        todo = [t for t in ctx["plan"] if t not in already]
        if not todo:
            return {"findings": [], "log": ["researcher[MCP]: nothing left to research"]}

        topic = todo[0]
        finding_id = f"F{len(ctx['findings']) + 1}"

        if topic == "policy_basis":
            # THE one changed line vs. the local researcher: retrieval
            # crosses a process boundary (JSON-RPC over stdio) instead of
            # an in-process function call.
            tools = await client.get_tools()
            by_name = {t.name: t for t in tools}
            raw = await by_name["search_policy"].ainvoke({"query": safe_ticket_text(ctx)[:200], "k": 3})
            hits = parse_tool_result(raw)
            text = " | ".join(h.get("text", "")[:200] for h in hits) if hits else "no policy match found"
            source = "MCP"
        else:
            text = topic_finding_text(topic, ctx, api=api, store=store) or f"no data available for topic '{topic}'"
            source = "local"

        findings = [{"topic": topic, "id": finding_id, "text": text}]
        return {"findings": findings, "log": [f"researcher[{source}]: retrieved 1 record for '{topic}'"]}

    return mcp_researcher_node
