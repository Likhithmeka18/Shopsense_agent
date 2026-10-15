"""
ShopSense runtime: assemble the ticket-handling system (capstone M5 workflow or M6 team) from its parts, once, so the pipeline,
the demos and the approver's resume script all build it the SAME way. Resuming a paused ticket in a new process only works if
that process builds the same graph topology the first one paused in, which is why this lives in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .core.contacts import ContactLedger
from .memory.memory_store import MemoryStore
from .observability.report import DEFAULT_TRACE_FILE
from .tools.action_tools import ActionLedger, ActionTools
from .tools.order_tools import MockAPI
from .tools.policy_tools import PolicyKnowledgeBase
from .workflows import ticket_resolution_graph as tg

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "shopsense" / "workflow_state"
TICKET_DB = STATE_DIR / "ticket_reviews.db"
ACTION_LEDGER = STATE_DIR / "actions.jsonl"
TRACE_FILE = DEFAULT_TRACE_FILE
TOPOLOGIES = ("team", "workflow")


def build_deps(*, kb: Any = None, store: Any = None, ledger_path: Optional[Path] = None, contacts: Optional[ContactLedger] = None,
               with_memo_team: bool = True, mcp_toolset: Any = None, mcp_secret: Optional[str] = None, extract: Any = None,
               api: Any = None, order_guard: Any = None) -> tg.Deps:
    """The dependencies the ticket graph runs on. `ledger_path=None` keeps the action ledger in memory (demos); give a path to
    make committed actions survive restarts (pipeline, approver). `mcp_toolset` moves the Order-Actions tools behind MCP."""
    api = api or MockAPI()
    kb = kb or PolicyKnowledgeBase()
    store = store or MemoryStore()
    kwargs: dict = {"actions": ActionTools(api, ActionLedger(ledger_path))}
    if mcp_toolset is not None:
        from .teams.mcp_ops import mcp_backed
        kwargs.update(mcp_backed(mcp_toolset, secret=mcp_secret))
    if extract is not None:
        kwargs["extract"] = extract
    if order_guard is not None:
        kwargs["order_guard"] = order_guard
    if with_memo_team:
        from .teams import escalation_review_team as ert
        from .teams.support_team import make_memo_runner
        if ert.LANGGRAPH_AVAILABLE:
            kwargs["memo_runner"] = make_memo_runner(ert.build_team(ert.make_local_researcher_node(kb, api, store)), api)
    return tg.Deps(api=api, kb=kb, store=store, ledger=contacts or ContactLedger(), **kwargs)


def build_graph(deps: tg.Deps, *, topology: str = "team", db_path: Optional[Path] = None):
    """Compile the graph with a durable SQLite checkpoint (db_path) or an in-memory one (None)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}")
    saver = None
    if db_path is not None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        saver = tg.build_sqlite_checkpointer(str(db_path))
    if topology == "team":
        from .teams.support_team import build_support_team
        return build_support_team(deps, checkpointer=saver)
    return tg.build_ticket_graph(deps, checkpointer=saver)


def pending_node(graph: Any, ticket_id: str) -> tuple:
    """Which node a ticket's thread is paused at ('approval' in either topology), or () when it is not paused."""
    return tuple(graph.get_state(tg.thread_config(ticket_id)).next)
