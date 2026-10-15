"""
ShopSense Teams: the support team (CAPSTONE Milestone 6).

The capstone's M6, verbatim: "multi-agent team (Triage / Policy RAG / Order-Actions / Escalation Reviewer) + an MCP server
exposing order/refund/shipping mock APIs."

    supervisor  <-- every agent reports back here; it picks the next one by a DETERMINISTIC policy
       |
       +-- triage               reads the ticket (LLM extraction + deterministic safety floor + contact history)
       +-- policy_rag           answers from the policy corpus (hybrid RAG, cited)
       +-- order_actions        reads order / shipment / refund state, PROPOSES an action, and -- only after routing and any
       |                        human approval -- COMMITS it. Its tools can live behind MCP (teams/mcp_ops.py).
       +-- escalation_reviewer  routes (auto | approval | human_review), writes the hand-off packet (with the M6 memo team's
       |                        auditable memo when one is wired in), and checks the customer-facing reply before it ships
       +-- human_approval       the interrupt: a person approves, denies or lowers the amount

Same decisions as the M5 workflow (they are the same node functions: shopsense/workflows/ticket_resolution_graph.py), organised
as a team with the properties that make a team safer than one big agent:

  * SCOPES   Each role may write ONLY its declared state keys; `scoped()` raises PermissionError otherwise. A policy-RAG agent
             cannot mark a refund committed; the order-actions agent cannot change the route; the reviewer cannot edit facts.
  * SUPERVISOR  `supervisor_policy` is plain code reading the state. There is exactly one legal next step at any point, so a
             model has nothing to choose -- and nothing to be talked into. Language models are used where language is the job
             (triage extraction, the policy answer, the escalation memo), not to decide who moves money.
  * TERMINATION  A step budget guarantees the team ends: past it, `escalate` hands the ticket to a human.
  * SEPARATION  The agent that proposes an action (order_actions/plan) is not the one that decides whether it needs a human
             (escalation_reviewer/route), and the agent that commits it is held to the role limit by the tool itself.

Like the other modules, everything except `build_support_team` imports nothing from langgraph; tests/support_team_sim.py runs
the same agents and policy without it.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from ..observability.tracing import traced_node
from ..workflows import ticket_resolution_graph as tg
from ..workflows.ticket_resolution_graph import Deps, TicketState, make_nodes

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

MAX_STEPS = 12
AGENTS = ["triage", "policy_rag", "order_actions", "escalation_reviewer", "human_approval"]

AGENT_SCOPES: dict[str, set[str]] = {
    "triage": {"customer_id", "triage", "reasons", "escalation_codes", "status", "log"},
    "policy_rag": {"policy", "log"},
    "order_actions": {"facts", "proposal", "reasons", "action_result", "log"},
    "escalation_reviewer": {"route", "route_reasons", "escalation_codes", "holding_reply", "handoff", "reply", "status", "log"},
    "human_approval": {"human_decision", "human_notes", "approver_role", "approved_amount_inr", "log"},
    "supervisor": {"next_agent", "steps", "log"},
    "escalate": {"route", "reply", "status", "log"},
}


class TeamState(TicketState, total=False):
    """Same state as the M5 workflow (the supervisor's `next_agent` / `steps` live on TicketState so one seed serves both)."""


def scoped(role: str) -> Callable:
    allowed = AGENT_SCOPES[role]

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            out = fn(state)
            extra = set(out) - allowed
            if extra:
                raise PermissionError(f"{role} tried to write {sorted(extra)}, outside its scope {sorted(allowed)}")
            return out
        return wrapper
    return deco


# ---- the supervisor (pure) ---------------------------------------------------------------------------------------------
def supervisor_policy(state: dict) -> str:
    """The one legal next step. Reads only what earlier agents already wrote."""
    if int(state.get("steps", 0)) >= MAX_STEPS:
        return "escalate"
    if not state.get("triage"):
        return "triage"
    if state.get("reply"):
        return "done"
    route = state.get("route")
    if not route:
        hard = [r for r in state.get("reasons", []) if r not in tg.SOFT_REASONS]
        if hard and state.get("proposal") is None:
            return "escalation_reviewer"             # a hard fact at intake: no need to search policy or read the order
        if state.get("policy") is None:
            return "policy_rag"
        if state.get("proposal") is None:
            return "order_actions"
        return "escalation_reviewer"
    if route == "human_review":
        return "escalation_reviewer"
    if route == "approval" and state.get("human_decision") is None:
        return "human_approval"
    if state.get("action_result") is None:
        return "order_actions"
    return "escalation_reviewer"


def supervisor_node(state: dict) -> dict:
    nxt = supervisor_policy(state)
    steps = int(state.get("steps", 0)) + (0 if nxt in ("done", "escalate") else 1)
    return {"next_agent": nxt, "steps": steps, "log": [f"supervisor: -> {nxt} (step {steps})"]}


def escalate_node(state: dict) -> dict:
    reply = tg.holding_reply([], [])
    return {"route": "human_review", "reply": reply, "status": "escalated_to_human",
            "log": [f"escalate: team exceeded its {MAX_STEPS}-step budget; handed to a human"]}


# ---- the agents --------------------------------------------------------------------------------------------------------
def make_agents(deps: Deps) -> dict[str, Callable]:
    n = make_nodes(deps)

    @scoped("triage")
    def triage(state):
        return n["triage"](state)

    @scoped("policy_rag")
    def policy_rag(state):
        return n["policy_check"](state)

    @scoped("order_actions")
    def order_actions(state):
        if state.get("proposal") is None:                 # phase 1: read-only, propose
            return n["action_plan"](state)
        return n["commit"](state)                         # phase 2: only reached after routing (+ approval)

    @scoped("escalation_reviewer")
    def escalation_reviewer(state):
        if not state.get("route"):                        # phase 1: route
            routed = n["route"](state)
            if routed["route"] != "human_review":
                return routed
            merged = {**state, **{k: v for k, v in routed.items() if k != "log"}}
            handed = n["human_review"](merged)            # hand-off packet (+ memo) and the holding reply
            return {**handed, "log": routed["log"] + handed["log"]}
        return n["reply"](state)                          # phase 2: compose + check the customer-facing text

    @scoped("human_approval")
    def human_approval(state):
        return n["approval"](state)

    agents = {"triage": triage, "policy_rag": policy_rag, "order_actions": order_actions,
              "escalation_reviewer": escalation_reviewer, "human_approval": human_approval}
    # One span per agent turn (M7); the node spans from make_nodes nest inside it, so the trace reads
    # ticket.run > agent:order_actions > node:action_plan > order_system.read.
    return {role: traced_node(f"agent:{role}", kind="agent")(fn) for role, fn in agents.items()}


def make_memo_runner(escalation_team: Any, api: Any) -> Callable[[dict], dict]:
    """Plug the five-agent memo team into the reviewer's hand-off: packet -> {"status", "memo"}."""
    from .escalation_review_team import seed_state as memo_seed

    def run_memo(packet: dict) -> dict:
        seed = memo_seed(ticket_id=packet["ticket_id"], customer_id=packet["customer_id"], ticket_text=packet["ticket_text"],
                         order_id=packet.get("order_id"), escalation_reasons=packet["reasons"], api=api)
        out = escalation_team.invoke(seed, {"recursion_limit": 50})
        return {"status": out["status"] or "converged", "memo": out["draft"] if out["status"] == "" else ""}

    return run_memo


# ---- the compiled graph -------------------------------------------------------------------------------------------------
def build_support_team(deps: Deps, checkpointer: Any = None):
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed. Run: pip install langgraph langgraph-checkpoint-sqlite  (see README.md)")
    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("escalate", escalate_node)
    for name, fn in make_agents(deps).items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    routes = {name: name for name in AGENTS} | {"done": END, "escalate": "escalate"}
    g.add_conditional_edges("supervisor", lambda s: s["next_agent"], routes)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# The run/resume helpers are the M5 ones: same thread ids, same interrupt payload, same outcome shape.
seed_state = tg.seed_state
run_ticket = tg.run_ticket
resume_ticket = tg.resume_ticket
outcome_of = tg.outcome_of
build_sqlite_checkpointer = tg.build_sqlite_checkpointer
thread_config = tg.thread_config
