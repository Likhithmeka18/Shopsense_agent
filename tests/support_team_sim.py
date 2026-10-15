"""
Pure-Python executor for the capstone-M6 support team (shopsense/teams/support_team.py): the SAME agents, scopes and
supervisor policy, driven by a loop instead of LangGraph. An approval pause returns a Paused handle exactly like
tests/ticket_sim.py, so one scenario can be run through both executors and must give the same answer.
"""
from __future__ import annotations

from ticket_sim import Paused, merge
from shopsense.teams.support_team import make_agents, supervisor_node


def run_team(deps, state: dict, *, max_loops: int = 40):
    agents = make_agents(deps)
    for _ in range(max_loops):
        state = merge(state, supervisor_node(state))
        nxt = state["next_agent"]
        if nxt == "done":
            return state
        if nxt == "escalate":
            from shopsense.teams.support_team import escalate_node
            return merge(state, escalate_node(state))
        if nxt == "human_approval" and state.get("human_decision") is None:
            p = state.get("proposal") or {}
            return Paused(state=state, interrupt={"kind": "ticket_approval", "ticket_id": state["ticket_id"], "order_id": p.get("order_id"),
                                                   "proposal": p, "escalation_codes": state.get("escalation_codes"),
                                                   "holding_reply": state.get("holding_reply")})
        state = merge(state, agents[nxt](state))
    raise RuntimeError("team simulation did not terminate")


def resume_team(deps, paused: Paused, *, decision: str, approver_role: str = "manager", notes: str = "", amount_inr=None):
    state = merge(paused.state, {"human_decision": decision, "human_notes": notes, "approver_role": approver_role,
                                  "approved_amount_inr": amount_inr, "log": ["human_approval: decision recorded"]})
    return run_team(deps, state)
