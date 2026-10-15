"""
A tiny pure-Python executor for the Milestone 6 team, used by the test
suite and dev_smoke_check.py to run the WHOLE supervisor loop (supervisor ->
specialist -> supervisor ... -> done/escalate) with the real node functions
and a scripted model, without `langgraph` installed.

It mirrors what build_team() wires: the star topology, the `findings`/`log`
accumulate-reducers, everything else overwritten. It does NOT prove the
LangGraph wiring (only a machine with langgraph can) -- it proves the node
functions, scopes, supervisor policy, citation check and revision cap
cooperate end to end, which is where the logic bugs would be.
"""
from __future__ import annotations

import json
import re

from shopsense.teams.escalation_review_team import escalate_node, supervisor_node

_ACCUMULATE = {"findings", "log"}


def merge(state: dict, update: dict) -> dict:
    out = dict(state)
    for k, v in update.items():
        out[k] = (state.get(k, []) + v) if k in _ACCUMULATE else v
    return out


def run_team(state: dict, nodes: dict, *, max_steps: int = 60) -> dict:
    """nodes: {"planner","researcher","writer","fact_checker","reviewer"} -> callables."""
    for _ in range(max_steps):
        state = merge(state, supervisor_node(state))
        nxt = state["next_agent"]
        if nxt == "done":
            return state
        if nxt == "escalate":
            return merge(state, escalate_node(state))
        state = merge(state, nodes[nxt](state))
    raise RuntimeError("team simulation did not terminate -- the revision cap is broken")


def make_responder(*, writer_memos: list[dict], seen_writer_prompts: list[str] | None = None):
    """A keyed fake model. `writer_memos` are returned in order (the last
    one repeats), so a test can script 'first draft bad, second good'."""
    memos = list(writer_memos)

    def respond(messages: list[dict]) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        if "supervisor of an escalation-review team" in system:
            legal = re.search(r"Legal next steps: \[(.*?)\]", user).group(1).replace("'", "")
            return json.dumps({"next_agent": legal.split(",")[0].strip()})
        if "escalation-review planner" in system:
            return json.dumps({"topics": ["policy_basis"]})
        if "drafting the body of an escalation-review memo" in system:
            if seen_writer_prompts is not None:
                seen_writer_prompts.append(user)
            return json.dumps(memos.pop(0) if len(memos) > 1 else memos[0])
        if "independent fact-checker" in system:
            return json.dumps({"ok": True, "note": ""})
        if "final reviewer" in system:
            return json.dumps({"approved": True, "notes": ""})
        return "Refund policy applies here [0]."  # M4 grounded answer

    return respond
