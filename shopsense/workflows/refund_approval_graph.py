"""
ShopSense Workflows: refund-approval LangGraph workflow (Milestone 5).

Maps Milestone 5's requirement -- "an orchestrated LangGraph workflow with
checkpointing" -- onto ShopSense's own domain: `tools/order_tools.py`'s
`calculate_refund_amount` already classifies a refund into an approval tier
(auto / manager_review / executive_review), and
`data/shopsense/policies/escalation_sop.md`'s ESC-1 table already assigns
those tiers escalation codes ESC-L1 (manager, INR 2,000-9,999) and ESC-L2
(executive, INR 10,000+), routed to a "Team Lead queue" / "Finance
Executive queue". Today nothing in the codebase actually represents that
human approval step -- M2's ShopSenseAgent just tells the customer it needs
sign-off. This file is that missing step, built as a real, durable,
resumable LangGraph workflow instead of a TODO comment:

    draft a recommendation (LLM)
        |
        v
    checks_node (deterministic critic -- never trust the model's own claim
         that its draft is compliant)
        |
        +-- issues, under the revision budget --> loop back to draft
        +-- issues, budget exhausted -----------> escalate_unresolved -> END
        +-- clean -------------------------------> human_approval_node
                                                        |
                                            interrupt() -- PAUSES here,
                                            durably, via a checkpointer.
                                            A real human (a team lead or
                                            finance exec) approves/denies/
                                            modifies, out of process, any
                                            time later -- even after a
                                            restart.
                                                        |
                                                        v
                                                  finalize_node -- the ONLY
                                            node that performs an
                                            irreversible action (recording
                                            the decision). It lives
                                            downstream of the pause on
                                            purpose: a node re-runs from the
                                            top on every resume, so any side
                                            effect placed *in*
                                            human_approval_node would fire
                                            once per resume attempt, not once.
                                                        |
                                                        v
                                                       END

Every decision function below (`check_recommendation`, `route_after_checks`,
`build_finalize_record`, `escalation_code_for_tier`) is a plain function
with no LangGraph import, so it is fully unit-testable (see
tests/test_workflows.py and dev_smoke_check.py) without `langgraph`
installed. Only building and running the actual graph needs the real
package -- the tests cover the pure functions without it.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Annotated, Any, Literal, Optional, TypedDict

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from ..core.safety_net import is_abuse_flagged
from ..tools.order_tools import MockAPI, refund_tier_for_amount

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # If `langgraph`/`langgraph-checkpoint-sqlite` are not installed,
    # every pure decision function in this file still imports and runs
    # fine -- only build_refund_approval_graph()/build_sqlite_checkpointer()
    # need the real package, and they say so clearly if it's missing,
    # rather than failing this whole module's import (which would also
    # break dev_smoke_check.py's offline coverage of the decision logic).
    LANGGRAPH_AVAILABLE = False


def _operator_add(a: list, b: list) -> list:
    return a + b


try:
    import operator

    _REDUCER_ADD = operator.add
except ImportError:  # pragma: no cover -- operator is stdlib, always available
    _REDUCER_ADD = _operator_add


class RefundApprovalState(TypedDict):
    """Milestone 5's state schema, with the control/audit split that
    prevents a common class of bugs: `issues`
    has NO reducer (each pass's checks_node must overwrite it, or a fixed
    problem would never clear and the loop would never exit); `issue_log`
    accumulates (Annotated + operator.add) because it's a read-only audit
    trail, not something the graph branches on."""

    ticket_id: str
    customer_id: str
    order_id: str
    refund_amount_inr: float
    approval_tier: Literal["manager_review", "executive_review"]
    escalation_code: Literal["ESC-L1", "ESC-L2"]
    case_facts: dict                                      # control -- real record facts the drafter may use
    recommendation: str                                   # control -- overwritten every draft
    issues: list[str]                                     # control -- overwritten every check
    issue_log: Annotated[list[str], _REDUCER_ADD]          # audit -- accumulates
    revision_count: int
    human_decision: Optional[str]                         # "approve" | "deny" | "modify"
    human_notes: Optional[str]
    final_outcome: str
    status: str                                           # "" while running


MAX_REVISIONS = 2

# ESC-1 table in escalation_sop.md, restated here as the single source of
# truth this module checks drafts against (never re-typed by the LLM).
ESCALATION_CODE_BY_TIER = {"manager_review": "ESC-L1", "executive_review": "ESC-L2"}


UNSUPPORTED_HISTORY_PHRASES = ("chargeback", "no prior", "prior partial", "previous refund", "no previous", "fraud")


def escalation_code_for_tier(approval_tier: str) -> str:
    """ESC-1 lookup: tier -> escalation code. Defaults to the stricter
    ESC-L2 for any tier this module doesn't recognize, rather than silently
    under-escalating an unfamiliar value."""
    return ESCALATION_CODE_BY_TIER.get(approval_tier, "ESC-L2")


def check_recommendation(state: RefundApprovalState) -> list[str]:
    """The deterministic critic behind checks_node. A drafted recommendation
    is compliant only if a human reading it alone -- with no other context
    -- can see the order, the amount, and the escalation code without
    having to trust the model got them right. Returns the list of problems;
    empty means clean. This is pure: no model call, no I/O, so it is
    trivially unit-testable."""
    rec = (state.get("recommendation") or "").strip()
    issues: list[str] = []
    if not rec:
        return ["recommendation is empty"]

    code = state["escalation_code"]
    if code not in rec:
        issues.append(f"missing required escalation code {code}")

    amount = state["refund_amount_inr"]
    amount_variants = {f"{amount:,.0f}", f"{amount:.0f}", f"{amount:,.2f}", f"{amount:.2f}", str(int(amount))}
    if not any(v in rec for v in amount_variants):
        issues.append("recommendation doesn't state the refund amount")

    if state["order_id"] not in rec:
        issues.append(f"recommendation doesn't reference order {state['order_id']}")

    # Heuristic floor against invented history. A draft can say "No
    # prior partial refunds or chargebacks have been recorded" -- a fact
    # nothing in the state supports. Flag phrases that assert history unless
    # the record facts mention them. Not a proof; the real fix is the prompt
    # only being given facts it may use.
    facts_text = json.dumps(state.get("case_facts") or {}).lower()
    for phrase in UNSUPPORTED_HISTORY_PHRASES:
        if phrase in rec.lower() and phrase not in facts_text:
            issues.append(f"states unsupported history ('{phrase}') -- only use the facts provided")

    return issues


def route_after_checks(
    state: RefundApprovalState, *, max_revisions: int = MAX_REVISIONS
) -> Literal["revise", "escalate_unresolved", "human_approval"]:
    """The pure router behind the conditional edge out of checks_node.
    'Decide in a node, route in an edge' -- this function only reads
    `issues`/`revision_count`, which the node already wrote; it never
    calls a model itself, so the decision stays inspectable and testable
    in isolation, and is exactly what a loop-cap bug (an unbounded
    revise<->checks cycle) would show up in first."""
    if not state["issues"]:
        return "human_approval"
    if state["revision_count"] >= max_revisions:
        return "escalate_unresolved"
    return "revise"


def build_finalize_record(state: RefundApprovalState) -> dict:
    """The exact shape finalize_node persists. Pure, so the "what gets
    written down" contract is testable without a MemoryStore, a model, or
    LangGraph itself."""
    return {
        "ticket_id": state["ticket_id"],
        "order_id": state["order_id"],
        "customer_id": state["customer_id"],
        "escalation_code": state["escalation_code"],
        "refund_amount_inr": state["refund_amount_inr"],
        "decision": state.get("human_decision"),
        "notes": state.get("human_notes") or "",
        "recommendation": state.get("recommendation", ""),
        "status": state.get("status", ""),
    }


_ORDER_ID_RE = re.compile(r"\bORD-\d+\b")


def find_order_id_in_text(text: str) -> Optional[str]:
    m = _ORDER_ID_RE.search(text or "")
    return m.group(0) if m else None


def decide_refund_review(
    api: MockAPI,
    *,
    customer_id: Optional[str],
    order_id: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """Should this ticket start a Milestone 5 approval workflow? Decided
    from DATA, not from whether the model happened to call a tool: a
    workflow opens only for a refund request that is actually PENDING
    (on file, not yet approved), belongs to the ticket's customer, sits in
    the manager/executive tier, and is not on a cancelled order or an
    abuse-flagged account (those go to a human via Milestone 6 / ESC-ABUSE,
    not to this workflow). Returns {"start": bool, "reason": str} plus the
    refund details when start is True.

    If the ticket names no order, a REFUND-category ticket may still match
    the customer's single pending refund; anything else does not."""
    candidates: list[str] = []
    if order_id:
        candidates = [order_id]
    elif category == "REFUND" and customer_id:
        candidates = [
            oid for oid, r in api.refunds.items()
            if r["customer_id"] == customer_id and r["approved"] not in ("True", True)
        ]
        if len(candidates) != 1:
            return {"start": False, "reason": "no single pending refund identifiable for this customer"}
    if not candidates:
        return {"start": False, "reason": "ticket references no order"}

    oid = candidates[0]
    order = api.orders.get(oid)
    if order is None:
        return {"start": False, "reason": f"order {oid} not found"}
    refund = api.refunds.get(oid)
    if refund is None:
        return {"start": False, "reason": f"no refund request on file for {oid}"}
    if customer_id and not customer_id.startswith("GUEST") and refund["customer_id"] != customer_id:
        return {"start": False, "reason": f"refund {refund['refund_id']} belongs to a different customer"}
    if refund["approved"] in ("True", True):
        return {"start": False, "reason": f"refund {refund['refund_id']} is already approved -- nothing pending"}
    if order["status"] == "cancelled":
        return {"start": False, "reason": f"order {oid} is cancelled -- a cancellation, not a refund approval"}
    if is_abuse_flagged(api.customers.get(refund["customer_id"])):
        return {"start": False, "reason": "abuse-flagged account -- human review via ESC-ABUSE, not this workflow"}
    amount = float(refund["amount_inr"])
    tier = refund_tier_for_amount(amount)
    if tier == "auto":
        return {"start": False, "reason": f"INR {amount:,.0f} is below the approval threshold"}
    return {
        "start": True,
        "reason": f"pending {tier} refund {refund['refund_id']}",
        "order_id": oid,
        "refund_id": refund["refund_id"],
        "refund_amount_inr": amount,
        "approval_tier": tier,
        "customer_id": refund["customer_id"],
        "case_facts": {
            "refund_id": refund["refund_id"],
            "refund_reason_on_file": refund["reason"],
            "refund_requested_date": refund["requested_date"],
            "order_status": order["status"],
            "order_total_inr": order["total_amount_inr"],
            "refund_covers_full_order_total": amount == float(order["total_amount_inr"]),
        },
    }


def find_pending_refunds(api: MockAPI) -> list[dict]:
    """Every pending refund that WOULD start a workflow -- used by the demo
    to pick a real case instead of scanning tool outputs."""
    out = []
    for oid, r in api.refunds.items():
        d = decide_refund_review(api, customer_id=r["customer_id"], order_id=oid)
        if d["start"]:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Node functions. These import nothing from langgraph except the one
# interrupt()/Command() call each needs -- the module-level try/except above
# already guards that.
# ---------------------------------------------------------------------------

DRAFT_SYSTEM_PROMPT = (
    "You are a ShopSense refund-approval drafting assistant. A refund has "
    "been flagged for mandatory human sign-off per escalation_sop.md. Write "
    "a short, factual recommendation (3-5 sentences) for the human "
    "approver. You MUST explicitly state, in plain text: the escalation "
    "code, the order id, and the refund amount in INR. Use ONLY the facts "
    "listed in the user message. If something is not listed (the customer's "
    "history, earlier refunds, chargebacks, what the customer said, evidence), "
    "you do not know it: do not state it, and do not say it is absent. "
    "Do not approve or deny anything yourself -- a human makes that call; "
    "you are only drafting the summary they will read before deciding."
)


def _draft_recommendation_node(state: RefundApprovalState) -> dict:
    user = (
        f"Order: {state['order_id']}\n"
        f"Customer: {state['customer_id']}\n"
        f"Refund amount: INR {state['refund_amount_inr']:.0f}\n"
        f"Approval tier: {state['approval_tier']} (escalation code: {state['escalation_code']})\n"
    )
    for key, value in (state.get("case_facts") or {}).items():
        user += f"{key.replace('_', ' ').capitalize()}: {value}\n"
    problems = state.get("issues") or []
    if problems:
        user += f"\nYour previous draft had these problems -- fix all of them: {problems}"

    resp = llm_completion(
        messages=[{"role": "system", "content": DRAFT_SYSTEM_PROMPT}, {"role": "user", "content": user}],
        model=DEFAULT_MODEL,
    )
    text = resp.choices[0].message.content or ""
    had_previous_draft = bool(state.get("recommendation"))
    return {
        "recommendation": text,
        "revision_count": state["revision_count"] + (1 if had_previous_draft else 0),
    }


def _checks_node(state: RefundApprovalState) -> dict:
    issues = check_recommendation(state)
    return {
        "issues": issues,
        "issue_log": [f"revision {state['revision_count']}: {issues if issues else 'clean'}"],
    }


def _route_after_checks_edge(state: RefundApprovalState) -> str:
    return route_after_checks(state)


def _human_approval_node(state: RefundApprovalState) -> dict:
    # Nothing but read state and pause. See the module docstring: any
    # side effect placed here (instead of in finalize_node) would re-fire
    # every time this node re-runs on resume.
    decision = interrupt({
        "ticket_id": state["ticket_id"],
        "order_id": state["order_id"],
        "customer_id": state["customer_id"],
        "escalation_code": state["escalation_code"],
        "refund_amount_inr": state["refund_amount_inr"],
        "recommendation": state["recommendation"],
        "ask": "Reply with a decision of 'approve', 'deny', or 'modify', plus optional notes.",
    })
    return {"human_decision": decision.get("decision"), "human_notes": decision.get("notes", "")}


def make_finalize_node(store: Any = None):
    """Build the finalize node, optionally bound to a MemoryStore. This is
    the ONLY node with side effects, and it sits downstream of interrupt()
    so it runs exactly once per review. With a store, it records the human's
    decision as episodic memory for the customer -- so the next ticket from
    that customer can recall it. (Previously the demo scripts wrote that
    memory themselves AFTER the graph returned, which meant the pipeline
    path -- and any resume from a fresh process -- silently skipped it.)"""

    def _finalize_node(state: RefundApprovalState) -> dict:
        status = state.get("human_decision") or "unknown"
        record = build_finalize_record({**state, "status": status})
        if store is not None:
            store.write(
                state["customer_id"],
                "episodic",
                f"Refund {state['order_id']} (INR {state['refund_amount_inr']:.0f}, {state['escalation_code']}) "
                f"human decision: {status}. {state.get('human_notes') or ''}".strip(),
            )
        return {"final_outcome": json.dumps(record), "status": status}

    return _finalize_node


_finalize_node = make_finalize_node(None)  # store-less default (tests, and graphs built without a store)


def _escalate_unresolved_node(state: RefundApprovalState) -> dict:
    # The AI couldn't produce a compliant draft within budget -- a human
    # starts this one from scratch, with no AI-drafted recommendation handed
    # to them (handing over a draft that failed its own compliance check
    # would be worse than handing over nothing).
    record = build_finalize_record({**state, "status": "escalated_no_ai_draft"})
    return {"status": "escalated_no_ai_draft", "final_outcome": json.dumps(record)}


def _require_langgraph() -> None:
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "langgraph is not installed in this environment. Run:\n"
            "    pip install langgraph langgraph-checkpoint-sqlite\n"
            "This module's decision logic (check_recommendation, "
            "route_after_checks, build_finalize_record) works without it "
            "and is covered by dev_smoke_check.py / tests/test_workflows.py "
            "-- only the actual graph wiring needs the real package. See "
            "README.md."
        )


def build_refund_approval_graph(checkpointer: Any = None, store: Any = None):
    """Compile the Milestone 5 graph. Pass a checkpointer from
    build_sqlite_checkpointer() for durability across restarts, or leave it
    None for an InMemorySaver (dies with the process -- fine for a quick
    smoke run, not for the real human-approval use case). Pass `store` (a
    MemoryStore) so finalize records the decision into the customer's memory."""
    _require_langgraph()
    if checkpointer is None:
        checkpointer = InMemorySaver()

    g = StateGraph(RefundApprovalState)
    g.add_node("draft", _draft_recommendation_node)
    g.add_node("checks", _checks_node)
    g.add_node("human_approval", _human_approval_node)
    g.add_node("finalize", make_finalize_node(store))
    g.add_node("escalate_unresolved", _escalate_unresolved_node)

    g.add_edge(START, "draft")
    g.add_edge("draft", "checks")
    g.add_conditional_edges(
        "checks",
        _route_after_checks_edge,
        {"revise": "draft", "escalate_unresolved": "escalate_unresolved", "human_approval": "human_approval"},
    )
    g.add_edge("human_approval", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("escalate_unresolved", END)
    return g.compile(checkpointer=checkpointer)


def build_sqlite_checkpointer(db_path: str):
    """A durable checkpointer backed by a real file -- survives a process
    restart, unlike InMemorySaver. `check_same_thread=False` is required by
    SqliteSaver's connection handling."""
    _require_langgraph()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def seed_state(
    *, ticket_id: str, customer_id: str, order_id: str, refund_amount_inr: float, approval_tier: str,
    case_facts: Optional[dict] = None,
) -> RefundApprovalState:
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "refund_amount_inr": refund_amount_inr,
        "approval_tier": approval_tier,  # type: ignore[typeddict-item]
        "escalation_code": escalation_code_for_tier(approval_tier),  # type: ignore[typeddict-item]
        "case_facts": case_facts or {},
        "recommendation": "",
        "issues": [],
        "issue_log": [],
        "revision_count": 0,
        "human_decision": None,
        "human_notes": None,
        "final_outcome": "",
        "status": "",
    }


def thread_config(ticket_id: str) -> dict:
    """The thread_id is the review's persistent identity across the pause.
    Reusing the same config dict against a checkpointer pointed at the same
    db file is what makes resume-after-restart work at all."""
    return {"configurable": {"thread_id": f"refund-review-{ticket_id}"}}


def start_refund_review(
    graph: Any, *, ticket_id: str, customer_id: str, order_id: str, refund_amount_inr: float, approval_tier: str,
    case_facts: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Run the graph up to its first pause (or straight to escalation, if
    the AI never produces a compliant draft within budget). Returns
    (result, config); result carries an '__interrupt__' key when paused."""
    seed = seed_state(
        ticket_id=ticket_id,
        customer_id=customer_id,
        order_id=order_id,
        refund_amount_inr=refund_amount_inr,
        approval_tier=approval_tier,
        case_facts=case_facts,
    )
    config = thread_config(ticket_id)
    result = graph.invoke(seed, config)
    return result, config


def resume_refund_review(graph: Any, config: dict, *, decision: str, notes: str = "") -> dict:
    """Resume a paused review -- from this process or, with a checkpointer
    reopened against the same db file in a brand-new process, after a real
    restart. `decision` becomes interrupt()'s return value inside
    _human_approval_node."""
    _require_langgraph()
    return graph.invoke(Command(resume={"decision": decision, "notes": notes}), config)
