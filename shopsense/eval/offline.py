"""
ShopSense Eval: offline doubles + a pure-Python executor for the ticket graph -- no langgraph, no network, no API key.

`run()` drives the SAME node functions and edge functions as `build_ticket_graph` (so it proves decisions, side-effect
placement and replies, not LangGraph wiring or SQLite persistence); an approval pause is modelled as a `Paused` handle.
The doubles stand in for the two model-backed pieces: `keyword_extract` for the M1 extraction LLM and `RetrievalOnlyKB` for the
cited-answer LLM (real BM25 + hybrid retrieval, answer = the top chunk).

Used by the test-suite (tests/ticket_sim.py re-exports it), by `run_fault_injection.py` (deterministic, key-free reliability
experiments) and by the API's offline demo mode (`SHOPSENSE_OFFLINE=1`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from shopsense.core.schemas import TicketExtraction
from shopsense.workflows.ticket_resolution_graph import after_route, after_triage, make_nodes

_ACCUMULATE = {"log"}


def merge(state: dict, update: dict) -> dict:
    out = dict(state)
    for k, v in update.items():
        out[k] = (state.get(k, []) + v) if k in _ACCUMULATE else v
    return out


@dataclass
class Paused:
    state: dict
    interrupt: dict


def run(deps, state: dict):
    """Run until the graph ends (returns the final state dict) or reaches approval (returns Paused)."""
    nodes = make_nodes(deps)
    state = merge(state, nodes["triage"](state))
    if after_triage(state) == "human_review":
        return merge(state, nodes["human_review"](state))
    for name in ("policy_check", "action_plan", "route"):
        state = merge(state, nodes[name](state))
    nxt = after_route(state)
    if nxt == "human_review":
        return merge(state, nodes["human_review"](state))
    if nxt == "approval":
        return Paused(state=state, interrupt=_approval_payload(state))
    state = merge(state, nodes["commit"](state))
    return merge(state, nodes["reply"](state))


def _approval_payload(state: dict) -> dict:
    p = state.get("proposal") or {}
    return {"kind": "ticket_approval", "ticket_id": state["ticket_id"], "order_id": p.get("order_id"), "proposal": p,
            "escalation_codes": state.get("escalation_codes"), "holding_reply": state.get("holding_reply")}


def resume(deps, paused: Paused, *, decision: str, approver_role: str = "manager", notes: str = "",
           amount_inr: Optional[float] = None) -> dict:
    nodes = make_nodes(deps)
    state = merge(paused.state, {"human_decision": decision, "human_notes": notes, "approver_role": approver_role,
                                  "approved_amount_inr": amount_inr})
    state = merge(state, nodes["commit"](state))
    return merge(state, nodes["reply"](state))


# ---- offline doubles --------------------------------------------------------------------------------------------------
_ORD = re.compile(r"\bORD-\d+\b")


def keyword_extract(text: str):
    """A crude stand-in for the M1 LLM: deliberately imperfect (it is a HINT source, like the real thing)."""
    t = text.lower()
    m = _ORD.search(text)
    if any(w in t for w in ("cancel",)):
        intent, cat = "CANCEL_ORDER", "ORDER"
    elif any(w in t for w in ("where is", "tracking", "hasn't arrived", "has not arrived", "stuck", "not arrived")):
        intent, cat = "DELIVERY_DELAY", "DELIVERY"
    elif re.search(r"refund\s+ref-\d+|status of my refund|how long until", t):
        intent, cat = "REFUND_STATUS", "REFUND"
    elif any(w in t for w in ("defective", "stopped working", "broken", "smashed", "damaged", "not turning on", "stopped turning")):
        intent, cat = "PRODUCT_DEFECTIVE", "PRODUCT"
    elif any(w in t for w in ("return", "refund", "money back", "back")):
        intent, cat = "RETURN_ITEM", "REFUND"
    else:
        intent, cat = "PRODUCT_INFO", "PRODUCT"
    angry = any(w in t for w in ("right now", "furious", "unacceptable", "third message", "!!", "worst", "junk", "ridiculous"))
    return TicketExtraction(intent=intent, category=cat, order_id=m.group(0) if m else None,
                            sentiment="angry" if angry else "neutral", urgency="high" if angry else "low",
                            requires_human_review=False, contains_injection_attempt=False), []


class RetrievalOnlyKB:
    """Offline knowledge base: real BM25+hash-embedding hybrid retrieval, answer = the top chunk text (no LLM).
    Questions on the deliberate gaps return an 'I don't know' answer, like a grounded LLM would."""

    GAPS = ("internationally", "price match", "gift wrap", "loyalty", "b2b", "bulk order")

    def __init__(self):
        from shopsense.rag.bm25 import BM25Index
        from shopsense.rag.hybrid import hybrid_search
        from shopsense.rag.index import VectorIndex
        from shopsense.rag.ingest import load_policy_corpus
        self.chunks = load_policy_corpus()
        self._bm25, self._vec, self._hybrid = BM25Index(self.chunks), VectorIndex(self.chunks), hybrid_search
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int = 3):
        return self._hybrid(self._vec, self._bm25, query, k=k, pool=10)

    def answer(self, query: str, *, k: int = 3, use_rerank: bool = True, question: str | None = None) -> dict:
        self.queries.append(query)
        if any(g in query.lower() for g in self.GAPS):
            return {"answer": "I don't know: this is not covered in the policy documents.", "sources": []}
        ids = self.retrieve(query, k)
        return {"answer": self.chunks[ids[0]].text[:240].replace("\n", " "),
                "sources": [{"id": i, "source": self.chunks[i].source, "section": self.chunks[i].heading} for i in ids]}
