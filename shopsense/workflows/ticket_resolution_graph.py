"""
ShopSense Workflows: the ticket-resolution graph (CAPSTONE Milestone 5).

The capstone's M5, verbatim: "LangGraph workflow: triage -> policy-check -> action execution -> conditional routing
(auto-resolve low-risk, human-approval interrupt for high refunds or angry-customer sentiment), with checkpointing."

    START
      |
    triage ---------------- hard escalation (legal / safety / injection / abuse flag / 3rd contact /
      |                      lost shipment / owner mismatch / extraction failure) ---------+
      v                                                                                    |
    policy_check   RAG over the policy corpus (category addenda included)                  |
      |                                                                                    |
    action_plan    READ-ONLY tool calls (order, shipment, refund status) + deterministic   |
      |            rules -> a PROPOSAL (refund / replace / cancel / warranty / decline /   |
      |            inform ...). Nothing has been changed yet.                              |
    route          decide the route, in a node, from data                                  |
      |                                                                                    |
      +--- auto ------------------------------------+                                      |
      |                                             |                                      |
      +--- approval: refund tier >= manager, or     |                                      |
      |    angry sentiment                          |                                      |
      |      approval --interrupt()--> (human) -----+                                      |
      v                                             v                                      |
    commit  <-- the ONLY node that changes the world, via tools/action_tools.py            |
      |         (which enforces role limits itself)                                        |
    reply   compose + deterministic check of the customer-facing text                      |
      |                                                                                    |
     END                                          human_review <---------------------------+
                                                   (hand-off packet, holding reply,
                                                    optional M6 team memo) --> END

Design decisions worth knowing (each is also tested):
  * LLMs are used where language is the job (ticket extraction, policy answer). Everything that decides who may move
    money -- tiers, windows, lost shipments, roles -- is plain code reading the tables and tools/policy_rules.py.
  * The model's labels are HINTS. A question ("how many days do I have to return them?") never opens a refund even if
    the intent label says RETURN_ITEM; a defect described in words is treated as a defect whatever the label says.
  * Side effects live only in `commit`, downstream of `interrupt()`, because a node re-runs from its top on resume.
  * "Angry sentiment" routes to the approval interrupt (the capstone wording), where the human sees the proposed reply;
    per refund_authorisation_matrix RA-2.2 anger raises PRIORITY and attention but never lowers anyone's required
    authority, which is why it only adds a human, never removes a limit.
  * Replies are assembled from structured facts and checked by `check_reply`; the only model-written text in a reply is
    the cited policy answer, which is checked too. No reply ever discloses an account flag.

Like the other workflow modules, every decision function imports nothing from langgraph, so it is fully testable
offline (tests/ticket_sim.py runs the same node functions without langgraph); only build_ticket_graph needs the package.
"""
from __future__ import annotations

import json
import operator
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Optional, TypedDict

from ..core.contacts import ContactLedger
from ..core.llm_client import DEFAULT_MODEL
from ..core.safety_net import combine_escalation_reasons, detect_injection_attempt, is_abuse_flagged, redact_injection
from ..core.ticket_intake import extract_ticket_info
from ..guardrails import APPROVAL_CLAIM_RE, FLAG_WORDS_RE, clean_model_text
from ..tools import policy_rules as pr
from ..observability.tracing import PROMPT_VERSION, add_event, get_tracer, traced_node
from ..tools.action_tools import ActionTools
from ..tools.order_tools import MockAPI, build_tools, refund_tier_for_amount
from ..tools.reliability import CircuitBreaker, DependencyGuard, DependencyUnavailable

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = ""
except ImportError as _e:                 # same graceful degradation as the other workflow modules
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

_ORDER_ID_RE = re.compile(r"\bORD-\d+\b")


def find_order_id_in_text(text: str) -> Optional[str]:
    m = _ORDER_ID_RE.search(text or "")
    return m.group(0) if m else None


# --------------------------------------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    # input
    ticket_id: str
    customer_id: str
    channel: str
    created_at: str
    text: str
    history: list                   # earlier contacts [{ticket_id, created_at, order_id, text}] (golden cases / replays)
    # triage
    triage: dict                    # intent/category/sentiment/urgency/order_id/request_type/defective/extraction_ok
    reasons: list                   # why a human is involved (control; overwritten by route)
    escalation_codes: list
    # evidence + plan
    policy: dict                    # {query, answer, sources, grounded, unknown}
    facts: dict                     # tool results
    proposal: dict                  # what we WOULD do
    # routing
    route: str                      # auto | approval | human_review
    route_reasons: list
    holding_reply: str              # what the customer can be told before a human decides
    # human decision (set by resume)
    human_decision: Optional[str]   # approve | deny | modify
    human_notes: Optional[str]
    approver_role: Optional[str]
    approved_amount_inr: Optional[float]
    # outcome
    action_result: dict
    handoff: dict                   # packet for the human queue (human_review route)
    reply: str
    status: str                     # "" while running
    next_agent: str                 # team supervisor's pick (unused by the plain M5 graph)
    steps: int
    log: Annotated[list, operator.add]   # audit trail only; nothing branches on it


MONEY_PROPOSALS = ("refund", "decide_pending_refund", "replace")
POLICY_INTENTS = {"REFUND_REQUEST", "REFUND_STATUS", "RETURN_ITEM", "PRODUCT_INFO", "PRODUCT_DEFECTIVE",
                  "WRONG_ITEM_DELIVERED", "CANCEL_ORDER", "CHANGE_ORDER", "DELIVERY_DELAY", None}
REFUND_FAMILY = {"REFUND_REQUEST", "RETURN_ITEM", "PRODUCT_DEFECTIVE", "WRONG_ITEM_DELIVERED"}
STATUS_FAMILY = {"ORDER_STATUS", "TRACK_DELIVERY", "DELIVERY_DELAY"}
CODE_BY_TIER = {"manager_review": "ESC-L1", "executive_review": "ESC-L2"}
ROLE_BY_TIER = {"manager_review": "manager", "executive_review": "executive"}
QUEUE_BY_CODE = {
    "ESC-SAFETY": "Product Safety queue", "ESC-LEGAL": "Legal queue", "ESC-ABUSE": "Trust & Safety queue",
    "ESC-LOST": "Logistics Recovery queue", "ESC-REPEAT": "Senior Support queue", "ESC-L1": "Team Lead queue",
    "ESC-L2": "Finance Executive queue",
}
CODE_PRIORITY = ["ESC-SAFETY", "ESC-LEGAL", "ESC-ABUSE", "ESC-LOST", "ESC-REPEAT", "ESC-L2", "ESC-L1"]

# Reasons that are ONLY a hint from the model; every other reason is a hard, deterministic fact.
SOFT_REASONS = {"requires_human_review"}


def codes_from_reasons(reasons: list[str]) -> list[str]:
    codes: list[str] = []
    for r in reasons:
        code = {"safety_net:ESC-LEGAL": "ESC-LEGAL", "safety_net:ESC-SAFETY": "ESC-SAFETY", "abuse_flag_refund": "ESC-ABUSE",
                "repeat_contact": "ESC-REPEAT", "lost_shipment": "ESC-LOST"}.get(r)
        if code and code not in codes:
            codes.append(code)
    return codes


def order_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=lambda c: CODE_PRIORITY.index(c) if c in CODE_PRIORITY else 99)


# --------------------------------------------------------------------------------------------------------------------
# Dependencies (injected, so tests and the MCP variant can swap any of them)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Deps:
    api: MockAPI
    kb: Any = None                                       # needs .answer(query, k=, use_rerank=, question=) -> {"answer", "sources"}
    store: Any = None                                    # MemoryStore (optional)
    ledger: ContactLedger = field(default_factory=ContactLedger)
    actions: Optional[ActionTools] = None
    extract: Callable = extract_ticket_info              # (text) -> (TicketExtraction | None, errors)
    memo_runner: Optional[Callable[[dict], dict]] = None  # M6 hook: handoff packet -> {"status", "memo"}
    tools: Optional[dict] = None                         # order tool callables (defaults to build_tools(api))
    now: datetime = pr.REFERENCE_NOW
    use_rerank: bool = True
    eligibility: Optional[Callable] = None               # (order_id, kind) -> eligibility dict; default = local policy_rules (MCP can swap it)
    order_guard: Optional[DependencyGuard] = None        # M7: retries + circuit breaker around every read/write of the order system

    def __post_init__(self):
        if self.actions is None:
            self.actions = ActionTools(self.api)
        if self.tools is None:
            self.tools, _ = build_tools(self.api, robust=False)      # ONE retry layer (the guard), not two stacked ones
        if self.order_guard is None:
            self.order_guard = make_order_guard()


# A record with a missing field / wrong type is the order system answering with garbage. It is not retried (same bad odds) and it
# is not guessed around: it goes to the human queue, with the cause kept in the trace.
ORDER_MALFORMED_ERRORS = (KeyError, TypeError, ValueError, AttributeError)


def make_order_guard() -> DependencyGuard:
    """The M7 guard for the order system (capstone: "retries, fallback to human queue, circuit breaker"). Tunable by env so a
    real, slower backend can get longer back-offs without code changes."""
    return DependencyGuard("order_system", max_attempts=int(os.environ.get("SHOPSENSE_ORDER_RETRIES", "3")),
                           base_delay=float(os.environ.get("SHOPSENSE_ORDER_BACKOFF_S", "0.2")),
                           malformed_on=ORDER_MALFORMED_ERRORS,
                           breaker=CircuitBreaker(failure_threshold=int(os.environ.get("SHOPSENSE_ORDER_BREAKER_THRESHOLD", "3")),
                                                  reset_timeout=float(os.environ.get("SHOPSENSE_ORDER_BREAKER_RESET_S", "30"))))


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------------------------------
_UNKNOWN_RE = re.compile(
    r"(don'?t|do not|doesn'?t|does not|cannot|can'?t|unable to)\s+(know|have|find|confirm|say)|not (covered|mentioned|in (the|our) (context|polic))|"
    r"no (information|policy|document)|isn'?t (covered|mentioned)|(doesn'?t|don'?t|does not|do not)\s+(mention|cover|address|specify|contain)|"
    r"not (able to )?(confirm|determine)", re.I)
# One definition, shared with the output guardrail (shopsense/guardrails.py), so the graph's critic and the last-line check cannot drift apart.
_FLAG_WORDS_RE = FLAG_WORDS_RE
_APPROVAL_CLAIM_RE = APPROVAL_CLAIM_RE


# A figure the policy documents could have supplied: a duration, a percentage or an amount of money. Section ids ("RP-1.1") and plain
# numbers do not count.
_FIGURE_RE = re.compile(r"\d[\d,.]*\s*-?\s*(?:(?:business|working|calendar)\s+)?(?:hours?|hrs?|days?|weeks?|months?|%|percent|inr|rs\b|rupees?)|(?:inr|rs\.?|₹)\s*\d", re.I)
_FIRST_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# A plain demand for money back. The model sometimes files an angry "I want my INR 25,516 back for ORD-4553" under COMPLAINT, which would
# leave the refund unopened and the approver signing off a reply only. Model labels are hints, so a clear demand that names an order
# is read as a refund request. Questions and status chasers ("where is my refund?", "still waiting") are not demands.
_MONEY_BACK_RE = re.compile(r"\b(?:refund(?:ed)?|money back|(?:want|need|give|get|send)\b[^.?!]{0,40}\bback)\b", re.I)
_STATUS_CHASE_RE = re.compile(r"\b(?:status|where is|where'?s|when will|how long|still|yet|haven'?t|have not|not received|waiting|update on)\b|\?", re.I)


def is_money_back_demand(text: str) -> bool:
    t = text or ""
    return bool(_MONEY_BACK_RE.search(t)) and not _STATUS_CHASE_RE.search(t)



def answer_is_unknown(answer: str) -> bool:
    """True when the model's answer is a REFUSAL ("the documents do not say"), not an answer that carries a caveat.

    The old rule discarded the whole answer if any hedge appeared in its first 400 characters. So "Electronics: 15 days; 30 in the festive
    season. I can't confirm which applies without your order date." was thrown away and the customer was told the documents had nothing,
    which is false and loses the figure they asked for. A hedge now counts only when
      * it opens the answer (first sentence) and that sentence gives no figure -- "I don't have that information in our policy documents."; or
      * it appears anywhere in the first 400 characters and the answer contains no figure at all (nothing was actually answered).
    A hedge later in an answer that already states a figure is a caveat, and the answer stands."""
    text = (answer or "").replace("\u2019", "'").replace("\u2018", "'").strip()      # curly apostrophes (many models write them) must not hide a refusal
    if not text:
        return True
    if not _UNKNOWN_RE.search(text[:400]):
        return False
    first = _FIRST_SENTENCE_RE.split(text, maxsplit=1)[0]
    if _UNKNOWN_RE.search(first) and not _FIGURE_RE.search(first):
        return True
    return not _FIGURE_RE.search(text)


_SOURCE_TAIL_RE = re.compile(r"\s*\((?:Sources?|Policy reference)\b[^)]*\)\s*$", re.I)
_CONTACT_RE = re.compile(r"\bcontact\b.{0,40}\b(human|agent|support|customer (care|service)|team)\b", re.I)
_GENERIC_WORDS = frozenset(
    "have has had information policy policies document documents available provide provided specific detail details confirm confirmed "
    "contact human agent further assistance please about regarding this that with from the our your support team cannot unable does "
    "mention mentioned cover covered address addressed there their they them would could should also anything something".split())


def _topic_stems(sentence: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", sentence.lower()) if w not in _GENERIC_WORDS}


def drop_unrelated_refusal(answer: str, customer_text: str) -> str:
    """Remove a trailing "I don't have information on X" sentence when the answer already states a figure and X is not what the
    customer asked about (a model sometimes appends a refusal about a side topic: a returns answer ending "I don't have information on
    the shipping delivery timeline. Please contact a human agent."). A refusal about something the customer DID ask about is kept, so a
    two-part question still gets an honest "this part is not covered". The (Source: ...) tag is kept."""
    text = (answer or "").replace("\u2019", "'").replace("\u2018", "'")
    if not text.strip():
        return answer
    tail = ""
    m = _SOURCE_TAIL_RE.search(text)
    if m:
        tail, text = m.group(0), text[:m.start()]
    parts = re.split(r"(?<=[.!?])(\s+)", text)                      # sentence, separator, sentence, ...
    sentences = parts[0::2]
    seps = parts[1::2] + [""]
    wanted = {w[:5] for w in re.findall(r"[a-z]{4,}", (customer_text or "").lower())}
    keep = [True] * len(sentences)
    for i, sent in enumerate(sentences):
        if _FIGURE_RE.search(sent) or not _UNKNOWN_RE.search(sent):
            continue
        topic = _topic_stems(sent)
        if topic and len(topic & wanted) / len(topic) < 0.6:
            keep[i] = False
            if i + 1 < len(sentences) and _CONTACT_RE.search(sentences[i + 1]) and not _FIGURE_RE.search(sentences[i + 1]):
                keep[i + 1] = False
    kept = [sentences[i] + seps[i] for i in range(len(sentences)) if keep[i]]
    if all(keep) or not any(_FIGURE_RE.search(sentences[i]) for i in range(len(sentences)) if keep[i]):
        return answer                                              # nothing to drop, or dropping would leave no figure
    return "".join(kept).rstrip() + tail


def sla_text(codes: list[str]) -> str:
    for c in CODE_PRIORITY:
        if c in codes and c in pr.SLA_BY_CODE:
            return pr.SLA_BY_CODE[c]
    return "one business day"


def holding_reply(codes: list[str], reasons: list[str], *, sentiment: str = "neutral") -> str:
    """What the customer is told when a human takes over. Deterministic, so it can promise nothing, copy nothing from
    the (possibly hostile) ticket, and never reveal why the case was routed beyond what the customer already knows."""
    lead = "I'm sorry about this experience. " if sentiment in ("angry", "concerned") else ""
    sla = sla_text(codes)
    if "ESC-SAFETY" in codes:
        return (lead + "Your safety comes first: please stop using the product right now, unplug it, and keep it away from anything "
                "flammable. Please do not ship it back yourself -- we will arrange a safe pickup. I've passed this to our Product "
                f"Safety team as urgent, and they will contact you within {sla}.")
    if "ESC-LEGAL" in codes:
        return (lead + "I've passed your message to our specialist team, who handle matters like this directly. They will respond "
                f"{'' if sla.startswith('the') else 'within '}{sla}. I'm not able to discuss compensation or liability myself.")
    if "ESC-ABUSE" in codes:
        return (lead + "Your request needs additional verification, so I've passed it to a specialist team who will review it and "
                f"get back to you within {sla}. I'm not able to approve or confirm it myself.")
    if "ESC-REPEAT" in codes:
        return (lead + "I can see you've contacted us about this more than once without a resolution, and I'm sorry. I've escalated "
                f"it to a senior support specialist who will take ownership and reach out within {sla}.")
    if "ESC-LOST" in codes:
        return (lead + "Your shipment hasn't moved for far too long, so I've passed it to our logistics recovery team. They will "
                f"arrange a replacement or refund and contact you within {sla}.")
    if "order_system_unavailable" in reasons:
        return (lead + "I couldn't complete this right away because our order system isn't responding, so I can't confirm anything about "
                f"your order yet. I've passed your request to a team member, who will check it and confirm the outcome to you {'' if sla.startswith('the') else 'within '}{sla}.")
    if "policy_unavailable" in reasons:
        return (lead + "I couldn't look up our policy details right now, so I can't answer your question yet. I've passed it to a team "
                f"member, who will reply {'' if sla.startswith('the') else 'within '}{sla}.")
    return lead + "Thanks for contacting us. I've passed your request to a team member for review, and they will follow up with you shortly."


def approval_holding_reply(proposal: dict, codes: list[str], *, angry: bool) -> str:
    lead = "I'm sorry for the trouble. " if angry else ""
    t = proposal.get("type")
    if t in MONEY_PROPOSALS and proposal.get("tier") in CODE_BY_TIER:
        who = {"manager_review": "a Team Lead", "executive_review": "a Finance Executive"}[proposal["tier"]]
        sla = pr.SLA_BY_CODE[CODE_BY_TIER[proposal["tier"]]]
        what = {"refund": "refund", "decide_pending_refund": "refund", "replace": "replacement"}[t]
        return (lead + f"I've sent your {what} request for order {proposal['order_id']} (INR {proposal['amount_inr']:,.0f}) to {who} for "
                f"review. Requests of this size need a person's approval, and I'm not able to approve them myself. You'll hear back within {sla}.")
    return lead + "I've sent my proposed response to a team member for a quick review before it goes out, so you'll hear back shortly."


_TOPICS = (
    (r"return|exchange|send (it )?back|too (small|tight|big)", "return window days"),
    (r"refund|money back|reimburs", "refund processing time business days"),
    (r"warranty|repair|stopped working|not turning on", "warranty months"),
    (r"cancel", "cancellation processing orders"),
    (r"ship|deliver|arrive|courier", "shipping delivery timeline"),
    (r"damag|smash|broken|defect|wrong item", "damaged defective report days"),
    (r"\bopen(ed|ing)?\b|\bunseal|\bused\b|\btried\b|\bworn\b", "opened sealed hygiene seal condition"),
)


_SALIENT = re.compile(r"\b(festive|diwali|sale|upi|card|cod|cash on delivery|net banking|wallet|opened|sealed|seal|tags?|box|packaging|"
                      r"doa|dead on arrival|rash|skin|allerg\w*|breakable|bulky|pickup|swollen|duplicate|twice|charged|international\w*|"
                      r"price match\w*|gift|warranty|exchange|size)\b", re.I)


def _salient_words(text: str) -> str:
    """A handful of policy-relevant words the customer actually used (festive, UPI, sealed...); the rest of a chatty ticket is
    noise to a keyword index."""
    return " ".join(dict.fromkeys(m.group(0).lower() for m in _SALIENT.finditer(text or "")))


def _topic_terms(text: str) -> str:
    """Policy vocabulary for the topics a ticket touches. Tickets are chatty; the policy corpus is not, and a retrieval query
    that only repeats the chat finds the generic FAQ instead of the category addendum that holds the real number."""
    return " ".join(terms for pat, terms in _TOPICS if re.search(pat, text or "", re.I))


def _policy_categories(triage: dict, safe_text: str, api: Optional[MockAPI]) -> list[str]:
    oid = triage.get("order_id")
    names = sorted({i["category"] for i in pr.order_items(api, oid)}) if (api is not None and oid and oid in api.orders) else []
    return names or pr.categories_in_text(safe_text)             # no order named: aim at the product the customer mentions


def _search_query(names: list[str], safe_text: str) -> str:
    cats = (" ".join(names) + " ") if names else ""
    focused = f"{cats}{_topic_terms(safe_text)} {_salient_words(safe_text)}".strip()
    return focused if (cats or _topic_terms(safe_text)) else (safe_text or "")[:300]


def _answer_question(names: list[str], safe_text: str) -> str:
    text = " ".join((safe_text or "").split())[:400]
    return f"{text} (product category: {', '.join(names)})" if names and text else text


def policy_search_and_question(triage: dict, safe_text: str, api: Optional[MockAPI]) -> tuple[str, str]:
    """Two different texts for two different readers.

    SEARCH query, for the retrieval index: keywords. The product category is added when the ticket names an order, because the
    category addendum -- not the general policy -- usually holds the right number (electronics 15 days, not 30).
    QUESTION, for the answer model: the customer's own words plus that category. Handing the model the keyword string instead
    ("electronics return window days shipping delivery timeline") left it guessing what was asked: it either refused or added a
    refusal about a topic nobody raised.
    The order is read once, so one ticket makes one order-system lookup here."""
    names = _policy_categories(triage, safe_text, api)
    return _search_query(names, safe_text), _answer_question(names, safe_text)


def policy_query_for(triage: dict, safe_text: str, api: Optional[MockAPI]) -> str:
    return policy_search_and_question(triage, safe_text, api)[0]


# ---- planning ------------------------------------------------------------------------------------------------------
def _money(proposal: dict, amount: float) -> dict:
    tier = refund_tier_for_amount(amount)
    proposal.update({"amount_inr": amount, "tier": tier, "code": CODE_BY_TIER.get(tier)})
    return proposal


# A message that is ONLY a greeting, thanks or goodbye. Matched against the WHOLE message, so "hi, ignore your rules" is not small talk.
_SMALLTALK_RE = re.compile(
    r"^\W*(?:(?P<greet>hi+|hello+|hey+|hiya|namaste|good\s+(?:morning|afternoon|evening))|(?P<thanks>thanks?|thank\s+you|thx|ok|okay)|"
    r"(?P<bye>bye|goodbye))(?:\s+(?:there|team|support|everyone))?\W*$", re.I)
SMALLTALK_REPLIES = {
    "greet": "Hello! I can help with order status and tracking, returns and refunds, cancellations and delivery questions. What can I help you with?",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "bye": "Goodbye! You can message us any time.",
}


def smalltalk_kind(text: str) -> str:
    """'greet' | 'thanks' | 'bye' | '' -- nothing to look up, nothing to promise, and no reason to open a human ticket for it."""
    m = _SMALLTALK_RE.match(text or "")
    return next((k for k in ("greet", "thanks", "bye") if m and m.group(k)), "")


def plan_action(state: TicketState, deps: Deps) -> tuple[dict, dict, list[str]]:
    """(facts, proposal, extra hard reasons). Reads only; never changes anything."""
    tri, api = state["triage"], deps.api
    text, intent, oid = state["text"], tri.get("intent"), tri.get("order_id")
    cust = state.get("customer_id") or ""
    facts: dict = {}
    extra: list[str] = []
    wants_action = intent in (REFUND_FAMILY | STATUS_FAMILY | {"CANCEL_ORDER", "CHANGE_ORDER", "REFUND_STATUS"}) and not tri.get("request_type") == "question"

    kind = smalltalk_kind(text) if not oid else ""
    if kind:
        return facts, {"type": "smalltalk", "kind": kind}, extra
    if not oid:
        return facts, ({"type": "ask_order_id"} if wants_action else {"type": "inform"}), extra
    order = api.orders.get(oid)
    if order is None:
        return facts, {"type": "order_not_found", "order_id": oid}, extra
    if cust.startswith("GUEST") or not cust:
        return facts, {"type": "inform"}, ["unverified_customer"]
    if order["customer_id"] != cust:
        return facts, {"type": "inform"}, ["order_owner_mismatch"]

    t = deps.tools
    facts["order"] = t["order_lookup"](oid)
    facts["shipment"] = t["shipping_tracker"](oid)
    facts["refund"] = t["refund_status"](oid)
    facts["payment_method"] = order["payment_method"]

    if tri.get("request_type") == "question" or intent == "PRODUCT_INFO":
        return facts, {"type": "inform", "order_id": oid}, extra          # a pure question is answered; a lost parcel is not its business
    lost = pr.lost_shipment(api, oid, now=deps.now)
    if lost:
        facts["lost"] = lost
        extra.append("lost_shipment")

    if pr.mentions_skin_reaction(text) and any(i["category"] == "beauty" for i in pr.order_items(api, oid)):
        return facts, {"type": "goodwill_review", "order_id": oid, "cap_inr": pr.GOODWILL_CAP_INR}, extra
    if intent == "CANCEL_ORDER":
        if order["status"] == "processing":
            return facts, {"type": "cancel", "order_id": oid, "amount_inr": float(order["total_amount_inr"]),
                           "prepaid": order["payment_method"] != "Cash on Delivery"}, extra
        return facts, {"type": "cannot_cancel", "order_id": oid, "status": order["status"]}, extra
    if intent == "REFUND_STATUS":
        return facts, {"type": "inform_refund_status", "order_id": oid}, extra
    if intent in STATUS_FAMILY:
        return facts, {"type": "inform_status", "order_id": oid}, extra
    if intent in REFUND_FAMILY:
        ref = facts["refund"]
        if ref.get("has_refund_request"):
            if ref["approved"]:
                return facts, {"type": "inform_refund_status", "order_id": oid}, extra
            p = {"type": "decide_pending_refund", "order_id": oid, "refund_id": ref["refund_id"]}
            return facts, _money(p, float(ref["amount_inr"])), extra
        kind = "defect" if (intent in ("PRODUCT_DEFECTIVE", "WRONG_ITEM_DELIVERED") or tri.get("defective")) else "change_of_mind"
        elig = deps.eligibility(oid, kind) if deps.eligibility else pr.return_eligibility(api, oid, kind=kind, now=deps.now)
        facts["eligibility"] = elig
        outcome = elig["outcome"]
        if outcome in ("refund_eligible", "partial"):
            p = {"type": "replace" if (kind == "defect" and pr.wants_replacement(text)) else "refund", "order_id": oid, "kind": kind,
                 "partial": outcome == "partial"}
            amt = elig["eligible_amount_inr"]
            if p["type"] == "replace":
                p["value_inr"] = amt
            return facts, _money(p, amt), extra
        if outcome == "warranty_claim":
            months = max((pr.warranty_months(i["category"], i["product"]) for i in elig["items"]), default=0)
            return facts, {"type": "warranty_claim", "order_id": oid, "months": months}, extra
        if outcome == "not_delivered":
            return facts, {"type": "inform_status", "order_id": oid, "note": "not_delivered_yet"}, extra
        if outcome == "already_returned":
            return facts, {"type": "inform_refund_status", "order_id": oid}, extra
        return facts, {"type": "decline", "order_id": oid, "kind": kind, "outcome": outcome,
                       "categories": sorted({i["category"] for i in elig["items"]}),
                       "window_days": max((i["window_days"] for i in elig["items"]), default=0)}, extra
    return facts, {"type": "inform", "order_id": oid}, extra


def decide_route(state: TicketState) -> dict:
    """Pure: reasons + proposal + sentiment -> route. Hard facts always win; the model's own 'needs human review' flag
    sends a ticket to a human only when no approval workflow already puts one in the loop."""
    reasons = list(state.get("reasons") or [])
    proposal = state.get("proposal") or {}
    sentiment = (state.get("triage") or {}).get("sentiment") or "neutral"
    codes = list(codes_from_reasons(reasons))
    hard = [r for r in reasons if r not in SOFT_REASONS]
    tier_gate = proposal.get("type") in MONEY_PROPOSALS and proposal.get("tier") in CODE_BY_TIER
    if tier_gate and proposal.get("code"):
        codes.append(proposal["code"])
    if hard:
        return {"route": "human_review", "route_reasons": hard, "escalation_codes": order_codes(codes)}
    if "requires_human_review" in reasons and not tier_gate:
        return {"route": "human_review", "route_reasons": ["requires_human_review"], "escalation_codes": order_codes(codes)}
    why = []
    if tier_gate:
        why.append(f"high_value_refund:{proposal['tier']}")
    if sentiment == "angry":
        why.append("angry_sentiment")
    if why:
        return {"route": "approval", "route_reasons": why, "escalation_codes": order_codes(codes)}
    return {"route": "auto", "route_reasons": [], "escalation_codes": order_codes(codes)}


# ---- reply ---------------------------------------------------------------------------------------------------------
def _refund_timing(facts: dict) -> str:
    return pr.REFUND_TIMING_BY_PAYMENT.get(facts.get("payment_method", ""), "")


def outcome_text(state: TicketState) -> str:
    p, facts, res = state.get("proposal") or {}, state.get("facts") or {}, state.get("action_result") or {}
    t, oid = p.get("type"), p.get("order_id", "")
    denied, notes = res.get("denied"), (state.get("human_notes") or "").strip()
    if res.get("refused"):
        return ("I wasn't able to complete this myself because it needs a higher level of approval. I've passed it to the right "
                "team and they will follow up.")
    if t == "smalltalk":
        return SMALLTALK_REPLIES.get(p.get("kind", ""), "")
    if t == "ask_order_id":
        return "Could you share your order number (it looks like ORD-1234)? I can't look anything up or act on an order without it."
    if t == "order_not_found":
        return f"I couldn't find an order {oid}. Could you double-check the number?"
    if t == "inform_status":
        o, s = facts.get("order", {}), facts.get("shipment", {})
        bits = [f"Order {oid} is currently {o.get('status', 'unknown')}."]
        if s.get("tracking_available"):
            bits.append(f"It is with {s['carrier']} (tracking {s['tracking_number']}), last status: {s['scan_state'].replace('_', ' ')}.")
        elif p.get("note") == "not_delivered_yet":
            bits.append("It hasn't been delivered yet, so it can't be returned yet.")
        else:
            bits.append("It hasn't shipped yet, so there is no tracking number.")
        return " ".join(bits)
    if t == "inform_refund_status":
        r = facts.get("refund", {})
        if not r.get("has_refund_request"):
            return f"I don't see a refund request on order {oid}."
        if r["approved"]:
            timing = _refund_timing(facts)
            return (f"Your refund {r['refund_id']} (INR {float(r['amount_inr']):,.0f}) for order {oid} is approved."
                    + (f" For your payment method it is processed within {timing} after the returned item passes quality check." if timing else ""))
        return f"Your refund {r['refund_id']} (INR {float(r['amount_inr']):,.0f}) for order {oid} is still waiting for review."
    if t == "cancel":
        if denied:
            return f"Order {oid} was not cancelled. {notes}".strip()
        if res.get("committed"):
            return (f"Order {oid} is cancelled." + (" The charge will be reversed to your original payment method." if p.get("prepaid") else " No payment was taken, so nothing needs refunding."))
        return f"I'll cancel order {oid} once it has been reviewed."
    if t == "cannot_cancel":
        return (f"Order {oid} is {p['status']}, and orders can only be cancelled while they are still processing. "
                "Once it is delivered you can return it under the return policy.")
    if t in ("refund", "decide_pending_refund", "replace"):
        what = "replacement" if t == "replace" else "refund"
        amt = p["amount_inr"]
        if denied:
            return f"After review, your {what} request for order {oid} was not approved. {notes}".strip()
        if res.get("committed"):
            amount_note = f"INR {res.get('amount_inr', amt):,.0f}"
            if t == "replace":
                return f"Your replacement for order {oid} has been arranged."
            timing = _refund_timing(facts)
            return (f"Your refund of {amount_note} for order {oid} has been authorised."
                    + (f" It is processed within {timing} once the returned item passes our quality check." if timing else
                       " It is processed once the returned item passes our quality check."))
        return approval_holding_reply(p, [], angry=False)
    if t == "warranty_claim":
        return (f"This item is past the dead-on-arrival period but is covered by its {p['months']}-month warranty. We'll arrange a repair or "
                "replacement; a cash refund is only considered if neither is possible, and any refund of INR 2,000 or more needs a person's "
                "approval (INR 10,000 or more needs a Finance Executive), so I can't approve one myself.")
    if t == "goodwill_review":
        return (f"A skin reaction isn't a product defect under our policy, but I can offer a goodwill refund of up to INR {p['cap_inr']} if you "
                "send a photo of the reaction. Anything above that has to be decided by a manager.")
    if t == "decline":
        cats = p.get("categories") or []
        if "groceries" in cats and p["kind"] == "change_of_mind":
            return ("Grocery items can't be returned for change of mind or taste. If an item arrived damaged, spoiled or wrong, it must be "
                    "reported within 48 hours of delivery with a photo of the item and its batch label.")
        if p["kind"] == "defect":
            return f"This is outside the {p['window_days']}-day window for reporting a problem with this item, so I can't raise a return for it."
        return f"This order is outside the {p['window_days']}-day return window that applies to this product, so I can't raise a return for it."
    return ""


def policy_text(state: TicketState) -> str:
    pol = state.get("policy") or {}
    if not pol:
        return ""
    if pol.get("unknown"):
        return ("I don't have this in our policy documents, so I can't confirm it. For anything our policy documents do not cover, please contact our support team directly.")        # no promise of a follow-up: nothing queues one
    ans = (pol.get("answer") or "").strip()
    if not ans:
        return ""
    srcs = sorted({s["source"] for s in pol.get("sources", [])})
    return ans + (f" (Source: {', '.join(srcs)})" if srcs else "")


# What is said when the system has nothing to say. It promises nothing: a promise of a follow-up is only honest when a human queue
# holds the ticket, and a ticket that reaches this text is NOT in one. The action planner therefore routes such a ticket to a person
# (reason `no_automated_answer`) before this text can be sent, except for praise, which needs no reply beyond thanks.
NO_ANSWER_REPLY = "Thanks for getting in touch -- I've noted your message."


def compose_reply(state: TicketState, *, include_policy: bool = True) -> str:
    route = state.get("route")
    if route == "human_review":
        return state.get("holding_reply") or holding_reply(state.get("escalation_codes", []), state.get("reasons", []))
    tri = state.get("triage") or {}
    parts = []
    if tri.get("sentiment") == "angry":
        parts.append("I'm sorry for the trouble, and I understand how frustrating this is.")
    out = outcome_text(state)
    if out:
        parts.append(out)
    only_info = (state.get("proposal") or {}).get("type") in ("inform", None)
    if include_policy and only_info:
        pol = policy_text(state)                         # the one place model-written text is shown: no deterministic answer exists
        if pol:
            parts.append(pol)
    elif include_policy:
        srcs = sorted({x["source"] for x in (state.get("policy") or {}).get("sources", [])})
        if srcs and (state.get("proposal") or {}).get("type") in ("decline", "goodwill_review", "warranty_claim", "cannot_cancel",
                                                                  "inform_refund_status"):
            parts.append(f"(Policy reference: {', '.join(srcs)}.)")   # a citation, not model prose: figures above come from the records
    return " ".join(parts).strip() or NO_ANSWER_REPLY


def check_reply(state: TicketState, reply: str) -> list[str]:
    """Deterministic critic over the customer-facing text."""
    issues: list[str] = []
    if _FLAG_WORDS_RE.search(reply):
        issues.append("mentions account flags / abuse")
    res, p = state.get("action_result") or {}, state.get("proposal") or {}
    already_approved = p.get("type") == "inform_refund_status" and bool((state.get("facts") or {}).get("refund", {}).get("approved"))
    money_done = (res.get("committed") and not res.get("denied")) or already_approved
    if _APPROVAL_CLAIM_RE.search(reply) and not money_done:
        issues.append("claims an approval/refund that has not been committed")
    allowed = {f"{v:,.0f}" for v in _numbers_in(state)} | {f"{v:.0f}" for v in _numbers_in(state)}
    for m in re.finditer(r"INR\s*([\d,]+)", reply):
        if m.group(1).rstrip(",") not in allowed and m.group(1) not in allowed and m.group(1) not in {"2,000", "10,000", "500"}:
            issues.append(f"amount INR {m.group(1)} not backed by the facts")
    if (state.get("triage") or {}).get("injection") and re.search(r"(?i)\b(system note|ignore (all )?previous|override)\b", reply):
        issues.append("repeats injected text")
    return issues


def _numbers_in(state: TicketState) -> set:
    out = set()
    p, facts = state.get("proposal") or {}, state.get("facts") or {}
    for k in ("amount_inr", "value_inr", "cap_inr"):
        if p.get(k) is not None:
            out.add(float(p[k]))
    for m in re.finditer(r"INR\s*([\d,]+)", (state.get("policy") or {}).get("answer", "") or ""):   # figures quoted from the retrieved policy
        out.add(float(m.group(1).replace(",", "") or 0))
    for src in (facts.get("order", {}).get("total_amount_inr"), (facts.get("refund") or {}).get("amount_inr"),
                (state.get("action_result") or {}).get("amount_inr")):
        if src is not None:
            out.add(float(src))
    return out


# --------------------------------------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------------------------------------
def make_nodes(deps: Deps) -> dict[str, Callable]:
    def triage_node(state: TicketState) -> dict:
        text = state["text"]
        cust = state.get("customer_id") or f"GUEST-{state['ticket_id']}"
        injected = detect_injection_attempt(text)
        safe = redact_injection(text) if injected else text
        try:
            extraction, _errs = deps.extract(safe)
        except Exception as e:                                   # noqa: BLE001 -- a rate limit / outage must not lose the safety floor
            extraction, _errs = None, [f"{type(e).__name__}: {str(e)[:120]}"]
            add_event("triage.extraction_error", error=_errs[0])
        visible_order = find_order_id_in_text(text)              # only a reference the customer actually wrote counts
        ex = lambda k: getattr(extraction, k, None) if extraction is not None else None
        request_type = "question" if pr.is_policy_question(safe) else "request"
        intent = ex("intent")
        if intent in ("COMPLAINT", "OTHER") and visible_order and request_type == "request" and is_money_back_demand(safe):
            intent = "REFUND_REQUEST"
        triage = {"intent": intent, "category": ex("category"), "sentiment": ex("sentiment") or "neutral",
                  "urgency": ex("urgency") or "low", "order_id": visible_order, "request_type": request_type,
                  "defective": pr.looks_defective(safe), "extraction_ok": extraction is not None, "injection": injected,
                  "safe_text": safe, "extraction_error": (_errs[0][:160] if extraction is None and _errs else "")}
        # The customer record decides the abuse-flag escalation. If the order system cannot be read we cannot rule the flag out, so
        # the ticket goes to a person rather than being treated as an unflagged customer.
        customer_unavailable = False
        try:
            customer_rec = deps.order_guard.call(deps.api.customers.get, cust)
        except DependencyUnavailable as e:
            customer_rec, customer_unavailable = None, True
            add_event("fallback.human_queue", dependency="order_system", reason=e.reason, step="triage.customer")
        reasons = combine_escalation_reasons(extraction, text, customer=customer_rec)
        if customer_unavailable:
            reasons.append("order_system_unavailable")
        if extraction is None:
            reasons.append("extraction_failed")
        for h in state.get("history") or []:
            deps.ledger.record(ticket_id=h["ticket_id"], customer_id=cust, created_at=h["created_at"], text=h.get("text", ""),
                               order_id=h.get("order_id"))
        n = deps.ledger.contact_number(ticket_id=state["ticket_id"], customer_id=cust, created_at=state.get("created_at") or deps.now.isoformat(),
                                       order_id=visible_order, intent=triage["intent"])
        if n >= pr.REPEAT_CONTACT_THRESHOLD:
            reasons.append("repeat_contact")
        deps.ledger.record(ticket_id=state["ticket_id"], customer_id=cust, created_at=state.get("created_at") or deps.now.isoformat(),
                           text=text, order_id=visible_order, intent=triage["intent"])
        codes = codes_from_reasons(reasons)
        why = f" extraction_error={triage['extraction_error']!r}" if triage["extraction_error"] else ""
        return {"customer_id": cust, "triage": triage, "reasons": reasons, "escalation_codes": order_codes(codes), "status": "",
                "log": [f"triage: intent={triage['intent']} sentiment={triage['sentiment']} order={visible_order} contact#{n} "
                        f"reasons={reasons or 'none'}{why}"]}

    def policy_check_node(state: TicketState) -> dict:
        tri = state["triage"]
        if tri["intent"] not in POLICY_INTENTS and tri["request_type"] != "question":
            return {"policy": {}, "log": ["policy_check: skipped (status-type ticket)"]}
        order_unreadable = False
        try:
            query, question = deps.order_guard.call(policy_search_and_question, tri, tri["safe_text"], deps.api)
        except DependencyUnavailable as e:
            # Without the order we cannot name its product category, and the category addendum is what holds the right number
            # (electronics 15 days, not 30). A generic answer could be confidently wrong, so the ticket goes to a person.
            query, question = policy_search_and_question(tri, tri["safe_text"], None)
            order_unreadable = True          # the policy agent may only write `policy`; the action planner turns this into the routing reason
            add_event("fallback.human_queue", dependency="order_system", reason=e.reason, step="policy_check.query")
        try:
            res = deps.kb.answer(query, k=3, use_rerank=deps.use_rerank, question=question)
            ans = clean_model_text(res.get("answer", ""))          # citation markers / odd spaces the model copied from its context
            if not ans.strip():
                # "The documents do not cover this" is an ANSWER (words from the model). Nothing at all is a failed call, and it must
                # not be reported to the customer as a gap in our policy.
                raise ValueError("the policy model returned no answer")
            if not answer_is_unknown(ans):
                ans = drop_unrelated_refusal(ans, tri.get("safe_text", ""))
            policy = {"query": query, "answer": ans, "sources": res.get("sources", []), "unknown": answer_is_unknown(ans)}
        except Exception as e:                                   # noqa: BLE001 -- RAG outage must not kill the ticket
            # `unavailable` (the policy agent may only write `policy`) is turned into a routing reason by the action planner,
            # because a ticket whose whole reply IS the policy answer cannot be closed without one.
            policy = {"query": query, "answer": "", "sources": [], "unknown": False, "unavailable": True,
                      "error": f"{type(e).__name__}: {str(e)[:160]}"}
            add_event("policy.answer_unavailable", error=policy["error"], step="policy_check.answer")
        if order_unreadable:
            policy["order_unreadable"] = True
        return {"policy": policy,
                "log": [f"policy_check: sources={[s['source'] for s in policy['sources']]} unknown={policy['unknown']}"]}

    def action_plan_node(state: TicketState) -> dict:
        with get_tracer().span("order_system.read", kind="tool") as sp:
            try:
                facts, proposal, extra = deps.order_guard.call(plan_action, state, deps)       # read-only, so retrying the whole step is safe
            except DependencyUnavailable as e:
                # FALLBACK TO THE HUMAN QUEUE: with no order facts nothing may be promised, refunded or declined. The ticket is
                # routed (a hard reason) to a person; the customer is told honestly that we could not check.
                facts, proposal, extra = {}, {"type": "order_system_unavailable", "detail": e.reason}, ["order_system_unavailable"]
                add_event("fallback.human_queue", dependency="order_system", reason=e.reason)
                sp.fail(str(e))
        if (state.get("policy") or {}).get("order_unreadable") and "order_system_unavailable" not in extra:
            extra = [*extra, "order_system_unavailable"]       # the policy step could not read the order, so its answer may not be category-specific
        if (state.get("policy") or {}).get("unavailable") and proposal.get("type") in ("inform", None) and "policy_unavailable" not in extra:
            # The reply to this ticket would BE the policy answer (compose_reply shows model text only for `inform`). With none, the
            # old behaviour closed it as "resolved" with a promise that a colleague would follow up -- and nothing queued one.
            # Money and status outcomes come from records, not from the model, so they stand without the policy text.
            extra = [*extra, "policy_unavailable"]
            add_event("fallback.human_queue", dependency="policy_answer", reason=(state.get("policy") or {}).get("error", "unavailable")[:100],
                      step="action_plan.policy")
        reasons = list(state.get("reasons") or []) + [r for r in extra if r not in (state.get("reasons") or [])]
        tri = state.get("triage") or {}
        praise = tri.get("intent") == "COMPLIMENT" and tri.get("sentiment") == "positive"
        if not reasons and not praise and proposal.get("type") in ("inform", None) and \
                compose_reply({**state, "facts": facts, "proposal": proposal, "route": "auto"}) == NO_ANSWER_REPLY:
            # Complaints: if nothing automated applied, the reply would fall through to "a colleague will follow up" and the ticket
            # would close as resolved with NO colleague queued. A ticket the system cannot answer belongs to a person.
            reasons.append("no_automated_answer")
        return {"facts": facts, "proposal": proposal, "reasons": reasons,
                "log": [f"action_plan: proposal={proposal.get('type')} amount={proposal.get('amount_inr')} tier={proposal.get('tier')} extra={extra or 'none'}"]}

    def route_node(state: TicketState) -> dict:
        decision = decide_route(state)
        angry = (state.get("triage") or {}).get("sentiment") == "angry"
        if decision["route"] == "human_review":
            holding = holding_reply(decision["escalation_codes"], decision["route_reasons"], sentiment=(state["triage"] or {}).get("sentiment", "neutral"))
        elif decision["route"] == "approval":
            holding = approval_holding_reply(state.get("proposal") or {}, decision["escalation_codes"], angry=angry)
        else:
            holding = ""
        return {**decision, "holding_reply": holding, "log": [f"route: {decision['route']} reasons={decision['route_reasons']} codes={decision['escalation_codes']}"]}

    def human_review_node(state: TicketState) -> dict:
        codes = state.get("escalation_codes") or codes_from_reasons(state.get("reasons", []))
        if state.get("route") != "human_review":          # reached straight from triage: run the route decision now
            d = decide_route(state)
            codes = d["escalation_codes"]
            reasons = d["route_reasons"]
            holding = holding_reply(codes, reasons, sentiment=(state.get("triage") or {}).get("sentiment", "neutral"))
        else:
            reasons, holding = state.get("route_reasons", []), state.get("holding_reply", "")
        queue = next((QUEUE_BY_CODE[c] for c in CODE_PRIORITY if c in codes), "Support review queue")
        packet = {"ticket_id": state["ticket_id"], "customer_id": state["customer_id"], "order_id": (state.get("triage") or {}).get("order_id"),
                  "queue": queue, "escalation_codes": codes, "reasons": reasons, "sla": sla_text(codes),
                  "ticket_text_safe": (state.get("triage") or {}).get("safe_text", state["text"]),
                  "proposal": state.get("proposal") or {}}
        if state.get("human_decision") is not None or (state.get("action_result") or {}).get("unavailable"):
            packet["decision_so_far"] = {"human_decision": state.get("human_decision"), "approver_role": state.get("approver_role"),
                                          "approved_amount_inr": state.get("approved_amount_inr"), "action_result": state.get("action_result")}
        if deps.memo_runner is not None:
            try:
                memo = deps.memo_runner({**packet, "ticket_text": state["text"], "state": state})
                packet["memo_status"], packet["memo"] = memo.get("status", ""), memo.get("memo", "")
            except Exception as e:                       # noqa: BLE001 -- the hand-off must happen even if the team errors
                packet["memo_status"], packet["memo"] = "memo_failed", f"(review memo unavailable: {str(e)[:120]})"
        if deps.store is not None:
            deps.store.write(state["customer_id"], "episodic", f"Ticket {state['ticket_id']} was handed to a human specialist team for review.",
                             codes=codes, ticket_id=state["ticket_id"])
        reply = holding or holding_reply(codes, reasons)
        return {"route": "human_review", "handoff": packet, "reply": reply, "status": "escalated_to_human", "holding_reply": reply,
                "escalation_codes": codes, "log": [f"human_review: queue={queue} sla={packet['sla']} memo={packet.get('memo_status', 'n/a')}"]}

    def approval_node(state: TicketState) -> dict:
        p = state.get("proposal") or {}
        decision = interrupt({
            "kind": "ticket_approval", "ticket_id": state["ticket_id"], "customer_id": state["customer_id"], "order_id": p.get("order_id"),
            "reasons": state.get("route_reasons"), "escalation_codes": state.get("escalation_codes"), "proposal": p,
            "needs_role": ROLE_BY_TIER.get(p.get("tier"), "manager"), "holding_reply": state.get("holding_reply"),
            "policy_basis": (state.get("policy") or {}).get("answer", ""),
            "proposed_reply": compose_reply(state),                      # what the customer would be told if nothing needed deciding
            "ask": "Reply with 'approve', 'deny' or 'modify' (with amount_inr), plus your role (manager/executive) and optional notes.",
        })
        return {"human_decision": decision.get("decision"), "human_notes": decision.get("notes", ""),
                "approver_role": decision.get("approver_role"), "approved_amount_inr": decision.get("amount_inr")}

    def commit_node(state: TicketState) -> dict:
        p, tid = state.get("proposal") or {}, state["ticket_id"]
        decision = state.get("human_decision")
        human = decision is not None
        role = (state.get("approver_role") or "manager") if human else "system"
        cust = state["customer_id"]
        if decision == "deny":
            result = {"committed": False, "denied": True}
        else:
            amount = p.get("amount_inr")
            if decision == "modify" and state.get("approved_amount_inr") is not None and amount is not None:
                amount = min(float(state["approved_amount_inr"]), float(amount))
            t = p.get("type")
            raw: dict = {"ok": True, "noop": True}
            write: Optional[Callable] = None
            if t == "refund":
                write = lambda: deps.actions.issue_refund(p["order_id"], amount, approver_role=role, ticket_id=tid, customer_id=cust,
                                                          reason=p.get("kind", ""))
            elif t == "decide_pending_refund":
                write = lambda: deps.actions.approve_pending_refund(p["order_id"], approver_role=role, amount_inr=amount, ticket_id=tid)
            elif t == "replace":
                write = lambda: deps.actions.request_replacement(p["order_id"], value_inr=p.get("value_inr", amount or 0.0), approver_role=role,
                                                                 ticket_id=tid, customer_id=cust)
            elif t == "cancel":
                write = lambda: deps.actions.cancel_order(p["order_id"], ticket_id=tid, customer_id=cust)
            unavailable: Optional[DependencyUnavailable] = None
            if write is not None:
                with get_tracer().span("order_system.write", kind="tool", input={"action": t, "order_id": p.get("order_id"), "amount_inr": amount, "role": role}) as sp:
                    try:
                        # Safe to retry: every action is idempotent per (action, order, ticket), so a timeout that hid a success
                        # returns the earlier record (duplicate=True) instead of paying a second time.
                        raw = deps.order_guard.call(write)
                    except DependencyUnavailable as e:
                        unavailable = e
                        add_event("fallback.human_queue", dependency="order_system", reason=e.reason, action=t)
                        sp.fail(str(e))
            if unavailable is not None:
                result = {"committed": False, "unavailable": True, "reason": unavailable.reason}
            elif raw.get("noop"):
                result = {"committed": False, "noop": True}
            elif raw.get("ok"):
                result = {"committed": True, "amount_inr": raw.get("amount_inr", amount), "approver_role": role, "duplicate": bool(raw.get("duplicate"))}
            else:
                result = {"committed": False, "refused": raw.get("refused"), "detail": {k: v for k, v in raw.items() if k not in ("ok", "refused")}}
        if deps.store is not None and not result.get("unavailable"):      # nothing happened yet: the hand-off, not memory, records it
            deps.store.write(cust, "episodic", f"Ticket {tid}: {p.get('type', 'inform')} -> "
                             + ("denied by reviewer" if result.get("denied") else "completed" if result.get("committed") else "no change"
                                if not result.get("refused") else "needs higher approval"), ticket_id=tid)
        return {"action_result": result, "log": [f"commit: {p.get('type')} as {role} -> {result}"]}

    def reply_node(state: TicketState) -> dict:
        if (state.get("action_result") or {}).get("unavailable"):
            # The order system failed AT COMMIT (after routing, maybe after a human approved). Nothing is promised to the customer;
            # the ticket goes to the human queue with the proposal and any decision so a person can finish it.
            reasons = list(state.get("reasons") or []) + ["order_system_unavailable"]
            holding = holding_reply([], reasons, sentiment=(state.get("triage") or {}).get("sentiment", "neutral"))
            handed = human_review_node({**state, "route": "human_review", "reasons": reasons, "route_reasons": ["order_system_unavailable"],
                                        "holding_reply": holding, "escalation_codes": []})
            return {**handed, "log": [f"reply: order system unavailable at commit -> {handed['log'][0]}"]}
        reply = compose_reply(state)
        issues = check_reply(state, reply)
        if issues and (state.get("policy") or {}).get("answer"):
            # The code-built part is trustworthy; the model-written policy sentence is what usually trips the checks. Drop just that.
            lean = compose_reply(state, include_policy=False)
            if not check_reply(state, lean):
                issues, reply = issues + ["policy text dropped"], lean
        if check_reply(state, reply):                           # still unsafe: minimal safe text, and the log says why
            reply = state.get("holding_reply") or "Thanks for contacting us. A colleague will review your request and follow up shortly."
        res = state.get("action_result") or {}
        if res.get("refused"):
            status = "needs_higher_approval"
        elif res.get("denied"):
            status = "denied_by_reviewer"
        elif state.get("route") == "approval":
            status = "resolved_after_approval"
        else:
            status = "resolved"
        return {"reply": reply, "status": status, "log": [f"reply: status={status} issues={issues or 'none'}"]}

    nodes = {"triage": triage_node, "policy_check": policy_check_node, "action_plan": action_plan_node, "route": route_node,
             "human_review": human_review_node, "approval": approval_node, "commit": commit_node, "reply": reply_node}
    kinds = {"policy_check": "retriever", "action_plan": "tool", "commit": "tool"}
    return {name: traced_node(f"node:{name}", kind=kinds.get(name, "agent"))(fn) for name, fn in nodes.items()}


# ---- edges (pure; read what a node already wrote) --------------------------------------------------------------------
def after_triage(state: TicketState) -> str:
    hard = [r for r in state.get("reasons", []) if r not in SOFT_REASONS]
    return "human_review" if hard else "policy_check"


def after_route(state: TicketState) -> str:
    return {"human_review": "human_review", "approval": "approval"}.get(state.get("route"), "commit")


# --------------------------------------------------------------------------------------------------------------------
# Graph, checkpointing, run/resume
# --------------------------------------------------------------------------------------------------------------------
def _require_langgraph() -> None:
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed. Run: pip install langgraph langgraph-checkpoint-sqlite  (see README.md)")


def build_ticket_graph(deps: Deps, checkpointer: Any = None):
    _require_langgraph()
    nodes = make_nodes(deps)
    g = StateGraph(TicketState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", after_triage, {"human_review": "human_review", "policy_check": "policy_check"})
    g.add_edge("policy_check", "action_plan")
    g.add_edge("action_plan", "route")
    g.add_conditional_edges("route", after_route, {"human_review": "human_review", "approval": "approval", "commit": "commit"})
    g.add_edge("approval", "commit")
    g.add_edge("commit", "reply")
    g.add_edge("reply", END)
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_sqlite_checkpointer(db_path: str):
    _require_langgraph()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def thread_config(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": f"ticket-{ticket_id}"}}


def seed_state(*, ticket_id: str, customer_id: Optional[str], text: str, channel: str = "chat", created_at: str = "",
               history: Optional[list] = None) -> TicketState:
    # Every control field is reset explicitly. A checkpointed thread MERGES a new input into its previous state, so without this a
    # re-run of the same ticket would inherit the last run's route, human decision and action result -- e.g. commit as a manager
    # who approved yesterday. Only `log` (the audit trail) is allowed to carry over.
    return {"ticket_id": ticket_id, "customer_id": customer_id or "", "channel": channel, "created_at": created_at or pr.REFERENCE_NOW.isoformat(),
            "text": text, "history": history or [], "triage": None, "reasons": [], "escalation_codes": [], "policy": None, "facts": None,
            "proposal": None, "route": "", "route_reasons": [], "holding_reply": "", "human_decision": None, "human_notes": None,
            "approver_role": None, "approved_amount_inr": None, "action_result": None, "handoff": None, "reply": "", "status": "",
            "next_agent": "", "steps": 0, "log": []}


def _run_tags(extra: tuple | list = ()) -> list[str]:
    return ["shopsense", f"model:{DEFAULT_MODEL}", f"prompt:{PROMPT_VERSION}", *extra]


def run_ticket(graph: Any, state: TicketState, *, tags: tuple | list = (), on_result: Optional[Callable[[dict], None]] = None) -> tuple[dict, dict]:
    """Run one ticket inside a root trace span (session id = ticket id, hashed customer id, model + prompt version tags). The
    result carries `_run`: what the run cost and whether it was served degraded (a fallback model, a retried or unreachable order
    system ...), so a rate-limited night does not look like a normal one. `on_result(result)` runs INSIDE the trace, so a check made on
    the finished state (the M8 output guardrail) is a span of the same trace and its events count toward `degraded`."""
    config = thread_config(state["ticket_id"])
    with get_tracer().run("ticket.run", session_id=state["ticket_id"], user_id=state.get("customer_id") or "", tags=_run_tags(tags),
                          input={"ticket_id": state["ticket_id"], "channel": state.get("channel"), "text": state.get("text", "")},
                          metadata={"model": DEFAULT_MODEL, "prompt_version": PROMPT_VERSION, "channel": state.get("channel")}) as run:
        result = dict(graph.invoke(state, config))
        if on_result is not None:
            on_result(result)
        out = outcome_of(result)
        run.update(output={"route": out["route"], "reply": out["answer"]},
                   metadata={"route": "paused" if out["paused"] else out["route"], "codes": out["escalation_codes"]})
        if run.stats is not None:
            result["_run"] = run.stats.as_dict()
    return result, config


_ROLE_RANK = {"system": 0, "ai_agent": 0, "manager": 1, "executive": 2, "trust_safety": 2}
_NEEDED_RANK = {"manager": 1, "executive": 2}


def approval_blocker(graph: Any, config: dict, *, decision: str, approver_role: str, amount_inr: Optional[float] = None) -> Optional[dict]:
    """A role that plainly lacks the authority for the pending refund is turned away BEFORE the ticket resumes, so a wrong-role
    attempt leaves the ticket paused for the right person instead of ending it. (Resuming would run `commit`, which refuses the
    money but then closes the ticket as handed on, and nobody could approve it afterwards.) `commit` still enforces the limit
    itself; this only keeps a mistaken attempt from consuming the pause. Returns None when the attempt may go ahead, including
    when the pending request cannot be read (the commit check is then the only gate, as before)."""
    try:
        tasks = graph.get_state(config).tasks
        payload = tasks[0].interrupts[0].value if tasks else None
        if not isinstance(payload, dict) or payload.get("kind") != "ticket_approval":
            return None
        needed = _NEEDED_RANK.get(payload.get("needs_role"), 0)
        proposal = payload.get("proposal") or {}
    except Exception:                                                      # noqa: BLE001 -- unreadable state: let the commit decide
        return None
    if decision == "deny":
        return None                                                       # declining a refund moves no money
    if decision == "modify" and amount_inr is not None:                   # a lowered amount may fall inside a lower role's limit
        lowered = min(float(amount_inr), float(proposal.get("amount_inr") or amount_inr))
        needed = _NEEDED_RANK.get(ROLE_BY_TIER.get(refund_tier_for_amount(lowered)), 0)
    if _ROLE_RANK.get(approver_role, 0) >= needed:
        return None
    return {"status": "needs_higher_approval", "paused": True, "reply": "",
            "action_result": {"committed": False, "refused": "above_role_limit",
                              "detail": {"role": approver_role, "needs_role": payload.get("needs_role"), "ticket_still_paused": True}}}


def resume_ticket(graph: Any, config: dict, *, decision: str, notes: str = "", approver_role: str = "manager",
                  amount_inr: Optional[float] = None) -> dict:
    _require_langgraph()
    tid = (config.get("configurable", {}).get("thread_id") or "").removeprefix("ticket-")
    blocked = approval_blocker(graph, config, decision=decision, approver_role=approver_role, amount_inr=amount_inr)
    if blocked is not None:
        add_event("approval.wrong_role", ticket_id=tid, role=approver_role)
        return blocked
    with get_tracer().run("ticket.resume", session_id=tid, user_id="", tags=_run_tags(["resume", f"decision:{decision}"]),
                          metadata={"decision": decision, "approver_role": approver_role}) as run:
        result = dict(graph.invoke(Command(resume={"decision": decision, "notes": notes, "approver_role": approver_role, "amount_inr": amount_inr}), config))
        run.update(metadata={"route": outcome_of(result)["route"]})
        if run.stats is not None:
            result["_run"] = run.stats.as_dict()
    return result


def outcome_of(result: dict) -> dict:
    """Normalise a graph result (finished or paused) to {route, answer, escalation_codes, paused} - the shape the golden
    scorer and the demos read."""
    paused = "__interrupt__" in result
    if paused:
        payload = result["__interrupt__"][0].value
        return {"route": "refund_approval", "answer": payload.get("holding_reply", ""), "escalation_codes": payload.get("escalation_codes", []), "paused": True}
    route = {"human_review": "human_review", "approval": "refund_approval"}.get(result.get("route"), "resolve")
    return {"route": route, "answer": result.get("reply", ""), "escalation_codes": result.get("escalation_codes", []), "paused": False}


def run_info_of(result: dict) -> dict:
    """What `run_ticket` recorded about the run itself: {llm_calls, tokens_in, tokens_out, cost_usd, models, degraded, degraded_reasons,
    events}. Empty when the ticket was not run through `run_ticket` (e.g. the offline simulators)."""
    return dict(result.get("_run") or {})
