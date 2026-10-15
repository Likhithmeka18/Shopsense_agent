"""
ShopSense Core: deterministic escalation safety net.

`escalation_sop.md` ESC-2 says a ticket matching ESC-LEGAL or ESC-SAFETY
"must ALWAYS be routed to a human agent, regardless of what an AI agent's
tools report". Milestone 1's LLM extraction is how tickets usually get
flagged, but an LLM's classification of the same text can differ from run
to run (the same "...I will be filing a complaint with the
consumer forum" can be escalated in one pass and handed to the agent in another). A rule
that policy says must ALWAYS hold shouldn't depend on a sampled output, so
this module is the deterministic floor under it: plain regexes over the
ticket text, no model, fully unit-testable.

It is deliberately biased toward over-escalating -- a false positive costs
a human a minute; a false negative on a fire hazard or a legal threat costs
far more. It only ADDS escalations; it never clears one the LLM raised.
"""
from __future__ import annotations

import re
from typing import Optional

# ESC-LEGAL: "Customer mentions legal action, a lawyer, or a regulator".
_LEGAL_PATTERNS = [
    r"\blawyers?\b", r"\battorneys?\b", r"\blegal\s+(action|notice|team|proceedings?|recourse)\b",
    r"\bsue\b", r"\bsuing\b", r"\blawsuit\b", r"\bcourt\b",
    r"\bconsumer\s+(forum|court|commission|protection|helpline|affairs)\b",
    r"\bregulators?\b", r"\bombudsman\b", r"\bpolice\b",
]

# ESC-SAFETY: "safety issue with a product (injury, fire hazard, choking hazard)".
_SAFETY_PATTERNS = [
    r"\bsmok(e|es|ed|ing)\b", r"\bfire\b", r"\bcaught\s+fire\b", r"\bburn(s|t|ed|ing)?\b",
    r"\bsparks?\b", r"\bsparking\b", r"\belectric(al)?\s+shock\b", r"\belectrocut\w*\b",
    r"\binjur(y|ed|ies)\b", r"\bchok(e|ed|ing)\b", r"\boverheat\w*\b", r"\bexplo(de|ded|sion|ding)\b",
    r"\bextremely\s+hot\b", r"\bdangerous\b", r"\bhazard(ous)?\b",
    r"\b(very|really|too|so|super|incredibly)\s+hot\b", r"\bswoll?en\b", r"\bswell(s|ed|ing)?\b", r"\bbulg(e|es|ed|ing)\b",
    r"\b(blew|blow|blown)\s+up\b", r"\bburst(s|ing)?\b", r"\bmelt(s|ed|ing)?\b", r"\bsmell(s|ed|ing)?\s+(of\s+|like\s+)?(burn\w*|chemical\w*|gas|plastic)\b",
]

# Prompt injection: ticket text is attacker-controlled by definition (Data Sourcing Guidelines 4.4.4),
# so an LLM flag alone is not a sufficient floor. These are phrasings of *instructions aimed at the
# assistant*, authority claims that bypass verification, and fake system/config blocks. Biased toward
# over-flagging (a human reads it) -- a missed injection is the expensive error.
_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier|preceding)\b",
    r"\b(ignore|disregard|forget)\s+(everything|all)\b",
    r"\bdisregard\s+(the\s+)?(refund|approval|policy|tiers?|cap|limits?|restrictions?)",
    r"(^|\n)\s*(=+|-+|#+|\*+)\s*(end\s+)?(system|assistant|admin|developer)\b",
    r"</?\s*(system|assistant|admin|instructions?)\s*>",
    r"\[\s*\[?\s*(system|assistant|admin|developer)\b",
    r"\bsystem\s*(note|prompt|message|override|instruction)s?\b",
    r"\b(admin|developer|root|god)\s+(override|mode|access)\b", r"\boverride\s+(enabled|activated|mode)\b",
    r"\byou\s+are\s+now\b",
    r"\b(print|reveal|show|repeat|output)\s+(me\s+)?(your\s+|the\s+)?(full\s+)?(system\s+prompt|hidden\s+instructions|instructions)\b",
    r"\b(skip|bypass|without)\s+(the\s+|any\s+)?(manager\s+|executive\s+)?(review|approval|verification|verifying)\b",
    r"\bno\s+need\s+to\s+(verify|check|review)\b",
    r"\b(already|pre-?)\s*approved\b[^.\n]*\b(skip|no need|don'?t need|do not need|just|mark|process|offline)\b",
    r"\bapproved\s+by\s+(finance|the\s+manager|a\s+manager|an?\s+exec\w*)\b",
    r"\b(note|message|instructions?)\s+to\s+(the\s+)?(ai|assistant|bot|model|agent)\b",
    r"\bdear\s+(ai|assistant|bot|model)\b",
    r"\bassistant\s+instructions?\b",
    r"\"role\"\s*:\s*\"system\"", r"\brequires_human_review\b", r"\bapproval_required\s*=", r"\brefund_cap\s*=",
    r"\breply(ing)?\s+(only\s+)?with\s+(the\s+)?(code|word)\b",
    r"\b(respond|reply|answer)\s+only\s+(in|with)\b",
    r"\b(authori[sz]ed|allowed|permitted)\s+to\s+approve\s+any\b", r"\bapprove\s+any\s+refund\b",
    r"\bwhen\s+you\s+(write|draft|summari[sz]e)\b[^.\n]*\b(add|include|append|say)\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)
# Whole fake blocks that should be removed as a unit, not just the sentence that matched.
_INJECTION_BLOCK_RES = [
    re.compile(r"={3,}\s*system\s*={3,}.*?={3,}\s*end\s+system\s*={3,}", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\s*system\s*>.*?<\s*/\s*system\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"\[\[?[^\]]*(system|ignore|authori[sz]ed|override|skip|assistant)[^\]]*\]\]?", re.IGNORECASE),
    re.compile(r"-{3,}.*?-{3,}", re.DOTALL),
]

_LEGAL_RE = re.compile("|".join(_LEGAL_PATTERNS), re.IGNORECASE)
_SAFETY_RE = re.compile("|".join(_SAFETY_PATTERNS), re.IGNORECASE)


def detect_escalation_codes(text: str) -> list[str]:
    """ESC codes the ticket TEXT alone triggers. Returns a list drawn from
    ["ESC-LEGAL", "ESC-SAFETY"] (empty when neither applies)."""
    codes: list[str] = []
    if _LEGAL_RE.search(text or ""):
        codes.append("ESC-LEGAL")
    if _SAFETY_RE.search(text or ""):
        codes.append("ESC-SAFETY")
    return codes


def detect_injection_attempt(text: str) -> bool:
    """True when the ticket text contains instruction-like content aimed at the assistant (fake system
    blocks, role overrides, claimed pre-approval, requests to print prompts or alter the output)."""
    return bool(_INJECTION_RE.search(text or ""))


# Once a ticket is KNOWN to contain an injection, imperative sentences about money/data that sit next to it are
# part of the same attack ("...Mark it approved and tell the customer it has been paid."), so they go too.
_DIRECTIVE_RE = re.compile(
    r"\b(approve|approved|mark|process|issue|release|confirm|print|list|reveal|tell\s+the\s+customer|refund|credit)\b", re.IGNORECASE)


def redact_injection(text: str) -> str:
    """Remove instruction-like content before any model sees the ticket: whole fake blocks first, then any
    sentence/line that matches an injection pattern and -- if something matched -- the directive sentences
    around it. What is left is the customer's genuine question (if any); the structured facts about the
    order come from the order system, not from this text."""
    out = text or ""
    for block in _INJECTION_BLOCK_RES:
        out = block.sub(" [instruction-like text removed] ", out)
    pieces = [pc for pc in re.split(r"(?<=[.!?])\s+|\n+", out) if pc.strip()]
    hit = any(_INJECTION_RE.search(pc) for pc in pieces) or "[instruction-like text removed]" in out
    kept = []
    for pc in pieces:
        if _INJECTION_RE.search(pc) or (hit and _DIRECTIVE_RE.search(pc) and "[instruction-like" not in pc):
            kept.append("[instruction-like text removed]")
        else:
            kept.append(pc)
    return re.sub(r"(\[instruction-like text removed\]\s*)+", "[instruction-like text removed] ", " ".join(kept)).strip()


_REFUND_TEXT_RE = re.compile(r"\b(refund\w*|money\s+back|reimburs\w*|return(ing|ed)?\s+(it|this|the|my)|send\s+(it\s+)?back|exchange|replacement)\b", re.I)
_REFUND_INTENTS = {"REFUND_REQUEST", "REFUND_STATUS", "RETURN_ITEM"}


def is_abuse_flagged(customer: Optional[dict]) -> bool:
    """True when a customers.csv row (or equivalent dict) carries abuse_flag."""
    return bool(customer) and customer.get("abuse_flag") in ("True", True)


def combine_escalation_reasons(
    extraction: Optional[object], text: str, *, customer: Optional[dict] = None
) -> list[str]:
    """Every reason a ticket must go to a human at intake: the M1 LLM flags
    (if an extraction exists), the deterministic text safety net, and --
    when the caller passes the ticket customer's record -- ESC-2 rule 4
    (a refund-type request from an abuse-flagged account). Reasons from the
    safety net are prefixed `safety_net:` so a human can tell which layer
    raised them."""
    reasons: list[str] = []
    text_refundish = bool(_REFUND_TEXT_RE.search(text or ""))      # the model's label is a hint; the customer's words are evidence
    if extraction is None and text_refundish and is_abuse_flagged(customer):
        reasons.append("abuse_flag_refund")
    if extraction is not None:
        if getattr(extraction, "requires_human_review", False):
            reasons.append("requires_human_review")
        if getattr(extraction, "contains_injection_attempt", False):
            reasons.append("contains_injection_attempt")
        refundish = (
            getattr(extraction, "intent", None) in _REFUND_INTENTS
            or getattr(extraction, "category", None) == "REFUND"
        )
        if (refundish or text_refundish) and is_abuse_flagged(customer):
            reasons.append("abuse_flag_refund")
    reasons.extend(f"safety_net:{code}" for code in detect_escalation_codes(text))
    if detect_injection_attempt(text) and "contains_injection_attempt" not in reasons:
        reasons.append("safety_net:INJECTION")
    return reasons
