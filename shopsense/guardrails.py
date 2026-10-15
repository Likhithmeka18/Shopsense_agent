"""
ShopSense guardrails (Milestone 8): the last check before a reply leaves the system.

Independent of whether an answer is *good* : the graph decides what to do; this layer only asks "is the text about
to be sent to a customer safe to send?". It is pure Python (no model, no network), so it is fast, testable and cannot be
talked out of its job. Two guardrails carry the capstone brief, the rest are the same idea applied to neighbouring failures:

  refund cap          no refund at or above INR 2,000 may be committed by the *system* role, and no ticket may take the auto route
                      with a proposal at or above it. `action_tools` already refuses such a write; this re-checks the OUTCOME, so a
                      bug anywhere upstream is still caught before it becomes a customer-visible promise.
  policy fabrication  the only model-written text that reaches a customer is the policy sentence. Every money amount, duration
                      ("15 days", "3-5 business days", "48 hours", "two weeks") and percentage in it must appear in what retrieval
                      actually returned (or in the structured order facts). A number the sources do not contain is invented.
  approval claims     "I've approved / refunded ..." is only allowed when money was actually committed (or already approved).
  flag disclosure     the account's abuse flag / trust-and-safety status is never mentioned to the customer.
  injection echo      text from a hostile ticket is never repeated back.

Tuning (*tune thresholds against the golden set, not intuition*): every check can be switched on or off in
`GuardrailConfig`, and `shopsense/eval/guardrail_eval.py` scores any configuration in BOTH directions -- false approvals (a bad
reply let through) and false rejections (a good reply blocked) -- on labelled cases plus every reply the real pipeline produces.
Too strict a fabrication check blocks honest paraphrases and floods the human queue; too loose lets an invented number through.

This is a pattern layer, not a classifier. It will not catch a fabricated claim that contains no number ("we always refund
within the same hour" is caught only through its '1 hour'), a wrong product category, or a subtly wrong policy name. It cannot judge
whether a *grounded* number answers the customer's question; that is what the golden set is for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .tools.policy_rules import AUTO_APPROVE_BELOW_INR, EXECUTIVE_FROM_INR

# ---- shared patterns (the ticket graph's reply critic uses the same two) ------------------------------------------------------
FLAG_WORDS_RE = re.compile(r"\b(flag(ged)?|abuse|trust\s*(&|and)\s*safety|restricted|blacklist\w*|watch\s*list)\b", re.I)
APPROVAL_CLAIM_RE = re.compile(
    r"\b(i\s*(have|'ve)\s+(approved|issued|processed|refunded)|(has|have)\s+been\s+(approved|issued|processed|refunded)|is\s+approved|"
    r"refund\s+(is|was)\s+(approved|issued|processed)|we\s*(have|'ve)\s+(approved|refunded))\b", re.I)
INJECTION_ECHO_RE = re.compile(r"(?i)\b(system note|ignore (all )?previous|override)\b")                 # broad: also fires on "I can't override our policy"
INJECTION_ECHO_NARROW_RE = re.compile(r"(?i)\b(system note|ignore (all |any )?(previous|prior|above)( instructions)?|you are now|new instructions?)\b")
# Narrow flag disclosure: words that are about THE CUSTOMER'S ACCOUNT, not the word "flag" anywhere ("I'll flag this to the courier" is fine).
FLAG_DISCLOSURE_NARROW_RE = re.compile(
    r"(?i)\b(abuse|trust\s*(&|and)\s*safety|blacklist\w*|watch\s*list)\b|"
    r"\b(account|profile)\b[^.!?]{0,60}\b(flag(ged)?|restricted|marked|blocked)\b|"
    r"\b(flag(ged)?|restricted|marked)\b[^.!?]{0,40}\b(account|profile)\b|"
    r"\byou('ve| have| are| were)?\s+(been\s+)?(flag(ged)?|marked|restricted)\b|"
    r"\bflag(ged)?\s+(your|the)\s+(orders?|returns?|history|refunds?)\b")

_AMOUNT_RE = re.compile(r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", re.I)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60, "ninety": 90}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(_WORD_NUM) + r")"
_UNIT = r"(business\s+days?|working\s+days?|calendar\s+days?|days?|hours?|hrs?|weeks?|months?|years?)"
_DURATION_RE = re.compile(rf"\b{_NUM}(?:\s*(?:-|–|to)\s*{_NUM})?(?:\s*\(\d+\))?[\s-]*{_UNIT}\b", re.I)
_HOURS_PER = {"day": 24, "week": 168, "month": 720, "year": 8760, "hour": 1}      # business days convert to nothing: they match only themselves

# Severity: a "block" replaces the reply; a "warn" is logged and traced but the reply goes out.
BLOCK, WARN = "block", "warn"


@dataclass(frozen=True)
class GuardrailConfig:
    """Which checks run, and how hard. `balanced()` is the default; `loose()` / `strict()` exist so the evaluation can show what the
    default is protecting against in each direction."""
    refund_cap: bool = True
    auto_refund_cap_inr: float = AUTO_APPROVE_BELOW_INR
    amounts: bool = True
    durations: bool = True
    percentages: bool = True
    approval_claims: bool = True
    flag_disclosure: bool = True
    injection_echo: bool = True
    # Convert units before comparing ("2 weeks" == "14 days"; "48 hours" == "2 days"). Off = literal "2 weeks" must appear.
    convert_units: bool = True
    # Accept a number grounded ONLY as a bare integer somewhere in the structured facts (window_days=15 backs "15-day window").
    accept_structured_integers: bool = True
    # Strict mode: any digit sequence of 2+ digits in the model text that is not grounded is flagged (catches "30% off" style claims
    # the other patterns miss, at the price of blocking things like "Section 14").
    strict_numbers: bool = False
    # "narrow": only wording that is about the customer's account / a hostile ticket. "broad": the word anywhere (graph critic's regex).
    flag_mode: str = "narrow"
    injection_mode: str = "narrow"

    @staticmethod
    def balanced() -> "GuardrailConfig":
        """The default, chosen from the evaluation in docs/GUARDRAIL_EVALUATION.md (see `PRESETS`)."""
        return GuardrailConfig()

    @staticmethod
    def loose() -> "GuardrailConfig":
        """Only money amounts, the refund cap and approval claims -- roughly what the graph's own critic did before M8."""
        return GuardrailConfig(durations=False, percentages=False, flag_disclosure=False, injection_echo=False)

    @staticmethod
    def strict() -> "GuardrailConfig":
        return GuardrailConfig(convert_units=False, accept_structured_integers=False, strict_numbers=True, flag_mode="broad", injection_mode="broad")

    @staticmethod
    def broad() -> "GuardrailConfig":
        """First draft: every check on, wide patterns, no strict-number mode."""
        return GuardrailConfig(flag_mode="broad", injection_mode="broad")


@dataclass
class Flag:
    code: str
    severity: str
    detail: str

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class GuardrailResult:
    flags: list[Flag] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == BLOCK for f in self.flags)

    @property
    def passed(self) -> bool:
        return not self.flags

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.flags]

    def as_dict(self) -> dict:
        return {"passed": self.passed, "blocked": self.blocked, "flags": [f.as_dict() for f in self.flags]}


@dataclass
class ReplyContext:
    """Everything the guardrail may compare a reply against. All fields are optional; a check with nothing to compare against only
    flags what is unconditionally unsafe (e.g. a flag word), never what it merely cannot verify."""
    reply: str
    model_text: Optional[str] = None                 # the model-written part of the reply (None = treat the whole reply as model text)
    source_texts: list[str] = field(default_factory=list)      # what retrieval returned (chunk text) -- the ground truth for policy figures
    policy_answer: str = ""                          # the model's own policy answer (also quotable: it already passed the unknown check)
    facts: dict = field(default_factory=dict)
    proposal: dict = field(default_factory=dict)
    action_result: dict = field(default_factory=dict)
    route: str = ""                                  # auto | approval | human_review
    refund_records: list[dict] = field(default_factory=list)   # ledger records for THIS ticket: {action, amount_inr, approver_role}
    refund_already_approved: bool = False
    injection_attempt: bool = False


# ---- helpers -----------------------------------------------------------------------------------------------------------------
def _to_num(tok: str) -> float:
    t = tok.lower()
    return float(_WORD_NUM[t]) if t in _WORD_NUM else float(t)


def _unit_family(unit: str) -> str:
    """'business day' / 'working day' are their own family: a policy that says "5 business days" does not back "5 days" (a model that
    confuses a processing time with a return window has invented a policy)."""
    u = unit.lower()
    if u.startswith(("business", "working")):
        return "bday"
    for k in ("hour", "hr", "week", "month", "year", "day"):
        if u.startswith(k):
            return "hour" if k == "hr" else k
    return "day"


def amounts_in(text: str) -> list[float]:
    out = []
    for m in _AMOUNT_RE.finditer(text or ""):
        raw = m.group(1).rstrip(",").rstrip(".")
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def durations_in(text: str) -> list[tuple[float, str, str]]:
    """(value, unit family, literal) for every duration phrase; a range yields both ends."""
    out = []
    for m in _DURATION_RE.finditer(text or ""):
        fam = _unit_family(m.group(3))
        for tok in (m.group(1), m.group(2)):
            if tok:
                out.append((_to_num(tok), fam, m.group(0).strip()))
    return out


def percentages_in(text: str) -> list[float]:
    return [float(m.group(1)) for m in _PERCENT_RE.finditer(text or "")]


def _structured_numbers(*objs) -> set[float]:
    """Every number that appears anywhere in the structured context (proposal / facts / action result)."""
    found: set[float] = set()

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            found.add(float(o))
        elif isinstance(o, str):
            for m in re.finditer(r"\d+(?:\.\d+)?", o.replace(",", "")):
                found.add(float(m.group(0)))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)

    for o in objs:
        walk(o)
    return found


def _support_texts(ctx: ReplyContext) -> list[str]:
    """What a model-written figure may be checked against: the retrieved policy text. A model's own answer cannot vouch for itself, so
    `policy_answer` is used ONLY when the caller could not supply the retrieved text (a weaker check, and the evaluation says so)."""
    return list(ctx.source_texts) if ctx.source_texts else ([ctx.policy_answer] if ctx.policy_answer else [])


def _grounded_amounts(ctx: ReplyContext) -> set[float]:
    ok: set[float] = {float(AUTO_APPROVE_BELOW_INR), float(EXECUTIVE_FROM_INR), 500.0}      # the published tier boundaries and goodwill cap
    for t in _support_texts(ctx):
        ok.update(amounts_in(t))
    for k in ("amount_inr", "value_inr", "cap_inr"):
        if ctx.proposal.get(k) is not None:
            ok.add(float(ctx.proposal[k]))
    facts = ctx.facts or {}
    for src in ((facts.get("order") or {}).get("total_amount_inr"), (facts.get("refund") or {}).get("amount_inr"), ctx.action_result.get("amount_inr")):
        if src is not None:
            try:
                ok.add(float(src))
            except (TypeError, ValueError):
                pass
    return ok


def _same_duration(v1: float, f1: str, v2: float, f2: str, convert: bool) -> bool:
    if "bday" in (f1, f2) or not convert:
        return f1 == f2 and abs(v1 - v2) < 1e-9
    h1, h2 = v1 * _HOURS_PER[f1], v2 * _HOURS_PER[f2]
    # months and years are not exact ("12 months" vs "one year", "30 days" vs "1 month"): allow 2% there and nowhere else, so
    # "two weeks" still does NOT back "15 days" (a 7% difference is a different policy number).
    tol = 0.02 if {f1, f2} & {"month", "year"} else 1e-6
    return abs(h1 - h2) <= tol * max(h1, h2)


# ---- the individual checks ---------------------------------------------------------------------------------------------------
def check_refund_cap(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    flags = []
    per_order: dict[str, float] = {}
    for rec in ctx.refund_records:
        role = str(rec.get("approver_role", "system")).lower()
        if role in ("system", "ai_agent"):
            order = str(rec.get("order_id", ""))
            per_order[order] = per_order.get(order, 0.0) + float(rec.get("amount_inr") or 0)       # RA-2.1: the tier is decided on the order's TOTAL
    for order, total in per_order.items():
        if total >= cfg.auto_refund_cap_inr:
            flags.append(Flag("refund_cap.committed_over_cap", BLOCK,
                              f"the system role committed INR {total:,.0f} in total on {order or 'one order'}; the cap is INR {cfg.auto_refund_cap_inr:,.0f}"))
    p = ctx.proposal or {}
    if ctx.route == "auto" and p.get("type") in ("refund", "decide_pending_refund", "replace"):
        amt = float(p.get("amount_inr") or p.get("value_inr") or 0)
        if amt >= cfg.auto_refund_cap_inr:
            flags.append(Flag("refund_cap.auto_route_over_cap", BLOCK, f"auto-routed {p.get('type')} of INR {amt:,.0f} (cap INR {cfg.auto_refund_cap_inr:,.0f})"))
    return flags


def check_amounts(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    grounded = _grounded_amounts(ctx)
    flags = []
    for amt in amounts_in(ctx.reply):
        if not any(abs(amt - g) < 0.5 for g in grounded):
            flags.append(Flag("fabrication.amount", BLOCK, f"INR {amt:,.0f} is not in the order facts or the retrieved policy"))
    return flags


def check_durations(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    text = ctx.model_text if ctx.model_text is not None else ctx.reply
    claims = durations_in(text)
    if not claims:
        return []
    support = [(v, f) for t in _support_texts(ctx) for v, f, _ in durations_in(t)]
    structured = _structured_numbers(ctx.proposal, ctx.facts, ctx.action_result) if cfg.accept_structured_integers else set()
    flags, seen = [], set()
    for v, fam, literal in claims:
        if (v, fam) in seen:
            continue
        seen.add((v, fam))
        ok = any(_same_duration(v, fam, sv, sf, cfg.convert_units) for sv, sf in support)
        if not ok and v in structured and fam != "bday":
            ok = True
        if not ok:
            flags.append(Flag("fabrication.duration", BLOCK, f"'{literal}' does not appear in the retrieved policy text"))
    return flags


def check_percentages(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    text = ctx.model_text if ctx.model_text is not None else ctx.reply
    support = {p for t in _support_texts(ctx) for p in percentages_in(t)}
    return [Flag("fabrication.percentage", BLOCK, f"{p:g}% does not appear in the retrieved policy text")
            for p in dict.fromkeys(percentages_in(text)) if p not in support]


def check_strict_numbers(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    text = ctx.model_text if ctx.model_text is not None else ctx.reply
    corpus = " ".join(_support_texts(ctx)).replace(",", "")
    structured = _structured_numbers(ctx.proposal, ctx.facts, ctx.action_result)
    flags = []
    for m in re.finditer(r"(?<![\w,-])(?:\d{1,3}(?:,\d{3})+|\d{2,})(?![\w-])", text):
        raw = m.group(0).replace(",", "")
        if raw not in corpus and float(raw) not in structured and not re.fullmatch(r"(ORD|TCK|REF|TRK)-?\d+", raw):
            flags.append(Flag("fabrication.number", BLOCK, f"'{m.group(0)}' is not in the retrieved policy text"))
    return flags


def check_approval_claims(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    if not APPROVAL_CLAIM_RE.search(ctx.reply):
        return []
    res, p = ctx.action_result or {}, ctx.proposal or {}
    already = p.get("type") == "inform_refund_status" and bool(((ctx.facts or {}).get("refund") or {}).get("approved")) or ctx.refund_already_approved
    done = (res.get("committed") and not res.get("denied")) or already
    return [] if done else [Flag("approval_claim.not_committed", BLOCK, "the reply says a refund/approval happened but nothing was committed")]


def check_flag_disclosure(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    m = (FLAG_WORDS_RE if cfg.flag_mode == "broad" else FLAG_DISCLOSURE_NARROW_RE).search(ctx.reply)
    return [Flag("disclosure.account_flag", BLOCK, f"the reply mentions '{m.group(0)}'")] if m else []


def check_injection_echo(ctx: ReplyContext, cfg: GuardrailConfig) -> list[Flag]:
    if ctx.injection_attempt and (INJECTION_ECHO_RE if cfg.injection_mode == "broad" else INJECTION_ECHO_NARROW_RE).search(ctx.reply):
        return [Flag("injection.echo", BLOCK, "the reply repeats text from a hostile ticket")]
    return []


def evaluate(ctx: ReplyContext, cfg: Optional[GuardrailConfig] = None) -> GuardrailResult:
    """Run every enabled check. Never raises: a guardrail that can crash is a guardrail that gets disabled."""
    cfg = cfg or GuardrailConfig.balanced()
    checks = []
    if cfg.refund_cap:
        checks.append(check_refund_cap)
    if cfg.amounts:
        checks.append(check_amounts)
    if cfg.durations:
        checks.append(check_durations)
    if cfg.percentages:
        checks.append(check_percentages)
    if cfg.strict_numbers:
        checks.append(check_strict_numbers)
    if cfg.approval_claims:
        checks.append(check_approval_claims)
    if cfg.flag_disclosure:
        checks.append(check_flag_disclosure)
    if cfg.injection_echo:
        checks.append(check_injection_echo)
    result = GuardrailResult()
    for check in checks:
        try:
            result.flags.extend(check(ctx, cfg))
        except Exception as e:  # noqa: BLE001 -- fail CLOSED: an erroring check blocks the reply rather than waving it through
            result.flags.append(Flag(f"guardrail_error.{check.__name__}", BLOCK, f"{type(e).__name__}: {str(e)[:100]}"))
    return result


# ---- input side ---------------------------------------------------------------------------------------------------------------
# ---- tidy model text before a customer sees it (not a safety check: it removes formatting artefacts) -----------------------------
_CITATION_MARK_RE = re.compile(r"[\u3010\[]\s*\d+(?:\s*[:,-]\s*\d+)*\s*(?:\u2020[^\u3011\]]*)?[\u3011\]]")       # 【24】 【3:1†source】 [24] -- some models add them
_ODD_SPACES_RE = re.compile(r"[\u00a0\u202f\u2009\u200a]")                                                                  # no-break / thin spaces -> a normal space
_TYPOGRAPHY = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}                  # hyphens, dashes, minus -> "-"
_TYPOGRAPHY.update({ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"', ord("\u2026"): "..."})


def clean_model_text(text: str) -> str:
    """Remove citation markers the model copied from its context (`【24】`), exotic spaces and typographic punctuation. The (Source: ...) line the graph appends is
    the customer-facing citation; a bare chunk number means nothing to a customer. Never changes a figure, so it cannot affect grounding."""
    if not text:
        return text or ""
    out = _CITATION_MARK_RE.sub("", text)
    out = _ODD_SPACES_RE.sub(" ", out)
    out = out.translate(_TYPOGRAPHY)                         # curly quotes / non-breaking hyphens: plain ASCII prints the same on every console
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)          # "word 【24】." -> "word ." -> "word."
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


MAX_MESSAGE_CHARS = 2000


def check_input(text: str) -> list[str]:
    """Flags about the customer's message that the caller should SURFACE (the graph itself already routes hostile text to a person)."""
    from .core.safety_net import detect_injection_attempt
    flags = []
    if detect_injection_attempt(text or ""):
        flags.append("prompt_injection_attempt")
    if len(text or "") > MAX_MESSAGE_CHARS:
        flags.append("message_too_long")
    return flags


SAFE_FALLBACK_REPLY = ("Thanks for contacting us. I want to be sure what I tell you is right, so I've passed your request to a team member "
                       "who will review it and get back to you within one business day.")


PRESETS = {"loose": GuardrailConfig.loose, "strict": GuardrailConfig.strict, "broad": GuardrailConfig.broad, "balanced": GuardrailConfig.balanced}
