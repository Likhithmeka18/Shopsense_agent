"""
ShopSense Observability: redact customer data BEFORE it leaves the process.

    "A trace store is a database of your customers' words -- treat it like one."

Prompts and completions carry real customer text. Every span input/output is passed through `redact_obj` at the tracer
boundary (tracing.py), so neither the local trace file nor LangFuse ever receives an e-mail address, a phone number, a card
number, a date of birth, an API key ... Redacting in the LangFuse dashboard would be too late: the data would already have
been sent. Identifiers we NEED for debugging (ticket ids, order ids, refund ids) are deliberately not touched; the customer id
is replaced by a stable salted hash (`hash_id`), so "all runs for this customer" is still searchable without knowing who it is.

This is a regex floor, not a classifier: it catches the structured identifiers listed below, not a street address written in
prose. The README says so; do not read "redacted" as "anonymous".
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any

REDACTED = "[REDACTED]"
MAX_TEXT_CHARS = 2000                    # a trace is for debugging, not a transcript archive

# Order matters: the most specific patterns first, so a card number is not half-eaten by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api_key", re.compile(r"\b(?:gsk_[A-Za-z0-9]{10,}|sk-lf-[\w-]{8,}|pk-lf-[\w-]{8,}|sk-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-]{20,})")),
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")),
    ("upi", re.compile(r"\b[\w.\-]{2,}@(?:ok(?:axis|icici|hdfcbank|sbi)|ybl|ibl|axl|paytm|upi|apl|oksbi)\b", re.I)),
    ("card_last4", re.compile(r"(?i)\b(card|credit card|debit card)\s+(ending|ends|number ending)(\s+(in|with))?\s+\d{4}\b")),
    ("card", re.compile(r"(?<![\w-])(?:\d[ -]?){13,19}(?![\w-])")),
    ("aadhaar", re.compile(r"(?<![\w-])\d{4}[ -]\d{4}[ -]\d{4}(?![\w-])")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("phone", re.compile(r"(?<![\w+])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\w)")),
    ("dob", re.compile(r"(?i)\b(dob|d\.o\.b\.?|date of birth|born on)\b[:\s]+(\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")),
]

# A dict value under one of these keys is replaced whatever it contains.
_SENSITIVE_KEYS = re.compile(r"(?i)^(name|customer_name|email|phone|mobile|address|password|passwd|secret|token|api[_-]?key|authorization|cookie)$")


def redact_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for name, pat in _PATTERNS:
        if name == "card_last4":
            out = pat.sub(lambda m: f"{m.group(1)} {m.group(2)}{m.group(3) or ''} {REDACTED}", out)
        elif name == "dob":
            out = pat.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
        else:
            out = pat.sub(REDACTED, out)
    if len(out) > MAX_TEXT_CHARS:
        out = out[:MAX_TEXT_CHARS] + f"...[truncated {len(out) - MAX_TEXT_CHARS} chars]"
    return out


def redact_obj(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact strings inside dicts / lists / tuples. Unknown objects are stringified (and redacted) rather than
    sent as-is, because an arbitrary object's repr is exactly where a customer record could hide."""
    if _depth > 8:
        return REDACTED
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {str(k): (REDACTED if _SENSITIVE_KEYS.match(str(k)) and v not in (None, "") else redact_obj(v, _depth + 1)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [redact_obj(v, _depth + 1) for v in obj]
    return redact_text(str(obj))


def hash_id(value: Any, *, salt: str | None = None) -> str:
    """Stable, salted, one-way id for user/customer ids ('u_' + 12 hex). Same input -> same output, so runs for one customer
    can still be found; the salt (SHOPSENSE_TRACE_SALT) keeps the hash from being reversed by trying every CUST-nnnn."""
    if value in (None, ""):
        return ""
    salt = salt if salt is not None else os.environ.get("SHOPSENSE_TRACE_SALT", "shopsense-dev-salt")
    return "u_" + hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:12]
