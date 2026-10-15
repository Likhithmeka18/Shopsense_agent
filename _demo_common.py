"""Shared helpers for the two Milestone 6 demo scripts (not part of the package)."""
from __future__ import annotations

import re
from pathlib import Path

from shopsense.core.safety_net import combine_escalation_reasons
from shopsense.core.ticket_intake import extract_ticket_info, parse_envelopes_jsonl
from shopsense.tools.order_tools import MockAPI
from shopsense.workflows.refund_approval_graph import find_order_id_in_text

TICKETS_PATH = Path(__file__).resolve().parent / "data" / "shopsense" / "intake" / "tickets.jsonl"


def find_escalated_ticket(api: MockAPI, prefer: str):
    """First ESC-2 ticket whose escalation reasons include `prefer`
    (e.g. 'safety_net:ESC-SAFETY' or 'contains_injection_attempt'); falls
    back to any escalated ticket. Reasons come from the SAME function
    run_pipeline.py uses (M1 flags + deterministic safety net)."""
    envelopes, _failed = parse_envelopes_jsonl(str(TICKETS_PATH))
    fallback = None
    for ticket in envelopes:
        extraction, _errors = extract_ticket_info(ticket.text)
        if extraction is None:
            continue
        reasons = combine_escalation_reasons(extraction, ticket.text, customer=api.customers.get(ticket.customer_id or ""))
        if not reasons:
            continue
        order_id = extraction.order_id or find_order_id_in_text(ticket.text)
        found = (ticket, order_id, reasons)
        if prefer in reasons:
            return found
        fallback = fallback or found
    return fallback if fallback else (None, None, [])


def assert_memo_invariants(final: dict) -> None:
    """The properties a free-text memo writer can break."""
    draft = final["draft"]
    risk = final["risk_profile"]
    assert final["ticket_id"] in draft, "memo must carry the REAL ticket id"
    assert not re.search(r"#\s*XYZ|#\s*\d{3,}\b", draft), "memo must not invent a ticket number"
    for code in risk["esc_codes"]:
        assert code in draft, f"routing header is missing {code}"
    upper = draft.upper()
    assert "SYSTEM NOTE" not in upper and "IGNORE ALL PREVIOUS" not in upper, "injected text leaked into the memo"
    if risk["injection_attempt"]:
        assert "WARNING" in draft, "injection tickets must carry the warning line"
