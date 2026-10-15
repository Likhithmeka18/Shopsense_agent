"""
ShopSense Core: contact history, for ESC-REPEAT ("third contact about the same unresolved issue within 7 days").

A single ticket cannot tell you it is the third. Counting needs the customer's earlier tickets, so the ticket graph
asks this ledger. "Same issue" is approximated as: same customer, and the same order if both tickets name one
(otherwise the same intent). "Unresolved" cannot be known from tickets alone, so every earlier contact in the window
counts -- biased toward escalating, which for a customer who has already written twice is the right bias.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from ..tools.policy_rules import REPEAT_CONTACT_WINDOW_DAYS

_ORDER_RE = re.compile(r"\bORD-\d+\b")


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


class ContactLedger:
    def __init__(self, contacts: Optional[list[dict]] = None):
        self.contacts: list[dict] = []
        for c in contacts or []:
            self.record(**c)

    def record(self, *, ticket_id: str, customer_id: str, created_at: str, text: str = "", order_id: Optional[str] = None,
               intent: Optional[str] = None) -> None:
        order_id = order_id or (_ORDER_RE.search(text).group(0) if _ORDER_RE.search(text or "") else None)
        if any(c["ticket_id"] == ticket_id for c in self.contacts):
            return
        self.contacts.append({"ticket_id": ticket_id, "customer_id": customer_id, "created_at": created_at,
                              "order_id": order_id, "intent": intent})

    @classmethod
    def from_tickets(cls, tickets: list[dict]) -> "ContactLedger":
        """Build from ticket dicts/objects with ticket_id/customer_id/created_at/text."""
        led = cls()
        for t in tickets:
            g = lambda k: t[k] if isinstance(t, dict) else getattr(t, k)
            if g("customer_id"):
                led.record(ticket_id=g("ticket_id"), customer_id=g("customer_id"), created_at=g("created_at"), text=g("text"))
        return led

    def prior(self, *, ticket_id: str, customer_id: str, created_at: str, order_id: Optional[str] = None,
              intent: Optional[str] = None, window_days: int = REPEAT_CONTACT_WINDOW_DAYS) -> list[dict]:
        """Earlier contacts by this customer, inside the window, about the same issue."""
        now = _ts(created_at)
        out = []
        for c in self.contacts:
            if c["ticket_id"] == ticket_id or c["customer_id"] != customer_id:
                continue
            when = _ts(c["created_at"])
            if not (now - timedelta(days=window_days) <= when < now):
                continue
            if order_id and c["order_id"]:
                same = c["order_id"] == order_id
            else:
                same = bool(intent) and c["intent"] == intent
            if same:
                out.append(c)
        return sorted(out, key=lambda c: c["created_at"])

    def contact_number(self, **kw) -> int:
        """1 + earlier contacts about the same issue in the window."""
        return 1 + len(self.prior(**kw))
