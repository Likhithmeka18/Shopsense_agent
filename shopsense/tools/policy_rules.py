"""
ShopSense Tools: the policy numbers that CODE must enforce, in one place.

The RAG pipeline answers policy questions in prose. But a few policy facts decide whether the system is allowed to
MOVE MONEY (is this return still inside its window? is this shipment lost? how many contacts is too many?), and a
rule that gates an irreversible action must not depend on a retrieved paragraph being phrased well. So those facts
live here as constants, each tagged with the policy section it comes from, and tests/test_policy_rules.py checks
every number against the actual policy documents -- if the handbook changes and this file does not, a test fails.

Two honest approximations, both stated where used:
  * the shipments table has a ship_date but no delivered date, so delivery is ESTIMATED as ship_date + 3 days;
  * "no tracking movement for N days" is approximated as "N days since ship_date while still in a non-delivered state".
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

# Frozen "now" (Data Sourcing Guidelines: never the wall clock). Everything time-based takes `now=` so tests can move it.
REFERENCE_NOW = datetime(2026, 10, 1, 9, 0, 0)
DELIVERY_TRANSIT_DAYS = 3

# --- return windows (category addenda override return_policy RP-1.1; RP-1.4 says the addendum wins) ---
RETURN_WINDOW_DAYS = {"electronics": 15, "apparel": 45, "groceries": 0, "home": 30, "beauty": 10}        # CA-EL/AP/GR/HM/BT-1.1
FESTIVE_WINDOW_DAYS = {"electronics": 30, "apparel": 60, "groceries": 0, "home": 45, "beauty": 10}       # CA-xx-1.2 (beauty: no extension)
FESTIVE_MONTHS = (10, 11, 12)                                                                          # orders placed 1 Oct - 31 Dec

# --- defective / wrong / damaged item report windows ---
DEFECT_REPORT_DAYS_DEFAULT = 60            # RP-1.3
DEFECT_REPORT_DAYS_GROCERIES = 2           # CA-GR-1.2: 48 hours
DEFECT_REPORT_DAYS_BREAKABLE = 7           # CA-HM-2.1
DEFECT_REPORT_DAYS_ELECTRONICS_DOA = 7     # CA-EL-1.3
BREAKABLE_KEYWORDS = ("ceramic", "glass", "porcelain")

# --- warranty (warranty_policy.md) ---
WARRANTY_MONTHS_ELECTRONICS = 12           # WP-1.1
WARRANTY_MONTHS_ELECTRICAL = 6             # WP-2 (home electrical) / WP-3 (beauty appliances)
ELECTRICAL_PRODUCTS = {"led desk lamp": WARRANTY_MONTHS_ELECTRICAL, "hair dryer": WARRANTY_MONTHS_ELECTRICAL}

# --- money limits (refund_policy RF-3, refund_authorisation_matrix RA-1/RA-4) ---
AUTO_APPROVE_BELOW_INR = 2000
EXECUTIVE_FROM_INR = 10000
GOODWILL_CAP_INR = 500

# --- escalation triggers ---
REPEAT_CONTACT_THRESHOLD = 3               # ESC-2 item 5 / ESC-REPEAT: third contact ...
REPEAT_CONTACT_WINDOW_DAYS = 7             # ... within 7 days
LOST_AFTER_DAYS = 10                       # SP-4.1
ABUSE_RETURNS = 3                          # FP-1.1
ABUSE_HIGH_VALUE_INR = 5000
ABUSE_WINDOW_DAYS = 90

# refund_policy RF-2: processing time after the returned item passes QC. Net banking is NOT listed there on purpose.
REFUND_TIMING_BY_PAYMENT = {
    "UPI": "2-3 business days",
    "Credit Card": "5-7 business days",
    "Debit Card": "5-7 business days",
    "Cash on Delivery": "7-10 business days",
}

# First-response SLA by escalation code (escalation_sop.md ESC-3).
SLA_BY_CODE = {
    "ESC-SAFETY": "1 hour", "ESC-LEGAL": "the same business day", "ESC-LOST": "24 hours", "ESC-L2": "24 hours",
    "ESC-L1": "48 hours", "ESC-ABUSE": "48 hours", "ESC-REPEAT": "24 hours",
}


def _date(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d")


def is_festive_order(order_date: str) -> bool:
    return _date(order_date).month in FESTIVE_MONTHS


def estimated_delivery_date(order: dict, shipment: Optional[dict]) -> Optional[datetime]:
    """Delivered orders only. ship_date + DELIVERY_TRANSIT_DAYS (the tables carry no delivered date)."""
    if order.get("status") != "delivered":
        return None
    base = _date(shipment["ship_date"]) if shipment else _date(order["order_date"]) + timedelta(days=2)
    return base + timedelta(days=DELIVERY_TRANSIT_DAYS)


def return_window_days(category: str, order_date: str) -> int:
    table = FESTIVE_WINDOW_DAYS if is_festive_order(order_date) else RETURN_WINDOW_DAYS
    return table.get(category, 30)


def defect_window_days(category: str, product_name: str) -> int:
    name = product_name.lower()
    if category == "groceries":
        return DEFECT_REPORT_DAYS_GROCERIES
    if any(k in name for k in BREAKABLE_KEYWORDS):
        return DEFECT_REPORT_DAYS_BREAKABLE
    if category == "electronics":
        return DEFECT_REPORT_DAYS_ELECTRONICS_DOA
    return DEFECT_REPORT_DAYS_DEFAULT


def warranty_months(category: str, product_name: str) -> int:
    if category == "electronics":
        return WARRANTY_MONTHS_ELECTRONICS
    return ELECTRICAL_PRODUCTS.get(product_name.lower(), 0)

# Words a customer uses for products in each category (checked against the catalogue by tests/test_policy_rules.py).
# Used only to AIM the policy search ("lip balm" -> the beauty addendum); never to decide an eligibility outcome.
CATEGORY_KEYWORDS = {
    "electronics": ("earbud", "smartwatch", "smart watch", "speaker", "keyboard", "charger", "power bank", "powerbank", "laptop",
                    "headphone", "earphone", "phone", "smartphone", "tablet", "electronic"),
    "apparel": ("shirt", "t-shirt", "jacket", "shoes", "sweater", "pants", "jeans", "dress", "clothes", "clothing", "apparel", "sneaker"),
    "groceries": ("rice", "almond", "honey", "cooking oil", "oil 1l", "green tea", "grocery", "groceries", "food"),
    "home": ("mug", "lamp", "pillow", "non-stick", "pan set", "organizer", "organiser", "cookware", "furniture"),
    "beauty": ("lip balm", "sunscreen", "serum", "hair dryer", "cosmetic", "skincare", "moisturi", "makeup", "perfume"),
}


def _word_start(word: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![a-z0-9])" + re.escape(word))


# A keyword must START a word ("phone" matches "phones" and "phone", "moisturi" matches "moisturiser"). The old plain-substring test made
# "price" match "rice" (groceries) and "address" match "dress" (apparel), which aimed the policy search at the wrong category addendum.
_CATEGORY_PATTERNS = {c: [_word_start(w) for w in words] for c, words in CATEGORY_KEYWORDS.items()}


def categories_in_text(text: str) -> list[str]:
    t = (text or "").lower()
    return [c for c, pats in _CATEGORY_PATTERNS.items() if any(p.search(t) for p in pats)]


def order_items(api, order_id: str) -> list[dict]:
    """Line items of an order. MockAPI doesn't load order_items.csv (the M2 tools never needed it), so rules that
    need per-item category/price read it lazily and cache it on the api object."""
    cache = getattr(api, "_items_by_order", None)
    if cache is None:
        import csv
        from .order_tools import DATA_DIR
        cache = {}
        with open(DATA_DIR / "order_items.csv") as f:
            for row in csv.DictReader(f):
                cache.setdefault(row["order_id"], []).append(row)
        api._items_by_order = cache
    return cache.get(order_id, [])


def return_eligibility(api, order_id: str, *, kind: str, now: datetime = REFERENCE_NOW) -> dict:
    """Which items of this order can still be returned/refunded, and why not for the others?

    kind="change_of_mind": category return window (festive-aware); groceries never.
    kind="defect":         defect/damage report window; if electronics miss the 7-day DOA window but are inside
                           the warranty period the outcome is a WARRANTY CLAIM, not a refund.
    outcome: refund_eligible | partial | outside_window | warranty_claim | not_returnable | not_delivered | already_returned
    """
    order = api.orders[order_id]
    shipment = api.shipments.get(order_id)
    status = order["status"]
    if status == "returned":
        return {"outcome": "already_returned", "eligible_amount_inr": 0.0, "items": []}
    if status != "delivered":
        return {"outcome": "not_delivered", "eligible_amount_inr": 0.0, "items": [], "order_status": status}
    delivered = estimated_delivery_date(order, shipment)
    days = max(0, (now - delivered).days) if delivered else None
    items_out, eligible, warranty_items = [], 0.0, 0
    for it in order_items(api, order_id):
        cat, name, price = it["category"], it["product_name"], float(it["unit_price_inr"])
        if kind == "change_of_mind":
            window = return_window_days(cat, order["order_date"])
            ok = window > 0 and days is not None and days <= window
            reason = "not returnable for change of mind" if window == 0 else (
                f"within {window}-day window" if ok else f"outside the {window}-day window")
        else:
            window = defect_window_days(cat, name)
            ok = days is not None and days <= window
            reason = f"reported within {window} days" if ok else f"outside the {window}-day report window"
            if not ok and days is not None and days <= warranty_months(cat, name) * 30:
                warranty_items += 1
                reason = f"outside the {window}-day window but covered by a {warranty_months(cat, name)}-month warranty"
        if ok:
            eligible += price
        items_out.append({"product": name, "category": cat, "price_inr": price, "window_days": window,
                          "within": ok, "reason": reason})
    if eligible > 0:
        outcome = "refund_eligible" if all(i["within"] for i in items_out) else "partial"
    elif warranty_items:
        outcome = "warranty_claim"
    elif items_out and all(i["window_days"] == 0 for i in items_out) and kind == "change_of_mind":
        outcome = "not_returnable"
    else:
        outcome = "outside_window"
    return {"outcome": outcome, "eligible_amount_inr": eligible, "items": items_out,
            "days_since_delivery": days, "delivery_estimated": True, "festive_order": is_festive_order(order["order_date"])}


def lost_shipment(api, order_id: str, *, now: datetime = REFERENCE_NOW) -> Optional[dict]:
    """SP-4.1 approximation: still-in-motion state for >= LOST_AFTER_DAYS since ship_date."""
    order, ship = api.orders.get(order_id), api.shipments.get(order_id)
    if not order or not ship or order["status"] != "shipped":
        return None
    if ship["scan_state"] not in ("in_transit", "delivery_exception", "picked_up", "label_created"):
        return None
    days = (now - _date(ship["ship_date"])).days
    return {"days_since_ship": days, "carrier": ship["carrier"], "scan_state": ship["scan_state"]} if days >= LOST_AFTER_DAYS else None


# --- text heuristics (deterministic; the M1 LLM label is a hint, not an authority for money-moving decisions) ---

_DEFECT_RE = re.compile(
    r"\b(stopped\s+(working|turning|charging)|not\s+(working|turning\s+on|switching\s+on|charging)|won'?t\s+(turn|switch|power)|dead\s+on\s+arrival|"
    r"doa|defective|faulty|damaged|broken|smashed|cracked|shattered|arrived\s+(broken|damaged)|kharab|not\s+as\s+described)\b", re.I)
_REACTION_RE = re.compile(r"\b(rash|allerg\w*|reaction|irritat\w*|itch\w*|breakout|burning\s+sensation)\b", re.I)
_ACTION_REQUEST_RE = re.compile(
    r"\b(i\s+want\s+(a\s+|the\s+|my\s+|full\s+)*(refund|money\s+back|to\s+return)|(please\s+|pls\s+)?(refund|process\s+the\s+refund)|refund\s+me|"
    r"give\s+me\s+(a\s+|my\s+|the\s+)*(refund|money)|i(?:'d|\s+would)\s+like\s+(a\s+refund|to\s+return|my\s+money)|requesting\s+(a\s+)?refund|"
    r"want\s+(the\s+|my\s+)*(full\s+)?(amount|money)\s+(refunded|back)|want\s+to\s+return|need\s+a\s+replacement|want\s+a\s+replacement)\b", re.I)
_QUESTION_RE = re.compile(
    r"\b(how\s+(many|long|much)|can\s+i|could\s+i|is\s+it\s+possible|am\s+i\s+allowed|do\s+i\s+have|what\s+is|what'?s|is\s+that|when\s+will|"
    r"still\s+possible|does\s+(it|the|this)|do\s+you)\b", re.I)


def looks_defective(text: str) -> bool:
    return bool(_DEFECT_RE.search(text or ""))


def mentions_skin_reaction(text: str) -> bool:
    return bool(_REACTION_RE.search(text or ""))


def is_policy_question(text: str) -> bool:
    """True when the customer is ASKING about policy rather than requesting an action. 'How many days do I have to
    return them?' must not open a refund workflow just because the intent label says RETURN_ITEM."""
    t = text or ""
    asks = bool(_QUESTION_RE.search(t)) and "?" in t
    requests = bool(_ACTION_REQUEST_RE.search(t))
    return asks and not requests


def wants_replacement(text: str) -> bool:
    return bool(re.search(r"\breplace(ment)?\b", text or "", re.I))
