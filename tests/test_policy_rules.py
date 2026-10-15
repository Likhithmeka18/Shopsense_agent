"""policy_rules.py holds the numbers code enforces; these tests pin each one to the policy documents so the two
cannot drift apart silently."""
import re
from datetime import datetime
from pathlib import Path

from shopsense.tools import policy_rules as pr
from shopsense.tools.order_tools import MockAPI, refund_tier_for_amount

POL = Path(__file__).resolve().parents[1] / "data" / "shopsense" / "policies"


def _doc(name):
    return " ".join((POL / f"{name}.md").read_text().split())


def test_category_return_windows_match_the_addenda():
    for cat, doc, code in (("electronics", "category_electronics", "CA-EL-1.1"), ("apparel", "category_apparel", "CA-AP-1.1"),
                           ("home", "category_home", "CA-HM-1.1"), ("beauty", "category_beauty", "CA-BT-1.1")):
        text = _doc(doc)
        assert f"**{pr.RETURN_WINDOW_DAYS[cat]} days**" in text, (cat, code)
    assert pr.RETURN_WINDOW_DAYS["groceries"] == 0 and "not returnable" in _doc("category_groceries")


def test_festive_windows_match_the_addenda():
    assert f"**{pr.FESTIVE_WINDOW_DAYS['electronics']} days**" in _doc("category_electronics")
    assert f"**{pr.FESTIVE_WINDOW_DAYS['apparel']} days**" in _doc("category_apparel")
    assert f"**{pr.FESTIVE_WINDOW_DAYS['home']} days**" in _doc("category_home")
    assert pr.FESTIVE_WINDOW_DAYS["beauty"] == pr.RETURN_WINDOW_DAYS["beauty"] and "does **not** apply" in _doc("category_beauty")
    assert "1 October and 31 December" in _doc("return_policy") and pr.FESTIVE_MONTHS == (10, 11, 12)


def test_defect_windows_match_the_documents():
    assert f"**{pr.DEFECT_REPORT_DAYS_DEFAULT} days**" in _doc("return_policy")
    assert "**48 hours**" in _doc("category_groceries") and pr.DEFECT_REPORT_DAYS_GROCERIES == 2
    assert f"**{pr.DEFECT_REPORT_DAYS_BREAKABLE} days**" in _doc("category_home")
    assert f"**{pr.DEFECT_REPORT_DAYS_ELECTRONICS_DOA} days**" in _doc("category_electronics")


def test_warranty_months_match_the_warranty_policy():
    w = _doc("warranty_policy")
    assert f"**{pr.WARRANTY_MONTHS_ELECTRONICS}-month**" in w and f"**{pr.WARRANTY_MONTHS_ELECTRICAL} months**".replace(" months", "-month") in w.replace("6-month", "6-month")


def test_money_limits_match_refund_policy_and_matrix():
    assert pr.AUTO_APPROVE_BELOW_INR == 2000 and pr.EXECUTIVE_FROM_INR == 10000
    assert refund_tier_for_amount(1999) == "auto" and refund_tier_for_amount(2000) == "manager_review"
    assert refund_tier_for_amount(9999) == "manager_review" and refund_tier_for_amount(10000) == "executive_review"
    assert f"INR {pr.GOODWILL_CAP_INR}" in _doc("refund_authorisation_matrix")


def test_escalation_triggers_match_the_sop_and_fraud_policy():
    sop, fraud, ship = _doc("escalation_sop"), _doc("fraud_prevention"), _doc("shipping_policy")
    assert "Third contact about the same unresolved issue within 7 days" in sop
    assert (pr.REPEAT_CONTACT_THRESHOLD, pr.REPEAT_CONTACT_WINDOW_DAYS) == (3, 7)
    assert f"**{pr.LOST_AFTER_DAYS} calendar days**" in ship
    assert f"**{pr.ABUSE_RETURNS} or more high-value returns within {pr.ABUSE_WINDOW_DAYS} days**" in fraud
    assert f"INR {pr.ABUSE_HIGH_VALUE_INR:,}" in fraud


def test_sla_table_matches_escalation_sop():
    sop = _doc("escalation_sop")
    for code, sla in (("ESC-SAFETY", "1 hour"), ("ESC-LEGAL", "Same business day"), ("ESC-LOST", "24 hours"),
                      ("ESC-L2", "24 hours"), ("ESC-L1", "48 hours"), ("ESC-ABUSE", "48 hours"), ("ESC-REPEAT", "24 hours")):
        assert f"| {code} | {sla} |" in sop, code
        assert pr.SLA_BY_CODE[code].lower() == sla.lower() or pr.SLA_BY_CODE[code].lower() == sla.lower().replace("same", "the same")


def test_refund_timing_matches_rf2():
    rf2 = _doc("refund_policy")
    assert "| UPI / wallet | 2-3 business days |" in rf2 and "| Original payment method (card) | 5-7 business days |" in rf2
    assert "7-10 business days" in rf2 and "Net Banking" not in pr.REFUND_TIMING_BY_PAYMENT


def test_golden_orders_resolve_to_the_expected_eligibility():
    api = MockAPI()
    assert pr.return_eligibility(api, "ORD-4445", kind="change_of_mind")["outcome"] == "refund_eligible"      # earbuds, inside 15 days
    assert pr.return_eligibility(api, "ORD-4143", kind="change_of_mind")["outcome"] == "refund_eligible"      # sweater, 36d < 45d (would be OUT of the general 30)
    assert pr.return_eligibility(api, "ORD-4465", kind="change_of_mind")["outcome"] == "not_returnable"       # rice
    assert pr.return_eligibility(api, "ORD-4313", kind="change_of_mind")["outcome"] == "outside_window"       # serum, 10-day window
    assert pr.return_eligibility(api, "ORD-4219", kind="defect")["outcome"] == "warranty_claim"               # smartwatch after DOA window
    assert pr.return_eligibility(api, "ORD-4589", kind="defect")["outcome"] == "refund_eligible"              # broken mugs inside 7 days
    assert pr.return_eligibility(api, "ORD-4228", kind="defect")["outcome"] == "not_delivered"


def test_festive_order_gets_the_extended_window_and_the_general_default_is_not_used():
    assert pr.return_window_days("electronics", "2026-11-05") == 30 and pr.return_window_days("electronics", "2026-08-05") == 15
    assert pr.return_window_days("apparel", "2026-12-31") == 60 and pr.return_window_days("beauty", "2026-11-05") == 10


def test_lost_shipment_detection():
    api = MockAPI()
    assert pr.lost_shipment(api, "ORD-4382") and pr.lost_shipment(api, "ORD-4228")        # stuck 14-16 days
    assert pr.lost_shipment(api, "ORD-4445") is None                                         # delivered
    assert pr.lost_shipment(api, "ORD-4228", now=datetime(2026, 9, 20)) is None             # earlier clock: not yet 10 days


def test_text_heuristics_separate_questions_from_requests():
    assert pr.is_policy_question("How many days do I have to return them?")
    assert pr.is_policy_question("Is a return still possible?")
    assert not pr.is_policy_question("I'd like to return it for a refund.")
    assert not pr.is_policy_question("I want my INR 25,516 back for ORD-4553 RIGHT NOW")
    assert pr.looks_defective("it stopped turning on last week") and pr.looks_defective("arrived smashed")
    assert not pr.looks_defective("it does not fit") and pr.mentions_skin_reaction("I got a bad rash")
    assert pr.wants_replacement("Want a replacement please")


def test_category_keywords_recognise_every_catalogue_product_and_nothing_crosses_over():
    import csv
    from shopsense.tools.order_tools import DATA_DIR
    with open(DATA_DIR / "order_items.csv") as f:
        rows = list(csv.DictReader(f))
    for name, cat in {(r["product_name"], r["category"]) for r in rows}:
        assert pr.categories_in_text(f"I bought the {name}") == [cat], (name, cat, pr.categories_in_text(name))
    assert pr.categories_in_text("Where is my order?") == []


def test_category_words_must_start_a_word_so_price_is_not_rice_and_address_is_not_dress():
    # live finding: "Do you offer price matching against other websites?" searched the policy for 'groceries'
    for text in ("Do you offer price matching against other websites?", "Please change my delivery address", "Is there a price drop refund?"):
        assert pr.categories_in_text(text) == [], text
    assert pr.categories_in_text("my smartphone screen cracked") == ["electronics"]
    assert pr.categories_in_text("the rice arrived spoiled") == ["groceries"]
    assert pr.categories_in_text("my moisturiser leaked") == ["beauty"]
