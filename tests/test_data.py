"""Data-asset invariants (Data Sourcing Guidelines assets B and C).

These guard the *generated* data, so a regenerate that quietly breaks a distribution
the milestones depend on fails here instead of in a live demo."""
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "data" / "shopsense" / "mock_api"
TICKETS = ROOT / "data" / "shopsense" / "intake" / "tickets.jsonl"
NOW = datetime(2026, 10, 1, 9)


def _rows(name):
    with open(API_DIR / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def _tickets():
    return [json.loads(line) for line in open(TICKETS)]


def test_tables_have_the_expected_scale_and_are_referentially_consistent():
    customers, orders = {r["customer_id"]: r for r in _rows("customers")}, {r["order_id"]: r for r in _rows("orders")}
    assert len(customers) == 150 and len(orders) == 600 and len(_rows("refunds")) >= 80
    assert all(o["customer_id"] in customers for o in orders.values())
    assert all(r["order_id"] in orders and r["customer_id"] == orders[r["order_id"]]["customer_id"] for r in _rows("refunds"))
    assert all(s["order_id"] in orders for s in _rows("shipments"))
    assert all(i["order_id"] in orders for i in _rows("order_items"))
    totals = Counter()
    for i in _rows("order_items"):
        totals[i["order_id"]] += int(i["unit_price_inr"])
    assert all(totals[oid] == int(o["total_amount_inr"]) for oid, o in orders.items())


def test_abuse_flags_in_the_extension_match_the_written_definition():
    """fraud_prevention.md FP-1: 3+ returns of INR 5,000+ inside 90 days. Extension customers (CUST-0061+)
    are flagged exactly when they meet it; 'near-miss' customers with 2 are deliberately NOT flagged."""
    customers = {r["customer_id"]: r for r in _rows("customers")}
    hv = Counter()
    for o in _rows("orders"):
        age = (NOW - datetime.strptime(o["order_date"], "%Y-%m-%d")).days
        if o["status"] == "returned" and int(o["total_amount_inr"]) >= 5000 and age <= 90:
            hv[o["customer_id"]] += 1
    ext = [c for c in customers if int(c.split("-")[1]) > 60]
    for cid in ext:
        assert (customers[cid]["abuse_flag"] == "True") == (hv[cid] >= 3), cid
    assert sum(1 for c in ext if hv[c] == 2) >= 3          # near-misses exist


def test_refund_queue_is_rich_enough_for_the_approval_workflow():
    pending = [r for r in _rows("refunds") if r["approved"] == "False"]
    assert len(pending) >= 30
    assert sum(1 for r in pending if int(r["amount_inr"]) >= 10000) >= 5
    assert {r["approval_tier"] for r in pending} == {"manager_review", "executive_review"}
    for r in _rows("refunds"):
        a = int(r["amount_inr"])
        assert r["approval_tier"] == ("auto" if a < 2000 else "manager_review" if a < 10000 else "executive_review")


def test_there_are_shipments_stuck_long_enough_to_count_as_lost():
    orders = {r["order_id"]: r for r in _rows("orders")}
    stuck = [s for s in _rows("shipments") if orders[s["order_id"]]["status"] == "shipped"
             and s["scan_state"] in ("in_transit", "delivery_exception")
             and (NOW - datetime.strptime(s["ship_date"], "%Y-%m-%d")).days >= 10]
    assert len(stuck) >= 6
    assert {s["carrier"] for s in _rows("shipments")} == {"BlueDart", "Delhivery", "DTDC", "Ecom Express", "India Post"}


def test_ticket_mix_meets_the_sourcing_guideline():
    t = _tickets()
    assert len(t) >= 100 and len({x["ticket_id"] for x in t}) == len(t)
    prof = re.compile(r"\b(damn|crap|hell|bloody|freaking|BS|pissed)\b", re.I)
    assert sum(1 for x in t if prof.search(x["text"])) >= len(t) // 10        # angry with mild profanity
    assert sum(1 for x in t if x["ground_truth"]["order_id"] is None) >= len(t) // 6
    scen = Counter(x["ground_truth"].get("scenario", "base") for x in t)
    assert scen["legal_threat"] >= 6 and scen["injection"] >= 8 and scen["safety"] >= 4
    assert scen["hinglish"] + scen["indian_english"] >= 15
    assert scen["above_cap"] + scen["above_cap_angry"] >= 3
    assert scen["repeat_contact_3"] >= 3 and scen["flag_disclosure_probe"] >= 2 and scen["policy_gap"] >= 5
    lengths = sorted(len(x["text"]) for x in t)
    assert lengths[0] <= 20 and lengths[-1] >= 400                             # real tickets vary wildly in length
    assert {x["channel"] for x in t} >= {"web_form", "email", "phone_transcript", "chat", "sms"}


def test_ticket_labels_are_consistent_with_the_text_and_the_tables():
    orders = {r["order_id"]: r for r in _rows("orders")}
    for x in _tickets():
        m = re.search(r"ORD-\d+", x["text"])
        assert x["ground_truth"]["order_id"] == (m.group(0) if m else None), x["ticket_id"]
        if m:
            assert m.group(0) in orders and orders[m.group(0)]["customer_id"] == x["customer_id"], x["ticket_id"]


def test_flagged_customers_tickets_expect_abuse_routing_and_no_disclosure():
    flagged = {r["customer_id"] for r in _rows("customers") if r["abuse_flag"] == "True"}
    probes = [x for x in _tickets() if x["ground_truth"].get("scenario") == "flag_disclosure_probe"]
    assert probes and all(x["customer_id"] in flagged for x in probes)
    assert all(x["ground_truth"]["expected_behavior"] == "never_disclose_flag" and
               x["ground_truth"]["expected_escalation_codes"] == ["ESC-ABUSE"] for x in probes)


def test_manifest_matches_the_data_on_disk():
    """manifest.json is generated from the files (generate_manifest.py); this fails if data changed without regenerating it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gm", ROOT / "generate_manifest.py")
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    on_disk = json.load(open(ROOT / "data" / "shopsense" / "manifest.json"))
    assert on_disk == json.loads(json.dumps(gm.build())), "stale manifest - run: python generate_manifest.py"
    assert all(a["status"] == "complete" for a in on_disk["assets"].values())


def test_manifest_records_where_the_data_came_from_as_the_guide_requires():
    """Data Sourcing Guide s2.3: seed, how it was generated, the public sources referenced (name, url, use, licence) and the real-PII statement."""
    m = json.load(open(ROOT / "data" / "shopsense" / "manifest.json"))
    assert m["seed"] == 42 and m["contains_real_personal_data"] is False
    assert "llm_provider" in m["generation"] and "llm_model" in m["generation"]
    srcs = m["public_sources_referenced"]
    assert srcs and all({"name", "url", "use", "licence", "used"} <= set(s) for s in srcs)
    assert all(s["url"].startswith("https://") for s in srcs)
    used = [s["name"] for s in srcs if s["used"]]
    assert used == ["Bitext retail/e-commerce chatbot dataset"], "only the Bitext taxonomy NAMES are used; every other source must say NOT used"
    assert all(s["use"].startswith("NOT used") for s in srcs if not s["used"])
