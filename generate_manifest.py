"""
ShopSense dev script: write data/shopsense/manifest.json (provenance, seed, counts, licences, known gaps) FROM the
files on disk, so it cannot go stale the way a hand-edited manifest did. tests/test_data.py compares it to the data.

    python generate_manifest.py
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "shopsense"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def count_rows(path):
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def build():
    pol = sorted((DATA / "policies").glob("*.md"))
    pdf = sorted((DATA / "policies_pdf").glob("*.pdf"))
    tickets = [json.loads(line) for line in open(DATA / "intake" / "tickets.jsonl")]
    golden = json.load(open(DATA / "eval" / "golden_set.json"))["cases"]
    qa = json.load(open(DATA / "eval" / "retrieval_qa.json"))["questions"]
    tables = {n: count_rows(DATA / "mock_api" / f"{n}.csv") for n in ("customers", "products", "orders", "order_items", "shipments", "refunds")}
    scen = Counter(t["ground_truth"].get("scenario", "base") for t in tickets)
    return {
        "domain": "shopsense",
        "generated_by": "generate_tables.py, generate_tickets.py, generate_corpus_pdfs.py, hand-written policies/ and eval/ (see `sources`)",
        "seed": 42,
        "reference_clock": "2026-10-01T09:00:00",
        "reference_clock_note": "All dates are anchored here, never the wall clock, so runs are reproducible.",
        "assets": {
            "A_policy_corpus": {
                "status": "complete", "needed_by": "M3 (index) -> M4 (hybrid RAG)",
                "markdown": {"dir": "policies/", "documents": len(pol), "files": {p.name: sha(p) for p in pol}},
                "pdf": {"dir": "policies_pdf/", "documents": len(pdf), "note": "Lossy on purpose: heading markers dropped, tables flattened, "
                        "running header + page number on every page. Produced by generate_corpus_pdfs.py; ingested by rag.ingest.load_policy_corpus_pdf.",
                        "files": {p.name: sha(p) for p in pdf}},
                "composition": "5 core (returns, refunds, shipping, escalation SOP, FAQ) + 5 category addenda + refund authorisation matrix, warranty, fraud prevention, cancellation",
                "deliberate_conflicts": [
                    "Return window: general 30 days (RP-1.1) vs electronics 15 (CA-EL-1.1), apparel 45 (CA-AP-1.1), groceries non-returnable (CA-GR-1.1), "
                    "home 30 (CA-HM-1.1), beauty 10 (CA-BT-1.1). RP-1.4 states that the category addendum wins.",
                    "Festive window: general 45 days (RP-1.2) vs electronics 30 (CA-EL-1.2), apparel 60 (CA-AP-1.2), beauty none (CA-BT-1.2).",
                    "Defective-item report window: 60 days (RP-1.3) vs groceries 48 hours (CA-GR-1.2) and breakables 7 days (CA-HM-2.1).",
                ],
                "near_duplicates": ["Opened-seal rules appear in RP-3.2, CA-EL-2.1, CA-BT-2.1, CA-AP-2.2 with different scope.",
                                    "Approval tiers appear in RF-3, ESC-1 and RA-1 and must agree."],
                "known_gaps": ["international shipping", "price matching", "gift wrapping", "returns at physical stores", "loyalty points", "bulk / B2B orders"],
                "provenance": "Generated for this capstone; no third-party text. No retailer publishes a redistributable returns policy.",
            },
            "B_intake_tickets": {
                "status": "complete", "needed_by": "M1", "file": "intake/tickets.jsonl", "records": len(tickets), "sha256_16": sha(DATA / "intake" / "tickets.jsonl"),
                "layers": {"base": scen["base"], "extension": len(tickets) - scen["base"]},
                "scenario_counts": dict(sorted(scen.items())),
                "ground_truth": "intent, category, order_id (the reference VISIBLE in the text, else null), customer_id, sentiment, urgency, requires_human_review, "
                                "contains_injection_attempt; extension adds scenario, expected_route, expected_escalation_codes, expected_behavior.",
                "taxonomy_source": "Intent names follow Bitext's retail/e-commerce taxonomy (CDLA-Sharing 1.0). No Bitext text is included - all ticket text is generated.",
            },
            "C_mock_api_tables": {
                "status": "complete", "needed_by": "M2 -> M6", "dir": "mock_api/", "rows": tables,
                "files": {n: sha(DATA / "mock_api" / f"{n}.csv") for n in tables},
                "layers": {"base": "60 customers / 120 orders / 15 refunds (byte-stable; tests and docs cite it)",
                           "extension": "+90 customers / +480 orders; own RNG; append-only; abuse flags meet the written FP-1 definition"},
                "note": "Equivalent five-table set (customers, products, orders, shipments, refunds) with Indian carriers and rupee amounts, as the guidelines allow instead of Olist.",
            },
            "D_golden_eval": {
                "status": "complete", "needed_by": "M4 (first 5) -> M8", "golden_set": "eval/golden_set.json", "cases": len(golden),
                "by_category": dict(Counter(c["category"] for c in golden)), "by_milestone": dict(sorted(Counter(c["added_at"] for c in golden).items())),
                "sha256_16": sha(DATA / "eval" / "golden_set.json"),
                "retrieval_qa": {"file": "eval/retrieval_qa.json", "questions": len(qa)},
                "hand_written": True, "scorer": "shopsense/eval/golden.py",
            },
        },
        "generation": {"method": "seeded Python generators (tables, tickets, PDFs) plus hand-written policies, extension tickets and eval cases",
                       "llm_provider": None, "llm_model": None,
                       "note": "No language model wrote any asset, so there is no provider or model to record; the seeded tables are byte-reproducible."},
        # Data Sourcing Guide s2.3: every public dataset the work consulted, what it was used for (including 'not used'), and its licence.
        "public_sources_referenced": [
            {"name": "Bitext retail/e-commerce chatbot dataset", "url": "https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
             "use": "Intent taxonomy NAMES only (e.g. TRACK_DELIVERY, REFUND_REQUEST) for the ticket labels. No Bitext text or rows are included.",
             "licence": "CDLA-Sharing 1.0", "used": True},
            {"name": "Olist Brazilian e-commerce dataset", "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
             "use": "NOT used. Replaced by the guide's allowed synthetic equivalent (Indian carriers, rupee amounts, abuse flags, refund approval tiers).",
             "licence": "CC BY-NC-SA 4.0 (non-commercial)", "used": False},
            {"name": "Customer Support on Twitter", "url": "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter",
             "use": "NOT used. Recommended by the guide to calibrate how angry real customers sound; open follow-up (sentiment labels were not checked against real tweets).",
             "licence": "Kaggle platform terms", "used": False},
            {"name": "Amazon Reviews 2023 (McAuley Lab)", "url": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
             "use": "NOT used. Product names in products.csv are invented.",
             "licence": "Research use (see dataset card)", "used": False},
        ],
        "contains_real_personal_data": False,
        "foreign_key_check": "enforced by tests/test_data.py (orders->customers, refunds/shipments/items->orders, ticket labels->visible order refs)",
        "licences": {"generated_content": "Original to this project", "Bitext": "taxonomy names only (CDLA-Sharing 1.0)", "Olist": "not used (CC BY-NC-SA 4.0)"},
    }


if __name__ == "__main__":
    m = build()
    (DATA / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest.json: {len(m['assets']['A_policy_corpus']['markdown']['files'])} md + {len(m['assets']['A_policy_corpus']['pdf']['files'])} pdf, "
          f"{m['assets']['B_intake_tickets']['records']} tickets, tables {m['assets']['C_mock_api_tables']['rows']}, {m['assets']['D_golden_eval']['cases']} golden cases")
