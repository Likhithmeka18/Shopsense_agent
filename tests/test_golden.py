"""Golden set (Asset D) - structure, real-data grounding, scorer behaviour and the LLM-free layers."""
import csv
import json
from pathlib import Path

from shopsense.eval.golden import (
    CATEGORY_RANGES, floor_check, load_golden, retrieval_check, score_case, summarize, validate_golden,
)
from shopsense.rag.bm25 import BM25Index
from shopsense.rag.ingest import load_policy_corpus
from shopsense.tools.order_tools import MockAPI

ROOT = Path(__file__).resolve().parents[1]


def test_golden_set_is_structurally_valid_and_has_twenty_cases():
    cases = load_golden()
    assert len(cases) == 20
    assert validate_golden(cases) == []


def test_category_counts_follow_the_sourcing_guideline():
    counts = {}
    for c in load_golden():
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    for cat, (lo, hi) in CATEGORY_RANGES.items():
        assert lo <= counts[cat] <= hi, (cat, counts[cat])
    assert counts["injection"] >= 4          # heaviest injection weighting of the five domains


def test_the_set_grows_five_cases_per_milestone():
    assert len(load_golden(up_to="M4")) == 5
    assert len(load_golden(up_to="M5")) == 10
    assert len(load_golden(up_to="M6")) == 15
    assert len(load_golden(up_to="M8")) == 20


def test_cases_reference_real_customers_and_orders_that_belong_together():
    api = MockAPI()
    for c in load_golden():
        t = c["ticket"]
        assert t["customer_id"] in api.customers, c["id"]
        if t["order_id"]:
            assert api.orders[t["order_id"]]["customer_id"] == t["customer_id"], c["id"]
            assert t["order_id"] in t["text"], c["id"]


def test_adversarial_cases_are_hand_written_and_numbered_9xx():
    for c in load_golden():
        if c["category"] in ("guardrail", "injection"):
            assert c["hand_written"] and c["id"].startswith("SS-EV-9"), c["id"]
            assert c["must_not_contain"], c["id"]


def test_guardrail_cases_cover_the_five_named_failure_modes():
    codes = {code for c in load_golden() if c["category"] == "guardrail" for code in c["expected_escalation_codes"]}
    assert codes == {"ESC-L2", "ESC-REPEAT", "ESC-LEGAL", "ESC-SAFETY", "ESC-ABUSE"}


def test_flag_disclosure_case_forbids_every_way_of_naming_the_flag():
    case = next(c for c in load_golden() if c["id"] == "SS-EV-905")
    assert {"flagged", "abuse"} <= set(case["must_not_contain"])


def _good_result(case):
    return {"answer": " ".join(case["answer_contains_all"] + case["answer_contains_any"][:1]),
            "route": case["expected_route"], "escalation_codes": case["expected_escalation_codes"]}


def test_scorer_passes_an_answer_built_from_the_expected_strings():
    for c in load_golden():
        s = score_case(c, _good_result(c))
        assert s["passed"], (c["id"], s)


def test_scorer_flags_a_forbidden_phrase_as_unsafe():
    c = next(c for c in load_golden() if c["id"] == "SS-EV-901")
    bad = {**_good_result(c), "answer": "Done - I have approved your refund. A reviewer will look."}
    s = score_case(c, bad)
    assert s["unsafe"] and not s["passed"] and s["must_not_hits"]


def test_scorer_checks_route_and_required_codes_separately():
    c = next(c for c in load_golden() if c["id"] == "SS-EV-903")
    wrong_route = score_case(c, {**_good_result(c), "route": "resolve"})
    assert not wrong_route["route_ok"] and wrong_route["codes_ok"]
    no_code = score_case(c, {**_good_result(c), "escalation_codes": []})
    assert no_code["route_ok"] and not no_code["codes_ok"]
    extra = score_case(c, {**_good_result(c), "escalation_codes": ["ESC-LEGAL", "ESC-LOST"]})
    assert extra["codes_ok"]                      # a superset of the expected codes is fine


def test_scorer_whitespace_and_case_are_normalised():
    c = next(c for c in load_golden() if c["id"] == "SS-EV-005")
    assert score_case(c, {**_good_result(c), "answer": "About 2-3   BUSINESS\nDAYS."})["contains_all_ok"]


def test_summary_reports_unsafe_outputs_and_per_category_rates():
    cases = load_golden()
    scores = [score_case(c, _good_result(c)) for c in cases]
    scores[0] = score_case(cases[0], {"answer": "within 30 days", "route": "resolve", "escalation_codes": []})
    s = summarize(scores)
    assert s["n"] == 20 and s["passed"] == 19 and s["unsafe_outputs"] == 1
    assert cases[0]["id"] in s["failed"] and set(s["by_category"]) == set(CATEGORY_RANGES)


def test_deterministic_floor_raises_what_the_data_says_is_mandatory():
    api = MockAPI()
    for c in load_golden():
        r = floor_check(c, customer=api.customers.get(c["ticket"]["customer_id"]))
        assert r["ok"], (c["id"], r["problems"])


def test_bm25_alone_retrieves_the_expected_policy_documents_for_most_cases():
    chunks = load_policy_corpus()
    bm25 = BM25Index(chunks)
    rows = [retrieval_check(c, lambda q, k: bm25.search(q, k=k), chunks, k=5) for c in load_golden()]
    applicable = [r for r in rows if r["applicable"]]
    assert len(applicable) == 18                       # the 2 unanswerable cases have no expected source
    assert sum(r["ok"] for r in applicable) >= 15


def test_retrieval_qa_lives_in_data_and_every_question_resolves():
    from shopsense.rag.evaluate import GOLDEN, gold_chunks
    data = json.load(open(ROOT / "data" / "shopsense" / "eval" / "retrieval_qa.json"))["questions"]
    assert GOLDEN == data and len(data) >= 20
    chunks = load_policy_corpus()
    assert all(gold_chunks(chunks, q["answer_contains"]) for q in data)
