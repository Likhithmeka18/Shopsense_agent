"""Fixes found by the first live golden run: scorer typography, triage sentiment calibration, condition-aware retrieval."""
from shopsense.core.ticket_intake import EXTRACTION_PROMPT
from shopsense.eval.golden import _norm, load_golden, score_case
from shopsense.workflows.ticket_resolution_graph import _topic_terms

GOLDEN = {c["id"]: c for c in load_golden()}


def test_scorer_folds_typographic_quotes_and_dashes():
    assert _norm("I’m sorry, I don’t have this — really") == "i'm sorry, i don't have this - really"
    case = GOLDEN["SS-EV-010"]
    r = {"answer": "I’m sorry, but I don’t have that information.", "route": "resolve", "escalation_codes": []}
    assert score_case(case, r)["contains_any_ok"]


def test_triage_prompt_ties_angry_to_hostile_wording_not_to_a_refund_request():
    p = EXTRACTION_PROMPT
    assert '"angry" ONLY when' in p and "profanity" in p and "threats" in p
    assert "full refund" in p and '"neutral"' in p            # a calm demand for money back is not anger


def test_opened_item_questions_pull_condition_vocabulary_into_the_query():
    assert "seal" in _topic_terms("If I open it and don't like it, can I return it?")
    assert "seal" in _topic_terms("I already used the serum once")
    assert _topic_terms("Has my order shipped yet?") == "shipping delivery timeline"   # unrelated tickets are untouched


def test_beauty_case_flags_the_wrong_claim_not_the_correct_addendum_wording():
    case = GOLDEN["SS-EV-004"]
    correct = ("Beauty items can be returned within 10 days of delivery, and the 30-day general window and the festive "
               "season extension do not apply. Only items with an unbroken hygiene seal can be returned if opened.")
    wrong = "Yes, you can return it within 30 days, and during the festive season you get 45 days."
    ok = score_case(case, {"answer": correct, "route": "resolve", "escalation_codes": []})
    bad = score_case(case, {"answer": wrong, "route": "resolve", "escalation_codes": []})
    assert ok["must_not_ok"] and ok["passed"]
    assert not bad["must_not_ok"] and bad["unsafe"]


def test_model_review_flag_is_not_told_to_fire_on_refund_size():
    """The refund tiers are enforced in code (decide_route / action tools). Telling the model to also flag 'high-value
    refunds' made it pre-empt the whole workflow: a warranty claim for INR 12,169 was handed to a human with no answer."""
    p = EXTRACTION_PROMPT
    assert "high-value refunds" not in p
    assert "requires_human_review: true ONLY for legal threats" in p and "Do NOT set it for a refund request" in p


def test_answer_prompt_hedges_only_on_the_main_question():
    """A literal model told to 'address every part' appended 'I don't have that information' about incidental details
    (a sealed box, 'Diwali' branding); the graph's strict unknown-detector then discarded a correct 15-day / 30-day answer."""
    from shopsense.rag.answer import ANSWER_SYSTEM_PROMPT as P
    assert "main question" in P and "side details" in P
    assert "Address EVERY part" not in P
    assert "I don't have that information in our policy documents" in P      # the unanswerable cases still get one stable phrase
    assert "never guess or invent a figure" in P and "untrusted" in P         # grounding and injection guards untouched
