"""
Milestone 8 hardening tests (found by running the demo API live): a chat request must not wait out a provider's rate limit for minutes
(it holds the service's one-ticket lock and the caller's connection), model citation markers must not reach a customer, the memo team
is off for chat by default, and the dashboard counts LLM attempts, failed ones included.
"""
import pytest

from conftest import FakeResponse
from shopsense.core import llm_client as lc
from shopsense.guardrails import clean_model_text
from shopsense.observability import tracing as T
from shopsense.observability.report import slo_summary
from shopsense.service import DEFAULT_REQUEST_BUDGET_S, ChatService


class Quota(lc.RateLimitError):
    pass


@pytest.fixture
def harness(monkeypatch):
    lc.reset_llm_state()
    for var in ("SHOPSENSE_LLM_FALLBACK_MODELS", "SHOPSENSE_LLM_MIN_INTERVAL", "SHOPSENSE_LLM_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    sleeps, calls, script = [], [], []

    def fake_completion(*, model, messages, **kwargs):
        calls.append(model)
        item = script.pop(0) if script else FakeResponse(content="ok")
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(lc, "completion", fake_completion)
    monkeypatch.setattr(lc, "_sleep", lambda s: sleeps.append(round(s, 3)))
    prev = T.set_tracer(T.Tracer())
    yield type("H", (), {"sleeps": sleeps, "calls": calls, "script": script})
    lc.set_request_deadline(None)
    T.set_tracer(prev)
    lc.reset_llm_state()


def _call():
    return lc.llm_completion([{"role": "user", "content": "hi"}], model="m/primary")


# ---- request deadline ------------------------------------------------------------------------------------------------------------
def test_a_long_rate_limit_wait_is_not_slept_when_it_would_pass_the_request_deadline(harness):
    harness.script.append(Quota("Please try again in 6m54s."))               # the 414 s wait seen live
    lc.set_request_deadline(25)
    with pytest.raises(Quota):
        _call()
    assert harness.sleeps == [] and len(harness.calls) == 1                   # gave up at once instead of sleeping 7 minutes


def test_short_waits_inside_the_deadline_are_still_waited_out(harness):
    harness.script.extend([Quota("Please retry in 3.0s."), FakeResponse(content="fine")])
    lc.set_request_deadline(25)
    assert _call().choices[0].message.content == "fine"
    assert harness.sleeps == [4.0]


def test_without_a_deadline_a_batch_still_waits_up_to_the_quota_limit(harness):
    harness.script.extend([Quota("Please try again in 6m54s."), FakeResponse(content="later")])
    assert _call().choices[0].message.content == "later"
    assert harness.sleeps == [415.0]


def test_an_expired_deadline_stops_before_calling_the_provider_at_all(harness):
    lc.set_request_deadline(-1)
    with pytest.raises(lc.LLMUnavailableError) as e:
        _call()
    assert e.value.reason == "deadline_exceeded" and harness.calls == []


def test_the_deadline_is_logged_loudly_and_marks_the_run_degraded(harness):
    t = T.Tracer()
    prev = T.set_tracer(t)
    try:
        harness.script.append(Quota("Please try again in 6m54s."))
        lc.set_request_deadline(25)
        with t.run(session_id="TCK-X"):
            with pytest.raises(Quota):
                _call()
        assert any(e["kind"] == "llm.deadline_exceeded" for tr in t.store.traces() for sp in tr["spans"] for e in sp.get("events", []))
        assert t.store.traces()[0]["degraded"] is True
    finally:
        T.set_tracer(prev)


def test_repeated_deadline_give_ups_open_the_breaker_so_the_next_requests_fail_fast(harness):
    for _ in range(3):
        harness.script.append(Quota("Please try again in 6m54s."))
        lc.set_request_deadline(25)
        with pytest.raises(Quota):
            _call()
    before = len(harness.calls)
    lc.set_request_deadline(25)
    with pytest.raises(lc.LLMUnavailableError):
        _call()
    assert len(harness.calls) == before                                        # breaker open: no provider call


# ---- the service's settings ----------------------------------------------------------------------------------------------------
def test_the_chat_service_defaults_to_a_short_request_budget_and_no_memo_team(monkeypatch):
    monkeypatch.delenv("SHOPSENSE_API_DEADLINE_S", raising=False)
    monkeypatch.delenv("SHOPSENSE_API_MEMO_TEAM", raising=False)
    svc = ChatService(mode="offline")
    assert svc.request_budget_s == DEFAULT_REQUEST_BUDGET_S and svc.memo_team is False
    assert svc.health()["limits"] == {"request_budget_s": DEFAULT_REQUEST_BUDGET_S, "memo_team": False}


def test_the_settings_come_from_the_environment_and_a_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("SHOPSENSE_API_DEADLINE_S", "12.5")
    monkeypatch.setenv("SHOPSENSE_API_MEMO_TEAM", "1")
    svc = ChatService(mode="offline")
    assert svc.request_budget_s == 12.5 and svc.memo_team is True
    monkeypatch.setenv("SHOPSENSE_API_DEADLINE_S", "soon")
    assert ChatService(mode="offline").request_budget_s == DEFAULT_REQUEST_BUDGET_S


def test_a_live_request_sets_and_always_clears_the_deadline(monkeypatch):
    seen = []

    class Graph:
        pass

    from shopsense.workflows import ticket_resolution_graph as tg

    def fake_run_ticket(graph, state, *, tags=(), on_result=None):
        seen.append(lc._deadline_left())
        raise RuntimeError("model went away")

    monkeypatch.setattr(tg, "run_ticket", fake_run_ticket)
    svc = ChatService(mode="live", deps=_offline_deps(), graph=Graph(), request_budget_s=30)
    with pytest.raises(RuntimeError):
        svc.chat("hello there", customer_id="CUST-0077")
    assert seen and 0 < seen[0] <= 30
    assert lc._deadline_left() is None                                         # cleared even though the request failed


def _offline_deps():
    from shopsense.eval.offline import RetrievalOnlyKB, keyword_extract
    from shopsense.runtime import build_deps
    return build_deps(kb=RetrievalOnlyKB(), extract=keyword_extract, with_memo_team=False)


# ---- tidy customer text ----------------------------------------------------------------------------------------------------------
def test_citation_markers_and_odd_spaces_are_removed_without_touching_figures():
    raw = ("Electronics can be returned within **15 days of delivery** (delivery day counts as day 0)【24】.  \n"
           "During the festive season (1 October – 31 December) the window is **30 days**【25】.")
    out = clean_model_text(raw)
    assert "【" not in out and "】" not in out and " " not in out
    assert "15 days" in out and "30 days" in out and "day 0)." in out and "1 October" in out
    assert clean_model_text("See the policy 【3:1†source】 and [2] too.") == "See the policy and too."
    assert clean_model_text("") == "" and clean_model_text("plain text.") == "plain text."
    assert "RP-1.2" in clean_model_text("The festive window in RP-1.2 does not apply.")        # policy codes are not citation markers


def test_the_graph_cleans_the_policy_answer_before_it_can_reach_a_customer():
    class MarkedKB:
        chunks = []

        def answer(self, query, k=3, use_rerank=False, question=None):
            return {"answer": "Electronics can be returned within 15 days of delivery【24】.", "sources": []}

    from shopsense.eval.offline import keyword_extract
    from shopsense.runtime import build_deps
    svc = ChatService(mode="offline", deps=build_deps(kb=MarkedKB(), extract=keyword_extract, with_memo_team=False))
    r = svc.chat("How many days after delivery can electronics be returned?")
    assert "【" not in r["reply"] and "15 days" in r["reply"]


# ---- dashboard -------------------------------------------------------------------------------------------------------------------
def test_the_dashboard_counts_every_llm_attempt_including_the_failed_ones():
    t = T.Tracer()
    with t.run(session_id="TCK-1") as run:
        run.update(metadata={"route": "human_review"})
        for ok in (False, False, True):
            try:
                with t.span("llm.generate", kind="generation"):
                    if not ok:
                        raise RuntimeError("503")
            except RuntimeError:
                pass
    slo = slo_summary(t.store.traces())
    assert slo["llm_calls_per_ticket"] == 3.0 and slo["llm_failed_calls"] == 2
