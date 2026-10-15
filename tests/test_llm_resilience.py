"""
Milestone 7 tests: the LLM client's hardening (shopsense/core/llm_client.py).

Every provider failure here is scripted, so the numbers are exact and nothing sleeps for real (`_sleep` is replaced by a
recorder). What these prove: the provider's own advice on how long to wait is honoured (Groq AND Gemini formats), only
transient failures are retried, a dead model trips its circuit breaker and stops being called, a configured fallback model is
used loudly (and flags the ticket as degraded), and the real error text survives instead of being replaced by a generic one.
"""
import pytest

from conftest import FakeResponse
from shopsense.core import llm_client as lc
from shopsense.observability import tracing as T
from shopsense.tools.reliability import CircuitBreaker


class Quota(lc.RateLimitError):
    pass


class Overloaded(Exception):
    status_code = 503


class NotFound(Exception):
    status_code = 404


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def harness(monkeypatch):
    """A scriptable `completion`, a recording `_sleep`, clean breakers/counters, a fresh tracer, no fallback models."""
    lc.reset_llm_state()
    for var in ("SHOPSENSE_LLM_FALLBACK_MODELS", "SHOPSENSE_LLM_MIN_INTERVAL", "SHOPSENSE_LLM_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    sleeps, calls, script = [], [], []

    def fake_completion(*, model, messages, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})
        item = script.pop(0) if script else FakeResponse(content="ok")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(model)
        return item

    monkeypatch.setattr(lc, "completion", fake_completion)
    monkeypatch.setattr(lc, "_sleep", lambda s: sleeps.append(round(s, 3)))
    tracer = T.Tracer()
    previous = T.set_tracer(tracer)
    yield type("H", (), {"sleeps": sleeps, "calls": calls, "script": script, "tracer": tracer, "monkeypatch": monkeypatch})
    T.set_tracer(previous)
    lc.reset_llm_state()


def _call(model="m/primary"):
    return lc.llm_completion([{"role": "user", "content": "hi"}], model=model)


# ---- reading the provider's advice ----------------------------------------------------------------------------------
def test_parse_retry_wait_reads_gemini_formats_too():
    assert lc.parse_retry_wait("Please retry in 46.4s.") == 46.4
    assert lc.parse_retry_wait("{'error': {'details': [{'retryDelay': '46s'}]}}") == 46.0
    assert lc.parse_retry_wait('"retryDelay": "7s"') == 7.0
    # the old formats still work
    assert lc.parse_retry_wait("try again in 2m33.36s") - 153.36 < 1e-6
    assert lc.parse_retry_wait("no hint here") is None


def test_classify_llm_error():
    assert lc.classify_llm_error(Quota("slow down")) == "rate_limit"
    assert lc.classify_llm_error(Exception("RESOURCE_EXHAUSTED: quota")) == "rate_limit"
    assert lc.classify_llm_error(Overloaded("model is overloaded")) == "transient"
    assert lc.classify_llm_error(TimeoutError("timed out")) == "transient"
    assert lc.classify_llm_error(NotFound("model not found")) == "permanent"
    assert lc.classify_llm_error(ValueError("something odd")) == "permanent"      # unknown: never loop on it


# ---- retries ---------------------------------------------------------------------------------------------------------
def test_rate_limit_waits_the_time_the_provider_names(harness):
    harness.script.extend([Quota("Please retry in 46.4s."), FakeResponse(content="fine")])
    resp = _call()
    assert resp.choices[0].message.content == "fine"
    assert harness.sleeps == [47.4]                    # 46.4 + the 1s safety margin, not a blind 15s guess
    assert len(harness.calls) == 2


def test_transient_errors_back_off_then_succeed(harness):
    harness.script.extend([Overloaded("503 overloaded"), Overloaded("503 overloaded"), FakeResponse(content="back")])
    assert _call().choices[0].message.content == "back"
    assert len(harness.sleeps) == 2 and harness.sleeps[0] < harness.sleeps[1]        # exponential


def test_permanent_error_is_not_retried_and_keeps_its_real_message(harness):
    harness.script.append(NotFound("model gemini-2.5-flash is no longer available to new users"))
    with pytest.raises(NotFound):
        _call()
    assert len(harness.calls) == 1 and harness.sleeps == []
    assert lc.LLM_STATS["permanent_errors"] == 1


def test_a_quota_wait_longer_than_ten_minutes_stops_immediately(harness):
    harness.script.append(Quota("Please try again in 2h4m1s. daily token cap"))
    with pytest.raises(Quota):
        _call()
    assert harness.sleeps == []


def test_default_timeout_is_set_so_one_hung_call_cannot_stall_a_batch(harness):
    _call()
    assert harness.calls[0]["kwargs"]["timeout"] == lc.DEFAULT_TIMEOUT_S


# ---- circuit breaker ----------------------------------------------------------------------------------------------------
def test_a_dead_model_trips_the_breaker_and_stops_being_called(harness):
    clock = FakeClock()
    lc._BREAKERS["m/primary"] = CircuitBreaker(failure_threshold=2, reset_timeout=60, clock=clock)
    for _ in range(2):                                  # two calls, each exhausting its 5 attempts on 503s
        harness.script.extend([Overloaded("503")] * lc.MAX_RATE_LIMIT_RETRIES)
        with pytest.raises(Overloaded):
            _call()
    attempts_so_far = len(harness.calls)
    with pytest.raises(lc.LLMUnavailableError) as err:
        _call()
    assert err.value.reason == "circuit_open"
    assert len(harness.calls) == attempts_so_far        # short-circuited: the provider was not touched
    assert lc.LLM_STATS["circuit_open"] == 1


def test_breaker_allows_one_trial_after_the_cool_down_then_closes_on_success(harness):
    clock = FakeClock()
    br = lc._BREAKERS["m/primary"] = CircuitBreaker(failure_threshold=1, reset_timeout=60, clock=clock)
    br.record_failure()
    with pytest.raises(lc.LLMUnavailableError):
        _call()
    clock.t += 61
    assert _call().choices[0].message.content == "ok"
    assert br.state == "closed"


# ---- fallback model ---------------------------------------------------------------------------------------------------
def test_fallback_model_is_used_loudly_and_flags_the_ticket(harness):
    harness.monkeypatch.setenv("SHOPSENSE_LLM_FALLBACK_MODELS", "m/backup")
    harness.script.extend([NotFound("primary model is gone"), FakeResponse(content="from backup")])
    with harness.tracer.run(session_id="TCK-1", user_id="CUST-1") as run:
        resp = _call()
        stats = run.stats
    assert resp.choices[0].message.content == "from backup"
    assert [c["model"] for c in harness.calls] == ["m/primary", "m/backup"]
    assert lc.LLM_STATS["fallbacks"] == 1
    assert stats.degraded and "llm.fallback" in stats.degraded_reasons()
    kept = harness.tracer.store.traces()[0]
    assert kept["degraded"] is True                      # a degraded run is always kept by the sampler


def test_when_every_model_fails_the_primary_error_is_raised_not_a_generic_one(harness):
    harness.monkeypatch.setenv("SHOPSENSE_LLM_FALLBACK_MODELS", "m/backup")
    harness.script.extend([NotFound("primary 404"), NotFound("backup 404")])
    with pytest.raises(NotFound) as err:
        _call()
    assert "primary 404" in str(err.value)


# ---- pacing + tracing ---------------------------------------------------------------------------------------------------
def test_min_interval_spaces_calls(harness):
    harness.monkeypatch.setenv("SHOPSENSE_LLM_MIN_INTERVAL", "4.5")
    _call()
    _call()
    assert harness.sleeps and 0 < harness.sleeps[0] <= 4.5


def test_each_call_is_a_generation_span_with_tokens_and_redacted_input(harness):
    class U:
        prompt_tokens, completion_tokens = 120, 30

    resp = FakeResponse(content="answer")
    resp.usage = U()
    harness.script.append(resp)
    with harness.tracer.run(session_id="TCK-2"):
        lc.llm_completion([{"role": "user", "content": "refund please, mail me at jane.doe@example.com"}], model="m/primary")
    trace = harness.tracer.store.traces()[0]
    gen = next(s for s in trace["spans"] if s["name"] == "llm.generate")
    assert gen["kind"] == "generation" and gen["model"] == "m/primary"
    assert gen["usage"] == {"input": 120, "output": 30, "total": 150}
    assert "jane.doe@example.com" not in str(gen["input"])         # redacted before it is stored anywhere
    assert trace["run"]["tokens_in"] == 120 and trace["run"]["llm_calls"] == 1
