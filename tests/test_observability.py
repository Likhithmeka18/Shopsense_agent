"""
Milestone 7 tests: tracing + redaction (shopsense/observability).

What these prove: customer data is redacted BEFORE a span is stored or exported; spans nest the way the code runs (ticket -> node
-> tool -> generation); the sampler drops only boring healthy runs and never a failed / degraded / escalated one; a LangFuse
outage can never fail a ticket. The LangFuse client here is a stand-in that mirrors the SDK calls the project makes
(`start_as_current_observation`, `update`, `propagate_attributes`, `flush`) -- it proves OUR use of that API, not the cloud
service; `python check_langfuse.py` is the check against the real one.
"""
from contextlib import contextmanager

import pytest

from shopsense.observability import tracing as T
from shopsense.observability.redact import REDACTED, hash_id, redact_obj, redact_text
from shopsense.observability.report import format_dashboard, per_span_table, percentile, slo_summary


# ---- redaction -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,leaked", [
    ("mail me at jane.doe@example.com please", "jane.doe@example.com"),
    ("call +91 98765 43210 now", "98765 43210"),
    ("call 9876543210 now", "9876543210"),
    ("card 4111 1111 1111 1111 was charged", "4111 1111 1111 1111"),
    ("my PAN is ABCDE1234F", "ABCDE1234F"),
    ("aadhaar 1234 5678 9012", "1234 5678 9012"),
    ("pay to rahul@okhdfcbank", "rahul@okhdfcbank"),
    ("key gsk_abcdefghijklmnop1234", "gsk_abcdefghijklmnop1234"),
    ("DOB: 12/03/1990", "12/03/1990"),
])
def test_personal_data_is_redacted(raw, leaked):
    assert leaked not in redact_text(raw)
    assert REDACTED in redact_text(raw)


def test_ids_we_need_for_debugging_survive_redaction():
    text = "Order ORD-4445, ticket TCK-0042, refund REF-0012, tracking TRK8569491238, INR 2,499 for card ending 4242"
    out = redact_text(text)
    for keep in ("ORD-4445", "TCK-0042", "REF-0012", "TRK8569491238", "INR 2,499"):
        assert keep in out
    assert "4242" not in out


def test_redact_obj_walks_nested_data_and_sensitive_keys():
    out = redact_obj({"customer_name": "Asha Rao", "items": [{"note": "write to a@b.co"}], "n": 3, "ok": True})
    assert out["customer_name"] == REDACTED and "a@b.co" not in str(out) and out["n"] == 3 and out["ok"] is True


def test_long_text_is_truncated_and_ids_are_hashed_stably():
    assert len(redact_text("x" * 5000)) < 2100
    assert hash_id("CUST-0001") == hash_id("CUST-0001") != hash_id("CUST-0002")
    assert "CUST-0001" not in hash_id("CUST-0001") and hash_id("") == ""


# ---- spans, nesting, run stats ----------------------------------------------------------------------------------------
def test_spans_nest_like_the_code_runs_and_inherit_the_session():
    t = T.Tracer()
    with t.run(session_id="TCK-1", user_id="CUST-9", tags=["shopsense"]):
        with t.span("node:triage", kind="agent"):
            with t.span("order_system.read", kind="tool"):
                pass
    trace = t.store.traces()[0]
    by = {s["name"]: s for s in trace["spans"]}
    assert by["order_system.read"]["parent_id"] == by["node:triage"]["span_id"]
    assert by["node:triage"]["parent_id"] == by["ticket.run"]["span_id"] and by["ticket.run"]["parent_id"] is None
    assert {s["trace_id"] for s in trace["spans"]} == {trace["trace_id"]}
    assert by["order_system.read"]["session_id"] == "TCK-1"
    assert trace["user_id"] == hash_id("CUST-9") and "CUST-9" not in str(trace)


def test_a_failing_span_is_marked_and_the_exception_still_propagates():
    t = T.Tracer()
    with pytest.raises(ValueError):
        with t.run(session_id="TCK-2"):
            with t.span("node:x"):
                raise ValueError("boom")
    trace = t.store.traces()[0]
    assert trace["ok"] is False
    assert next(s for s in trace["spans"] if s["name"] == "node:x")["error"].startswith("ValueError")


def test_a_langgraph_pause_is_not_recorded_as_a_failure():
    class GraphInterrupt(Exception):
        pass

    t = T.Tracer()
    with pytest.raises(GraphInterrupt):
        with t.run(session_id="TCK-3"):
            with t.span("node:approval"):
                raise GraphInterrupt("waiting for a human")
    trace = t.store.traces()[0]
    node = next(s for s in trace["spans"] if s["name"] == "node:approval")
    assert node["ok"] is True and node["metadata"]["paused"] is True


def test_span_inputs_are_redacted_before_they_are_stored():
    t = T.Tracer()
    with t.run(session_id="TCK-4", input={"text": "my number is 9876543210"}):
        with t.span("x", input={"email": "me@example.com"}) as sp:
            sp.update(output={"reply": "we will mail you at me@example.com"})
    assert "9876543210" not in str(t.store.traces()) and "me@example.com" not in str(t.store.traces())


def test_events_count_toward_degraded_but_a_plain_retry_does_not():
    t = T.Tracer()
    with t.run(session_id="TCK-5") as run:
        T.add_event("guard.retry", dependency="order_system")
        assert not run.stats.degraded
        T.add_event("fallback.human_queue", dependency="order_system")
        assert run.stats.degraded and "fallback.human_queue" in run.stats.degraded_reasons()
    assert t.store.traces()[0]["degraded"] is True


def test_traced_node_records_what_the_node_wrote():
    t = T.Tracer()
    prev = T.set_tracer(t)
    try:
        node = T.traced_node("node:demo")(lambda state: {"reply": "hi", "log": ["demo ran"]})
        with t.run(session_id="TCK-6"):
            node({"ticket_id": "TCK-6"})
    finally:
        T.set_tracer(prev)
    span = next(s for s in t.store.traces()[0]["spans"] if s["name"] == "node:demo")
    assert span["output"] == {"wrote": ["reply"], "log": ["demo ran"]}


def test_disabled_tracer_costs_nothing_and_changes_nothing():
    t = T.Tracer(enabled=False)
    with t.run(session_id="TCK-7"):
        with t.span("x") as sp:
            sp.update(output={"a": 1})
    assert len(t.store) == 0


# ---- tail sampling ----------------------------------------------------------------------------------------------------
class _Dice:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_sampling_drops_only_boring_healthy_runs():
    t = T.Tracer(keep_healthy_rate=0.1, rng=_Dice(0.99))        # the dice always says "drop"
    with t.run(session_id="healthy", metadata={"route": "resolve"}):
        pass
    with t.run(session_id="escalated") as run:
        run.update(metadata={"route": "human_review"})
    with t.run(session_id="degraded"):
        T.add_event("llm.fallback", model="x")
    with pytest.raises(RuntimeError):
        with t.run(session_id="failed"):
            raise RuntimeError("x")
    with t.run(session_id="paused", metadata={"paused": True}):
        pass
    kept = {tr["session_id"] for tr in t.store.traces()}
    assert kept == {"escalated", "degraded", "failed", "paused"} and t.dropped_healthy == 1


def test_trace_file_is_jsonl_one_trace_per_line(tmp_path):
    path = tmp_path / "traces.jsonl"
    t = T.Tracer(trace_file=path)
    for i in range(3):
        with t.run(session_id=f"TCK-{i}"):
            pass
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["session_id"] for r in rows] == ["TCK-0", "TCK-1", "TCK-2"]


# ---- LangFuse stand-in (mirrors the SDK calls) --------------------------------------------------------
class FakeObservation:
    def __init__(self, client, as_type, kwargs, parent):
        self.client, self.as_type, self.kwargs, self.parent, self.updates = client, as_type, kwargs, parent, []

    def update(self, **kw):
        self.updates.append(kw)


class FakeLangfuse:
    def __init__(self, unsupported=()):
        self.observations, self.stack, self.flushed, self.propagated = [], [], 0, []
        self.unsupported = set(unsupported)

    @contextmanager
    def start_as_current_observation(self, *, as_type="span", name, input=None, model=None, **kw):
        if as_type in self.unsupported:
            raise ValueError(f"unsupported as_type {as_type}")
        obs = FakeObservation(self, as_type, {"name": name, "input": input, "model": model}, self.stack[-1] if self.stack else None)
        self.observations.append(obs)
        self.stack.append(obs)
        try:
            yield obs
        finally:
            self.stack.pop()

    def flush(self):
        self.flushed += 1

    def auth_check(self):
        return True


def fake_propagate_factory(client):
    @contextmanager
    def propagate_attributes(**kw):
        client.propagated.append(kw)
        yield
    return propagate_attributes


def test_langfuse_receives_nested_observations_with_session_and_hashed_user():
    lf = FakeLangfuse()
    t = T.Tracer(langfuse_client=lf, langfuse_propagate=fake_propagate_factory(lf))
    with t.run(session_id="TCK-8", user_id="CUST-8", tags=["shopsense", "model:x"], input={"text": "mail a@b.co"}):
        with t.span("order_system.read", kind="tool"):
            with t.span("llm.generate", kind="generation", model="gemini/x", input=[{"role": "user", "content": "call 9876543210"}]):
                pass
    names = [(o.as_type, o.kwargs["name"]) for o in lf.observations]
    assert names == [("agent", "ticket.run"), ("tool", "order_system.read"), ("generation", "llm.generate")]
    assert lf.observations[2].parent is lf.observations[1] and lf.observations[1].parent is lf.observations[0]
    assert lf.observations[2].kwargs["model"] == "gemini/x"
    assert lf.propagated == [{"session_id": "TCK-8", "tags": ["shopsense", "model:x"], "user_id": hash_id("CUST-8")}]
    assert "a@b.co" not in str(lf.observations[0].kwargs) and "9876543210" not in str(lf.observations[2].kwargs)


def test_langfuse_gets_the_error_level_when_a_span_fails():
    lf = FakeLangfuse()
    t = T.Tracer(langfuse_client=lf, langfuse_propagate=fake_propagate_factory(lf))
    with pytest.raises(KeyError):
        with t.run(session_id="TCK-9"):
            raise KeyError("missing")
    assert any(u.get("level") == "ERROR" for u in lf.observations[0].updates)


def test_an_sdk_without_a_richer_observation_type_falls_back_to_a_plain_span():
    lf = FakeLangfuse(unsupported={"tool", "retriever"})
    t = T.Tracer(langfuse_client=lf, langfuse_propagate=fake_propagate_factory(lf))
    with t.run(session_id="TCK-10"):
        with t.span("order_system.read", kind="tool"):
            pass
    assert [o.as_type for o in lf.observations] == ["agent", "span"]


def test_a_broken_langfuse_never_fails_a_ticket():
    class Broken:
        def start_as_current_observation(self, **kw):
            raise ConnectionError("langfuse cloud unreachable")

        def flush(self):
            raise ConnectionError("flush failed")

    def bad_propagate(**kw):
        raise RuntimeError("no")

    t = T.Tracer(langfuse_client=Broken(), langfuse_propagate=bad_propagate)
    with t.run(session_id="TCK-11"):
        with t.span("x"):
            pass
    t.flush()
    assert t.langfuse_errors >= 1 and len(t.store) == 1          # the local record is still there


def test_flush_reaches_langfuse():
    lf = FakeLangfuse()
    T.Tracer(langfuse_client=lf).flush()
    assert lf.flushed == 1


# ---- dashboard -------------------------------------------------------------------------------------------------------
def test_dashboard_reports_percentiles_degraded_share_and_costly_spans():
    assert percentile([10, 20, 30, 40, 50], 50) == 30 and percentile([], 95) == 0.0
    t = T.Tracer()
    for i in range(4):
        with t.run(session_id=f"TCK-{i}") as run:
            run.update(metadata={"route": "resolve"})
            with t.span("node:triage"):
                pass
            if i == 3:
                T.add_event("fallback.human_queue", dependency="order_system")
    traces = t.store.traces()
    slo = slo_summary(traces)
    assert slo["tickets"] == 4 and slo["degraded_rate"] == 0.25 and slo["routes"] == {"resolve": 4}
    assert any(row["span"] == "node:triage" and row["calls"] == 4 for row in per_span_table(traces))
    text = format_dashboard(traces)
    assert "degraded" in text.lower() and "node:triage" in text
