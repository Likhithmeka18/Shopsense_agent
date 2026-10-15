"""
Milestone 8 tests: the FastAPI wrapper (shopsense/api.py).

FastAPI could not be installed where this was built, so these run against a recording stand-in that implements the few FastAPI calls
api.py uses (`FastAPI(...)`, `@app.get`, `@app.post(response_model=...)`, `HTTPException`). They prove OUR code: which routes exist,
that handlers delegate to the service, how failures map to status codes, and what the request / response contracts accept and omit.
They do not prove FastAPI's own behaviour -- `python demo_api_client.py` against a running server does.
"""
import asyncio
import importlib
import sys
import types

import pytest


class HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class FakeFastAPI:
    def __init__(self, **kw):
        self.kw, self.routes, self.state = kw, {}, types.SimpleNamespace()

    def _register(self, method, path, **opts):
        def deco(fn):
            self.routes[(method, path)] = (fn, opts)
            return fn
        return deco

    def get(self, path, **opts):
        return self._register("GET", path, **opts)

    def post(self, path, **opts):
        return self._register("POST", path, **opts)


@pytest.fixture
def api(monkeypatch):
    fake = types.ModuleType("fastapi")
    fake.FastAPI, fake.HTTPException = FakeFastAPI, HTTPException
    monkeypatch.setitem(sys.modules, "fastapi", fake)
    sys.modules.pop("shopsense.api", None)
    mod = importlib.import_module("shopsense.api")
    yield mod
    sys.modules.pop("shopsense.api", None)


class StubService:
    def __init__(self, result=None, error=None):
        self.result = result or {"ticket_id": "CHAT-000001", "session_id": "s", "outcome": "resolved", "reply": "hi", "awaiting_approval": False,
                                 "guardrail": {"passed": True, "blocked": False, "flags": []}, "degraded": False, "input_flags": []}
        self.error, self.calls = error, []

    def chat(self, message, **kw):
        self.calls.append((message, kw))
        if self.error:
            raise self.error
        return self.result

    def health(self):
        return {"status": "ok"}


def test_exactly_two_routes_are_exposed(api):
    app = api.create_app(StubService())
    assert set(app.routes) == {("GET", "/health"), ("POST", "/chat")}
    assert app.routes[("POST", "/chat")][1]["response_model"] is api.ChatResponse


def test_chat_delegates_and_returns_the_public_contract(api):
    stub = StubService()
    app = api.create_app(stub)
    chat = app.routes[("POST", "/chat")][0]
    out = chat(api.ChatRequest(message="hello", customer_id="CUST-1", session_id="s"))
    assert out.reply == "hi" and out.guardrail.passed and stub.calls == [("hello", {"customer_id": "CUST-1", "session_id": "s"})]
    assert app.routes[("GET", "/health")][0]() == {"status": "ok"}


def test_failures_map_to_status_codes_without_leaking_internals(api):
    def status(err):
        app = api.create_app(StubService(error=err))
        with pytest.raises(HTTPException) as e:
            app.routes[("POST", "/chat")][0](api.ChatRequest(message="x"))
        return e.value.status_code, str(e.value.detail)

    assert status(ValueError("message is longer than 2000 characters"))[0] == 422
    assert status(RuntimeError("live mode needs langgraph"))[0] == 503
    code, detail = status(KeyError("secret internal table name"))
    assert code == 500 and "secret" not in detail and "internal error" in detail


def test_request_contract_rejects_empty_and_oversized_messages(api):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        api.ChatRequest(message="")
    with pytest.raises(ValidationError):
        api.ChatRequest(message="x" * 2001)
    assert api.ChatRequest(message="x" * 2000).customer_id is None


def test_response_contract_omits_internal_fields(api):
    fields = set(api.ChatResponse.model_fields)
    assert fields == {"ticket_id", "session_id", "outcome", "reply", "awaiting_approval", "guardrail", "degraded", "input_flags"}
    for internal in ("escalation_codes", "reasons", "facts", "policy", "trace", "handoff"):
        assert internal not in fields


def test_public_fields_in_the_service_and_the_api_agree(api):
    from shopsense.service import PUBLIC_FIELDS
    assert set(PUBLIC_FIELDS) == set(api.ChatResponse.model_fields)


def test_lifespan_installs_the_tracer_and_flushes_it_on_shutdown(api):
    from shopsense.observability import tracing as T
    flushed = []
    app = api.create_app(StubService())

    async def run():
        async with app.kw["lifespan"](app):
            assert T.get_tracer() is not None
            T.get_tracer().flush = lambda: flushed.append(True)

    prev = T.get_tracer()
    try:
        asyncio.run(run())
    finally:
        T.set_tracer(prev)
    assert flushed == [True]
