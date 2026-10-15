"""
Milestone 2 tests: reliability wrappers and the ShopSense order tools.
These never touch litellm, so they run against the real implementations
with no mocking needed.
"""
import time

import pytest

from shopsense.tools.order_tools import MockAPI, build_tools
from shopsense.tools.reliability import CircuitBreaker, make_robust_tool, retry_with_backoff


# --------------------------------------------------------------------------
# Reliability wrappers
# --------------------------------------------------------------------------

def test_retry_with_backoff_retries_then_succeeds():
    calls = {"n": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("simulated failure")
        return "recovered"

    assert flaky() == "recovered"
    assert calls["n"] == 3


def test_retry_with_backoff_raises_after_exhausting_retries():
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    def always_fails():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        always_fails()


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
    assert not breaker.is_open
    breaker.record_failure()
    assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open
    time.sleep(0.15)
    assert not breaker.is_open  # reset after timeout


def test_make_robust_tool_logs_calls():
    from shopsense.tools.reliability import TOOL_CALL_LOG

    def ok(x):
        return x * 2

    wrapped = make_robust_tool(ok, "double_tool")
    before = len(TOOL_CALL_LOG)
    result = wrapped(5)
    assert result == 10
    assert len(TOOL_CALL_LOG) == before + 1
    assert TOOL_CALL_LOG[-1]["outcome"] == "success"


# --------------------------------------------------------------------------
# Order tools (against the real generated mock data)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api():
    return MockAPI()


@pytest.fixture(scope="module")
def tools(api):
    callables, schemas = build_tools(api)
    return callables, schemas


def test_mock_api_loads_data(api):
    assert len(api.customers) > 0
    assert len(api.orders) > 0


def test_tool_schemas_match_callables(tools):
    callables, schemas = tools
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == set(callables.keys())


def test_order_lookup_known_order(api, tools):
    callables, _ = tools
    order_id = next(iter(api.orders.keys()))
    result = callables["order_lookup"](order_id)
    assert result["order_id"] == order_id
    assert "status" in result


def test_order_lookup_unknown_order(tools):
    callables, _ = tools
    result = callables["order_lookup"]("ORD-99999")
    assert "error" in result


def test_shipping_tracker_shape(api, tools):
    callables, _ = tools
    order_id = next(iter(api.shipments.keys()))
    result = callables["shipping_tracker"](order_id)
    assert result["tracking_available"] is True
    assert "carrier" in result


def test_calculate_refund_amount_shape(api, tools):
    callables, _ = tools
    order_id = next(iter(api.orders.keys()))
    result = callables["calculate_refund_amount"](order_id)
    assert "eligible" in result
