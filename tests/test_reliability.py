"""
Milestone 7 tests: the reliability building blocks (shopsense/tools/reliability.py).

Time is a fake clock and sleeping is a recorder, so every number is exact and nothing waits. What these prove:
retries are for TRANSIENT errors only; a breaker opens after N consecutive failures, rejects calls without touching the
dependency, lets exactly ONE trial through after the cool-down, re-opens at once if that trial fails and closes if it works;
bad data and bugs are never retried; and every give-up is visible (typed error + counters) instead of silent.
"""
import random
import threading

import pytest

from shopsense.tools.reliability import (CircuitBreaker, DependencyGuard, DependencyUnavailable, TransientDependencyError)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_guard(*, attempts=3, threshold=3, reset=30.0, clock=None, malformed_on=(), jitter=0.0):
    sleeps = []
    clock = clock or Clock()
    g = DependencyGuard("orders", max_attempts=attempts, base_delay=0.2, max_delay=2.0, jitter=jitter, sleep=sleeps.append,
                        rng=random.Random(1), malformed_on=malformed_on, breaker=CircuitBreaker(threshold, reset, clock=clock))
    return g, sleeps, clock


def flaky(failures, exc=ConnectionError, result="ok"):
    state = {"calls": 0}

    def fn(*a, **k):
        state["calls"] += 1
        if state["calls"] <= failures:
            raise exc("down")
        return result
    fn.state = state
    return fn


# ---- retries ---------------------------------------------------------------------------------------------------------
def test_transient_errors_are_retried_with_exponential_backoff():
    g, sleeps, _ = make_guard()
    fn = flaky(2)
    assert g.call(fn) == "ok" and fn.state["calls"] == 3
    assert sleeps == [0.2, 0.4]                         # base * 2**(attempt-1), no jitter in this test
    assert g.stats["retries"] == 2 and g.stats["successes"] == 1 and g.breaker.state == "closed"


def test_jitter_stays_within_its_band():
    g, sleeps, _ = make_guard(jitter=0.25)
    g.call(flaky(2))
    assert 0.15 <= sleeps[0] <= 0.25 and 0.30 <= sleeps[1] <= 0.50


def test_all_transient_error_types_retry():
    for exc in (ConnectionError, TimeoutError, TransientDependencyError):
        g, _, _ = make_guard()
        assert g.call(flaky(1, exc)) == "ok"


def test_giving_up_raises_a_typed_error_that_names_the_cause_and_counts_the_failure():
    g, sleeps, _ = make_guard()
    with pytest.raises(DependencyUnavailable) as err:
        g.call(flaky(99, TimeoutError))
    e = err.value
    assert (e.dependency, e.reason, e.attempts) == ("orders", "retries_exhausted", 3) and isinstance(e.cause, TimeoutError)
    assert len(sleeps) == 2 and g.stats["failures"] == 1


def test_a_bug_is_not_retried_and_does_not_count_against_the_dependency():
    g, sleeps, _ = make_guard(threshold=1)
    fn = flaky(99, KeyError)
    with pytest.raises(KeyError):
        g.call(fn)
    assert fn.state["calls"] == 1 and sleeps == [] and g.breaker.state == "closed" and g.stats["failures"] == 0


def test_malformed_data_is_never_retried_and_trips_the_breaker_like_an_outage():
    g, sleeps, _ = make_guard(threshold=2, malformed_on=(KeyError, TypeError))
    fn = flaky(99, KeyError)
    for _ in range(2):
        with pytest.raises(DependencyUnavailable) as err:
            g.call(fn)
        assert err.value.reason == "malformed" and isinstance(err.value.cause, KeyError)
    assert fn.state["calls"] == 2 and sleeps == [] and g.stats["malformed"] == 2 and g.breaker.state == "open"


def test_validate_hook_rejects_a_record_that_came_back_wrong():
    g, _, _ = make_guard()
    with pytest.raises(DependencyUnavailable) as err:
        g.call(lambda: {"status": None}, validate=lambda r: r["status"] is not None)
    assert err.value.reason == "malformed" and g.stats["malformed"] == 1


def test_a_nested_unavailable_verdict_is_not_retried_again():
    inner, _, _ = make_guard(attempts=1)
    outer, sleeps, _ = make_guard(attempts=3)

    def fn():
        return inner.call(flaky(99))

    with pytest.raises(DependencyUnavailable):
        outer.call(fn)
    assert sleeps == [] and outer.stats["failures"] == 1


# ---- circuit breaker -------------------------------------------------------------------------------------------------
def test_breaker_opens_after_consecutive_failures_and_stops_calling_the_dependency():
    g, _, _ = make_guard(attempts=1, threshold=3)
    fn = flaky(99)
    for _ in range(3):
        with pytest.raises(DependencyUnavailable):
            g.call(fn)
    assert g.breaker.state == "open" and g.breaker.opened_count == 1
    before = fn.state["calls"]
    with pytest.raises(DependencyUnavailable) as err:
        g.call(fn)
    assert err.value.reason == "circuit_open" and fn.state["calls"] == before and g.stats["short_circuited"] == 1


def test_a_success_resets_the_failure_count_so_scattered_errors_do_not_open_it():
    g, _, _ = make_guard(attempts=1, threshold=3)
    for ok in (False, False, True, False, False):
        fn = flaky(0 if ok else 99)
        try:
            g.call(fn)
        except DependencyUnavailable:
            pass
    assert g.breaker.state == "closed"


def test_half_open_allows_one_trial_a_failed_trial_reopens_at_once_a_success_closes():
    clock = Clock()
    g, _, _ = make_guard(attempts=1, threshold=1, reset=30, clock=clock)
    with pytest.raises(DependencyUnavailable):
        g.call(flaky(99))
    assert g.breaker.state == "open"
    clock.t += 29
    assert g.breaker.state == "open"                       # not yet
    clock.t += 2
    assert g.breaker.state == "half_open"
    fn = flaky(99)
    with pytest.raises(DependencyUnavailable) as err:
        g.call(fn)                                          # the trial, and it fails
    assert err.value.reason == "retries_exhausted" and fn.state["calls"] == 1 and g.breaker.state == "open"
    with pytest.raises(DependencyUnavailable) as err:
        g.call(flaky(0))                                    # re-opened immediately: this call is refused, not tried
    assert err.value.reason == "circuit_open"
    clock.t += 31
    assert g.call(flaky(0)) == "ok" and g.breaker.state == "closed"


def test_only_one_caller_gets_the_half_open_trial():
    clock = Clock()
    br = CircuitBreaker(1, 10, clock=clock)
    br.record_failure()
    clock.t += 11
    grants = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        grants.append(br.allow())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert grants.count(True) == 1


def test_a_bug_during_the_trial_gives_the_slot_back():
    clock = Clock()
    g, _, _ = make_guard(attempts=1, threshold=1, reset=10, clock=clock)
    with pytest.raises(DependencyUnavailable):
        g.call(flaky(99))
    clock.t += 11
    with pytest.raises(ValueError):
        g.call(flaky(99, ValueError))                       # our own bug, not the dependency's
    assert g.breaker.state == "half_open"                   # still waiting for a real trial
    assert g.call(flaky(0)) == "ok"


def test_snapshot_reports_what_a_health_endpoint_needs():
    g, _, _ = make_guard(attempts=1, threshold=1)
    with pytest.raises(DependencyUnavailable):
        g.call(flaky(99))
    snap = g.snapshot()
    assert snap["name"] == "orders" and snap["state"] == "open" and snap["opened_count"] == 1 and snap["failures"] == 1
