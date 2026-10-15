"""
ShopSense Tools: reliability building blocks -- retry with backoff, a real three-state circuit breaker, a
`DependencyGuard` that combines them, and call logging (Milestones 2 and 7).

Milestone 7 (capstone wording): "reliability hardening for order-system API flakiness (retries, fallback to human queue,
circuit breaker)". This module implements four ideas, and the pitfalls it avoids:

    retries         only for TRANSIENT errors (connection, timeout). A bug (KeyError) or malformed data is not transient, and
                    retrying it only repeats the same bad odds -- so it is never retried.
    circuit breaker CLOSED -> OPEN after N consecutive failures (stop calling a dependency that is down: protect it and our
                    own latency budget) -> HALF_OPEN after a cool-down, where exactly ONE trial call is allowed; a failure
                    re-opens at once, a success closes. ("A recovered breaker that immediately re-opens" is what happens when
                    half-open is treated like closed.)
    fallback        the guard never decides what the fallback is -- it raises `DependencyUnavailable`, a typed, visible
                    signal. The caller degrades (the ticket graph hands the ticket to the human queue) instead of guessing.
    visibility      every retry, short-circuit and give-up is counted (`guard.stats`) and reported to the tracer, because "a
                    system that silently degrades is worse than one that fails loudly".

Nothing here knows about tickets or orders, so the same guard protects the order system, the policy index and the LLM.
"""
from __future__ import annotations

import functools
import random
import threading
import time
from typing import Any, Callable, Optional

TOOL_CALL_LOG: list[dict] = []


class TransientDependencyError(Exception):
    """A failure worth retrying (a real client should raise this, or ConnectionError / TimeoutError, for a blip)."""


TRANSIENT_ERRORS: tuple = (ConnectionError, TimeoutError, TransientDependencyError)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.5) -> Callable:
    """Retry a flaky call with exponential backoff."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 -- intentionally broad: any tool failure retries
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


class CircuitBreaker:
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    Opens after `failure_threshold` consecutive failures and rejects every call for `reset_timeout` seconds. After that it is
    HALF_OPEN: `allow()` hands out exactly ONE trial call. `record_success()` closes it; `record_failure()` re-opens it at once.
    `clock` is injectable so tests can move time without sleeping."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 30.0, *, clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock
        self._lock = threading.RLock()
        self._failures = 0
        self._opened_at: float | None = None
        self._state = self.CLOSED
        self._trial_in_flight = False
        self.opened_count = 0                     # how many times it has tripped (a metric worth alerting on)

    # -- internal: OPEN turns into HALF_OPEN once the cool-down has passed
    def _refresh(self) -> None:
        if self._state == self.OPEN and self._opened_at is not None and self._clock() - self._opened_at >= self.reset_timeout:
            self._state = self.HALF_OPEN
            self._trial_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh()
            return self._state

    @property
    def is_open(self) -> bool:
        """Would a call be rejected right now? (Does NOT claim the half-open trial; use `allow()` before calling.)"""
        with self._lock:
            self._refresh()
            return self._state == self.OPEN or (self._state == self.HALF_OPEN and self._trial_in_flight)

    def allow(self) -> bool:
        """Ask permission to call. In HALF_OPEN only the first caller gets it (that call IS the trial)."""
        with self._lock:
            self._refresh()
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN and not self._trial_in_flight:
                self._trial_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = self.CLOSED
            self._trial_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._refresh()
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                if self._state != self.OPEN:
                    self.opened_count += 1
                self._state = self.OPEN
                self._opened_at = self._clock()
                self._trial_in_flight = False

    def abort_trial(self) -> None:
        """The trial call died of something that says nothing about the dependency (a bug in OUR code): give the slot back."""
        with self._lock:
            self._trial_in_flight = False

    def snapshot(self) -> dict:
        with self._lock:
            self._refresh()
            return {"state": self._state, "consecutive_failures": self._failures, "opened_count": self.opened_count}


def make_robust_tool(fn: Callable, tool_name: str, *, max_retries: int = 3) -> Callable:
    """Wrap a plain function with retry + circuit breaker + call logging.
    Every invocation (success, failure, or circuit-open rejection) is
    appended to TOOL_CALL_LOG for observability."""
    breaker = CircuitBreaker()
    robust_fn = retry_with_backoff(max_retries=max_retries)(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {"tool": tool_name, "args": args, "kwargs": kwargs, "ts": time.time()}
        if not breaker.allow():
            entry["outcome"] = "circuit_open"
            TOOL_CALL_LOG.append(entry)
            raise RuntimeError(f"Circuit breaker open for tool '{tool_name}'")
        try:
            result = robust_fn(*args, **kwargs)
            breaker.record_success()
            entry["outcome"] = "success"
            TOOL_CALL_LOG.append(entry)
            return result
        except Exception:
            breaker.record_failure()
            entry["outcome"] = "failure"
            TOOL_CALL_LOG.append(entry)
            raise

    return wrapper


# --------------------------------------------------------------------------------------------------------------------
# DependencyGuard: retries (transient only) + breaker + a typed "unavailable" signal, with counters
# --------------------------------------------------------------------------------------------------------------------
class DependencyUnavailable(RuntimeError):
    """The dependency could not be used. `reason` is one of: circuit_open | retries_exhausted | malformed.
    Catch this where you can degrade safely (e.g. hand the ticket to a human); it is deliberately NOT a transient error."""

    def __init__(self, dependency: str, reason: str, *, attempts: int = 0, cause: Optional[BaseException] = None):
        self.dependency, self.reason, self.attempts, self.cause = dependency, reason, attempts, cause
        detail = f": {type(cause).__name__}: {str(cause)[:120]}" if cause is not None else ""
        super().__init__(f"{dependency} unavailable ({reason}, {attempts} attempt(s){detail})")


def _emit(kind: str, **attrs: Any) -> None:
    """Tell the tracer (if one is configured) what the guard just did. Never allowed to break the call it describes."""
    try:
        from ..observability.tracing import add_event

        add_event(kind, **attrs)
    except Exception:  # noqa: BLE001
        pass


class DependencyGuard:
    """One guard per dependency (order system, policy index, an LLM model). `call(fn, ...)` returns fn's result or raises
    `DependencyUnavailable`; exceptions that are not transient (a KeyError in our own code) propagate untouched and do not
    count against the dependency."""

    def __init__(self, name: str, *, max_attempts: int = 3, base_delay: float = 0.2, max_delay: float = 2.0, jitter: float = 0.25,
                 retry_on: tuple = TRANSIENT_ERRORS, malformed_on: tuple = (), breaker: Optional[CircuitBreaker] = None,
                 sleep: Callable[[float], None] = time.sleep, rng: Optional[random.Random] = None):
        self.name, self.max_attempts = name, max(1, int(max_attempts))
        self.base_delay, self.max_delay, self.jitter = base_delay, max_delay, jitter
        self.retry_on = retry_on
        # Exceptions that mean "the dependency answered with garbage" (a record missing a field, a wrong type). Never retried;
        # they count against the dependency and surface as DependencyUnavailable("malformed"). Empty by default so that a bug
        # in OUR code is not blamed on the dependency -- only guards that wrap a pure data fetch opt in.
        self.malformed_on = tuple(malformed_on)
        self.breaker = breaker or CircuitBreaker()
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.stats = {"calls": 0, "successes": 0, "retries": 0, "failures": 0, "short_circuited": 0, "malformed": 0}

    def _delay(self, attempt: int) -> float:
        base = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return base * (1.0 + self.jitter * (2 * self._rng.random() - 1))      # exponential backoff with +/- jitter

    def call(self, fn: Callable, *args: Any, validate: Optional[Callable[[Any], bool]] = None, **kwargs: Any) -> Any:
        self.stats["calls"] += 1
        if not self.breaker.allow():
            self.stats["short_circuited"] += 1
            _emit("guard.short_circuit", dependency=self.name, breaker=self.breaker.state)
            raise DependencyUnavailable(self.name, "circuit_open")
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = fn(*args, **kwargs)
            except self.retry_on as e:
                last = e
                if attempt < self.max_attempts:
                    self.stats["retries"] += 1
                    _emit("guard.retry", dependency=self.name, attempt=attempt, error=f"{type(e).__name__}: {str(e)[:100]}")
                    self._sleep(self._delay(attempt))
                    continue
                break
            except DependencyUnavailable:
                self.stats["failures"] += 1                         # a nested guard already retried; do not retry its verdict
                self.breaker.record_failure()
                raise
            except self.malformed_on as e:                          # garbage from the dependency: never retried
                self.stats["malformed"] += 1
                self.stats["failures"] += 1
                self.breaker.record_failure()
                _emit("guard.malformed", dependency=self.name, error=f"{type(e).__name__}: {str(e)[:100]}")
                raise DependencyUnavailable(self.name, "malformed", attempts=attempt, cause=e) from e
            except BaseException:
                self.breaker.abort_trial()                          # our own bug says nothing about the dependency
                raise
            if validate is not None and not validate(result):       # malformed data: retrying repeats the same bad odds
                self.stats["malformed"] += 1
                self.stats["failures"] += 1
                self.breaker.record_failure()
                _emit("guard.malformed", dependency=self.name)
                raise DependencyUnavailable(self.name, "malformed", attempts=attempt)
            self.stats["successes"] += 1
            self.breaker.record_success()
            return result
        self.stats["failures"] += 1
        self.breaker.record_failure()
        _emit("guard.gave_up", dependency=self.name, attempts=self.max_attempts, error=f"{type(last).__name__}: {str(last)[:100]}")
        raise DependencyUnavailable(self.name, "retries_exhausted", attempts=self.max_attempts, cause=last)

    def snapshot(self) -> dict:
        return {"name": self.name, **self.stats, **self.breaker.snapshot()}
