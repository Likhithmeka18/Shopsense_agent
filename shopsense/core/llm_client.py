"""
ShopSense Core: provider-agnostic LLM client (Milestone 1, part 1).

Wraps LiteLLM's completion() so the rest of the codebase never talks to a
specific provider's SDK directly. Swapping providers/models is a matter of
changing the `model=` string (e.g. "groq/llama-3.1-8b-instant", "gpt-4o-mini",
"groq/qwen/qwen3.8-27b") -- nothing downstream changes.

Also provides a SessionStore: LLM APIs are stateless, so "memory" across a
conversation is just us resending the growing message list on every call.
ShopSense uses this to hold one history per ticket thread, in case a ticket
needs a multi-turn clarification exchange with the customer.
"""
from __future__ import annotations

import os
import random
import re
import sys
import threading
import time
from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel

import litellm
from litellm import completion

from ..observability.tracing import add_event, get_tracer, note_llm_usage
from ..tools.reliability import CircuitBreaker

# LiteLLM prints a multi-line "Give Feedback / Get Help" banner on every
# provider error -- including each 429 the retry wrapper below handles
# silently. It is pure noise in a run log, so turn it off. (Setting an
# attribute is also safe against the test suite's fake litellm module.)
litellm.suppress_debug_info = True

try:
    from litellm import RateLimitError
except ImportError:  # the test suite's fake litellm module has no exception
    # classes -- the retry logic below simply never triggers under tests,
    # which is correct: the scripted fake never simulates a 429.
    class RateLimitError(Exception):  # type: ignore[no-redef]
        pass


def _exc_types(*names: str) -> tuple:
    """litellm exception classes by name (the fake litellm used in tests has none, so missing ones are skipped)."""
    out = []
    for n in names:
        c = getattr(litellm, n, None)
        if isinstance(c, type) and issubclass(c, BaseException):
            out.append(c)
    return tuple(out)


_TRANSIENT_TYPES = _exc_types("Timeout", "APITimeoutError", "APIConnectionError", "ServiceUnavailableError", "InternalServerError",
                              "BadGatewayError")
_PERMANENT_TYPES = _exc_types("AuthenticationError", "NotFoundError", "BadRequestError", "PermissionDeniedError",
                              "UnprocessableEntityError", "ContextWindowExceededError", "ContentPolicyViolationError")

# Default is a Groq model (free tier, no billing setup) -- override via env var
# or by passing model= explicitly, just change the string.
DEFAULT_MODEL = os.getenv("SHOPSENSE_LLM_MODEL", "groq/openai/gpt-oss-20b")
DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful assistant."

MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_FALLBACK_WAIT = 15.0  # seconds -- used only if the provider's own
# error message doesn't name a wait time
TRANSIENT_BASE_WAIT = 2.0        # seconds -- backoff base for connection errors / timeouts / 5xx (doubles, +/- jitter, capped)
TRANSIENT_MAX_WAIT = 30.0
MAX_AUTO_WAIT_SECONDS = 600.0  # a wait longer than this is a quota problem, not a blip
DEFAULT_TIMEOUT_S = 60.0         # litellm's own default is 10 minutes; one hung call must not stall a ticket batch that long


# A request deadline for interactive callers (the chat API): a MONOTONIC timestamp after which no further waiting is allowed. A batch job
# leaves it unset and waits out rate limits (up to MAX_AUTO_WAIT_SECONDS); a customer waiting on a chat reply cannot wait seven minutes.
# Module-level (not a contextvar) on purpose: the chat service runs one ticket at a time and graph nodes run on worker threads.
_request_deadline: list = [None]


def set_request_deadline(budget_s: Optional[float]) -> None:
    """Allow at most `budget_s` more seconds of WAITING (retries / rate-limit sleeps) from now; None clears it."""
    _request_deadline[0] = None if budget_s is None else time.monotonic() + float(budget_s)


def _deadline_left() -> Optional[float]:
    d = _request_deadline[0]
    return None if d is None else d - time.monotonic()


class LLMUnavailableError(RuntimeError):
    """The model could not be used and there was nothing left to try (circuit open / every model failed). The ticket graph catches
    every exception around the model, so this becomes a visible hand-off to a human, never a made-up answer."""

    def __init__(self, model: str, reason: str, detail: str = ""):
        self.model, self.reason = model, reason
        super().__init__(f"{model} unavailable ({reason}){': ' + detail if detail else ''}")


# --------------------------------------------------------------------------------------------------------------------
# Reading the provider's own advice on how long to wait
# --------------------------------------------------------------------------------------------------------------------
_DURATION = r"([\d.]+ms|(?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?)"
_WAIT_RE = re.compile(r"(?:try again|retry) in\s+" + _DURATION)
_GEMINI_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def parse_retry_wait(message: str) -> Optional[float]:
    """Seconds the provider says to wait, from text like "try again in
    45.2s", "...in 2m33.36s", "...in 1h2m3s", "...in 850ms", Gemini's "Please
    retry in 46.4s" or its JSON "'retryDelay': '46s'". Groq switches to the
    minutes form for daily-token (TPD) limits; the old parser only understood
    plain seconds, fell back to a 15s guess, and burned all its retries a
    minute before the quota actually reset. Gemini says "retry in" (not "try
    again in"), which the old parser did not read at all."""
    message = message or ""
    m = _WAIT_RE.search(message)
    if m and m.group(1):
        text = m.group(1)
        if text.endswith("ms"):
            return float(text[:-2]) / 1000.0
        total = 0.0
        for value, unit in re.findall(r"([\d.]+)([hms])", text):
            total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        if total:
            return total
    g = _GEMINI_DELAY_RE.search(message)
    if g:
        return float(g.group(1)) or None
    return None


def _retry_after_header(e: BaseException) -> Optional[float]:
    try:
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def classify_llm_error(e: BaseException) -> str:
    """'rate_limit' (wait the provider's time, then retry), 'transient' (connection / timeout / 5xx: back off and retry) or
    'permanent' (bad key, unknown model, malformed request: retrying cannot help, so say so at once)."""
    status = getattr(e, "status_code", None)
    text = f"{type(e).__name__} {e}".lower()
    if isinstance(e, RateLimitError) or status == 429 or any(t in text for t in ("ratelimit", "rate limit", "resource_exhausted", "too many requests")):
        return "rate_limit"
    if isinstance(e, _PERMANENT_TYPES) or status in (400, 401, 403, 404, 422):
        return "permanent"
    if (isinstance(e, _TRANSIENT_TYPES) or isinstance(e, (ConnectionError, TimeoutError)) or status in (408, 500, 502, 503, 504, 529)
            or any(t in text for t in ("timeout", "timed out", "connection", "service unavailable", "overloaded", "bad gateway",
                                       "internal server error", "temporarily unavailable"))):
        return "transient"
    return "permanent"          # something we do not understand: do not loop on it


# --------------------------------------------------------------------------------------------------------------------
# Process-wide resilience state: one breaker per model, counters for /health and the dashboard, optional pacing
# --------------------------------------------------------------------------------------------------------------------
LLM_STATS: Counter = Counter()
_BREAKERS: dict[str, CircuitBreaker] = {}
_STATE_LOCK = threading.Lock()
_PACE_LOCK = threading.Lock()
_last_call_at = [0.0]
_announced: set[str] = set()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _breaker_for(model: str) -> CircuitBreaker:
    with _STATE_LOCK:
        if model not in _BREAKERS:
            _BREAKERS[model] = CircuitBreaker(failure_threshold=int(os.environ.get("SHOPSENSE_LLM_BREAKER_THRESHOLD", "3")),
                                              reset_timeout=float(os.environ.get("SHOPSENSE_LLM_BREAKER_RESET_S", "60")))
        return _BREAKERS[model]


def reset_llm_state() -> None:
    """Forget breakers, counters and pacing (tests; also useful after fixing a key)."""
    with _STATE_LOCK:
        _BREAKERS.clear()
        _announced.clear()
    LLM_STATS.clear()
    _last_call_at[0] = 0.0


def llm_health() -> dict:
    """What /health and the pipeline summary report: per-model breaker state plus call counters."""
    with _STATE_LOCK:
        breakers = {m: b.snapshot() for m, b in _BREAKERS.items()}
    return {"models": breakers, "counters": dict(LLM_STATS), "fallback_models": _fallback_models()}


def _fallback_models() -> list[str]:
    return [m.strip() for m in os.environ.get("SHOPSENSE_LLM_FALLBACK_MODELS", "").split(",") if m.strip()]


def _announce_once(key: str, message: str) -> None:
    with _STATE_LOCK:
        if key in _announced:
            return
        _announced.add(key)
    print(f"    [{message}]")


def _pace() -> None:
    """Optional minimum gap between model calls (SHOPSENSE_LLM_MIN_INTERVAL seconds). A free tier is a requests-per-minute
    budget; spacing calls evenly is cheaper than hitting the limit and sleeping on the provider's penalty."""
    interval = float(os.environ.get("SHOPSENSE_LLM_MIN_INTERVAL", "0") or 0)
    if interval <= 0:
        return
    with _PACE_LOCK:
        gap = interval - (time.monotonic() - _last_call_at[0])
        if gap > 0:
            _sleep(gap)
        _last_call_at[0] = time.monotonic()


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else ''}"


def _attempt_model(model: str, messages: list[dict], kwargs: dict, breaker: CircuitBreaker) -> Any:
    """Call ONE model, waiting out rate limits and backing off on transient errors. Raises when it cannot succeed."""
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        left = _deadline_left()
        if left is not None and left <= 0:
            add_event("llm.deadline_exceeded", model=model, attempt=attempt)
            raise LLMUnavailableError(model, "deadline_exceeded")
        _pace()
        LLM_STATS["attempts"] += 1
        try:
            resp = completion(model=model, messages=messages, **kwargs)
        except Exception as e:  # noqa: BLE001 -- classified below
            kind = classify_llm_error(e)
            if kind == "permanent":
                breaker.abort_trial()                       # a bad key / unknown model is OUR configuration, not an outage
                LLM_STATS["permanent_errors"] += 1
                add_event("llm.permanent_error", model=model, error=_describe(e))
                raise
            if kind == "rate_limit":
                LLM_STATS["rate_limited"] += 1
                advised = parse_retry_wait(str(e))
                if advised is None:
                    advised = _retry_after_header(e)
                wait_s = advised + 1.0 if advised is not None else RATE_LIMIT_FALLBACK_WAIT          # +1s safety margin
                what = "rate limited by the provider"
            else:
                LLM_STATS["transient_errors"] += 1
                wait_s = min(TRANSIENT_MAX_WAIT, TRANSIENT_BASE_WAIT * (2 ** (attempt - 1))) * (0.75 + 0.5 * random.random())
                what = f"provider error ({type(e).__name__})"
            if wait_s > MAX_AUTO_WAIT_SECONDS:
                print(f"    [provider asks for a {wait_s / 60:.0f}-minute wait -- that is a quota limit "
                      f"(e.g. a daily token cap), not a blip; stopping instead of waiting]")
                breaker.record_failure()
                add_event("llm.quota_exhausted", model=model, wait_s=round(wait_s), error=_describe(e))
                raise
            left = _deadline_left()
            if left is not None and wait_s > left:
                # Waiting would run past the caller's deadline: give up now so the ticket can be handed to a person while the
                # customer is still there. Counts as a failure -- a provider that keeps rate-limiting us IS struggling.
                print(f"    [provider asks for a {wait_s:.0f}s wait but only {max(left, 0):.0f}s of this request's time is left -- giving up]")
                breaker.record_failure()
                add_event("llm.deadline_exceeded", model=model, wait_s=round(wait_s), left_s=round(max(left, 0), 1), error=_describe(e))
                raise
            if attempt == MAX_RATE_LIMIT_RETRIES:
                breaker.record_failure()
                add_event("llm.retries_exhausted", model=model, attempts=attempt, error=_describe(e))
                raise
            LLM_STATS["retries"] += 1
            print(f"    [{what} -- waiting {wait_s:.0f}s, retry {attempt}/{MAX_RATE_LIMIT_RETRIES}]")
            add_event("llm.retry", model=model, error_kind=kind, wait_s=round(wait_s, 1), attempt=attempt)
            _sleep(wait_s)
            continue
        breaker.record_success()
        return resp
    raise LLMUnavailableError(model, "retries_exhausted")   # unreachable: the loop returns or raises


def _usage_of(resp: Any) -> tuple[int, int, Optional[float]]:
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            return 0, 0, None
        get = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
        tin, tout = int(get("prompt_tokens") or 0), int(get("completion_tokens") or 0)
    except Exception:  # noqa: BLE001
        return 0, 0, None
    cost = None
    try:
        cost_fn = getattr(litellm, "completion_cost", None)
        if cost_fn is not None:
            cost = float(cost_fn(completion_response=resp))
    except Exception:  # noqa: BLE001 -- unknown price (free tier / new model): report tokens, not a made-up cost
        cost = None
    return tin, tout, cost


def _completion_with_retry(*, model: str, messages: list[dict], **kwargs: Any):
    """Provider call with Milestone 7 hardening, in this order:

      1. each model has its OWN circuit breaker: after a few consecutive exhausted calls it stops being called for a cool-down,
         so a dead provider costs one fast refusal per ticket instead of minutes of waiting;
      2. rate limits wait the time the provider names (Groq, Gemini, Retry-After header); connection errors, timeouts and 5xx back
         off exponentially with jitter; bad keys / unknown models / malformed requests are NOT retried and raise at once;
      3. if SHOPSENSE_LLM_FALLBACK_MODELS names other models, they are tried in order -- loudly (an event on the trace, a line
         in the log, the ticket flagged degraded), because a silent fallback hides a dead primary;
      4. when nothing works the PRIMARY model's own error is raised (callers and the graph's `extraction_failed` path behave as
         before, and the real error text is preserved instead of being replaced by a generic message).

    One `llm.generate` span per logical call carries model, token usage, cost when the price is known, and what it took."""
    kwargs.setdefault("timeout", float(os.environ.get("SHOPSENSE_LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S)))
    candidates = [model] + [m for m in _fallback_models() if m != model]
    LLM_STATS["calls"] += 1
    with get_tracer().span("llm.generate", kind="generation", model=model, input=messages) as span:
        errors: list[tuple[str, BaseException]] = []
        for i, m in enumerate(candidates):
            breaker = _breaker_for(m)
            if not breaker.allow():
                LLM_STATS["circuit_open"] += 1
                add_event("llm.circuit_open", model=m)
                errors.append((m, LLMUnavailableError(m, "circuit_open", "too many recent failures; will retry after the cool-down")))
                continue
            try:
                resp = _attempt_model(m, messages, kwargs, breaker)
            except Exception as e:  # noqa: BLE001 -- recorded, then the next candidate (if any) gets its turn
                errors.append((m, e))
                continue
            tin, tout, cost = _usage_of(resp)
            note_llm_usage(m, tin, tout, cost)
            if i > 0:
                LLM_STATS["fallbacks"] += 1
                add_event("llm.fallback", model=m, primary=model, primary_error=_describe(errors[0][1]))
                _announce_once(f"fallback:{model}->{m}", f"primary model {model} failed ({_describe(errors[0][1])}); using fallback {m}")
            try:
                text = resp.choices[0].message.content
            except Exception:  # noqa: BLE001
                text = None
            span.update(output=text, usage={"input": tin, "output": tout, "total": tin + tout} if (tin or tout) else None, cost_usd=cost,
                        metadata={"model_used": m, "fallback": i > 0})
            return resp
        LLM_STATS["failures"] += 1
        primary_error = next((e for _m, e in errors if not isinstance(e, LLMUnavailableError)), errors[0][1])
        add_event("llm.failed", model=model, error=_describe(primary_error), tried=[m for m, _e in errors])
        raise primary_error


def llm_chat(
    user_input: str,
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    **kwargs: Any,
) -> str:
    """One-shot provider-agnostic chat call. Returns the reply text only."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    resp = _completion_with_retry(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content


def llm_completion(messages: list[dict], *, model: str = DEFAULT_MODEL, **kwargs: Any):
    """Pass-through to litellm.completion (with rate-limit retry) for callers
    that need the full response object (e.g. to read tool_calls, reasoning,
    finish_reason)."""
    return _completion_with_retry(model=model, messages=messages, **kwargs)


def ask_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
) -> Optional[BaseModel]:
    """JSON-mode LLM call validated against a Pydantic schema, with one
    repair retry on failure -- the exact same "extract -> validate -> repair
    by re-prompting with the error" shape core/ticket_intake.py already uses
    for Milestone 1, reused here (Milestones 5 and 6) so a planner's topic
    list, a drafted recommendation, or a routing choice can be structured
    output without pulling in LangChain's ChatModel/with_structured_output()
    just for this -- litellm's JSON mode plus Pydantic is already the
    project's established pattern.

    Returns None (never raises) if both the original attempt and the one
    repair attempt still fail validation. Every caller of this function
    MUST have a deterministic fallback for the None case -- this is
    intentionally not a "trust the model" helper. """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = _completion_with_retry(model=model, messages=messages, response_format={"type": "json_object"})
    raw = resp.choices[0].message.content or ""

    for attempt in range(max_retries + 1):
        try:
            return schema.model_validate_json(raw)
        except Exception as e:  # noqa: BLE001 -- json.JSONDecodeError or pydantic.ValidationError
            if attempt == max_retries:
                return None
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That failed validation: {e}. Return corrected JSON only, no commentary.",
            })
            resp = _completion_with_retry(model=model, messages=messages, response_format={"type": "json_object"})
            raw = resp.choices[0].message.content or ""
    return None


class SessionStore:
    """Maps session_id -> that session's own message history, so independent
    conversations never bleed into each other. One history per ticket thread."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, model: str = DEFAULT_MODEL):
        self._sessions: dict[str, list[dict]] = {}
        self._system_prompt = system_prompt
        self._model = model

    def get_or_create(self, session_id: str) -> list[dict]:
        if session_id not in self._sessions:
            self._sessions[session_id] = [{"role": "system", "content": self._system_prompt}]
        return self._sessions[session_id]

    def chat(self, session_id: str, user_input: str, **kwargs: Any) -> str:
        history = self.get_or_create(session_id)
        history.append({"role": "user", "content": user_input})
        resp = _completion_with_retry(model=self._model, messages=history, **kwargs)
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())
