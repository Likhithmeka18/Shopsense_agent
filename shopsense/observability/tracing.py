"""
ShopSense Observability: tracing (Milestone 7, capstone: "LangFuse tracing of ticket-resolution flows").

Tracing in one module:

  nested spans      `tracer.run(...)` opens the root span for a ticket; every agent / node / tool / LLM call inside it
                    opens a child with `tracer.span(...)`. Nesting follows the Python call stack (contextvars), so no
                    span object is ever passed around by hand.
  tagged at write   session id (= the ticket id, so a ticket's run and its later approval resume line up), a HASHED
                    user id, model, prompt version and topology are recorded when the span is written, because you
                    cannot filter on a field you never recorded.
  per-agent view    every finished span also lands in a local TraceStore, so `observability/report.py` can print the
                    per-agent calls / latency / tokens / fallback-rate table without LangFuse being reachable.
  overhead          `Tracer(enabled=False)` is a no-op, so tracing's own cost can be measured against an untraced run.
  sample + redact   customer text is redacted at this boundary (redact.py) before it reaches the local file OR LangFuse;
                    the local trace file keeps EVERY failed / degraded / escalated / slow run and a configurable share
                    of healthy ones ("sample, but keep the failures").

LangFuse is an optional exporter. If the `langfuse` package or the keys are missing, or LangFuse errors at any point, tracing
quietly degrades to the local recorder: an observability outage must never take a ticket down with it. (Verified offline against
a stand-in that mirrors the SDK's API -- `get_client`, `start_as_current_observation`, `propagate_attributes`, `update`,
`flush`; run `python check_langfuse.py` once with your keys to see a real trace arrive.)
"""
from __future__ import annotations

import contextvars
import json
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .redact import hash_id, redact_obj

PROMPT_VERSION = os.environ.get("SHOPSENSE_PROMPT_VERSION", "2026-10-13")

# Events that mean "the system served this ticket in a degraded way": the ticket is flagged, never silently passed off as healthy.
DEGRADED_EVENTS = frozenset({"guardrail.blocked", "guard.gave_up", "guard.short_circuit", "guard.malformed", "llm.fallback", "llm.failed",
                             "llm.circuit_open", "fallback.human_queue"})

_CURRENT: contextvars.ContextVar[Optional["SpanRecord"]] = contextvars.ContextVar("shopsense_current_span", default=None)
_RUN: contextvars.ContextVar[Optional["RunStats"]] = contextvars.ContextVar("shopsense_run_stats", default=None)


# --------------------------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class SpanRecord:
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    kind: str = "span"                      # span | generation | agent | tool | retriever
    start: float = 0.0                      # wall clock (time.time) -- for ordering and display
    t0: float = 0.0                         # perf_counter -- for duration
    end: Optional[float] = None
    duration_ms: float = 0.0
    ok: bool = True
    error: str = ""
    model: str = ""
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost_usd: Optional[float] = None
    session_id: str = ""
    user_id: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"span_id": self.span_id, "trace_id": self.trace_id, "parent_id": self.parent_id, "name": self.name, "kind": self.kind,
                "start": round(self.start, 3), "duration_ms": round(self.duration_ms, 2), "ok": self.ok, "error": self.error,
                "model": self.model, "input": self.input, "output": self.output, "metadata": self.metadata, "events": self.events,
                "usage": self.usage, "cost_usd": self.cost_usd, "session_id": self.session_id, "user_id": self.user_id, "tags": self.tags}


class RunStats:
    """What happened to ONE ticket while it ran: LLM calls, retries, fallbacks, order-system trouble, tokens. Cheap, always
    on, and the source of the `degraded` flag the pipeline and the API report."""

    def __init__(self) -> None:
        self.counters: Counter = Counter()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.models: list[str] = []
        self._lock = threading.Lock()

    def note(self, kind: str, **attrs: Any) -> None:
        with self._lock:
            self.counters[kind] += 1
            dep = attrs.get("dependency")
            if dep:
                self.counters[f"{kind}:{dep}"] += 1

    def note_llm(self, model: str, tokens_in: int = 0, tokens_out: int = 0, cost_usd: Optional[float] = None) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += int(tokens_in or 0)
            self.tokens_out += int(tokens_out or 0)
            self.cost_usd += float(cost_usd or 0.0)
            if model and model not in self.models:
                self.models.append(model)

    @property
    def degraded(self) -> bool:
        return any(self.counters.get(k, 0) for k in DEGRADED_EVENTS)

    def degraded_reasons(self) -> list[str]:
        return sorted(k for k in DEGRADED_EVENTS if self.counters.get(k, 0) and ":" not in k)

    def as_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": round(self.cost_usd, 6), "models": list(self.models), "degraded": self.degraded,
                "degraded_reasons": self.degraded_reasons(),
                "events": {k: v for k, v in sorted(self.counters.items()) if ":" not in k}}


def current_run_stats() -> Optional[RunStats]:
    return _RUN.get()


def add_event(kind: str, **attrs: Any) -> None:
    """Attach a point-in-time event (a retry, a fallback, a short-circuit) to the current span and count it for the run.
    Safe to call anywhere, with or without a tracer; never raises."""
    try:
        stats = _RUN.get()
        if stats is not None:
            stats.note(kind, **attrs)
        rec = _CURRENT.get()
        if rec is not None:
            rec.events.append({"kind": kind, "ts": round(time.time(), 3), **redact_obj(attrs)})
    except Exception:  # noqa: BLE001
        pass


def note_llm_usage(model: str, tokens_in: int = 0, tokens_out: int = 0, cost_usd: Optional[float] = None) -> None:
    try:
        stats = _RUN.get()
        if stats is not None:
            stats.note_llm(model, tokens_in, tokens_out, cost_usd)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------------------------------------------------
# Span handle (what the `with` block receives)
# --------------------------------------------------------------------------------------------------------------------
class SpanHandle:
    def __init__(self, rec: Optional[SpanRecord], lf: Any = None, stats: Optional[RunStats] = None):
        self._rec, self._lf, self.stats = rec, lf, stats

    @property
    def trace_id(self) -> str:
        return self._rec.trace_id if self._rec else ""

    @property
    def span_id(self) -> str:
        return self._rec.span_id if self._rec else ""

    def _lf_call(self, **kwargs: Any) -> None:
        if self._lf is None:
            return
        try:
            self._lf.update(**kwargs)
        except Exception:  # noqa: BLE001 -- LangFuse trouble must never reach the ticket
            pass

    def update(self, *, input: Any = None, output: Any = None, metadata: Optional[dict] = None, usage: Optional[dict] = None,
               cost_usd: Optional[float] = None, level: Optional[str] = None, status_message: Optional[str] = None) -> None:
        rec = self._rec
        if rec is None:
            return
        if input is not None:
            rec.input = redact_obj(input)
        if output is not None:
            rec.output = redact_obj(output)
        if metadata:
            rec.metadata.update(redact_obj(metadata))
        if usage:
            rec.usage.update(usage)
        if cost_usd is not None:
            rec.cost_usd = cost_usd
        if level == "ERROR":
            rec.ok = False
        if status_message:
            rec.error = rec.error or str(status_message)[:200]
        io = {}
        if input is not None:
            io["input"] = rec.input
        if output is not None:
            io["output"] = rec.output
        if io:
            self._lf_call(**io)
        if metadata:
            self._lf_call(metadata=redact_obj(metadata))
        if usage:
            self._lf_call(usage_details={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))})
        if cost_usd is not None:
            self._lf_call(cost_details={"total": float(cost_usd)})
        if level or status_message:
            self._lf_call(**{k: v for k, v in (("level", level), ("status_message", status_message)) if v})

    def event(self, kind: str, **attrs: Any) -> None:
        add_event(kind, **attrs)

    def fail(self, message: str) -> None:
        """Mark the span failed without an exception (e.g. a degraded fallback was used)."""
        self.update(level="ERROR", status_message=message)


_NULL = SpanHandle(None)

_CONTROL_FLOW = frozenset({"GraphInterrupt", "NodeInterrupt", "GraphBubbleUp", "ParentCommand", "GeneratorExit"})


def _is_control_flow(e: BaseException) -> bool:
    return any(c.__name__ in _CONTROL_FLOW for c in type(e).__mro__)


# --------------------------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------------------------
class TraceStore:
    """The last `max_traces` KEPT traces, in memory (and optionally appended to a JSONL file by the tracer)."""

    def __init__(self, max_traces: int = 2000):
        self._traces: deque = deque(maxlen=max_traces)
        self._lock = threading.Lock()

    def add(self, trace: dict) -> None:
        with self._lock:
            self._traces.append(trace)

    def traces(self) -> list[dict]:
        with self._lock:
            return list(self._traces)

    def spans(self) -> list[dict]:
        return [s for t in self.traces() for s in t["spans"]]

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def __len__(self) -> int:
        return len(self._traces)


# --------------------------------------------------------------------------------------------------------------------
# Tracer
# --------------------------------------------------------------------------------------------------------------------
class Tracer:
    def __init__(self, *, enabled: bool = True, store: Optional[TraceStore] = None, trace_file: Optional[str | Path] = None,
                 langfuse_client: Any = None, langfuse_propagate: Optional[Callable] = None, keep_healthy_rate: float = 1.0,
                 slow_ms: float = 120_000.0, rng: Optional[random.Random] = None):
        self.enabled = enabled
        self.store = store or TraceStore()
        self.trace_file = Path(trace_file) if trace_file else None
        self._lf, self._lf_propagate = langfuse_client, langfuse_propagate
        self.keep_healthy_rate, self.slow_ms = keep_healthy_rate, slow_ms
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._open: dict[str, list[dict]] = {}          # trace_id -> finished spans, held until the root closes (tail sampling)
        self.dropped_healthy = 0
        self.kept = 0
        self.langfuse_errors = 0

    # ---- introspection -------------------------------------------------------------------------------------------
    @property
    def langfuse_enabled(self) -> bool:
        return self._lf is not None

    def status(self) -> dict:
        return {"enabled": self.enabled, "langfuse": self.langfuse_enabled, "kept_traces": self.kept, "dropped_healthy": self.dropped_healthy,
                "keep_healthy_rate": self.keep_healthy_rate, "trace_file": str(self.trace_file) if self.trace_file else None}

    def flush(self) -> None:
        if self._lf is not None:
            try:
                self._lf.flush()
            except Exception:  # noqa: BLE001
                self.langfuse_errors += 1

    # ---- LangFuse plumbing ---------------------------------------------------------------------------------------
    _LF_KINDS = {"generation": "generation", "agent": "agent", "tool": "tool", "retriever": "retriever"}

    def _enter_lf(self, stack: ExitStack, rec: SpanRecord, input_: Any) -> Any:
        if self._lf is None:
            return None
        base: dict = {"name": rec.name}
        if input_ is not None:
            base["input"] = input_
        if rec.model and rec.kind == "generation":
            base["model"] = rec.model
        for as_type in dict.fromkeys([self._LF_KINDS.get(rec.kind, "span"), "span"]):      # richer type first, plain span as the fallback
            try:
                return stack.enter_context(self._lf.start_as_current_observation(as_type=as_type, **base))
            except Exception:  # noqa: BLE001
                self.langfuse_errors += 1
        return None

    @contextmanager
    def _propagate(self, session_id: str, user_id: str, tags: list[str]) -> Iterator[None]:
        stack = ExitStack()
        if self._lf_propagate is not None:
            try:
                kwargs: dict = {"session_id": session_id, "tags": tags}
                if user_id:
                    kwargs["user_id"] = user_id
                stack.enter_context(self._lf_propagate(**kwargs))
            except Exception:  # noqa: BLE001
                self.langfuse_errors += 1
        try:
            yield
        except BaseException:
            try:
                stack.__exit__(*sys.exc_info())
            except Exception:  # noqa: BLE001
                pass
            raise
        else:
            try:
                stack.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- spans ---------------------------------------------------------------------------------------------------
    @contextmanager
    def span(self, name: str, *, kind: str = "span", input: Any = None, metadata: Optional[dict] = None, model: str = "",
             _stats: Optional[RunStats] = None) -> Iterator[SpanHandle]:
        if not self.enabled:
            yield _NULL
            return
        parent = _CURRENT.get()
        rec = SpanRecord(span_id=uuid.uuid4().hex[:16], trace_id=parent.trace_id if parent else uuid.uuid4().hex, parent_id=parent.span_id if parent else None,
                         name=name, kind=kind, start=time.time(), t0=time.perf_counter(), model=model,
                         input=redact_obj(input) if input is not None else None, metadata=redact_obj(metadata or {}))
        if parent is not None:
            rec.session_id, rec.user_id = parent.session_id, parent.user_id
        stack = ExitStack()
        lf = self._enter_lf(stack, rec, rec.input)
        token = _CURRENT.set(rec)
        handle = SpanHandle(rec, lf, _stats or _RUN.get())
        try:
            yield handle
        except BaseException as e:
            if _is_control_flow(e):                       # LangGraph's interrupt() unwinds through the node: a pause, not a failure
                rec.metadata["paused"] = True
            else:
                rec.ok = False
                rec.error = rec.error or f"{type(e).__name__}: {str(e)[:200]}"
            self._finish(rec, token, stack, lf, exc_info=sys.exc_info())
            raise
        else:
            self._finish(rec, token, stack, lf)

    def _finish(self, rec: SpanRecord, token: Any, stack: ExitStack, lf: Any, exc_info: Any = None) -> None:
        rec.end = time.time()
        rec.duration_ms = (time.perf_counter() - rec.t0) * 1000.0
        try:
            _CURRENT.reset(token)
        except ValueError:                       # finished in a different context than it was opened in: drop the pointer instead
            _CURRENT.set(None)
        if lf is not None:
            try:
                if rec.events:
                    lf.update(metadata={"events": redact_obj(rec.events)})
                if not rec.ok:
                    lf.update(level="ERROR", status_message=rec.error)
            except Exception:  # noqa: BLE001
                self.langfuse_errors += 1
        try:
            stack.__exit__(*(exc_info or (None, None, None)))
        except Exception:  # noqa: BLE001
            self.langfuse_errors += 1
        self._collect(rec)

    # ---- tail sampling + persistence -----------------------------------------------------------------------------
    def _collect(self, rec: SpanRecord) -> None:
        with self._lock:
            self._open.setdefault(rec.trace_id, []).append(rec.to_dict())
            if rec.parent_id is not None:
                return
            spans = self._open.pop(rec.trace_id)
        stats = _RUN.get()
        keep = self._keep(rec, stats)
        if not keep:
            self.dropped_healthy += 1
            return
        trace = {"trace_id": rec.trace_id, "name": rec.name, "session_id": rec.session_id, "user_id": rec.user_id, "tags": rec.tags,
                 "started": round(rec.start, 3), "duration_ms": round(rec.duration_ms, 2), "ok": rec.ok,
                 "route": rec.metadata.get("route", ""), "degraded": bool(stats.degraded) if stats else False,
                 "run": stats.as_dict() if stats else {}, "spans": spans}
        self.kept += 1
        self.store.add(trace)
        if self.trace_file is not None:
            try:
                self.trace_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace, default=str) + "\n")
            except OSError:
                pass

    def _keep(self, root: SpanRecord, stats: Optional[RunStats]) -> bool:
        """Every run that errored, degraded, escalated, paused or ran slow is kept; only clean auto-resolved runs are sampled."""
        if not root.ok or (stats is not None and stats.degraded) or root.duration_ms >= self.slow_ms:
            return True
        if root.metadata.get("route") not in (None, "", "resolve") or root.metadata.get("paused"):
            return True
        return self.keep_healthy_rate >= 1.0 or self._rng.random() < self.keep_healthy_rate

    # ---- a ticket run ---------------------------------------------------------------------------------------------
    @contextmanager
    def run(self, name: str = "ticket.run", *, session_id: str, user_id: str = "", tags: tuple | list = (), metadata: Optional[dict] = None,
            input: Any = None) -> Iterator[SpanHandle]:
        """Root span for one ticket. `session_id` is the ticket id; `user_id` is hashed here, so a caller cannot forget to."""
        if not self.enabled:
            yield _NULL
            return
        stats = RunStats()
        token = _RUN.set(stats)
        hashed = hash_id(user_id) if user_id else ""
        tag_list = [str(t)[:120] for t in tags]
        try:
            with self.span(name, kind="agent", input=input, metadata=metadata, _stats=stats) as root:
                root._rec.session_id, root._rec.user_id, root._rec.tags = session_id, hashed, tag_list     # type: ignore[union-attr]
                with self._propagate(session_id, hashed, tag_list):
                    yield root
        finally:
            try:
                _RUN.reset(token)
            except ValueError:
                _RUN.set(None)


# --------------------------------------------------------------------------------------------------------------------
# Decorators
# --------------------------------------------------------------------------------------------------------------------
def traced_node(name: str, *, kind: str = "agent", tracer_fn: Optional[Callable[[], Tracer]] = None) -> Callable:
    """Wrap a state-dict node (`fn(state) -> update dict`) in a span that records what it read and wrote."""
    def deco(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            t = (tracer_fn or get_tracer)()
            if not t.enabled:
                return fn(state)
            with t.span(name, kind=kind, input={k: state.get(k) for k in ("ticket_id", "route", "steps") if k in state}) as sp:
                result = fn(state)
                try:
                    sp.update(output={"wrote": sorted(set(result) - {"log"}), "log": result.get("log", [])})
                except Exception:  # noqa: BLE001
                    pass
                return result
        return wrapper
    return deco


# --------------------------------------------------------------------------------------------------------------------
# The process-wide tracer
# --------------------------------------------------------------------------------------------------------------------
_TRACER: Optional[Tracer] = None
_TRACER_LOCK = threading.Lock()
_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[tracing] {msg}", file=sys.stderr)


def make_langfuse() -> tuple[Any, Optional[Callable]]:
    """(client, propagate_attributes) when LangFuse is configured and importable, else (None, None). Never raises."""
    if os.environ.get("SHOPSENSE_LANGFUSE", "1") == "0":
        return None, None
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None, None
    try:
        from langfuse import get_client, propagate_attributes
        return get_client(), propagate_attributes
    except Exception as e:  # noqa: BLE001 -- not installed / incompatible version: keep going without it
        _warn_once(f"LangFuse keys are set but the SDK could not be loaded ({type(e).__name__}: {str(e)[:100]}); tracing locally only. pip install langfuse")
        return None, None


def tracer_from_env(*, trace_file: Optional[str | Path] = None) -> Tracer:
    enabled = os.environ.get("SHOPSENSE_TRACING", "1") != "0"
    client, propagate = make_langfuse() if enabled else (None, None)
    rate = float(os.environ.get("SHOPSENSE_TRACE_SAMPLE_RATE", "1.0"))
    return Tracer(enabled=enabled, langfuse_client=client, langfuse_propagate=propagate, keep_healthy_rate=rate,
                  trace_file=trace_file or os.environ.get("SHOPSENSE_TRACE_FILE") or None)


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        with _TRACER_LOCK:
            if _TRACER is None:
                _TRACER = tracer_from_env()
    return _TRACER


def set_tracer(tracer: Optional[Tracer]) -> Optional[Tracer]:
    """Install a tracer (pipeline / API start-up / tests); returns the previous one. `None` resets to 'build from env on next use'."""
    global _TRACER
    with _TRACER_LOCK:
        prev, _TRACER = _TRACER, tracer
    return prev
