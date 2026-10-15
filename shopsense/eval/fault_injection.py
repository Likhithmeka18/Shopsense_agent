"""
ShopSense Eval: seeded fault injection applied to the ticket graph (Milestone 7).

The point: do not ASSUME the system survives a flaky dependency, break it on purpose, in a repeatable way, and put a number on
what hardening bought. Everything here is deterministic: a `random.Random(seed)` decides every fault, so the same seed gives the
same numbers, the tests can assert them exactly, and a regression shows up as a changed number instead of an opinion.

Three experiments, none of which needs a network, an API key or langgraph:

  order_system_experiment   the same tickets through (a) a healthy order system, (b) a flaky one with NO hardening, (c) the same
                            flaky one with the guard (retries + breaker + human-queue fallback). Reports crashes, tickets served
                            normally / degraded to a person, and -- the number that matters most -- UNSAFE outcomes (an action the
                            healthy run would not have taken) and duplicate payments. Both must be zero.
  outage_experiment         a sustained outage in the middle of the run: shows the breaker opening (so we stop hammering a dead
                            system), every affected ticket reaching a person, and normal service resuming by itself afterwards.
  llm_experiment            a flaky model provider (503s and rate limits) with a bare call versus the hardened client; and a dead
                            primary model with and without a fallback model.

What is faulted: POINT LOOKUPS on the order tables (`orders`, `customers`, `shipments`, `refunds` -- `.get`, `[]`, `in`), like a
real API call. Bulk iteration is not randomly faulted, but during an outage every access fails.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..tools.order_tools import MockAPI
from ..tools.reliability import CircuitBreaker, DependencyGuard


# --------------------------------------------------------------------------------------------------------------------
# The fault model
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FaultConfig:
    """Per-lookup probabilities. `fail_rate` = connection error, `timeout_rate` = timeout, `malformed_rate` = the record comes back
    with a field missing. They are mutually exclusive draws from ONE random number, so they must sum to < 1."""
    seed: int = 42
    fail_rate: float = 0.0
    timeout_rate: float = 0.0
    malformed_rate: float = 0.0
    lost_ack_rate: float = 0.0        # a WRITE succeeds but the caller sees a timeout (own RNG stream, so lookups are unaffected)

    def __post_init__(self) -> None:
        if self.fail_rate + self.timeout_rate + self.malformed_rate >= 1.0:
            raise ValueError("fault rates must sum to less than 1")


class FaultInjector:
    """Decides, for each lookup, whether it works. One RNG draw per lookup, always, so the schedule does not depend on which
    branch the code under test happens to take."""

    def __init__(self, cfg: FaultConfig):
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self.down = False                                   # a sustained outage: everything fails until it is switched off
        self.counts: Counter = Counter()

    def roll(self) -> Optional[str]:
        self.counts["lookups"] += 1
        draw = self._rng.random()                          # always drawn, outage or not
        if self.down:
            self.counts["outage"] += 1
            return "outage"
        c = self.cfg
        if draw < c.timeout_rate:
            kind = "timeout"
        elif draw < c.timeout_rate + c.fail_rate:
            kind = "connection"
        elif draw < c.timeout_rate + c.fail_rate + c.malformed_rate:
            kind = "malformed"
        else:
            return None
        self.counts[kind] += 1
        return kind

    def raise_if_down(self, table: str) -> None:
        if self.down:
            self.counts["outage"] += 1
            raise ConnectionError(f"{table}: order system unreachable (injected outage)")


def _corrupt(row: dict, rng: random.Random) -> dict:
    """The record, with one of its fields missing -- what a half-written or schema-drifted response looks like."""
    keys = sorted(row)
    if not keys:
        return {}
    drop = keys[int(rng.random() * len(keys)) % len(keys)]
    return {k: v for k, v in row.items() if k != drop}


class FlakyTable(dict):
    """A dict whose point lookups can fail the way a network call does. Subclassing dict keeps every existing caller working
    (`.get`, `[]`, `in`, `.items()`), and writes go straight to the real data."""

    def __init__(self, name: str, data: dict, injector: FaultInjector):
        super().__init__(data)
        self._name, self._inj = name, injector

    def _fault(self, found: bool, value: Any) -> Any:
        kind = self._inj.roll()
        if kind in ("connection", "outage"):
            raise ConnectionError(f"{self._name}: connection refused (injected {kind})")
        if kind == "timeout":
            raise TimeoutError(f"{self._name}: request timed out (injected)")
        if kind == "malformed" and found and isinstance(value, dict):
            return _corrupt(value, self._inj._rng)
        return value

    def get(self, key, default=None):
        found = dict.__contains__(self, key)
        return self._fault(found, dict.get(self, key, default))

    def __getitem__(self, key):
        found = dict.__contains__(self, key)
        value = self._fault(found, dict.__getitem__(self, key) if found else None)
        if not found:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        self._fault(False, None)                           # existence checks are calls too; "malformed" cannot apply to them
        return dict.__contains__(self, key)

    # bulk reads: not randomly faulted, but an outage takes them down too
    def items(self):
        self._inj.raise_if_down(self._name)
        return dict.items(self)

    def values(self):
        self._inj.raise_if_down(self._name)
        return dict.values(self)

    def keys(self):
        self._inj.raise_if_down(self._name)
        return dict.keys(self)


class FlakyMockAPI(MockAPI):
    """A MockAPI whose four tables are `FlakyTable`s sharing one injector."""

    def __init__(self, cfg: FaultConfig = FaultConfig(), *, base: Optional[MockAPI] = None):
        base = base or MockAPI()
        self.injector = FaultInjector(cfg)
        self.customers = FlakyTable("customers", base.customers, self.injector)
        self.orders = FlakyTable("orders", base.orders, self.injector)
        self.shipments = FlakyTable("shipments", base.shipments, self.injector)
        self.refunds = FlakyTable("refunds", base.refunds, self.injector)
        items = getattr(base, "_items_by_order", None)
        if items is not None:
            self._items_by_order = items


class NoGuard:
    """The 'before' picture: the same interface as `DependencyGuard`, with no retries, no breaker and no fallback."""
    name = "order_system"
    stats: dict = {}

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("validate", None)
        return fn(*args, **kwargs)

    def snapshot(self) -> dict:
        return {"name": "none", "state": "n/a"}


class LossyActions:
    """Wraps `ActionTools`: with probability `rate` a write is carried out and THEN reports a timeout -- the dangerous failure, where
    a retry could pay twice if the action were not idempotent. Reads and attributes (the ledger) pass straight through."""
    WRITES = ("issue_refund", "approve_pending_refund", "request_replacement", "cancel_order")

    def __init__(self, inner: Any, rate: float, seed: int):
        self._inner, self._rate, self._rng = inner, rate, random.Random(seed + 1000)
        self.lost_acks = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name not in self.WRITES or not callable(attr) or self._rate <= 0:
            return attr

        def write(*a: Any, **k: Any) -> Any:
            result = attr(*a, **k)
            if self._rng.random() < self._rate:
                self.lost_acks += 1
                raise TimeoutError(f"{name}: acknowledgement lost (injected) -- the write DID happen")
            return result
        return write


class FakeClock:
    """Time that only moves when the experiment says so (the breaker's cool-down is measured on it)."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_experiment_guard(seed: int, clock: Optional[FakeClock] = None, *, max_attempts: int = 3, threshold: int = 5,
                          reset_s: float = 30.0) -> DependencyGuard:
    """The production guard, with no real sleeping and a seeded jitter RNG."""
    from ..workflows.ticket_resolution_graph import ORDER_MALFORMED_ERRORS
    return DependencyGuard("order_system", max_attempts=max_attempts, base_delay=0.2, sleep=lambda s: None, rng=random.Random(seed),
                           malformed_on=ORDER_MALFORMED_ERRORS,
                           breaker=CircuitBreaker(failure_threshold=threshold, reset_timeout=reset_s, clock=clock or FakeClock()))


# --------------------------------------------------------------------------------------------------------------------
# Running tickets and judging outcomes
# --------------------------------------------------------------------------------------------------------------------
def load_intake_tickets(limit: Optional[int] = None) -> list[dict]:
    import json
    from ..runtime import ROOT
    path = ROOT / "data" / "shopsense" / "intake" / "tickets.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


@dataclass
class TicketOutcome:
    ticket_id: str
    route: str = ""                    # resolve / refund_approval / human_review ...
    status: str = ""
    crashed: str = ""                  # exception type, when the ticket blew up
    fell_back: bool = False            # served by a person because the order system could not be used
    committed: int = 0                 # money / order actions actually taken (ledger records)
    approved: bool = False             # paused for approval and resumed with "approve"


def route_of(result: Any) -> tuple[str, str, list]:
    from ..workflows import ticket_resolution_graph as g
    from .offline import Paused
    if isinstance(result, Paused):
        return "refund_approval", "awaiting_approval", result.state.get("reasons") or []
    out = g.outcome_of(result)
    return out.get("route", ""), result.get("status", ""), result.get("reasons") or []


def run_tickets(tickets: Iterable[dict], *, api: MockAPI, guard: Any, kb: Any, extract: Callable, on_ticket: Optional[Callable] = None,
                approve: bool = True, lost_ack_rate: float = 0.0, seed: int = 0) -> tuple[list[TicketOutcome], Any]:
    """Run each ticket through the pure executor on ONE shared set of dependencies (so the breaker keeps its state between
    tickets, like a real service). A ticket that stops for approval is approved at once (as an executive) so the WRITE path is
    exercised under faults too. Returns (outcomes, deps)."""
    from ..runtime import build_deps
    from ..workflows import ticket_resolution_graph as g
    from . import offline

    deps = build_deps(kb=kb, extract=extract, api=api, order_guard=guard, with_memo_team=False)
    if lost_ack_rate > 0:
        deps.actions = LossyActions(deps.actions, lost_ack_rate, seed)
    out: list[TicketOutcome] = []
    for i, t in enumerate(tickets):
        if on_ticket:
            on_ticket(i, t)
        o = TicketOutcome(ticket_id=t["ticket_id"])
        before = len(deps.actions.ledger.records)
        try:
            result = offline.run(deps, g.seed_state(ticket_id=t["ticket_id"], customer_id=t.get("customer_id"), text=t["text"],
                                                    channel=t.get("channel", "chat"), created_at=t.get("created_at", "")))
            o.route, o.status, reasons = route_of(result)           # the ROUTING decision (before any approval)
            if isinstance(result, offline.Paused) and approve:
                o.approved = True
                result = offline.resume(deps, result, decision="approve", approver_role="executive", notes="fault-injection auto-approve")
                o.status = result.get("status", "")
            state = result.state if isinstance(result, offline.Paused) else result
            o.fell_back = "order_system_unavailable" in (state.get("reasons") or []) or bool((state.get("action_result") or {}).get("unavailable"))
        except Exception as e:                              # noqa: BLE001 -- a crash IS the measurement
            o.crashed = type(e).__name__
        o.committed = len(deps.actions.ledger.records) - before
        out.append(o)
    return out, deps


def summarize(outcomes: list[TicketOutcome], clean: Optional[list[TicketOutcome]] = None, *, deps: Any = None) -> dict:
    """Counts for one stage. With the healthy run as `clean`, also the UNSAFE count: tickets where the faulty run took an action
    (committed money / changed an order) that the healthy run did not, or took a different non-escalated decision."""
    n = len(outcomes)
    crashed = [o for o in outcomes if o.crashed]
    fell_back = [o for o in outcomes if o.fell_back and not o.crashed]
    served = [o for o in outcomes if not o.crashed and not o.fell_back]
    s: dict = {"tickets": n, "crashed": len(crashed), "crash_types": dict(Counter(o.crashed for o in crashed)),
               "served_normally": len(served), "fell_back_to_human": len(fell_back), "actions_committed": sum(o.committed for o in outcomes),
         "approved_after_pause": sum(1 for o in outcomes if o.approved),
         "crashed_after_money_moved": sum(1 for o in crashed if o.committed > 0)}
    if deps is not None and hasattr(deps.actions, "lost_acks"):
        s["lost_acknowledgements"] = deps.actions.lost_acks
    if clean is not None:
        base = {o.ticket_id: o for o in clean}
        unsafe = [o.ticket_id for o in outcomes if not o.crashed and o.committed > base[o.ticket_id].committed]
        # a ticket served "normally" must have reached the same route the healthy run reached
        diverged = [o.ticket_id for o in served if o.route != base[o.ticket_id].route]
        s["unsafe_actions"] = len(unsafe)
        s["unsafe_ticket_ids"] = unsafe
        s["diverged_from_healthy"] = len(diverged)
        s["diverged_ticket_ids"] = diverged
    if deps is not None:
        keys = [r["key"] for r in deps.actions.ledger.records]
        s["duplicate_actions"] = len(keys) - len(set(keys))
    return s


# --------------------------------------------------------------------------------------------------------------------
# Experiment 1: random flakiness, with and without hardening
# --------------------------------------------------------------------------------------------------------------------
def order_system_experiment(cfg: FaultConfig = FaultConfig(seed=42, fail_rate=0.05, timeout_rate=0.03, malformed_rate=0.02, lost_ack_rate=0.25), *,
                            tickets: Optional[list[dict]] = None, kb: Any = None, extract: Optional[Callable] = None) -> dict:
    from .offline import RetrievalOnlyKB, keyword_extract
    tickets = tickets if tickets is not None else load_intake_tickets()
    kb = kb or RetrievalOnlyKB()
    extract = extract or keyword_extract

    healthy, hdeps = run_tickets(tickets, api=MockAPI(), guard=make_experiment_guard(cfg.seed), kb=kb, extract=extract)
    api_b = FlakyMockAPI(cfg)
    before, bdeps = run_tickets(tickets, api=api_b, guard=NoGuard(), kb=kb, extract=extract, lost_ack_rate=cfg.lost_ack_rate, seed=cfg.seed)
    api_c = FlakyMockAPI(cfg)
    guard = make_experiment_guard(cfg.seed)
    after, adeps = run_tickets(tickets, api=api_c, guard=guard, kb=kb, extract=extract, lost_ack_rate=cfg.lost_ack_rate, seed=cfg.seed)
    return {"config": cfg.__dict__, "healthy": summarize(healthy, deps=hdeps), "no_hardening": summarize(before, healthy, deps=bdeps),
            "hardened": summarize(after, healthy, deps=adeps), "faults_injected": dict(api_c.injector.counts),
            "guard": guard.snapshot()}


# --------------------------------------------------------------------------------------------------------------------
# Experiment 2: a sustained outage, the breaker, and recovery
# --------------------------------------------------------------------------------------------------------------------
def _outage_run(tickets, kb, extract, healthy, *, outage, seconds_per_ticket, reset_s, threshold, seed):
    start, end = outage
    clock, api = FakeClock(), FlakyMockAPI(FaultConfig(seed=seed))
    guard = make_experiment_guard(seed, clock, threshold=threshold, reset_s=reset_s)

    def on_ticket(i: int, t: dict) -> None:
        api.injector.down = start <= i < end
        clock.advance(seconds_per_ticket)

    outcomes, deps = run_tickets(tickets, api=api, guard=guard, kb=kb, extract=extract, on_ticket=on_ticket)
    during, after = outcomes[start:end], outcomes[end:]
    base_after = {o.ticket_id: o for o in healthy[end:]}
    return {
        "crashed": sum(1 for o in outcomes if o.crashed),
        "outage_tickets_served_normally": sum(1 for o in during if not o.fell_back and not o.crashed),
        "outage_tickets_fell_back": sum(1 for o in during if o.fell_back),
        "actions_committed_during_outage": sum(o.committed for o in during),
        "lookups_hitting_the_dead_system": api.injector.counts["outage"],
        "breaker_opened_times": guard.breaker.opened_count, "short_circuited_calls": guard.stats["short_circuited"],
        "final_breaker_state": guard.breaker.state,
        "after_recovery_still_fell_back": sum(1 for o in after if o.fell_back),
        "after_recovery_diverged_from_healthy": sum(1 for o in after if (not o.crashed) and not o.fell_back and o.route != base_after[o.ticket_id].route),
    }


def outage_experiment(*, outage: tuple[int, int] = (30, 60), seconds_per_ticket: float = 2.0, reset_s: float = 20.0, threshold: int = 3,
                      tickets: Optional[list[dict]] = None, kb: Any = None, extract: Optional[Callable] = None, seed: int = 42) -> dict:
    """The order system is completely down for tickets [start, end). Time passes `seconds_per_ticket` per ticket on a fake clock.
    Run twice: with the breaker, and with a breaker that can never open (retries only) -- the difference is what the breaker buys."""
    from .offline import RetrievalOnlyKB, keyword_extract
    tickets = tickets if tickets is not None else load_intake_tickets()
    kb = kb or RetrievalOnlyKB()
    extract = extract or keyword_extract
    healthy, _ = run_tickets(tickets, api=MockAPI(), guard=make_experiment_guard(seed), kb=kb, extract=extract)
    kw = dict(outage=outage, seconds_per_ticket=seconds_per_ticket, reset_s=reset_s, seed=seed)
    return {"outage_tickets": f"{outage[0]}-{outage[1] - 1}", "tickets_in_outage": outage[1] - outage[0],
            "with_breaker": _outage_run(tickets, kb, extract, healthy, threshold=threshold, **kw),
            "retries_only": _outage_run(tickets, kb, extract, healthy, threshold=10 ** 9, **kw)}


# --------------------------------------------------------------------------------------------------------------------
# Experiment 3: the model provider
# --------------------------------------------------------------------------------------------------------------------
class _Msg:
    def __init__(self, content): self.content, self.tool_calls = content, None


class _Resp:
    def __init__(self, content="ok"):
        self.choices = [type("C", (), {"message": _Msg(content)})()]
        self.usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 20})()


class _Overloaded(Exception):
    status_code = 503


class _NotFound(Exception):
    status_code = 404


def llm_experiment(*, seed: int = 42, calls: int = 200, overload_rate: float = 0.15, rate_limit_rate: float = 0.05) -> dict:
    """Four scenarios against a scripted provider (no network, no sleeping):
      flaky_provider      15% 503s + 5% rate limits: one bare call vs the hardened client;
      total_outage        every request fails with a 503: the breaker stops us hammering it (provider requests are counted);
      dead_primary        the primary model is gone (404): without a fallback model every call fails; with one every call is
                          served, each flagged as degraded. (404s fail fast and do not trip the breaker -- they are configuration,
                          not an outage -- so the primary keeps being tried and recovers the moment the config is fixed.)"""
    import contextlib
    import io
    import os

    from ..core import llm_client as lc
    from ..observability import tracing as T

    def provider(rng: random.Random, log: list, *, dead_models: tuple = (), always_overloaded: bool = False):
        def completion(*, model, messages, **kw):
            log.append(model)
            if model in dead_models:
                raise _NotFound(f"model {model} is no longer available")
            if always_overloaded:
                raise _Overloaded("503 the model is overloaded")
            r = rng.random()
            if r < overload_rate:
                raise _Overloaded("503 the model is overloaded")
            if r < overload_rate + rate_limit_rate:
                raise lc.RateLimitError("429 rate limit. Please retry in 2s.")
            return _Resp("answer")
        return completion

    msgs = [{"role": "user", "content": "hello"}]
    saved = (lc.completion, lc._sleep, os.environ.get("SHOPSENSE_LLM_FALLBACK_MODELS"), os.environ.get("SHOPSENSE_LLM_MIN_INTERVAL"),
             os.environ.get("SHOPSENSE_LLM_BREAKER_THRESHOLD"))
    result: dict = {"calls": calls, "overload_rate": overload_rate, "rate_limit_rate": rate_limit_rate}
    previous_tracer = T.set_tracer(T.Tracer())

    def hardened_loop(label: str, log: list, tracer_runs: bool = True) -> dict:
        served = failed = degraded = 0
        tracer = T.Tracer()
        T.set_tracer(tracer)
        for i in range(calls):
            with tracer.run(session_id=f"call-{i}") as run:
                try:
                    lc.llm_completion(msgs, model="m/primary")
                    served += 1
                except Exception:       # noqa: BLE001
                    failed += 1
                degraded += 1 if run.stats.degraded else 0
        h = lc.llm_health()
        return {"served": served, "failed": failed, "flagged_degraded": degraded, "provider_requests": len(log),
                "primary_requests": log.count("m/primary"), "short_circuited": lc.LLM_STATS["circuit_open"],
                "breaker": h["models"].get("m/primary", {}).get("state")}

    try:
        with contextlib.redirect_stdout(io.StringIO()):            # the client announces each wait; the experiment prints a table instead
            lc._sleep = lambda s_: None
            for var in ("SHOPSENSE_LLM_FALLBACK_MODELS", "SHOPSENSE_LLM_MIN_INTERVAL"):
                os.environ.pop(var, None)
            os.environ["SHOPSENSE_LLM_BREAKER_THRESHOLD"] = "3"

            log: list = []
            lc.completion = provider(random.Random(seed), log)
            bare_failed = 0
            for _ in range(calls):
                try:
                    lc.completion(model="m/primary", messages=msgs)
                except Exception:       # noqa: BLE001
                    bare_failed += 1
            result["bare_call"] = {"failed": bare_failed, "provider_requests": len(log)}

            lc.reset_llm_state()
            log = []
            lc.completion = provider(random.Random(seed), log)
            result["flaky_provider"] = {**hardened_loop("flaky", log), "retries": lc.LLM_STATS["retries"]}

            lc.reset_llm_state()
            log = []
            lc.completion = provider(random.Random(seed), log, always_overloaded=True)
            result["total_outage"] = hardened_loop("outage", log)
            result["total_outage"]["without_breaker_would_send"] = calls * lc.MAX_RATE_LIMIT_RETRIES

            for label, fallback in (("dead_primary_no_fallback", None), ("dead_primary_with_fallback", "m/backup")):
                lc.reset_llm_state()
                if fallback:
                    os.environ["SHOPSENSE_LLM_FALLBACK_MODELS"] = fallback
                else:
                    os.environ.pop("SHOPSENSE_LLM_FALLBACK_MODELS", None)
                log = []
                lc.completion = provider(random.Random(seed), log, dead_models=("m/primary",))
                result[label] = hardened_loop(label, log)
    finally:
        lc.completion, lc._sleep = saved[0], saved[1]
        for var, val in zip(("SHOPSENSE_LLM_FALLBACK_MODELS", "SHOPSENSE_LLM_MIN_INTERVAL", "SHOPSENSE_LLM_BREAKER_THRESHOLD"), saved[2:]):
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
        lc.reset_llm_state()
        T.set_tracer(previous_tracer)
    return result


# --------------------------------------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------------------------------------
def format_order_experiment(r: dict) -> str:
    rows = [("tickets", "tickets"), ("crashed", "crashed (unhandled error)"), ("served_normally", "served normally"),
            ("fell_back_to_human", "handed to a person"), ("diverged_from_healthy", "served normally, different route than healthy"),
            ("approved_after_pause", "paused for approval, then approved"), ("actions_committed", "actions committed (refunds etc.)"),
            ("lost_acknowledgements", "write done but caller saw a timeout"), ("crashed_after_money_moved", "crashed AFTER money moved"),
            ("unsafe_actions", "UNSAFE actions (not taken when healthy)"), ("duplicate_actions", "duplicate actions (double pay)")]
    stages = [("healthy", "healthy"), ("no_hardening", "flaky, no hardening"), ("hardened", "flaky, hardened")]
    out = ["ORDER-SYSTEM FAULT INJECTION  " + ", ".join(f"{k}={v}" for k, v in r["config"].items()),
           f"  faults injected: {r['faults_injected']}", "", f"  {'':46}" + "".join(f"{label:>22}" for _, label in stages)]
    for key, label in rows:
        out.append(f"  {label:46}" + "".join(f"{str(r[st].get(key, '-')):>22}" for st, _ in stages))
    if r["no_hardening"]["crash_types"]:
        out.append(f"\n  crashes without hardening: {r['no_hardening']['crash_types']}")
    g = r["guard"]
    out.append(f"  guard: calls={g['calls']} retries={g['retries']} failures={g['failures']} malformed={g['malformed']} "
               f"short_circuited={g['short_circuited']} breaker_opened={g['opened_count']}")
    return "\n".join(out)


def format_outage(r: dict) -> str:
    w, n = r["with_breaker"], r["retries_only"]
    rows = [("crashed", "crashed tickets"), ("outage_tickets_fell_back", "outage tickets handed to a person"),
            ("outage_tickets_served_normally", "outage tickets served normally"),
            ("actions_committed_during_outage", "actions committed during the outage"),
            ("lookups_hitting_the_dead_system", "calls that hit the dead system"), ("breaker_opened_times", "breaker opened"),
            ("short_circuited_calls", "calls refused instantly by the breaker"), ("after_recovery_still_fell_back", "after recovery: still handed to a person"),
            ("after_recovery_diverged_from_healthy", "after recovery: differ from healthy run"), ("final_breaker_state", "breaker at the end")]
    out = [f"SUSTAINED OUTAGE  order system completely down for tickets {r['outage_tickets']} ({r['tickets_in_outage']} tickets)",
           f"  {'':44}{'retries only':>16}{'retries + breaker':>20}"]
    for key, label in rows:
        out.append(f"  {label:44}{str(n[key]):>16}{str(w[key]):>20}")
    return "\n".join(out)


def format_llm(r: dict) -> str:
    n, b, h, o = r["calls"], r["bare_call"], r["flaky_provider"], r["total_outage"]
    out = [f"LLM PROVIDER FAULT INJECTION  {n} calls per scenario",
           f"  flaky provider ({r['overload_rate']:.0%} 503 + {r['rate_limit_rate']:.0%} rate limit)",
           f"    bare call, no retries : {b['failed']}/{n} failed   ({b['provider_requests']} provider requests)",
           f"    hardened client       : {h['failed']}/{n} failed   ({h['provider_requests']} provider requests, {h['retries']} retries)",
           "  total outage (every request 503)",
           f"    hardened client       : {o['failed']}/{n} failed fast; {o['provider_requests']} provider requests instead of ~{o['without_breaker_would_send']}"
           f" ({o['short_circuited']} calls refused by the breaker, breaker {o['breaker']})",
           "  primary model removed (404)"]
    for key in ("dead_primary_no_fallback", "dead_primary_with_fallback"):
        d = r[key]
        out.append(f"    {key.replace('dead_primary_', ''):22}: served {d['served']}/{n}, flagged degraded {d['flagged_degraded']}, "
                   f"{d['provider_requests']} provider requests")
    return "\n".join(out)
