"""
Milestone 7 tests: the order system and the model provider misbehave on purpose (shopsense/eval/fault_injection.py).

Two kinds of test. (1) Scenario tests drive one ticket through the REAL node functions while the order system is scripted to
fail at a chosen moment, and check the contract: no crash, nothing promised, nothing paid unless a person approved it, the
ticket lands in the human queue with a truthful holding reply, and a retried write never pays twice. (2) Experiment tests pin the
seeded before/after NUMBERS from the seeded method, so a regression shows up as a changed number rather than an opinion.
What they cannot show: behaviour of a real order-system API (this one is the in-memory mock) or LangGraph's checkpointing.
"""
import pytest

from shopsense.eval import fault_injection as fi
from shopsense.eval.golden import load_golden
from shopsense.observability import tracing as T
from shopsense.tools.order_tools import MockAPI
from shopsense.workflows import ticket_resolution_graph as g
from ticket_sim import Paused, RetrievalOnlyKB, keyword_extract, resume, run

GOLDEN = {c["id"]: c for c in load_golden()}
HOLDING = "order system isn't responding"


class ScriptedKB:
    def __init__(self, on_answer=None):
        self.on_answer = on_answer

    def answer(self, query, *, k=3, use_rerank=True, question=None):
        if self.on_answer:
            self.on_answer()
        return {"answer": "Electronics can be returned within 15 days of delivery.", "sources": [{"id": 1, "source": "category_electronics", "section": "x"}]}


class ScriptedInjector(fi.FaultInjector):
    """Fails exactly the lookups the test names (by call number), succeeds otherwise."""

    def __init__(self, plan):
        super().__init__(fi.FaultConfig())
        self.plan, self.n = dict(plan), 0

    def roll(self):
        self.n += 1
        self.counts["lookups"] += 1
        kind = self.plan.get(self.n)
        if kind:
            self.counts[kind] += 1
        return kind


def flaky_api(injector=None):
    api = fi.FlakyMockAPI(fi.FaultConfig())
    if injector is not None:
        api.injector = injector
        for table in (api.customers, api.orders, api.shipments, api.refunds):
            table._inj = injector
    return api


def deps_for(api, *, kb=None, guard=None, lost_ack_rate=0.0, seed=0):
    from shopsense.runtime import build_deps
    d = build_deps(kb=kb or ScriptedKB(), extract=keyword_extract, api=api, order_guard=guard or fi.make_experiment_guard(1), with_memo_team=False)
    if lost_ack_rate:
        d.actions = fi.LossyActions(d.actions, lost_ack_rate, seed)
    return d


def go(deps, case_id, *, tid=None):
    t = GOLDEN[case_id]["ticket"]
    return run(deps, g.seed_state(ticket_id=tid or t["ticket_id"], customer_id=t["customer_id"], text=t["text"], history=GOLDEN[case_id]["history"]))


def ledger(deps):
    return deps.actions.ledger.records


# ---- reads ---------------------------------------------------------------------------------------------------------------
def test_a_blip_is_retried_and_the_customer_never_notices():
    # the 1st lookup of the ticket (the customer record in triage) times out once; the guard retries and it works
    api = flaky_api(ScriptedInjector({1: "timeout"}))
    deps = deps_for(api)
    r = go(deps, "SS-EV-001")
    assert r["route"] == "auto" and r["status"] == "resolved" and "order_system_unavailable" not in r["reasons"]
    assert deps.order_guard.stats["retries"] == 1 and deps.order_guard.stats["failures"] == 0


def test_order_system_down_at_triage_hands_the_ticket_to_a_person_and_promises_nothing():
    api = flaky_api()
    api.injector.down = True
    deps = deps_for(api)
    r = go(deps, "SS-EV-001")
    assert r["route"] == "human_review" and r["status"] == "escalated_to_human"
    assert "order_system_unavailable" in r["reasons"] and HOLDING in r["reply"]
    assert not ledger(deps) and not g._APPROVAL_CLAIM_RE.search(r["reply"])
    assert r["handoff"]["queue"] == "Support review queue" and "order_system_unavailable" in r["handoff"]["reasons"]


def test_order_system_dying_midway_through_the_ticket_is_caught_at_the_action_planner():
    api = flaky_api()
    deps = deps_for(api, kb=ScriptedKB(on_answer=lambda: setattr(api.injector, "down", True)))   # goes down after triage + policy query
    r = go(deps, "SS-EV-001")
    assert r["route"] == "human_review" and r["proposal"]["type"] == "order_system_unavailable"
    assert HOLDING in r["reply"] and not ledger(deps)
    assert any("action_plan" in line for line in r["log"])


def test_unreadable_order_means_the_policy_answer_is_not_trusted_to_resolve_the_ticket():
    # triage + action planning would succeed, but the policy step cannot read the order's category: the answer could be the wrong
    # window (30 days instead of the electronics 15), so a person takes it
    api = flaky_api(ScriptedInjector({2: "connection", 3: "connection", 4: "connection"}))        # lookups 2-4 = policy query, 3 attempts
    deps = deps_for(api)
    r = go(deps, "SS-EV-001")
    assert r["route"] == "human_review" and "order_system_unavailable" in r["reasons"]


def test_a_malformed_record_goes_to_a_person_instead_of_crashing_with_a_keyerror():
    class Garbled(fi.FaultInjector):
        def roll(self):
            self.counts["lookups"] += 1
            return "malformed"
    api = flaky_api(Garbled(fi.FaultConfig()))
    deps = deps_for(api)
    unguarded = deps_for(flaky_api(Garbled(fi.FaultConfig())), guard=fi.NoGuard())
    outcomes = []
    for cid in ("SS-EV-001", "SS-EV-002", "SS-EV-003", "SS-EV-007"):
        outcomes.append(go(deps, cid)["route"])
    assert set(outcomes) <= {"auto", "human_review"} and not ledger(deps)
    assert deps.order_guard.stats["malformed"] >= 1
    with pytest.raises(Exception):                      # the "before": the same garbage crashes an unguarded ticket
        for cid in ("SS-EV-001", "SS-EV-002", "SS-EV-003", "SS-EV-007"):
            go(unguarded, cid)


def test_the_breaker_stops_us_hammering_a_dead_order_system():
    api = flaky_api()
    api.injector.down = True
    clock = fi.FakeClock()
    deps = deps_for(api, guard=fi.make_experiment_guard(1, clock, threshold=3, reset_s=60))
    for i in range(5):
        r = go(deps, "SS-EV-001", tid=f"TCK-D{i}")
        assert r["route"] == "human_review" and HOLDING in r["reply"]
    assert deps.order_guard.breaker.state == "open"
    seen = api.injector.counts["lookups"]
    r = go(deps, "SS-EV-001", tid="TCK-D9")
    assert r["route"] == "human_review" and api.injector.counts["lookups"] == seen          # refused instantly: the system was not touched
    # recovery: cool-down passes, the system is back, the single trial succeeds and service resumes by itself
    api.injector.down = False
    clock.advance(61)
    assert go(deps, "SS-EV-001", tid="TCK-D10")["route"] == "auto" and deps.order_guard.breaker.state == "closed"


def test_nothing_is_auto_committed_for_an_escalated_ticket_even_when_the_system_flaps():
    api = fi.FlakyMockAPI(fi.FaultConfig(seed=3, fail_rate=0.3, timeout_rate=0.2))
    deps = deps_for(api)
    for cid in GOLDEN:
        r = go(deps, cid, tid=f"TCK-F-{cid}")
        state = r.state if isinstance(r, Paused) else r
        assert isinstance(r, Paused) or state["status"], cid          # every ticket ends somewhere defined, never in an exception
    assert not ledger(deps)


# ---- writes --------------------------------------------------------------------------------------------------------------
def test_order_system_failing_right_after_a_human_approves_commits_nothing_and_keeps_the_decision():
    api = flaky_api()
    deps = deps_for(api)
    paused = go(deps, "SS-EV-901")
    assert isinstance(paused, Paused)
    api.injector.down = True                                    # the approver clicks "approve" and the order system is down
    r = resume(deps, paused, decision="approve", approver_role="executive")
    assert r["status"] == "escalated_to_human" and r["action_result"]["unavailable"] is True and not ledger(deps)
    assert HOLDING in r["reply"] and not g._APPROVAL_CLAIM_RE.search(r["reply"])
    assert r["handoff"]["decision_so_far"]["human_decision"] == "approve"          # a person can finish it without asking again
    # the system comes back: the SAME approval, resumed again, pays exactly once
    api.injector.down = False
    r2 = resume(deps, paused, decision="approve", approver_role="executive")
    assert r2["action_result"]["committed"] is True and len(ledger(deps)) == 1
    r3 = resume(deps, paused, decision="approve", approver_role="executive")        # and a double click does not pay twice
    assert r3["action_result"].get("duplicate") is True and len(ledger(deps)) == 1


def test_a_lost_acknowledgement_is_retried_without_paying_twice():
    api = flaky_api()
    deps = deps_for(api, lost_ack_rate=0.999, seed=5)           # every write succeeds and then reports a timeout
    paused = go(deps, "SS-EV-009")
    r = resume(deps, paused, decision="approve", approver_role="manager")
    assert len(ledger(deps)) == 1                               # three attempts, one payment
    assert r["status"] == "escalated_to_human" and r["action_result"]["unavailable"]       # we never got an ack, so a person confirms it
    assert deps.actions.lost_acks == 3


def test_one_lost_acknowledgement_then_success_is_a_normal_resolved_ticket_with_one_payment():
    api = flaky_api()
    deps = deps_for(api)
    deps.actions = fi.LossyActions(deps.actions, 0.5, 0)

    class First:                                                # lose the first ack only
        n = 0

        def random(self):
            First.n += 1
            return 0.0 if First.n == 1 else 0.99
    deps.actions._rng = First()
    paused = go(deps, "SS-EV-009")
    r = resume(deps, paused, decision="approve", approver_role="manager")
    assert r["status"] == "resolved_after_approval" and r["action_result"]["committed"] and len(ledger(deps)) == 1


# ---- tracing sees the degradation ---------------------------------------------------------------------------------------------
def test_a_fallback_is_visible_in_the_trace_and_always_kept():
    tracer = T.Tracer(keep_healthy_rate=0.0, rng=type("D", (), {"random": lambda self: 0.99})())
    prev = T.set_tracer(tracer)
    try:
        api = flaky_api()
        api.injector.down = True
        deps = deps_for(api)
        t = GOLDEN["SS-EV-001"]["ticket"]
        with tracer.run(session_id=t["ticket_id"], user_id=t["customer_id"]):
            run(deps, g.seed_state(ticket_id=t["ticket_id"], customer_id=t["customer_id"], text=t["text"]))
        with tracer.run(session_id="TCK-OK", metadata={"route": "resolve"}):
            pass
    finally:
        T.set_tracer(prev)
    traces = tracer.store.traces()
    assert [tr["session_id"] for tr in traces] == ["GS-001"] and traces[0]["degraded"] is True and tracer.dropped_healthy == 1
    names = {s["name"] for s in traces[0]["spans"]}
    assert {"node:triage", "node:human_review"} <= names


# ---- the seeded experiments: pinned numbers ----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def order_report():
    return fi.order_system_experiment(kb=RetrievalOnlyKB())


def test_order_experiment_is_reproducible_and_hardening_removes_every_crash(order_report):
    r = order_report
    assert r["faults_injected"] == {"lookups": 971, "timeout": 29, "malformed": 20, "connection": 35}
    assert r["healthy"]["crashed"] == 0 and r["healthy"]["actions_committed"] == 11 and r["healthy"]["approved_after_pause"] == 18
    assert r["no_hardening"]["crashed"] == 48 and r["no_hardening"]["crash_types"] == {"TimeoutError": 23, "KeyError": 4, "ConnectionError": 21}
    h = r["hardened"]
    assert h["crashed"] == 0 and h["served_normally"] == 117 and h["fell_back_to_human"] == 9


def test_hardened_run_is_safe_not_just_quiet(order_report):
    h, b = order_report["hardened"], order_report["no_hardening"]
    assert h["unsafe_actions"] == 0 and h["duplicate_actions"] == 0 and h["diverged_from_healthy"] == 0
    assert h["actions_committed"] == order_report["healthy"]["actions_committed"]          # every approved refund still got paid, once
    assert h["lost_acknowledgements"] >= 1                                                 # and the dangerous fault really was exercised
    assert b["crashed_after_money_moved"] >= 1                                             # without hardening: money moved, ticket died


def test_hardened_run_stays_safe_across_other_seeds_and_harsher_rates():
    kb = RetrievalOnlyKB()
    for seed in (1, 7, 99):
        cfg = fi.FaultConfig(seed=seed, fail_rate=0.15, timeout_rate=0.05, malformed_rate=0.05, lost_ack_rate=0.3)
        h = fi.order_system_experiment(cfg, kb=kb)["hardened"]
        assert h["crashed"] == 0 and h["unsafe_actions"] == 0 and h["duplicate_actions"] == 0, (seed, h)


def test_outage_experiment_breaker_cuts_the_load_on_the_dead_system():
    r = fi.outage_experiment(kb=RetrievalOnlyKB())
    w, n = r["with_breaker"], r["retries_only"]
    assert w["crashed"] == n["crashed"] == 0
    assert w["outage_tickets_fell_back"] == n["outage_tickets_fell_back"] == 30 and w["actions_committed_during_outage"] == 0
    assert (n["lookups_hitting_the_dead_system"], w["lookups_hitting_the_dead_system"]) == (90, 15)
    assert (w["breaker_opened_times"], w["short_circuited_calls"], w["final_breaker_state"]) == (3, 27, "closed")
    assert w["after_recovery_diverged_from_healthy"] == 0 and w["after_recovery_still_fell_back"] == 2        # the cost: a short recovery lag


def test_llm_experiment_numbers():
    r = fi.llm_experiment()
    assert r["bare_call"]["failed"] == 38 and r["flaky_provider"]["failed"] == 0
    assert r["flaky_provider"]["provider_requests"] == 246 and r["flaky_provider"]["retries"] == 46
    o = r["total_outage"]
    assert o["failed"] == 200 and o["provider_requests"] == 15 and o["short_circuited"] == 197 and o["breaker"] == "open"
    assert r["dead_primary_no_fallback"]["served"] == 0 and r["dead_primary_with_fallback"]["served"] == 200
    assert r["dead_primary_with_fallback"]["flagged_degraded"] == 200          # a fallback is never silent


def test_fault_config_rejects_impossible_rates():
    with pytest.raises(ValueError):
        fi.FaultConfig(fail_rate=0.6, timeout_rate=0.5)
