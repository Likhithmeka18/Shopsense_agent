"""
ShopSense service core (Milestone 8): one customer message in, one guarded reply out. No web framework in here.

`api.py` is a thin FastAPI wrapper over `ChatService`; everything that matters -- running the ticket graph, applying the output
guardrails, deciding what the customer is allowed to see, health reporting -- lives here so it can be tested without FastAPI,
a model or a network.

Two modes:
  live     the real thing: the M6 support team (or M5 workflow) on LangGraph with the durable SQLite checkpoint, the real model, the
           real policy index. Needs a model key + `pip install langgraph langgraph-checkpoint-sqlite`.
  offline  SHOPSENSE_OFFLINE=1: the same node functions, tables, policy corpus, ledger and guardrails, but the extraction model is a
           keyword stand-in and the policy answer is the top retrieved chunk. For demos, the Postman walkthrough and CI with no key.
           It proves the plumbing and the safety behaviour; it says nothing about the real model's answers.

The public response is an ALLOW-LIST (internal trace fields are deliberately NOT returned). Escalation codes,
reasons and facts stay inside: `ESC-ABUSE` in a response body would tell a customer their account is flagged, which the reply itself
is forbidden to say.

Demo shortcuts that production must not keep (also in docs/ARCHITECTURE_REVIEW.md):
  * `customer_id` comes from the request body. Anyone can claim to be anyone. Production derives it from an authenticated session.
  * One ticket runs at a time (a lock), because the demo shares one in-memory ledger and one checkpoint file.
  * A ticket that needs a person's approval stays paused; the demo API has no approver endpoint (use resume_ticket_approval.py).
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from collections import Counter
from datetime import timedelta
from typing import Any, Optional

from .guardrails import (MAX_MESSAGE_CHARS, SAFE_FALLBACK_REPLY, GuardrailConfig, GuardrailResult, ReplyContext, check_input, evaluate)
from .observability.tracing import add_event, get_tracer

PUBLIC_FIELDS = ("ticket_id", "session_id", "outcome", "reply", "awaiting_approval", "guardrail", "degraded", "input_flags")


def offline_requested() -> bool:
    return os.environ.get("SHOPSENSE_OFFLINE", "0") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# How long ONE chat request may spend WAITING on the model provider (rate-limit sleeps, retries) before the ticket is handed to a person.
DEFAULT_REQUEST_BUDGET_S = 40.0


class ChatService:
    def __init__(self, *, mode: Optional[str] = None, deps: Any = None, graph: Any = None, topology: str = "team",
                 guardrails: Optional[GuardrailConfig] = None, request_budget_s: Optional[float] = None, memo_team: Optional[bool] = None):
        self.mode = mode or ("offline" if offline_requested() else "live")
        if self.mode not in ("live", "offline"):
            raise ValueError("mode must be 'live' or 'offline'")
        self.topology = topology
        self.guardrails = guardrails or GuardrailConfig.balanced()
        self.deps, self.graph = deps, graph
        # Seconds of model-provider waiting one request may use (live mode). SHOPSENSE_API_DEADLINE_S; 0 or negative = no limit.
        self.request_budget_s = request_budget_s if request_budget_s is not None else _env_float("SHOPSENSE_API_DEADLINE_S", DEFAULT_REQUEST_BUDGET_S)
        # The 5-agent memo team writes a reviewed memo for each escalation (~50 s of model calls). Right for a batch, too slow for a chat
        # reply, so the API leaves it off unless SHOPSENSE_API_MEMO_TEAM=1. The hand-off packet and the customer reply do not depend on it.
        self.memo_team = memo_team if memo_team is not None else os.environ.get("SHOPSENSE_API_MEMO_TEAM", "0") == "1"
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._started = time.time()
        self.counters: Counter = Counter()
        self.last_state: Optional[dict] = None        # developer view of the latest ticket (ask.py --debug); not in the public response
        self.contexts: Optional[list] = None          # evaluation hook: when a list, every guarded reply's ReplyContext is appended

    # ---- lazy construction (so importing the API never loads a model or an index) ---------------------------------------------
    def _ensure_ready(self) -> None:
        if self.deps is not None:
            return
        from .runtime import ACTION_LEDGER, TICKET_DB, build_deps, build_graph
        if self.mode == "offline":
            from .eval.offline import RetrievalOnlyKB, keyword_extract
            self.deps = build_deps(kb=RetrievalOnlyKB(), extract=keyword_extract, with_memo_team=False)
            return
        from .workflows import ticket_resolution_graph as tg
        if not tg.LANGGRAPH_AVAILABLE:
            raise RuntimeError("live mode needs langgraph (pip install langgraph langgraph-checkpoint-sqlite); "
                               "set SHOPSENSE_OFFLINE=1 for the offline demo")
        self.deps = build_deps(ledger_path=ACTION_LEDGER, with_memo_team=self.memo_team)
        self.graph = build_graph(self.deps, topology=self.topology, db_path=TICKET_DB)

    # ---- the one operation -------------------------------------------------------------------------------------------------------
    def chat(self, message: str, *, customer_id: Optional[str] = None, session_id: Optional[str] = None, channel: str = "chat") -> dict:
        from .workflows import ticket_resolution_graph as tg

        message = (message or "").strip()
        if not message:
            raise ValueError("message must not be empty")
        input_flags = check_input(message)
        if "message_too_long" in input_flags:
            raise ValueError(f"message is longer than {MAX_MESSAGE_CHARS} characters")
        with self._lock:
            self._ensure_ready()
            n = next(self._ids)
            ticket_id = f"CHAT-{n:06d}"
            session = session_id or f"sess-{ticket_id}"
            cust = (customer_id or "").strip() or f"GUEST-{session}"
            # Contact history lives in `deps.ledger` (keyed by CUSTOMER, so opening a new session does not reset the repeat-contact rule).
            # The deployment's clock is the dataset's reference "now" (policy windows are judged against it), advanced one second per
            # message so a customer's earlier messages are strictly EARLIER -- without that, the contact ledger sees every message as
            # simultaneous and the third-contact-in-seven-days rule can never fire.
            created_at = (self.deps.now + timedelta(seconds=n)).isoformat()
            state = tg.seed_state(ticket_id=ticket_id, customer_id=cust, text=message, channel=channel, created_at=created_at)
            box: dict = {}

            def after(result: dict) -> None:
                box["state"], box["guard"] = self._guard(result, ticket_id)

            llm_client = None
            if self.mode == "live":
                from .core import llm_client
                if self.request_budget_s > 0:
                    llm_client.set_request_deadline(self.request_budget_s)
            try:
                if self.mode == "live":
                    result, _cfg = tg.run_ticket(self.graph, state, tags=("api", f"topology:{self.topology}"), on_result=after)
                    info = tg.run_info_of(result)
                else:
                    result, info = self._run_offline(state, after)
            except Exception:
                self.counters["errors"] += 1
                raise
            finally:
                if llm_client is not None:
                    llm_client.set_request_deadline(None)
            final, guard = box["state"], box["guard"]
            self.last_state = final            # developer view only (ask.py --debug); never part of the public response
            response = self._public(final, guard, ticket_id, session, info, input_flags)
            self.counters["requests"] += 1
            self.counters[f"outcome:{response['outcome']}"] += 1
            if guard.blocked:
                self.counters["guardrail_blocked"] += 1
            return response

    # ---- internals -------------------------------------------------------------------------------------------------------------------
    def _run_offline(self, state: dict, after) -> tuple[dict, dict]:
        from .eval import offline
        with get_tracer().run("ticket.run", session_id=state["ticket_id"], user_id=state["customer_id"], tags=["shopsense", "api", "offline"],
                              input={"ticket_id": state["ticket_id"], "text": state["text"]}) as run:
            out = offline.run(self.deps, state)
            result = dict(out.state if isinstance(out, offline.Paused) else out)
            if isinstance(out, offline.Paused):
                result["__paused__"] = out.interrupt
            after(result)
            run.update(metadata={"route": "paused" if "__paused__" in result else result.get("route", ""),
                                 "codes": result.get("escalation_codes", [])})        # same trace metadata the live path records
            info = run.stats.as_dict() if run.stats is not None else {}
        return result, info

    def _source_texts(self, state: dict) -> list[str]:
        """The retrieved policy chunks behind this ticket's policy answer: the ground truth its figures are checked against."""
        chunks = {c.cid: c.text for c in getattr(self.deps.kb, "chunks", [])} if self.deps is not None else {}
        return [chunks[s["id"]] for s in (state.get("policy") or {}).get("sources", []) if s.get("id") in chunks]

    def _guard(self, result: dict, ticket_id: str) -> tuple[dict, GuardrailResult]:
        """Run the output guardrails on the finished (or paused) state. On a block the reply is REPLACED, never edited."""
        final = dict(result)
        paused = "__interrupt__" in result or "__paused__" in result
        if paused:
            payload = result["__paused__"] if "__paused__" in result else result["__interrupt__"][0].value
            final["reply"] = payload.get("holding_reply") or final.get("reply") or ""
            final["status"] = "awaiting_approval"
            final["needs_role"] = payload.get("needs_role") or payload.get("proposal", {}).get("tier", "")
        reply = final.get("reply", "") or ""
        policy = final.get("policy") or {}
        answer = policy.get("answer", "") or ""
        records = [r for r in self.deps.actions.ledger.records if r.get("ticket_id") == ticket_id]
        ctx = ReplyContext(reply=reply, model_text=answer if (answer and answer[:60] in reply) else "", source_texts=self._source_texts(final),
                           policy_answer=answer, facts=final.get("facts") or {}, proposal=final.get("proposal") or {},
                           action_result=final.get("action_result") or {}, route=final.get("route", ""), refund_records=records,
                           injection_attempt=bool((final.get("triage") or {}).get("injection")))
        if self.contexts is not None:
            self.contexts.append((ticket_id, ctx))
        with get_tracer().span("guardrail.check", kind="span", input={"reply_chars": len(reply)}) as sp:
            guard = evaluate(ctx, self.guardrails)
            sp.update(output=guard.as_dict())
            if guard.blocked:
                add_event("guardrail.blocked", codes=guard.codes)
        if guard.blocked:
            final["unsafe_draft_withheld"] = True
            final["reply"] = SAFE_FALLBACK_REPLY
            final["status"] = "blocked_by_guardrail"
            final["route"] = "human_review"
        return final, guard

    @staticmethod
    def _outcome(final: dict) -> str:
        if final.get("status") == "awaiting_approval":
            return "awaiting_approval"
        if final.get("status") == "blocked_by_guardrail" or final.get("route") == "human_review" or final.get("status") == "escalated_to_human":
            return "escalated"
        return "resolved"

    def _public(self, final: dict, guard: GuardrailResult, ticket_id: str, session: str, info: dict, input_flags: list[str]) -> dict:
        outcome = self._outcome(final)
        return {"ticket_id": ticket_id, "session_id": session, "outcome": outcome, "reply": final.get("reply", ""),
                "awaiting_approval": outcome == "awaiting_approval",
                "guardrail": {"passed": guard.passed, "blocked": guard.blocked, "flags": [f.code for f in guard.flags]},
                "degraded": bool(info.get("degraded")), "input_flags": input_flags}

    # ---- health ------------------------------------------------------------------------------------------------------------------------
    def health(self) -> dict:
        """Cheap liveness + dependency state. Never calls a model, so it cannot hit a rate limit."""
        from .core import llm_client
        h: dict = {"status": "ok", "mode": self.mode, "uptime_s": round(time.time() - self._started, 1), "requests": dict(self.counters)}
        problems: list[str] = []
        if self.mode == "live":
            llm = llm_client.llm_health()
            h["llm"] = {"model": llm_client.DEFAULT_MODEL, "breakers": llm["models"], "fallback_models": llm["fallback_models"]}
            problems += [f"llm breaker open: {m}" for m, b in llm["models"].items() if b.get("state") != "closed"]
        if self.deps is not None:
            snap = self.deps.order_guard.snapshot()
            h["order_system"] = {k: snap[k] for k in ("state", "calls", "retries", "failures", "short_circuited", "opened_count") if k in snap}
            if snap.get("state") != "closed":
                problems.append("order system breaker " + str(snap.get("state")))
        h["tracing"] = get_tracer().status()
        h["guardrails"] = {"blocked_total": self.counters.get("guardrail_blocked", 0)}
        h["limits"] = {"request_budget_s": self.request_budget_s, "memo_team": self.memo_team}
        if problems:
            h["status"], h["problems"] = "degraded", problems          # still HTTP 200: the service is up and is routing to humans
        return h
