"""
ShopSense: the end-to-end pipeline (capstone M1 -> M6).

    tickets.jsonl
        |
        v
    [M1] envelope validation + repair (core.ticket_intake)
        |
        v   one thread per ticket, durable SQLite checkpoint
    [M5/M6] the support TEAM (teams/support_team.py) running the ticket-resolution graph
        |
        |   triage -------- LLM extraction (+ repair loop), deterministic safety floor (legal / safety / injection / abuse flag),
        |      |            contact history (3rd contact in 7 days -> ESC-REPEAT)
        |      +-- hard fact? ------------------------------------------+
        |   policy_rag ---- hybrid RAG over the 14 policy documents      |
        |   order_actions - read order / shipment / refund state, PROPOSE|
        |   escalation_reviewer: route                                   |
        |      +-- auto ---------> commit -> reply                       v
        |      +-- approval: refund >= INR 2,000, or angry   human_review: hand-off packet + M6 memo team
        |            customer --> PAUSE (interrupt) --> a person        (legal, safety, injection, abuse, repeat,
        |            decides with  resume_ticket_approval.py            lost shipment, owner mismatch ...)
        v
    done

Needs a live model behind SHOPSENSE_LLM_MODEL (default: a Groq model -- export GROQ_API_KEY first) and
`pip install langgraph langgraph-checkpoint-sqlite`. Run with:  python run_pipeline.py [--limit N] [--workflow]

Free-tier friendly (Milestone 7):
    python run_pipeline.py --start 42 --min-interval 4      # continue from ticket 42, at most one model call every 4 s
    python run_pipeline.py --pause 20                       # sleep 20 s between tickets
Every ticket is traced (data/shopsense/traces/traces.jsonl, plus LangFuse when LANGFUSE_* keys are set). The summary lists the
tickets that were served DEGRADED (fallback model, unreachable order system ...) so a rate-limited night is never mistaken for
a normal one, and ends with a per-step latency / token dashboard.

Money only moves in the `commit` step, through tools/action_tools.py, which enforces the role limit itself: the system role can
refund under INR 2,000; a manager under INR 10,000; an executive anything. Nothing here can raise its own authority.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.core import llm_client  # noqa: E402
from shopsense.core.contacts import ContactLedger  # noqa: E402
from shopsense.core.ticket_intake import parse_envelopes_jsonl  # noqa: E402
from shopsense.observability import set_tracer, tracer_from_env  # noqa: E402
from shopsense.observability.report import format_dashboard  # noqa: E402
from shopsense.runtime import ACTION_LEDGER, TICKET_DB, TRACE_FILE, build_deps, build_graph  # noqa: E402
from shopsense.workflows import ticket_resolution_graph as tg  # noqa: E402

TICKETS_PATH = Path(__file__).resolve().parent / "data" / "shopsense" / "intake" / "tickets.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N tickets (handy on a free-tier token budget)")
    ap.add_argument("--start", type=int, default=1, metavar="N", help="begin at the Nth ticket (1-based) -- continue after a rate-limited run")
    ap.add_argument("--pause", type=float, default=0.0, metavar="SECONDS", help="sleep this long between tickets (free-tier token budgets)")
    ap.add_argument("--min-interval", type=float, default=None, metavar="SECONDS",
                    help="space model calls at least this far apart (sets SHOPSENSE_LLM_MIN_INTERVAL)")
    ap.add_argument("--tickets", default=str(TICKETS_PATH), metavar="FILE",
                    help="a JSON-lines file of tickets (ticket_id, channel, created_at, customer_id, text), one per line; default: the bundled 126")
    ap.add_argument("--no-trace", action="store_true", help="turn tracing off (SHOPSENSE_TRACING=0)")
    ap.add_argument("--workflow", action="store_true", help="run the plain M5 workflow instead of the M6 team")
    args = ap.parse_args()
    if args.min_interval is not None:
        os.environ["SHOPSENSE_LLM_MIN_INTERVAL"] = str(args.min_interval)
    if args.no_trace:
        os.environ["SHOPSENSE_TRACING"] = "0"
    tracer = tracer_from_env(trace_file=TRACE_FILE)
    set_tracer(tracer)
    topology = "workflow" if args.workflow else "team"

    print("=" * 70)
    print(f"ShopSense pipeline: M1 intake -> {'M5 workflow' if args.workflow else 'M6 support team'} (triage / policy / actions / routing)")
    print("=" * 70)
    if not tg.LANGGRAPH_AVAILABLE:
        print("langgraph is not installed -- run: pip install langgraph langgraph-checkpoint-sqlite")
        sys.exit(1)

    tickets_path = Path(args.tickets)
    if not tickets_path.is_file():
        print(f"Tickets file not found: {tickets_path}")
        sys.exit(2)
    print(f"\n[M1] Parsing envelopes from {tickets_path} ...")
    envelopes, failed = parse_envelopes_jsonl(str(tickets_path))
    print(f"  envelopes valid: {len(envelopes)}  failed: {len(failed)}")
    if args.start > 1:
        envelopes = envelopes[args.start - 1:]
        print(f"  starting at ticket #{args.start} ({envelopes[0].ticket_id if envelopes else 'none left'})")
    if args.limit:
        envelopes = envelopes[: args.limit]
    st = tracer.status()
    print(f"  tracing: {'on' if st['enabled'] else 'off'}; LangFuse {'on' if st['langfuse'] else 'off'}; local trace file {st['trace_file']}")

    print("\n[M4] Loading + indexing the policy corpus, building the graph ...")
    # Contact history comes from the ticket file itself, so ESC-REPEAT ("3rd contact within 7 days") is judged on real timestamps.
    deps = build_deps(ledger_path=ACTION_LEDGER, contacts=ContactLedger.from_tickets(envelopes))
    graph = build_graph(deps, topology=topology, db_path=TICKET_DB)
    print(f"  {len(deps.kb.chunks)} chunks across {len({c.source for c in deps.kb.chunks})} policy documents; "
          f"checkpoint {TICKET_DB}; memo team {'on' if deps.memo_runner else 'off (langgraph missing)'}")

    counts = {"resolved": 0, "escalated": 0, "awaiting_approval": 0, "errors": 0}
    waiting: list[str] = []
    degraded: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    tokens = [0, 0]
    for n, ticket in enumerate(envelopes):
        if n and args.pause:
            time.sleep(args.pause)
        state = tg.seed_state(ticket_id=ticket.ticket_id, customer_id=ticket.customer_id, text=ticket.text,
                              channel=ticket.channel, created_at=ticket.created_at)
        try:
            result, _cfg = tg.run_ticket(graph, state, tags=(f"topology:{topology}",))
        except Exception as e:  # noqa: BLE001 -- one failing ticket (rate limit, outage) must not stop the batch
            counts["errors"] += 1
            errors[ticket.ticket_id] = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  {ticket.ticket_id}: ERROR {errors[ticket.ticket_id]}")
            continue
        out = tg.outcome_of(result)
        info = tg.run_info_of(result)
        tokens[0] += info.get("tokens_in", 0)
        tokens[1] += info.get("tokens_out", 0)
        if info.get("degraded"):
            degraded[ticket.ticket_id] = info.get("degraded_reasons", [])
            print(f"  {ticket.ticket_id}: DEGRADED ({', '.join(info.get('degraded_reasons', []))})")
        if out["paused"]:
            counts["awaiting_approval"] += 1
            waiting.append(ticket.ticket_id)
            payload = result["__interrupt__"][0].value
            print(f"  {ticket.ticket_id}: PAUSED for {payload['needs_role']} approval ({payload['escalation_codes'] or payload['reasons']}, "
                  f"order {payload['order_id']}). Resume with:\n      python resume_ticket_approval.py {ticket.ticket_id} approve|deny|modify "
                  f"--role {payload['needs_role']}")
        elif out["route"] == "human_review":
            counts["escalated"] += 1
            h = result.get("handoff") or {}
            print(f"  {ticket.ticket_id}: ESCALATED -> {h.get('queue')} {out['escalation_codes']} memo={h.get('memo_status', 'n/a')}")
        else:
            counts["resolved"] += 1
            print(f"  {ticket.ticket_id}: {result.get('status')}: {out['answer'][:140]}")

    tracer.flush()
    print("\n" + "=" * 70)
    print(f"Done. {len(envelopes)} tickets: {counts['resolved']} resolved automatically, {counts['escalated']} handed to a human team, "
          f"{counts['awaiting_approval']} waiting for approval ({waiting or 'none'}), {counts['errors']} errored.")
    print(f"Served DEGRADED: {len(degraded)} ticket(s) {sorted(degraded) or ''}  -- these ran on a fallback model, or while the order system / model was struggling.")
    if errors:
        print(f"Errored (re-run with --start to continue; no partial state was committed): {sorted(errors)}")
    print(f"Tokens this run: {tokens[0]} in / {tokens[1]} out")
    health = llm_client.llm_health()
    print(f"LLM client: {health['counters']}  breakers: {health['models']}")
    print(f"Order-system guard: {deps.order_guard.snapshot()}")
    print("=" * 70)
    if tracer.enabled:
        print("\n" + format_dashboard(tracer.store.traces()))
        print(f"\nFull trace file: {TRACE_FILE}   (python run_trace_report.py to re-read it; python check_langfuse.py to test LangFuse)")


if __name__ == "__main__":
    main()
