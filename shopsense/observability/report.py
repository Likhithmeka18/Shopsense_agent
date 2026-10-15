"""
ShopSense Observability: the local dashboard (per-step latency, tokens, degraded runs and the SLO list).

Built from the kept traces (in memory or from the JSONL trace file), so a grading harness or a laptop with no LangFuse access
can still answer: which agent is slow, how many tokens did a ticket cost, how often did we degrade? In production the same
aggregation would query LangFuse's observations API for a session id; the shape (group by span name, sum/avg/p95 of duration,
sum of tokens) does not change.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Where run_pipeline.py / resume_ticket_approval.py write traces and run_trace_report.py reads them.
DEFAULT_TRACE_FILE = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "traces" / "traces.jsonl"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (p50 / p95), 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def load_traces(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def per_span_table(traces: Iterable[dict]) -> list[dict]:
    """One row per span name: calls, errors, total/avg/p95 ms, tokens. Sorted by total time, biggest first."""
    acc: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "ms": [], "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
    for t in traces:
        for s in t["spans"]:
            a = acc[s["name"]]
            a["calls"] += 1
            a["errors"] += 0 if s["ok"] else 1
            a["ms"].append(s["duration_ms"])
            a["tokens_in"] += int((s.get("usage") or {}).get("input", 0) or 0)
            a["tokens_out"] += int((s.get("usage") or {}).get("output", 0) or 0)
            a["cost_usd"] += float(s.get("cost_usd") or 0.0)
    rows = [{"span": n, "calls": a["calls"], "errors": a["errors"], "total_ms": round(sum(a["ms"]), 1), "avg_ms": round(sum(a["ms"]) / a["calls"], 1),
             "p95_ms": round(percentile(a["ms"], 95), 1), "tokens_in": a["tokens_in"], "tokens_out": a["tokens_out"], "cost_usd": round(a["cost_usd"], 6)}
            for n, a in acc.items()]
    return sorted(rows, key=lambda r: r["total_ms"], reverse=True)


def slo_summary(traces: list[dict]) -> dict:
    """The SLO list, computed from what we recorded: reliability, performance, cost."""
    roots = [t for t in traces if t.get("name") == "ticket.run"] or traces
    n = len(roots)
    if n == 0:
        return {"tickets": 0}
    durations = [t["duration_ms"] for t in roots]
    degraded = sum(1 for t in roots if t.get("degraded"))
    errors = sum(1 for t in roots if not t.get("ok"))
    routes: dict[str, int] = defaultdict(int)
    for t in roots:
        routes[t.get("route") or "unknown"] += 1
    tok_in = sum(t.get("run", {}).get("tokens_in", 0) for t in roots)
    tok_out = sum(t.get("run", {}).get("tokens_out", 0) for t in roots)
    cost = sum(t.get("run", {}).get("cost_usd", 0.0) for t in roots)
    # Count ATTEMPTS (every llm.generate span, failed ones included): a provider that is failing is exactly when the load matters. The run's own
    # counter only counts calls that returned, so it is the fallback for traces that carry no spans.
    llm_spans = [sp for t in roots for sp in t.get("spans", []) if sp.get("name") == "llm.generate"]
    llm_calls = len(llm_spans) if llm_spans else sum(t.get("run", {}).get("llm_calls", 0) for t in roots)
    llm_failed = sum(1 for sp in llm_spans if not sp.get("ok", True))
    return {"tickets": n, "error_rate": round(errors / n, 4), "degraded_rate": round(degraded / n, 4), "fallback_rate": round(degraded / n, 4),
            "latency_p50_ms": round(percentile(durations, 50), 1), "latency_p95_ms": round(percentile(durations, 95), 1),
            "routes": dict(routes), "llm_calls_per_ticket": round(llm_calls / n, 2), "llm_failed_calls": llm_failed, "tokens_per_ticket": round((tok_in + tok_out) / n, 1),
            "cost_usd_per_ticket": round(cost / n, 6)}


def format_dashboard(traces: list[dict], *, top: int = 12) -> str:
    rows = per_span_table(traces)
    slo = slo_summary(traces)
    lines = ["Per-agent / per-step view (from the kept traces)",
             f"{'span':32s} {'calls':>6s} {'err':>4s} {'total_ms':>10s} {'avg_ms':>9s} {'p95_ms':>9s} {'tok_in':>8s} {'tok_out':>8s}"]
    for r in rows[:top]:
        lines.append(f"{r['span'][:32]:32s} {r['calls']:6d} {r['errors']:4d} {r['total_ms']:10.1f} {r['avg_ms']:9.1f} {r['p95_ms']:9.1f} "
                     f"{r['tokens_in']:8d} {r['tokens_out']:8d}")
    lines += ["", "SLOs"]
    if slo["tickets"] == 0:
        lines.append("  (no traces recorded)")
    else:
        lines += [f"  tickets traced        {slo['tickets']}", f"  error rate            {slo['error_rate']:.1%}",
                  f"  degraded/fallback rate {slo['degraded_rate']:.1%}   <- a ticket served while the model/order system was struggling",
                  f"  latency p50 / p95     {slo['latency_p50_ms']:.0f} ms / {slo['latency_p95_ms']:.0f} ms",
                  f"  LLM calls / ticket    {slo['llm_calls_per_ticket']}   ({slo.get('llm_failed_calls', 0)} failed in total)", f"  tokens / ticket       {slo['tokens_per_ticket']}",
                  f"  cost / ticket (USD)   {slo['cost_usd_per_ticket']}  (0 when the provider's price is unknown, e.g. free tiers)",
                  f"  routes                {slo['routes']}"]
    return "\n".join(lines)
