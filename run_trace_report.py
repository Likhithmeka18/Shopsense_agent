"""
Milestone 7: read the local trace file and print the per-step / SLO dashboard , without calling any model.

    python run_trace_report.py                     # everything in data/shopsense/traces/traces.jsonl
    python run_trace_report.py --last 50           # only the 50 most recent traces
    python run_trace_report.py --degraded          # list tickets served degraded, with the reasons
    python run_trace_report.py --ticket TCK-0042   # one ticket: its span tree, durations, events (customer data already redacted)
    python run_trace_report.py --file other.jsonl

The same numbers exist in LangFuse when LANGFUSE_* keys are set; this works offline, on a laptop, and in a grading harness.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.observability.report import DEFAULT_TRACE_FILE as TRACE_FILE, format_dashboard, load_traces  # noqa: E402


def print_tree(trace: dict) -> None:
    spans = sorted(trace["spans"], key=lambda s: s["start"])
    children: dict = {}
    for s in spans:
        children.setdefault(s["parent_id"], []).append(s)

    def walk(parent, depth):
        for s in children.get(parent, []):
            flag = "" if s["ok"] else f"  !! {s['error']}"
            ev = f"  events={[e.get('kind') for e in s.get('events', [])]}" if s.get("events") else ""
            tok = f"  tokens={s['usage']}" if s.get("usage") else ""
            print(f"{'  ' * depth}{s['name']:{max(1, 34 - 2 * depth)}} {s['duration_ms']:9.1f} ms [{s['kind']}]{tok}{ev}{flag}")
            walk(s["span_id"], depth + 1)

    print(f"trace {trace['trace_id']}  session={trace['session_id']}  user={trace['user_id']}  ok={trace['ok']}  degraded={trace['degraded']}  "
          f"route={trace.get('route')}")
    print(f"tags: {trace.get('tags')}")
    walk(None, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(TRACE_FILE))
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--degraded", action="store_true")
    ap.add_argument("--ticket")
    args = ap.parse_args()
    traces = load_traces(args.file)
    if not traces:
        print(f"No traces in {args.file}. Run run_pipeline.py first (tracing is on by default).")
        return 1
    if args.ticket:
        hits = [t for t in traces if t["session_id"] == args.ticket]
        if not hits:
            print(f"No trace for {args.ticket} (healthy runs may have been sampled out: SHOPSENSE_TRACE_SAMPLE_RATE).")
            return 1
        for t in hits:
            print_tree(t)
            print()
        return 0
    if args.last:
        traces = traces[-args.last:]
    if args.degraded:
        rows = [t for t in traces if t.get("degraded")]
        for t in rows:
            print(f"{t['session_id']:12} route={t.get('route'):14} reasons={t['run'].get('degraded_reasons')} events={t['run'].get('events')}")
        print(f"\n{len(rows)} degraded of {len(traces)} traces")
        return 0
    print(format_dashboard(traces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
