"""
Milestone 8: evaluate the output guardrails in BOTH directions (false approvals AND false rejections) and print the comparison.

    python run_guardrail_eval.py                  # all presets: labelled cases + 146 real pipeline replies + mutated real replies
    python run_guardrail_eval.py --json out.json  # the same report as JSON
    python run_guardrail_eval.py --regenerate     # rebuild data/shopsense/eval/guardrail_cases.json first

No API key, no network: replies come from the offline pipeline (real tables, real policy chunks, real ledger). The labelled cases are
the in-sample set the default thresholds were tuned on -- the mutation and real-reply columns are the less biased ones.
"""
import argparse
import json
import subprocess
import sys

sys.path.insert(0, ".")
import os
os.environ.setdefault("SHOPSENSE_OFFLINE", "1")
os.environ.setdefault("SHOPSENSE_TRACING", "0")


def _install_litellm_stub_if_missing() -> None:
    """Offline evaluation never calls a model, so the model client only has to be importable."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("litellm")
        stub.completion = lambda **kw: None
        sys.modules["litellm"] = stub


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="FILE", help="also write the raw report as JSON")
    ap.add_argument("--regenerate", action="store_true", help="rebuild the labelled cases file first")
    args = ap.parse_args()
    if args.regenerate:
        subprocess.run([sys.executable, "generate_guardrail_cases.py"], check=True)
    _install_litellm_stub_if_missing()
    from shopsense.eval.guardrail_eval import evaluate_presets, format_report
    report = evaluate_presets()
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
