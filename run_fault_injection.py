"""
Milestone 7: seeded fault injection against ShopSense -- before/after numbers for the reliability hardening.

    python run_fault_injection.py                     # all three experiments, default seed
    python run_fault_injection.py --seed 7 --fail-rate 0.10
    python run_fault_injection.py --only order        # order | outage | llm
    python run_fault_injection.py --json report.json

No API key, no network, no langgraph: the model-backed pieces are offline doubles and the provider is scripted, so the numbers are
exactly reproducible (same seed -> same numbers). That is the point: a regression shows up as a changed number.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")


def _install_litellm_stub_if_missing() -> None:
    """The LLM experiment drives the client's retry logic with a scripted provider, so it only needs `litellm` to be importable."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("litellm")
        stub.completion = lambda **kw: None
        sys.modules["litellm"] = stub


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-rate", type=float, default=0.05, help="chance a lookup raises a connection error")
    ap.add_argument("--timeout-rate", type=float, default=0.03)
    ap.add_argument("--malformed-rate", type=float, default=0.02, help="chance a record comes back with a field missing")
    ap.add_argument("--lost-ack-rate", type=float, default=0.25, help="chance a successful WRITE reports a timeout (tests idempotency)")
    ap.add_argument("--only", choices=("order", "outage", "llm"))
    ap.add_argument("--json", metavar="FILE", help="also write the raw numbers to FILE")
    args = ap.parse_args()

    _install_litellm_stub_if_missing()
    from shopsense.eval import fault_injection as fi

    report: dict = {}
    if args.only in (None, "order"):
        cfg = fi.FaultConfig(seed=args.seed, fail_rate=args.fail_rate, timeout_rate=args.timeout_rate,
                             malformed_rate=args.malformed_rate, lost_ack_rate=args.lost_ack_rate)
        report["order_system"] = fi.order_system_experiment(cfg)
        print(fi.format_order_experiment(report["order_system"]), "\n")
    if args.only in (None, "outage"):
        report["outage"] = fi.outage_experiment(seed=args.seed)
        print(fi.format_outage(report["outage"]), "\n")
    if args.only in (None, "llm"):
        report["llm"] = fi.llm_experiment(seed=args.seed)
        print(fi.format_llm(report["llm"]), "\n")

    unsafe = report.get("order_system", {}).get("hardened", {})
    bad = unsafe.get("unsafe_actions", 0) + unsafe.get("duplicate_actions", 0) + unsafe.get("crashed", 0)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"raw numbers written to {args.json}")
    print("RESULT:", "PASS -- hardened run had 0 crashes, 0 unsafe actions, 0 duplicate payments" if not bad and "order_system" in report
          else ("FAIL -- see the table above" if bad else "(order-system experiment not run)"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
