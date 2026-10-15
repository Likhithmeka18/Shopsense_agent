"""
ShopSense: run the golden set (Asset D).

    python run_golden_eval.py                 # offline: structure + deterministic floor + retrieval (no LLM, no key)
    python run_golden_eval.py --up-to M5      # only the cases that exist by that milestone
    python run_golden_eval.py --pdf           # run the retrieval layer on the lossy PDF corpus too
    python run_golden_eval.py --live          # END-TO-END: every case through the real ticket graph (needs a live LLM)
    python run_golden_eval.py --live --only SS-EV-007,SS-EV-009   # re-run just those cases (saves free-tier tokens)
    python run_golden_eval.py --live --show-answers               # also print the agent's reply for every failed case

Offline mode answers "is the data sound and does the cheap machinery behave?" in seconds and for free:
  1. the golden set itself is valid (guideline category ranges, hand-written, must_not_contain populated);
  2. the deterministic safety net raises the mandatory codes / injection label the data expects;
  3. the expected policy DOCUMENTS come back from retrieval for each case's policy_query (BM25 and hybrid).
Live mode adds the layer that matters: route, wording and forbidden-phrase checks on the system's actual answer.
Embeddings use the project's embedding client; if no embedding key is configured the dense half falls back to a
weak offline embedding, so treat hybrid numbers from an offline run as a floor, not as the real figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopsense.eval.golden import floor_check, load_golden, retrieval_check, summarize, validate_golden  # noqa: E402
from shopsense.rag.bm25 import BM25Index  # noqa: E402
from shopsense.rag.hybrid import hybrid_search  # noqa: E402
from shopsense.rag.index import VectorIndex  # noqa: E402
from shopsense.rag.ingest import load_policy_corpus, load_policy_corpus_pdf  # noqa: E402
from shopsense.tools.order_tools import MockAPI  # noqa: E402


def offline(cases: list[dict], include_pdf: bool) -> bool:
    ok = True
    problems = validate_golden(cases) if len(cases) == 20 else validate_golden(cases)
    print(f"[1] golden set structure: {'OK' if not problems else 'PROBLEMS'} ({len(cases)} cases)")
    for p in problems:
        print("    -", p)
    ok &= not problems

    api = MockAPI()
    floor = [floor_check(c, customer=api.customers.get(c["ticket"]["customer_id"])) for c in cases]
    bad = [f for f in floor if not f["ok"]]
    print(f"[2] deterministic floor (safety net / injection / abuse flag): {len(floor) - len(bad)}/{len(floor)} cases agree with the data")
    for f in bad:
        print("    -", f["id"], f["problems"])
    ok &= not bad

    corpora = [("markdown", load_policy_corpus())]
    if include_pdf:
        corpora.append(("pdf", load_policy_corpus_pdf()))
    for label, chunks in corpora:
        bm25 = BM25Index(chunks)
        from shopsense.memory.embeddings import DEFAULT_EMBED_MODEL
        print(f"[3] dense embedder: {DEFAULT_EMBED_MODEL or 'offline hashed bag-of-words (floor)'}")
        try:
            vec = VectorIndex(chunks)
        except Exception as e:  # noqa: BLE001 -- a missing key / wrong model name should read as a message, not a traceback
            print(f"    could not build the dense index with SHOPSENSE_EMBED_MODEL={DEFAULT_EMBED_MODEL!r}: {type(e).__name__}: {str(e)[:300]}")
            print("    check the model string and that its API key is set in THIS terminal; unset SHOPSENSE_EMBED_MODEL for the offline run")
            return False
        for name, fn in (("bm25", lambda q, k: bm25.search(q, k=k)),
                         ("hybrid", lambda q, k: hybrid_search(vec, bm25, q, k=k, pool=10))):
            rows = [r for r in (retrieval_check(c, fn, chunks, k=5) for c in cases) if r["applicable"]]
            hit = sum(r["ok"] for r in rows)
            print(f"[3] retrieval ({label}, {name}, top-5): expected policy documents found for {hit}/{len(rows)} cases")
            for r in rows:
                if not r["ok"]:
                    print(f"    - {r['id']}: missing {r['missing']}")
    return ok


def live(cases: list[dict], *, show_answers: bool = False) -> bool:
    try:
        from shopsense.eval.live import run_cases_live
    except ImportError:
        print("Live mode needs the ticket-resolution graph (shopsense/eval/live.py) - not available in this build.")
        return False
    scores = run_cases_live(cases)
    summary = summarize(scores)
    print("\nLIVE RESULTS")
    print(f"  passed {summary['passed']}/{summary['n']}  route accuracy {summary['route_accuracy']}  unsafe outputs {summary['unsafe_outputs']}")
    for cat, v in summary["by_category"].items():
        print(f"  {cat:13s} n={v['n']}  pass_rate={v['pass_rate']}")
    if summary["failed"]:
        print("  failed:", ", ".join(summary["failed"]))
    if show_answers:
        for sc in scores:
            if not sc["passed"]:
                bad = [k for k, v in sc.items() if k.endswith("_ok") and not v]
                print(f"\n--- {sc['id']} failed {bad or ''} {sc.get('error', '')}\n{sc.get('answer', '')}")
    return summary["unsafe_outputs"] == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run every case end-to-end through the ticket graph (needs a live LLM)")
    ap.add_argument("--only", default=None, metavar="IDS", help="with --live: comma-separated case ids to run, e.g. SS-EV-007,SS-EV-009")
    ap.add_argument("--show-answers", action="store_true", help="with --live: print the reply of every failed case")
    ap.add_argument("--up-to", choices=["M4", "M5", "M6", "M8"], default=None)
    ap.add_argument("--pdf", action="store_true", help="also run retrieval on the lossy PDF corpus")
    args = ap.parse_args()
    cases = load_golden(up_to=args.up_to)
    if args.only:
        if not args.live:
            ap.error("--only is for --live runs (the offline checks validate the whole set's category balance)")
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = sorted(set(wanted) - {c["id"] for c in cases})
        if unknown:
            ap.error(f"unknown case id(s): {', '.join(unknown)}")
        cases = [c for c in cases if c["id"] in set(wanted)]
    print("=" * 70)
    print(f"ShopSense golden set ({len(cases)} cases{', up to ' + args.up_to if args.up_to else ''})")
    print("=" * 70)
    good = live(cases, show_answers=args.show_answers) if args.live else offline(cases, args.pdf)
    print("\n" + ("PASS" if good else "FAIL"))
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
