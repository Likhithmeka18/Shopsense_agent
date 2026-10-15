"""
Milestone 7: prove LangFuse tracing works from THIS machine, in about ten seconds, before spending model tokens on a pipeline run.

PowerShell (keys are typed into the window, never into a file or a chat):
    $env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
    $env:LANGFUSE_SECRET_KEY = "sk-lf-..."
    $env:LANGFUSE_BASE_URL   = "https://cloud.langfuse.com"      # or https://us.cloud.langfuse.com for the US region
    python check_langfuse.py

It checks the three things that usually go wrong, in order -- keys present, SDK importable, credentials accepted -- then sends one
nested demo trace through ShopSense's own tracer (so the redaction and hashed user id are exercised too) and tells you what to
look for in the LangFuse UI. It never prints a key.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        print(f"{name:22} {'set' if os.environ.get(name) else 'MISSING'}")
    base = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    print(f"{'LANGFUSE_BASE_URL':22} {base or 'not set (the SDK defaults to https://cloud.langfuse.com -- set it if your project is in the US region)'}")
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("\nSet both keys in this PowerShell window first (see the top of this file). Nothing was sent.")
        return 1
    try:
        import langfuse
        from langfuse import get_client
    except ImportError:
        print("\nThe langfuse package is not installed. Run:  pip install langfuse")
        return 1
    print(f"langfuse SDK           {getattr(langfuse, '__version__', 'installed')}")
    lf = get_client()
    try:
        ok = lf.auth_check()
    except Exception as e:  # noqa: BLE001
        print(f"\nauth_check raised {type(e).__name__}: {str(e)[:200]}\nUsually a wrong LANGFUSE_BASE_URL region or no internet from this network.")
        return 1
    if not ok:
        print("\nauth_check returned False: the keys were rejected. Re-copy BOTH keys from the same LangFuse project (Settings -> API Keys).")
        return 1
    print("auth_check             OK")

    from shopsense.observability import tracer_from_env
    tracer = tracer_from_env()
    if not tracer.langfuse_enabled:
        print("\nShopSense's tracer did not pick LangFuse up (SHOPSENSE_LANGFUSE=0, or SHOPSENSE_TRACING=0 is set).")
        return 1
    session = f"check-{int(time.time())}"
    with tracer.run(session_id=session, user_id="CUST-DEMO", tags=["shopsense", "check_langfuse"],
                    input={"text": "Where is ORD-1234? Mail me at demo.person@example.com or call 9876543210."}) as run:
        with tracer.span("node:triage", kind="agent"):
            with tracer.span("order_system.read", kind="tool", input={"order_id": "ORD-1234"}):
                pass
            with tracer.span("llm.generate", kind="generation", model="demo/model", input=[{"role": "user", "content": "hello"}]) as g:
                g.update(output="hi", usage={"input": 12, "output": 3})
        run.update(output={"reply": "demo"}, metadata={"route": "resolve"})
    tracer.flush()
    print(f"\nSent one demo trace. Open LangFuse -> Sessions (or Tracing) and find session id:  {session}")
    print("You should see ticket.run > node:triage > order_system.read + llm.generate, the user as a hashed 'u_...' id, and the e-mail /\n"
          "phone in the input shown as [REDACTED]. If the trace is missing after a minute, call this script again and check LANGFUSE_BASE_URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
