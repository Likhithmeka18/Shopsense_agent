"""
Diagnose the ShopSense MCP server in isolation (no LangGraph, no model, no langchain adapters).

    python check_mcp_server.py

Starts shopsense/mcp/shopsense_server.py exactly as the demo does, sends one MCP `initialize` request, and prints what comes back
plus everything the server wrote to stderr. If the demo says "Connection closed", the reason is in the stderr block below.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "shopsense" / "mcp" / "shopsense_server.py"


def version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print("python", sys.version.split()[0], "|", sys.executable)
    for pkg in ("fastmcp", "fastmcp-slim", "mcp", "mcp-types", "langchain-mcp-adapters", "langgraph"):
        print(f"  {pkg:24s} {version(pkg)}")
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "check", "version": "0"}}}
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    lst = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    call = lambda i, name, args: {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": args}}  # noqa: E731
    payload = "\n".join(json.dumps(m) for m in (
        init, note, lst,
        call(3, "order_lookup", {"order_id": "ORD-4445"}),
        call(4, "issue_refund", {"order_id": "ORD-4445", "amount_inr": 5000, "ticket_id": "TCK-CHECK", "approval_token": "forged"}),
    )) + "\n"
    import os
    import queue
    import threading
    import time
    env = {**os.environ, "SHOPSENSE_ACTION_LEDGER": str(Path(tempfile.mkdtemp()) / "a.jsonl")}
    proc = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    lines: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in proc.stdout], daemon=True).start()
    errs: list[str] = []
    threading.Thread(target=lambda: errs.extend(proc.stderr), daemon=True).start()
    try:
        proc.stdin.write(payload); proc.stdin.flush()
    except OSError:
        pass                                                   # the server already died; its stderr explains why
    stdout_lines, wanted, deadline = [], {1, 2, 3, 4}, time.time() + 30
    while wanted and time.time() < deadline and (proc.poll() is None or not lines.empty()):
        try:
            ln = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        stdout_lines.append(ln)
        try:
            wanted.discard(json.loads(ln).get("id"))
        except ValueError:
            pass
    try:
        proc.stdin.close()
    except OSError:
        pass
    time.sleep(0.3)
    if proc.poll() is None:
        proc.kill()
    class r:                                                   # noqa: N801 -- mimic the subprocess.run result used below
        returncode = proc.returncode
        stderr = "".join(errs)
        stdout = "".join(stdout_lines)
    print(f"\nserver exit code: {r.returncode} (-9 / 1 after a kill is expected once the checks are done)")
    print("\n--- server stderr (the reason, if it crashed) ---")
    print(r.stderr.strip() or "(empty)")
    print("\n--- server stdout (MCP replies) ---")
    replies = [ln for ln in r.stdout.splitlines() if ln.strip()]
    ok_init = ok_tools = False
    for ln in replies:
        try:
            m = json.loads(ln)
        except ValueError:
            print("NOT JSON (this corrupts the protocol):", ln[:200]); continue
        if m.get("id") == 1 and "result" in m:
            ok_init = True; print("initialize OK, server:", m["result"].get("serverInfo"))
        if m.get("id") in (3, 4) and "result" in m:
            text = " ".join(c.get("text", "") for c in m["result"].get("content", []))[:300]
            print({3: "order_lookup ORD-4445 ->", 4: "5,000 refund (needs a manager) with a forged token ->"}[m["id"]], text)
            if m["id"] == 4:
                refused = "refus" in text.lower() or "not authori" in text.lower() or '"ok": false' in text.lower() or "error" in text.lower() or "denied" in text.lower()
                print("  (the server must refuse this)", "refused" if refused else "!! NOT REFUSED -- tell me")
        if m.get("id") == 2 and "result" in m:
            names = sorted(t["name"] for t in m["result"]["tools"]); ok_tools = True; print("tools:", names)
    if not replies:
        print("(empty)")
    print("\nRESULT:", "server is healthy" if ok_init and ok_tools else "server is NOT working -- send me this whole output")
    return 0 if ok_init and ok_tools else 1


if __name__ == "__main__":
    sys.exit(main())
