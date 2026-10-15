# ShopSense — Deployment checklist (Milestone 8)

Work top to bottom. A box you cannot tick is a decision to record, not to skip. "Demo" = the capstone walkthrough; "Production" = real customers.

## A. Before you start the server (both)

- [ ] `pytest tests/ -v` passes (2 tests are skipped without the real `langgraph`).
- [ ] `python run_fault_injection.py` prints the pinned numbers (48 → 0 crashed, 0 unsafe, 0 duplicate payments).
- [ ] `python run_guardrail_eval.py` shows the default preset at 0 / 0 / 0 / 0 (see `docs/GUARDRAIL_EVALUATION.md`).
- [ ] `python run_golden_eval.py --live` pass rate recorded as the baseline (reference run: 20/20). Re-run after any model, prompt or corpus change.
- [ ] Secrets are environment variables only (PowerShell: `$env:GEMINI_API_KEY = "..."` typed in the window). None in the repo, a chat, a screenshot or a log.
- [ ] `python check_langfuse.py` reports keys → SDK → auth → demo trace OK (or you have decided on local traces only).

## B. Demo walkthrough (ngrok + Postman)

**This needs no model quota.** Run the server with `SHOPSENSE_OFFLINE=1` and every step below works: ngrok and Postman test the HTTP layer
(routes, validation, what the public response contains), not the model. Use the live model afterwards only if you want to see real wording.

Open three PowerShell terminals in the project folder (activate your virtual environment in each).

**Terminal 1: the server**
```powershell
pip install fastapi uvicorn          # once
$env:SHOPSENSE_OFFLINE = "1"
python -m uvicorn shopsense.api:app --port 8000
```
Check in a browser: `http://127.0.0.1:8000/health` (JSON, `"mode": "offline"`) and `http://127.0.0.1:8000/docs` (interactive page: try `POST /chat`).

**Postman, local first** (prove it locally before exposing anything)
1. Postman -> Import -> drop both files from `docs/postman/`: `ShopSense.postman_collection.json` and `ShopSense.postman_environment.json`.
2. Top-right environment drop-down -> **ShopSense (local)** (`baseUrl` = `http://127.0.0.1:8000`).
3. Open the collection -> **Run** -> Run ShopSense demo API. Expect **all tests green** (65 assertions across 10 requests). Click any request to read its JSON.
   The collection works offline and live: a healthy case must give the expected outcome, a degraded one (model rate-limited) must still end safely.

**Terminal 2: the public URL**
1. Create a free account at ngrok.com, install ngrok (`winget install ngrok.ngrok`, or download the zip from the ngrok dashboard).
2. Copy your authtoken from the ngrok dashboard (Your Authtoken) and register it once: `ngrok config add-authtoken <YOUR_TOKEN>`. Keep the token to yourself.
3. `ngrok http 8000`. Copy the **Forwarding** address it prints (`https://....ngrok-free.app` or `.ngrok-free.dev`).

**Postman, public**
1. In the environment, change `baseUrl` to that Forwarding address (no trailing slash) and save.
2. Run the collection again. Same 65 assertions, now through the tunnel. Terminal 1 logs each request.
3. The script client works too: `python demo_api_client.py --url https://<your-forwarding-address>`.

**Expected results** (also what the collection asserts)
- Policy question -> `resolved`; order status -> `resolved`; refund of INR 25,516 -> `awaiting_approval`, `awaiting_approval: true`.
- Flagged account and prompt injection -> `escalated`; the injection case has `input_flags: ["prompt_injection_attempt"]`; **no field or sentence mentions the flag**, and the response contains only the 8 documented fields.
- Guest asking to cancel an order -> `escalated`; nothing is cancelled.
- Empty, over-long (2,001 characters) and missing `message` -> **422**, with no stack trace in the body.

**When you are done:** press Ctrl+C in the ngrok terminal. The demo API has no authentication, so do not leave the tunnel open.
If a Postman request hangs on the public URL, the proxy timed out before the service did: keep `SHOPSENSE_API_DEADLINE_S` below the tunnel's timeout (section C).

## C. Production gates (do not go live without these)

**Identity and access**
- [ ] `customer_id` derived from an authenticated session; the request-body field removed or ignored.
- [ ] API authentication (token or session) and per-customer rate limiting at the proxy; request size limit at the proxy as well as the 2,000-character check.
- [ ] CORS limited to your own front-end origin. HTTPS only.
- [ ] Approver endpoint with real authentication and a role check, replacing `resume_ticket_approval.py`. Replace `SHOPSENSE_APPROVAL_SECRET` (and the MCP token secret) with a credential service.

**State and concurrency**
- [ ] Contact ledger, action ledger and checkpoint moved from local files/memory to a shared store (the demo's lock allows one ticket at a time).
- [ ] Dependencies built per request or per worker, not shared; one worker per checkpoint file until the store is shared.
- [ ] Order writes confirmed idempotent against the real order API (the mock's idempotency key is tested; yours is not).

**Reliability**
- [ ] `SHOPSENSE_API_DEADLINE_S` chosen for your front-end's own timeout (default 40 s; keep it below the proxy/ngrok timeout so the customer gets the hand-off reply, not a gateway error). The memo team stays off for chat (`SHOPSENSE_API_MEMO_TEAM=0`) unless you accept ~1 minute escalations.
- [ ] `SHOPSENSE_ORDER_*` retry/breaker settings tuned to the real order API's latency and error rate; timeouts set on every client.
- [ ] `SHOPSENSE_LLM_FALLBACK_MODELS` chosen deliberately (a fallback model is a quality change; those tickets are flagged degraded).
- [ ] The human queue has an owner, working hours and a backlog alert. Every failure path ends there.

**Safety**
- [ ] Guardrail thresholds re-tuned on labelled **live** replies (the offline numbers use retrieved chunks, not model prose); a person samples blocked replies weekly and labels them right/wrong.
- [ ] Refund cap (`AUTO_APPROVE_BELOW_INR`, INR 2,000) confirmed by the business owner.
- [ ] Data retention for traces and checkpoints decided; redaction reviewed against your real ticket text (it is a regex floor).
- [ ] Error responses checked: a failure returns a generic message, never a stack trace or an escalation code.

## D. Service-level objectives (starting points — adjust to your traffic)

| Signal | Target | Source |
|---|---|---|
| Availability of `/health` | 99.5 % | uptime probe |
| p95 `/chat` latency | set after a week of real traffic (it is dominated by model calls) | trace dashboard |
| Crashed tickets (HTTP 5xx) | < 0.5 % | access log + `errors` counter |
| Unsafe actions / duplicate payments | **0** (any is an incident) | action ledger, `guardrail` events |
| Degraded share | alert if > 5 % over 1 h | `degraded` flag, `run_trace_report.py --degraded` |
| Guardrail blocks | alert on a jump; review the sample | `guardrail_blocked` in `/health` |

## E. Monitoring and alerts

- [ ] Probe `GET /health` (it never calls a model, so it cannot hit a rate limit). `status: degraded` means a breaker is not closed — page when it persists.
- [ ] Dashboards: tickets by outcome, degraded reasons, breaker opens, tokens and cost per ticket, guardrail blocks by code.
- [ ] LangFuse: sessions = tickets; filter by tag `degraded`, by prompt version after a change.

## F. Rollback

- [ ] Keep the previous release and its `requirements.txt`; rollback = redeploy it. State files are append-only and compatible across M7→M8.
- [ ] Emergency off-switches that need no deploy: stop routing traffic to `/chat` (tickets queue to humans); `SHOPSENSE_LLM_MODEL` back to the previous model; `SHOPSENSE_TRACING=0` if tracing itself misbehaves.
- [ ] After a rollback re-run section A.

## G. First 48 hours

- [ ] Shadow or limited traffic first; a person reads every escalated and blocked reply.
- [ ] Compare golden pass rate, escalation rate and degraded rate with the baseline daily.
