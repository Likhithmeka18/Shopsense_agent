# ShopSense — Architecture review (Milestone 8)

Structure: system overview · components and data flow with failure modes · evaluation results · guardrails and safety · known limitations · deployment notes. Everything here is what the repository does today; "not verified" marks what has not been exercised end to end.

## 1. System overview

ShopSense answers customer-care messages for an e-commerce store. A message arrives, is turned into a structured ticket, checked against the
store's 14 policy documents and the customer's real order/shipment/refund records, and then one of three things happens:

1. **Resolved** — a reply is sent (and, for small refunds, the action is committed).
2. **Paused for approval** — a refund of INR 2,000 or more waits for a person (manager < INR 10,000, executive above).
3. **Handed to a person** — legal/safety wording, an injection attempt, an abuse-flagged account, a third contact in seven days, a lost parcel, someone else's order, a failed extraction, an unavailable order system or policy answer, or a blocked reply.

Design rule held through every milestone: **the model does language; code decides authority.** The model extracts fields and phrases one policy
sentence. Routing, refund tiers, role limits, idempotency and the final reply check are deterministic Python, so a clever customer cannot talk the system into a refund.

Entry point: `POST /chat` (FastAPI, `shopsense/api.py`) → `ChatService` (`shopsense/service.py`, framework-free) → the M6 support team (or the M5 workflow) → output guardrails → allow-listed response.

## 2. Components and data flow, with failure modes

```
request ─► input check ─► triage ─► policy_rag ─► order_actions (read, PROPOSE) ─► escalation_reviewer ─► route
 (422 if                     │ safety floor +        │ hybrid RAG, answer              auto ─► commit ─► reply
  empty / >2000)             │ contact ledger        │ from retrieved text             approval ─► PAUSE ─► person ─► commit ─► reply
                             └────────────── hard facts ──────────────────────────► human_review (packet + memo)
                                                    reply ─► OUTPUT GUARDRAILS ─► allow-listed JSON
every step: nested trace span (redacted)  ·  every order-system call: DependencyGuard (retry + breaker)
```

| Component | Job | What goes wrong | What the system does |
|---|---|---|---|
| Input check | reject empty / oversize messages | huge or empty body | 422 before any model call |
| Triage (`core/ticket_intake.py`, `safety_net.py`) | extract structure; deterministic legal/safety/injection floor; contact ledger | model returns bad JSON; provider down | repair loop; then `extraction failed` → human queue. The safety floor runs on the text itself, not on the model's output |
| Policy RAG (`rag/*`, `tools/policy_tools.py`) | hybrid BM25 + embedding retrieval, rerank, grounded answer | wrong chunk retrieved; model paraphrases a figure wrongly | figures are re-checked by the fabrication guardrail against the retrieved text |
| Order system (`tools/order_tools.py`, `reliability.py`) | customers, orders, shipments, refunds | timeouts, connection errors, malformed records, lost write acknowledgements | retried (transient only) with backoff; breaker opens after repeated failure; unavailable → ticket goes to the human queue with reason `order_system_unavailable`; writes are idempotent so a lost ack cannot pay twice |
| Policy answer (`rag/answer.py`, `policy_check_node`) | retrieval + a model-written answer | model down, rate-limited past the request budget, or an empty reply | an empty or failed answer is an OUTAGE, not a gap in the policy (a real "not covered" is words from the model and is still admitted honestly). When the reply would be the policy text itself (information questions) the ticket goes to the human queue with reason `policy_unavailable` and a deterministic holding reply, flagged degraded. Refund/status outcomes come from records, so they stand without the policy text |
| Order actions (`tools/action_tools.py`) | refund / replace / cancel | model asks for a bigger refund; wrong role | role limits enforced in code; ledger; the agent only *proposes* |
| Approval (`interrupt()` + SQLite checkpoint) | pause until a person decides | process restarts mid-pause | state is in the checkpoint file; `resume_ticket_approval.py` resumes it from a new process; role re-checked at commit |
| LLM client (`core/llm_client.py`) | all model calls | rate limit, overload, dead model, a 7-minute quota wait | provider-aware retry, per-model breaker, optional fallback models (use is logged and flags the ticket **degraded**). In the chat API each request has a waiting budget (`SHOPSENSE_API_DEADLINE_S`, default 40 s): a wait that would pass it is not slept, the ticket goes to a person and the failure counts toward the breaker |
| Output guardrails (`guardrails.py`) | last check before a customer sees text | invented figures, unearned promises, disclosure, echo | block → fixed safe reply + human review + `guardrail.blocked` event; check error → fail closed |
| Tracing (`observability/*`) | one trace per ticket, nested spans, redaction, tail sampling | LangFuse unreachable | local JSONL still written; tracing never fails a ticket |
| API (`api.py`) | HTTP surface | model/graph failure | ValueError → 422, RuntimeError → 503, anything else → generic 500 with no internals |

**Silent degradation is treated as a bug.** A fallback model, an unavailable order system or a blocked reply each set the `degraded` flag, show up
in the trace dashboard (`run_trace_report.py --degraded`) and in `/health`.

## 3. Evaluation results summary

| What | Result | Where reproduced | Status |
|---|---|---|---|
| Golden set, 20 cases (route, escalation codes, key facts) | **20/20** on `gemini/gemini-3.1-flash-lite` | `run_golden_eval.py --live` | live run |
| Order-system flakiness, seeded (seed 42; 5 % connection, 3 % timeout, 2 % malformed, 25 % lost write acks) | no hardening: 48 tickets crashed, 1 after money moved. Hardened: **0 crashed**, 0 unsafe, 0 duplicate payments, 9 handed to a person | `run_fault_injection.py --only order` | offline, exact |
| Order-system outage | retries only: 90 calls hit the dead system. With breaker: **15** hit it, 27 short-circuited | `--only outage` | offline, exact |
| Model provider faults (200 calls) | bare: 38 failed. Hardened: **0** failed. Dead primary: 0/200 served without a fallback model, 200/200 with one (all flagged degraded) | `--only llm` | scripted provider, exact |
| Guardrails, both directions | default: 0/40 false approvals, 0/44 false rejections (in-sample), 0/146 real replies blocked, 0/741 mutations missed. Looser/stricter presets are worse in the directions shown | `run_guardrail_eval.py`, `docs/GUARDRAIL_EVALUATION.md` | offline; labelled numbers in-sample |
| Test suite | 493 tests; all pass (2 are skipped without the real `langgraph`) | `pytest tests/ -v` | automated |

## 4. Guardrails and safety

* **Refund cap** — nothing at or above INR 2,000 is committed by the system role, checked at the action layer *and* again on the outcome (summed per order, so splitting a refund does not evade it).
* **Fabrication block** — amounts, durations and percentages in the model's sentence must exist in the retrieved policy text or the structured facts.
* **Approval claims, flag disclosure, injection echo** — see `docs/GUARDRAIL_EVALUATION.md` §1.
* **Allow-listed responses** — the API returns `ticket_id, session_id, outcome, reply, awaiting_approval, guardrail, degraded, input_flags`. No escalation codes, reasons, facts or flag state: an `ESC-ABUSE` code in a response would tell the customer their account is flagged.
* **Prompt injection** — a hostile ticket is routed to a person by the deterministic safety floor, not by asking the model to ignore it.
* **Privacy** — traces are redacted before storage or export (e-mail, UPI, card/Aadhaar/PAN/phone, API keys); customer ids are salted and hashed. A regex floor, not anonymisation.
* **Secrets** — only environment variables; `.env.example` lists them with no values.

## 5. Known limitations

Demo shortcuts that production must not keep:

1. **`customer_id` comes from the request body.** Anyone can claim to be anyone and read another customer's order status. Production must derive it from an authenticated session and ignore the body field.
2. **One ticket at a time** (a lock): the demo shares one in-memory ledger and one checkpoint file. Concurrency needs a shared ledger store and per-request dependencies.
3. **No approver endpoint.** A paused ticket is resumed with `resume_ticket_approval.py`. A real approval UI needs authentication and a role check on the person.
4. **No rate limiting, no auth, no CORS policy** on the API.
5. **Model-dependent behaviour seen in live API runs** (`groq/openai/gpt-oss-20b`): the policy answer carried raw citation markers (`【24】`, now stripped by `clean_model_text`) and a hedge about shipping times the customer never asked about (such a hedge, after a stated figure, does not make the detector discard the answer). The guardrail correctly let the grounded 15- and 30-day figures through; wording quality is measured by the golden set, not by the guardrails.
6. **Offline mode** (`SHOPSENSE_OFFLINE=1`) uses a keyword extractor and returns the top retrieved chunk as the policy answer. It proves plumbing and safety behaviour, not model quality.

Quality limits:

7. Guardrails are a **pattern layer**; five known gaps (no figure, paraphrased disclosure, right number but wrong category) are listed with the evaluation. Labelled-case numbers are in-sample, and offline "policy sentences" are retrieved chunks, not model prose — expect more false rejections with a real model.
8. The safety-net and redaction regexes are tuned to the shapes in the sample data.
9. Delivery date and lost-shipment age are estimates from `ship_date`.
10. Mock data: 150 customers, 600 orders, 126 tickets. No real order API was integrated.
11. **Exercised end to end:** the LangFuse export and the nested trace tree through the real LangGraph; real FastAPI/uvicorn serving in offline and live mode; the full test suite. **Not exercised:** the ngrok/Postman path against a public URL, the injection and guest cases against the live model (the deterministic floor decides them), and a full 126-ticket live pipeline run.

## 6. Deployment notes

See `docs/DEPLOYMENT_CHECKLIST.md` for the gate-by-gate list. In short: run `uvicorn shopsense.api:app` behind HTTPS with an authenticating proxy, set model keys and `LANGFUSE_*` as secrets, keep `SHOPSENSE_TRACE_SAMPLE_RATE` at 1.0 for failures (tail sampling already keeps them), alert on breaker state, `degraded` rate and `guardrail_blocked`, and keep the human queue staffed — it is the fallback for every failure path above.
