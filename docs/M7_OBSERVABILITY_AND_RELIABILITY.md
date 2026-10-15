# Milestone 7 — LangFuse tracing and reliability hardening

Capstone wording: *"LangFuse tracing; reliability hardening for order-system API flakiness (retries, fallback to human queue, circuit breaker)."*
Method: instrument first, then break the system on purpose and measure.

## 1. What is traced

One **trace per ticket** (`ticket.run`; session id = ticket id, user id = a salted one-way hash of the customer id, tags = model, prompt
version, topology). Under it, nested spans exactly as the code runs:

```
ticket.run
  node:triage / agent:triage          (+ triage.extraction_error event when the model call failed)
  node:policy_check
      rag.retrieve · rag.rerank · rag.answer · llm.generate (a *generation*: model, tokens in/out, cost when the provider's price is known)
  node:action_plan
      order_system.read               (tool span; guard.retry / guard.gave_up / fallback.human_queue events)
  node:route
  node:approval        (a LangGraph interrupt() is a PAUSE, not an error)
  node:commit
      order_system.write
  node:reply / node:human_review
```

* **Redaction happens before anything is stored or exported** (`observability/redact.py`): e-mail, UPI ids, card/Aadhaar/PAN/phone numbers,
  dates of birth, API keys; sensitive dict keys; long text truncated. Order, ticket, refund and tracking ids are deliberately kept (you need
  them to debug). It is a regex floor, **not** anonymisation — a street address written in prose is not caught.
* **Sampling is decided at the end of a run (tail sampling).** Failed, degraded, escalated, paused and slow runs are always kept locally;
  only clean auto-resolved runs are sampled (`SHOPSENSE_TRACE_SAMPLE_RATE`, default 1.0). LangFuse itself receives every run — the SDK's
  own head sampling cannot know a run is about to fail, so leave `LANGFUSE_SAMPLE_RATE` at 1.0.
* **A tracing failure never fails a ticket.** If LangFuse is unreachable the local trace is still written (`tracer.langfuse_errors` counts it).
* Local file: `data/shopsense/traces/traces.jsonl`. Read it with `python run_trace_report.py` (`--degraded`, `--ticket TCK-0042`, `--last 50`).

`python check_langfuse.py` is the first thing to run: it checks keys → SDK → `auth_check()` → sends one nested demo trace.

## 2. What is hardened, and how

| Dependency | Failure | What happens |
|---|---|---|
| Order system (every read in triage, policy query, action planning; every write in commit) | connection error, timeout | retried with exponential backoff + jitter (3 attempts, `SHOPSENSE_ORDER_*`) |
| | record with a missing field / wrong type | **not retried** (same bad odds); counts against the dependency; ticket goes to a person |
| | still failing / breaker open | `DependencyUnavailable` → **fallback to the human queue**: nothing is promised, refunded or declined; the customer is told the truth ("our order system isn't responding … a team member will confirm"); the hand-off packet carries the proposal and, after approval, the decision |
| | repeated failures | **circuit breaker** (closed → open → half-open with exactly one trial → closed): a dead system is not hammered and a ticket costs one instant refusal instead of three timeouts |
| Order writes | success whose acknowledgement is lost (caller sees a timeout) | the retry is safe because every action is idempotent per (action, order, ticket): it returns the earlier record instead of paying again |
| Model provider | 429 / quota | waits the time the provider names (Groq `try again in 2m33s`, Gemini `retry in 46s` / `retryDelay`, `Retry-After`); a >10-minute ask (daily cap) stops at once |
| | 5xx, timeout, connection | exponential backoff with jitter; default 60 s timeout per call |
| | 401 / 403 / 404 / 400 | **not retried** — it is configuration, not an outage (a retry loop would hide a wrong key for minutes) |
| | a model that keeps failing | per-model circuit breaker |
| | primary unusable | optional `SHOPSENSE_LLM_FALLBACK_MODELS`; every use is logged, an event on the trace, and the ticket is flagged **degraded** |
| Policy reranker | LLM rerank fails | falls back to the fused retrieval order (event `rag.rerank_failed`) |

**Silent degradation is treated as a bug.** A ticket is *degraded* if it used a fallback model, a dependency gave up or short-circuited, a record
was malformed, or it fell back to the human queue. `run_pipeline.py` lists degraded tickets by id, and the dashboard reports the rate.

**Design rule kept:** the guard never decides *what* the fallback is — it raises a typed error and the graph decides (here: a person).
A model failure never reaches the decision logic as "approve".

## 3. The measurements (seeded fault injection — `python run_fault_injection.py`)

Same 126 tickets, same seed (42), faults on every point lookup of the order tables: 5% connection errors, 3% timeouts, 2% malformed records,
and 25% of successful writes reporting a timeout. Any ticket that stops for approval is approved so the write path is exercised too.
Offline doubles stand in for the model (see limits). Numbers are reproducible and pinned by `tests/test_fault_injection.py`.

| | healthy | flaky, **no hardening** | flaky, **hardened** |
|---|---:|---:|---:|
| tickets crashed | 0 | **48** (23 timeouts, 21 connection errors, 4 KeyErrors) | **0** |
| served normally | 126 | 78 | 117 |
| handed to a person (fallback) | 0 | 0 | 9 |
| served normally but routed differently from healthy | – | 0 | **0** |
| refunds/actions committed | 11 | 2 | 11 |
| tickets that crashed *after* money moved | 0 | 1 | **0** |
| unsafe actions (taken when healthy run did not) | – | 0 | **0** |
| duplicate payments | 0 | 0 | **0** |

The hardened run also holds at 0 crashes / 0 unsafe / 0 duplicates for seeds 1, 7, 99 at 15% connection errors (tested).

**Sustained outage** (order system fully down for 30 consecutive tickets, then back):

| | retries only | retries + breaker |
|---|---:|---:|
| calls that hit the dead system | 90 | **15** |
| calls refused instantly by the breaker | 0 | 27 |
| outage tickets handed to a person / crashed | 30 / 0 | 30 / 0 |
| actions committed during the outage | 0 | 0 |
| after recovery: tickets still handed over (breaker cool-down) | 0 | **2** |

The last row is the honest cost of a breaker: normal service resumes one cool-down after the outage ends, not instantly.

**Model provider** (200 calls): a bare call fails 38 times (15% 503 + 5% rate limit); the hardened client fails 0 (246 provider requests, 46 retries).
Total outage: 15 provider requests instead of ~1,000. Primary model removed (404): 0/200 served without a fallback model, 200/200 with one — each flagged degraded.

## 4. Limits (read before relying on this)

* **LangFuse is verified against a stand-in that mirrors the SDK calls the project makes, not against the real SDK or cloud.** Run `check_langfuse.py`.
  The real LangGraph wiring (context propagation into node threads) is likewise unverified offline.
* Fault experiments use the in-memory order tables and offline model doubles; a real order API has failure modes this does not model
  (partial writes, slow-but-successful responses, schema changes). Rates are inputs, not measurements of any real system.
* Malformed-record handling treats `KeyError/TypeError/ValueError/AttributeError` raised while *reading* orders as a bad record. A genuine bug in the planner that raises one
  of those would also be sent to a person (loudly: the cause is on the trace) rather than crash the ticket.
* If a write fails *after* a human approved it, the ticket goes to the human queue with the decision recorded; there is no automatic replay queue yet.
  Actions are idempotent, so applying it later cannot pay twice.
* A fallback model can label sentiment or intent differently from the primary; that is why its use flags the ticket instead of being invisible.
* The redaction layer is a pattern floor; the trace file is still a store of customer text and should be treated (permissions, retention) like one.
