# ShopSense — Customer Care & Order Operations Assistant

ShopSense is a multi-agent AI assistant for an e-commerce support desk. A customer message comes in as plain text; the system works out
what is being asked, checks the store's policy documents and the customer's real order, shipment and refund records, and then does one
of three things:

1. **Resolves it** — answers from the policy documents or the order records, and completes small actions (refunds under INR 2,000) itself.
2. **Pauses for approval** — a larger refund, or an angry customer, stops the ticket until a manager or executive decides. The decision
   can be made later, from a different process.
3. **Hands it to a person** — legal threats, safety issues, prompt-injection attempts, abuse-flagged accounts, repeat contacts, lost parcels
   and anything the system cannot answer safely go to a human queue with a short hand-off packet.

The guiding rule: **the model handles language, code handles authority.** Language models read the message, phrase the policy answer and
draft the escalation memo. Who may refund how much, return windows, routing and approvals are plain, tested code, so a customer cannot
talk the assistant into a refund.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # Mac/Linux: cp .env.example .env
```

Open `.env` and set **one** of the following (only the first section of the file is needed):

| Way to run | What to put in `.env` |
|---|---|
| Live model, Groq (free tier) | `GROQ_API_KEY=...` (default model `groq/openai/gpt-oss-20b`) |
| Live model, Gemini (free tier) | `GEMINI_API_KEY=...` and `SHOPSENSE_LLM_MODEL=gemini/gemini-3.1-flash-lite` |
| No key (offline mode) | `SHOPSENSE_OFFLINE=1` |

In offline mode a keyword extractor and a retrieval-only policy answer stand in for the model. Routing, refund limits, approvals and
guardrails are the real code, so the behaviour is the same; only the wording of answers is simpler. A variable set in your terminal
takes priority over `.env`. Every setting is listed, with its default, in `.env.example`.

Check the installation:

```bash
pytest tests/ -q                       # unit and integration tests, no key needed
python dev_smoke_check.py              # the same logic checks with no test framework
```

## Running it: which script does what

All entry points feed customer messages into the **same multi-agent system**. They differ only in how the message arrives.

| Script | Use it to | Input | Needs a key? |
|---|---|---|---|
| `ask.py` | Try any message yourself, one at a time or as a chat | typed text | optional (`--offline`) |
| `python -m uvicorn shopsense.api:app --port 8000` | Run the assistant as an HTTP API (`POST /chat`, `GET /health`, docs at `/docs`) | JSON: `message`, `customer_id`, `session_id` | optional |
| `demo_api_client.py` | Send six fixed demo messages to a running API and print each response | built-in messages | optional |
| `run_pipeline.py` | Process a batch of tickets and print a summary and dashboard; paused tickets print their resume command | a JSON-lines ticket file (126 bundled tickets by default) | yes |
| `resume_ticket_approval.py` | Act as the approver for a ticket that is paused for approval | ticket id, decision, role | no |
| `run_ticket_team_demo.py` | Walk through the agent team on golden tickets: auto-resolve, pause, restart, resume, role refusal, escalation | bundled tickets | yes |
| `run_ticket_team_mcp_demo.py` | The same team with its order tools served by a real MCP server | bundled tickets | yes |
| `run_golden_eval.py` | Score the 20-case golden set (offline, or `--live` end to end) | `data/shopsense/eval/golden_set.json` | `--live` only |
| `run_fault_injection.py` | Break the order system and the model provider on purpose (seeded) and print before/after results | none | no |
| `run_guardrail_eval.py` | Compare guardrail presets in both directions (false approvals vs false rejections) | labelled replies | no |
| `run_trace_report.py` | Show a latency / token / degraded dashboard from the saved traces | `data/shopsense/traces/traces.jsonl` | no |
| `check_langfuse.py`, `check_mcp_server.py` | Quick connectivity checks for LangFuse and for the MCP server | none | LangFuse keys |

### Try your own messages

```bash
python ask.py "How many days after delivery can electronics be returned?"
python ask.py "Where is my order ORD-4445?" --customer CUST-0077
python ask.py "I want my INR 25,516 back for ORD-4553 right now" --customer CUST-0142
python ask.py                          # chat mode (empty line to quit)
python ask.py --examples               # six messages worth trying, with what to expect
python ask.py --debug "..."            # also show the policy query, sources and the raw model answer
python ask.py --offline "..."          # no key needed
```

The response shows the reply, the `outcome` (`resolved`, `escalated` or `awaiting_approval`), whether a person must approve, the
guardrail verdict, and whether the answer was served `degraded` (a fallback was used). Customer ids and orders come from the bundled
mock data (150 customers, 600 orders).

### Run your own tickets in a batch

Create a file with one JSON object per line (`ticket_id`, `channel`, `created_at`, `customer_id`, `text`) and run:

```bash
python run_pipeline.py --tickets my_tickets.jsonl --limit 5
```

Without `--tickets` the 126 bundled tickets are used. On a free model tier, use `--limit`, `--start N`, `--pause SECONDS` or
`--min-interval SECONDS` to stay within rate limits.

### Approving a paused ticket

When a ticket needs approval, the run prints its id and the command to continue. In a separate terminal:

```bash
python run_pipeline.py --start 105 --limit 1                           # pauses TCK-0105: an INR 25,516 refund that needs a Finance Executive
python resume_ticket_approval.py TCK-0105 approve --role executive     # a manager is refused here: INR 10,000 and above needs an executive
python resume_ticket_approval.py TCK-0105 deny --role executive        # (or deny instead)
```

An approval from a role without enough authority is refused and the ticket stays paused for the right person, an already-decided ticket cannot be decided again, and a refund is never
paid twice. Approvals work for tickets paused by `run_pipeline.py` and by the live chat (they share one checkpoint database). In offline mode a pause is kept in memory only and cannot be resumed from another process.

### Chat API

```bash
python -m uvicorn shopsense.api:app --port 8000
python demo_api_client.py              # in a second terminal
```

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType "application/json" `
  -Body '{"message":"What is your return policy for apparel?","customer_id":"CUST-0077","session_id":"s1"}'
```

An importable Postman collection with 65 assertions is in `docs/postman/` (regenerate it with `make_postman_collection.py`).
`docs/DEPLOYMENT_CHECKLIST.md` describes running it through ngrok.

## How it works

```
ticket ─► triage ─► policy_rag ─► order_actions (read, PROPOSE) ─► escalation_reviewer: route
            │                                                         │
            │ hard fact (legal / safety / injection /                 ├─ auto ─────────────► commit ─► reply
            │ abuse flag / 3rd contact / lost parcel /                ├─ approval: refund ≥ INR 2,000, or angry customer
            │ someone else's order / extraction failed)               │      └─► PAUSE (interrupt) ─► a person decides ─► commit ─► reply
            └──────────────────────────────────────────────────────── └─ human_review: hand-off packet + reviewed memo
```

| Agent | Responsibility |
|---|---|
| **Triage** | Extracts structure from the raw text (intent, order, amount, sentiment), with a repair loop for malformed model output and a deterministic safety floor for legal, safety and injection wording |
| **Policy RAG** | Finds the relevant policy sections (BM25 + embeddings fused with RRF, then a reranker) and writes a cited answer that treats retrieved text as untrusted |
| **Order-Actions** | Reads the customer's order, shipment and refund state and *proposes* an action; it cannot move money itself |
| **Escalation Reviewer** | Decides the route (auto, approval, human review) and writes the hand-off packet; for human review a five-agent memo team (planner, researcher, writer, fact-checker, reviewer) drafts a cited memo |

A **supervisor** (`shopsense/teams/support_team.py`) picks the next agent at each step. It is rule-based on purpose: in this workflow only
one next step is ever legal, and a model would only add a way to be talked out of it. Each agent can write only its own fields.

Design decisions:

* **Authority is code.** INR under 2,000 may be refunded by the system, under 10,000 by a manager, 10,000 and above by an executive. Splitting
  a refund does not avoid the limit. Anger never changes anyone's authority; it only adds a human.
* **Model labels are hints.** A question ("how many days do I have to return this?") never opens a refund, even if the model says RETURN_ITEM.
  Legal, safety and injection wording escalates even if the model misses it or fails.
* **Side effects live in one place.** Refunds, replacements and cancellations happen only in the `commit` step after the approval
  interrupt, through tools that enforce the role limit themselves and are idempotent per action, order and ticket (an on-disk ledger).
* **Over MCP a caller cannot raise its own role.** The order / refund / shipping server runs writes as `system`; a manager or executive action
  needs an HMAC approval token minted for exactly that order, ticket and amount.
* **Hostile text never reaches a model or a reply.** Injection attempts are detected and redacted before extraction; every reply passes
  output guardrails (refund cap, invented amounts or durations, unearned "approved" claims, account-flag disclosure, injection echo).
* **Failures are loud.** Fallback models, an unavailable order system or a blocked reply set a `degraded` flag; tickets that cannot be served
  safely go to the human queue instead of failing.
* **Memory per customer.** Episodic, semantic and procedural memory are stored per customer, summarised with a recall self-check and
  consolidated between sessions.
* **Vector index.** Retrieval uses a local `VectorIndex` with a swappable interface (`shopsense/rag/index.py`) rather than a hosted
  Qdrant service, so the project runs with no extra account; BM25, hybrid fusion, reranking and cited answers are fully implemented.

## What was built (milestone map)

| Milestone | Delivered | Where |
|---|---|---|
| M1 | Provider-agnostic LLM client, structured ticket intake with validation and repair | `shopsense/core/` |
| M2 | Order tools with retry and circuit breaker; tool-calling and ReAct agents | `shopsense/tools/`, `shopsense/agents/` |
| M3 | Persistent per-customer memory and semantic index | `shopsense/memory/` |
| M4 | Production RAG over 14 policy documents plus an evaluation harness | `shopsense/rag/` |
| M5 | LangGraph ticket-resolution workflow: triage → policy → action → routing, approval interrupt, checkpointing | `shopsense/workflows/ticket_resolution_graph.py` |
| M6 | Four-agent team with supervisor plus an MCP server for order / refund / shipping | `shopsense/teams/`, `shopsense/mcp/` |
| M7 | LangFuse tracing with redaction, retries, circuit breaker and human-queue fallback, seeded fault injection | `shopsense/observability/`, `shopsense/tools/reliability.py`, `shopsense/eval/fault_injection.py` |
| M8 | 20-case golden set, output guardrails, FastAPI chat, architecture review, deployment checklist | `shopsense/guardrails.py`, `shopsense/service.py`, `shopsense/api.py`, `docs/` |

`MILESTONES.md` has the detailed requirement-to-module map, the judgement calls and the known limits.

### Measured results

| Check | Result | Reproduce with |
|---|---|---|
| Golden set, 20 cases (route, escalation codes, key facts), live model | 20/20 | `python run_golden_eval.py --live` |
| Order-system flakiness (seeded): crashed tickets | 48 without hardening → 0 with it, 0 duplicate payments | `python run_fault_injection.py` |
| Order-system outage: calls reaching the dead system | 90 with retries only → 15 with the circuit breaker | `python run_fault_injection.py` |
| Model provider faults (200 calls): failed calls | 38 bare → 0 hardened | `python run_fault_injection.py` |
| Guardrails (default preset): false approvals / false rejections / real replies blocked / mutated replies missed | 0/40, 0/44, 0/146, 0/741 | `python run_guardrail_eval.py` |
| Tests | all passing (493 with the real `langgraph` installed) | `pytest tests/ -q` |

Guardrail numbers on the labelled cases are measured on the cases the presets were tuned with. The method and the five known gaps are in
`docs/GUARDRAIL_EVALUATION.md`.

## Project layout

```
ask.py · run_pipeline.py · resume_ticket_approval.py · demo_api_client.py     main entry points
run_ticket_team_demo.py · run_ticket_team_mcp_demo.py                         agent-team demos
run_golden_eval.py · run_fault_injection.py · run_guardrail_eval.py          evaluations
run_trace_report.py · check_langfuse.py · check_mcp_server.py                 observability and connectivity checks
run_refund_approval_demo.py · resume_refund_review.py                         single-workflow refund approval demo
run_escalation_team_demo.py · run_escalation_team_mcp_demo.py                 memo-team demos
dev_smoke_check.py                                                            dependency-free logic checks
generate_*.py · make_postman_collection.py                                    seeded data generators and the Postman collection

shopsense/
  core/          llm_client (retries, per-model breaker, fallback models) · schemas · ticket_intake · safety_net · contacts
  tools/         order_tools · reliability · memory_tools · policy_tools · policy_rules · action_tools
  memory/        embeddings · memory_store · summarizer · consolidation
  rag/           ingest · index · bm25 · hybrid · rerank · answer · evaluate
  agents/        tool_calling_agent · react_agent · memory_agent · shopsense_agent
  workflows/     ticket_resolution_graph (main workflow) · refund_approval_graph
  teams/         support_team (four agents + supervisor) · escalation_review_team (memo team) · mcp_ops · mcp_researcher
  mcp/           shopsense_server + ops_tools (order / refund / shipping) · policy_server
  observability/ tracing · redact · report
  eval/          golden · live · offline · fault_injection · guardrail_eval
  guardrails.py · service.py · api.py · runtime.py · envfile.py
data/shopsense/
  policies/ (14 markdown) · policies_pdf/ (14 lossy PDF versions) · intake/tickets.jsonl (126 tickets)
  mock_api/ (customers, orders, items, shipments, refunds as CSV) · eval/ (golden set, retrieval questions, guardrail cases)
  manifest.json (provenance, counts and hashes of the data)
docs/            architecture review · deployment checklist · guardrail evaluation · observability and reliability · Postman collection
tests/           pytest suite, pure-Python executors and a fake LangGraph for offline wiring checks
```

The data is synthetic and generated by seeded, append-only scripts (`generate_tables.py`, `generate_tickets.py`,
`generate_corpus_pdfs.py`, `generate_manifest.py`); it contains no real personal data. Runtime files (memory, embedding cache,
checkpoints, ledger, traces) are created on demand.

## Configuration notes

* `SHOPSENSE_LLM_MODEL` selects the chat model; any LiteLLM model string works. `SHOPSENSE_LLM_FALLBACK_MODELS` lists backups
  (every use is logged and flags the ticket as degraded).
* Groq has no embeddings endpoint. For real embeddings set another provider, e.g. `GEMINI_API_KEY` and
  `SHOPSENSE_EMBED_MODEL=gemini/gemini-embedding-001`; without it a deterministic hashed bag-of-words embedder is used, which makes
  offline retrieval numbers a floor rather than the real figure.
* Free tiers rate-limit. The client waits and retries, gives up on very long waits, and the graph turns a failing model call into a
  hand-off to a person instead of crashing. In the chat API each request waits at most `SHOPSENSE_API_DEADLINE_S` (default 40) seconds.
* LangFuse tracing is optional: set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST`, then run `python check_langfuse.py`.
  Without keys, traces are written to `data/shopsense/traces/traces.jsonl` (customer data is redacted first).
* `SHOPSENSE_APPROVAL_SECRET` signs MCP approval tokens. A development default is built in; set your own anywhere it matters.
* On Windows, if piped output (for example `| Tee-Object`) shows odd characters, run `$env:PYTHONUTF8=1` first.
* If `pytest` fails during collection with an import error from a folder outside this project, run it from the project root; the bundled
  `pytest.ini` pins the root.
* If the MCP demo fails, `python check_mcp_server.py` starts the server alone and prints its own error and package versions.

## Limits

* The demo API trusts the `customer_id` in the request body, handles one ticket at a time and has no authentication or approver endpoint.
  `docs/DEPLOYMENT_CHECKLIST.md` lists what a production deployment must change.
* Guardrails are a pattern layer with documented gaps; the safety-net and redaction patterns are a floor, not a classifier.
* Delivery dates and lost-shipment ages are estimates from the ship date; the order system is a mock.
* The contact ledger (third contact in seven days escalates) lives in the running process.
* Offline mode proves routing, limits, approvals and guardrails, not the quality of model-written wording.
