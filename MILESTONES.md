# ShopSense — milestone-to-module map

ShopSense is one system built across eight milestones. The code is organised by what each module *does*, not by the milestone that
introduced it; this file is the map between the two, plus the judgement calls, data assets, verification method and known limits.

## Capstone milestones

| Milestone | Requirement | Where it lives |
|---|---|---|
| **M5** | LangGraph workflow: triage → policy-check → action execution → conditional routing (auto-resolve low-risk, human-approval interrupt for high refunds or angry-customer sentiment), with checkpointing | `shopsense/workflows/ticket_resolution_graph.py` (+ `tools/policy_rules.py`, `tools/action_tools.py`, `core/contacts.py`, `runtime.py`), `run_ticket_team_demo.py --workflow`, `run_pipeline.py --workflow`, `resume_ticket_approval.py --workflow` |
| **M6** | Multi-agent team (Triage / Policy RAG / Order-Actions / Escalation Reviewer) plus an MCP server exposing order / refund / shipping APIs | `shopsense/teams/support_team.py` (team, supervisor, write scopes), `shopsense/mcp/shopsense_server.py` + `ops_tools.py` (server), `shopsense/teams/mcp_ops.py` (client adapters), `shopsense/teams/escalation_review_team.py` (memo team), `run_ticket_team_demo.py`, `run_ticket_team_mcp_demo.py` |
| **M7** | LangFuse tracing; reliability hardening for order-system flakiness (retries, fallback to the human queue, circuit breaker) | `shopsense/observability/*` (tracing, redaction, dashboard), `shopsense/tools/reliability.py` (`DependencyGuard`, `CircuitBreaker`), guarded order-system calls and the human-queue fallback in `workflows/ticket_resolution_graph.py`, `core/llm_client.py`, `eval/fault_injection.py`, `run_fault_injection.py`, `check_langfuse.py`, `run_trace_report.py`; write-up in `docs/M7_OBSERVABILITY_AND_RELIABILITY.md` |
| **M8** | 20-case golden set, refund-cap and fabrication guardrails, FastAPI demo chat, architecture review, deployment checklist | `data/shopsense/eval/golden_set.json`, `shopsense/eval/golden.py`, `run_golden_eval.py`; `shopsense/guardrails.py`, `shopsense/eval/guardrail_eval.py`, `run_guardrail_eval.py`; `shopsense/service.py`, `shopsense/api.py`, `demo_api_client.py`; `docs/GUARDRAIL_EVALUATION.md`, `docs/ARCHITECTURE_REVIEW.md`, `docs/DEPLOYMENT_CHECKLIST.md` |

## Foundation milestones

| Milestone | What it covers | Where it lives |
|---|---|---|
| **M1** | Provider-agnostic LLM client with session state; Pydantic schema; LLM extraction; validation; LLM-driven repair loop | `core/llm_client.py`, `core/schemas.py`, `core/ticket_intake.py` |
| **M2** | Tool-enabled agent: four order tools with retry, circuit breaker and logging; the model chooses the tool; ReAct and reflection patterns | `tools/reliability.py`, `tools/order_tools.py`, `agents/tool_calling_agent.py`, `agents/react_agent.py` |
| **M3** | Persistent memory and a semantic index: embeddings; episodic, semantic and procedural memory per customer; rolling summarisation with a recall self-check; consolidation; agent-managed memory | `memory/*`, `tools/memory_tools.py`, `agents/memory_agent.py` |
| **M4** | Production RAG: structure-aware chunking, vector index, hybrid dense + BM25 (RRF), reranker, grounded cited injection-resistant answers, P@k / R@k / MRR evaluation harness | `rag/*`, `tools/policy_tools.py`, `agents/shopsense_agent.py`; the loader also handles the lossy PDF corpus (`rag/ingest.py: load_policy_corpus_pdf`) |
| **Refund-approval workflow** | StateGraph with control/audit split, deterministic critic, loop cap, `interrupt()`, SQLite checkpoint that survives a restart | `workflows/refund_approval_graph.py`, `run_refund_approval_demo.py`, `resume_refund_review.py` |
| **Escalation memo team** | Five-agent supervisor team, write scopes and read scoping, deterministic supervisor policy, revision cap, policy server consumed over MCP | `teams/escalation_review_team.py`, `teams/mcp_researcher.py`, `mcp/policy_server.py`, `run_escalation_team_demo.py`, `run_escalation_team_mcp_demo.py` |

## Judgement calls

* **Angry-customer sentiment → the approval interrupt.** Anger routes a ticket to the approval step. Under `refund_authorisation_matrix`
  RA-2.2 anger raises priority but never lowers the authority required, so in the graph it only *adds a human*; it never changes the
  amount, the tier or the role needed. The interrupt payload carries a `proposed_reply` so the approver sees what would be sent.
* **Delivery date = ship date + 3 days.** The tables have no `delivered_date`; return windows run from this estimate. A real deployment
  should store the carrier's delivery scan.
* **Lost shipment = 10 days since ship date without delivery** (SOP ESC-LOST). Pure questions ("do you ship internationally?") are not
  escalated for it.
* **"Same issue" for ESC-REPEAT** = same customer and same order if both name one (else same intent), within 7 days. "Unresolved" cannot
  be known from tickets alone, so every earlier contact counts, which biases toward escalating.
* **The supervisor is deterministic code, not an LLM.** In this workflow there is exactly one legal next step in any state; a model would
  only add a way to be talked out of it. Models are used where language is the job (extraction, policy answer, memo).
* **Authority is not a tool argument.** The MCP server's writes run as `system`; manager and executive actions need a signed approval token.
* **`ShopSenseAgent`** (the M2–M4 tool-calling loop) is kept as a library and test target but is not on the pipeline's path: the graph's
  deterministic planner replaced "the model decides which tool to call" for money-moving steps. Memory (M3) is written by the graph
  (`Deps.store`).
* **Complaints and small talk.** A message that is a complaint with no answer or action to give goes to a person (`no_automated_answer`)
  instead of closing with a promise nothing keeps; greetings, thanks and goodbyes get a plain reply instead of a policy snippet.
* **The "not covered" reply.** When the policy documents do not answer a question, the reply says so and points to the support team; it
  does not promise a follow-up that nothing would queue.

## The agents and how they relate

* `agents/tool_calling_agent.py` and `agents/react_agent.py` are the M2 artifacts: the two tool-use patterns in isolation, order tools only.
* `agents/memory_agent.py` is the M3 artifact: the tool-calling loop plus recall-before-respond, agent-managed remember/recall and
  working-memory compaction.
* `agents/shopsense_agent.py` is the integrated agent: order tools (M2) + memory (M3) + `policy_lookup` (M4) in one loop.
* The refund-approval workflow picks up where `ShopSenseAgent` stops ("if a ticket needs human review … say so plainly") and gives that
  hand-off a durable, resumable shape. The escalation memo team gives the person who receives an escalated ticket an investigated, cited memo.
* The production path is the support team in `teams/support_team.py` on top of the ticket-resolution graph.

## How each milestone is verified

The test that matters: **remove the model call and see whether anything still works.** If the output is unchanged without a model in the
loop, the code is not demonstrating that milestone; it is ordinary automation.

* M1: extraction in `core/ticket_intake.py` has no non-model fallback path.
* M2: tool *selection* in `agents/tool_calling_agent.py` is the model's decision, not a rule table.
* M3: `agents/memory_agent.py` never decides what to remember on its own (that is the model's `remember` call); compaction in
  `memory/summarizer.py` is a real summarisation call, not truncation.
* M4: `rag/rerank.py` and `rag/answer.py` are real model calls; only the metrics in `rag/evaluate.py` are model-free by design.
* M5: delete the `interrupt()` call in the approval node and the workflow becomes an ordinary chain with no human; the checkpoint is
  what makes the pause durable. `run_refund_approval_demo.py` discards the graph and the SQLite connection, opens new ones on the same file
  and resumes from there.
* M6: `supervisor_policy()` and `check_citations()` are plain functions; the model's job is to agree with them, never to decide alone.
  Removing the model from the supervisor or the fact-checker breaks nothing, because they never trusted its verdict.
* M7: `python run_fault_injection.py` reproduces the before/after numbers exactly (seeded).
* M8: `python run_golden_eval.py`, `python run_guardrail_eval.py`, the Postman collection and `python demo_api_client.py`.

Layers of testing: (1) pure decision logic (routing, tiers, windows, role limits, reply checks, supervisor policy, token verification)
is unit-tested directly; (2) whole-flow executors (`tests/ticket_sim.py`, `support_team_sim.py`) run the real node functions against the
real tables, corpus and ledger, with a keyword stand-in for the extractor; all 20 golden cases get the expected route and escalation codes,
and the team produces exactly the workflow's outcome on every one; (3) compiled-graph wiring is exercised against `tests/fake_langgraph`
(a small stand-in for the LangGraph semantics the project uses; it is not LangGraph); (4) the real LangGraph, MCP server and model are
exercised by the demo scripts and `run_golden_eval.py --live`.

Totals: `dev_smoke_check.py` runs about 205 checks (209 with `pypdf`); the pytest suite has 493 tests (2 skip without the real `langgraph`).

## Data assets

All four are built to the data-sourcing guidelines, synthetic by design (no real personal data), seeded and reproducible, and
**append-only**: extending a table or ticket file never changes an existing row.

| Asset | Contents | Where |
|---|---|---|
| **A — Policy corpus** | 14 documents: return, refund, shipping, escalation SOP, FAQ, five category addenda (electronics, apparel, groceries, home, beauty), refund authorisation matrix, warranty, fraud prevention, cancellation, with section codes (RF-3.1, CA-BT-1.2 …) and cross-references. Also 14 lossy PDFs (headings dropped, tables flattened, running headers) with a heading-recovery ingestion path. Deliberate gaps (international shipping, price matching, gift wrap, loyalty, B2B) so "unanswerable" cases are real | `data/shopsense/policies/`, `policies_pdf/` |
| **B — Tickets** | 126 tickets: Hinglish and Indian-English, profanity, legal, safety, eight injection shapes, repeat-contact chains, flag-disclosure probes, above-cap and angry above-cap, policy-gap questions, category return windows, terse and long | `data/shopsense/intake/tickets.jsonl` |
| **C — Mock tables** | 150 customers (16 abuse-flagged under FP-1: three or more returns of INR 5,000+ in 90 days), 600 orders, 1,210 items, 549 shipments, 97 refunds (38 pending, 20 of them INR 10,000+) | `data/shopsense/mock_api/` |
| **D — Golden set** | 20 cases (6 factual, 3 multi-hop, 2 unanswerable, 5 guardrail, 4 injection) with expected route, escalation codes, required and forbidden text and the policy documents to cite; plus 25 retrieval questions | `data/shopsense/eval/` |

`manifest.json` records the seed, how the data was generated (no language model wrote any asset), the inventory with counts and hashes, the
public sources referenced (Bitext taxonomy names used; Olist, the Twitter support corpus and Amazon Reviews listed as not used, with URL
and licence) and `contains_real_personal_data: false`. Sentiment labels on the synthetic tickets are not calibrated against real support
tweets. Memory (`data/shopsense/memory/`), the embedding cache (`rag_cache/`), checkpoints and the ledger (`workflow_state/`) are written
at runtime and empty in a fresh checkout.

## Deliberate adaptations

1. **Memory store.** A hand-built store following the CoALA taxonomy (`memory/memory_store.py`) backed by plain per-customer JSONL files,
   so no hosted memory service or extra API key is needed. The interface can be swapped for a hosted store.
2. **Reranker.** An LLM-as-reranker (`rag/rerank.py`) instead of a local cross-encoder, which would need `sentence-transformers` and
   `torch`. The trade-off is a model call instead of local CPU inference.
3. **Vector storage and keyword search.** `rag/index.py` and `rag/bm25.py` are implemented in-repo rather than using a hosted Qdrant service
   or `rank_bm25`, so the suite and the smoke check run with no extra account or dependency. The interfaces (`VectorIndex.dense_search`,
   `BM25Index.search`) are what the rest of the code uses, so swapping either is a one-file change.
4. **Real frameworks for M5 and M6.** The graph uses the real `langgraph`, and the MCP half uses the official `mcp` SDK
   (`mcp.server.fastmcp`) with `langchain-mcp-adapters`, because those milestones are about those frameworks' own checkpointer, interrupt
   and protocol APIs. The standalone `fastmcp` package is deliberately not used (its 4.x line needs `mcp` 2.x, which breaks
   `langchain-mcp-adapters`); `shopsense/mcp/compat.py` picks whichever is installed.
5. **Structured outputs.** Graph and team structured outputs go through `core/llm_client.ask_structured()`, which reuses the project's
   JSON-mode + Pydantic + one-repair-retry pattern from M1, so every model call goes through one provider-agnostic client and `langchain`
   is not pulled in for a call type `litellm` already handles.
6. **Tighter legal routes.** `legal_routes()` in the memo team allows exactly one next step at each state (the pipeline is sequential),
   so the model confirms the policy's choice and has no discretion over it.

## Known limits

* `check_citations` proves a citation points at a real finding, not that the finding supports the sentence.
* Safety-net and redaction patterns are tuned to the shapes in the sample data; they are a floor, not a classifier.
* Delivery date and lost-shipment age are estimates from `ship_date`.
* Replies are not channel-aware (a phone transcript and a web form get the same register).
* A cancelled order never opens a refund approval (an assumption about the business rule).
* Over MCP, human-approved writes are authorised by an HMAC token whose secret both processes share; a production system would use a
  credential service. The built-in development secret must be replaced anywhere it matters.
* Guardrails are a pattern layer with five documented gaps (`docs/GUARDRAIL_EVALUATION.md`); labelled-case numbers are in-sample.
* The demo API trusts a body `customer_id`, runs one ticket at a time and has no approver endpoint or authentication
  (`docs/ARCHITECTURE_REVIEW.md` §5).
* Offline retrieval numbers use a hashed bag-of-words embedder and are a floor; set `SHOPSENSE_EMBED_MODEL` for real embeddings.
