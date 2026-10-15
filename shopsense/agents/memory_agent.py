"""
ShopSense Agents: memory-aware agent (Milestone 3).

Builds on the Milestone 2 tool-calling loop (tool_calling_agent.py) without
modifying that file -- it stays a clean, minimal M2 artifact, and its own
tests keep passing unchanged. This agent adds the three pieces Milestone 3
is actually about:

  1. working memory that survives a long thread: the per-ticket message
     history is compacted (memory.summarizer.compact_if_needed) once it
     outgrows the token budget, instead of being resent in full forever --
     the fix for a support chat that slows to a crawl.
  2. relevant long-term memory is recalled and injected as context BEFORE
     the model's first turn, so the agent isn't starting cold on a
     returning customer.
  3. the model can itself decide, mid-conversation, to call remember/recall
     -- agent-managed memory, not an app-hardcoded save ("let the agent decide what's worth keeping").

The Milestone 2 order tools are still available in the same loop, so one
agent can both look up an order AND manage what it remembers about the
customer who asked -- memory augments the tool-calling agent, it doesn't
replace it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from ..memory.memory_store import MemoryStore
from ..memory.summarizer import TOKEN_BUDGET, compact_if_needed
from ..tools.memory_tools import build_memory_tools
from ..tools.order_tools import MockAPI, build_tools

SYSTEM_PROMPT = (
    "You are a ShopSense customer support agent with memory of past "
    "interactions. Relevant facts recalled about this customer are given "
    "to you as context below -- use them if helpful, but trust what the "
    "customer says in THIS conversation over an old memory if the two "
    "conflict. Use the order/shipping/refund tools before answering "
    "questions about an order -- never guess. Use the remember tool to "
    "save anything durable the customer tells you (a preference, a "
    "standing issue), but only things they actually said, never something "
    "you inferred. Keep your final reply short and customer-facing."
)


@dataclass
class MemoryAgentRun:
    ticket_id: str
    customer_id: str
    final_answer: str
    transcript: list[dict] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    recalled_memories: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_cap: bool = False
    compacted: bool = False


class MemoryAwareAgent:
    """Tool-calling loop + Milestone 3 memory: recall-before-respond,
    agent-managed remember/recall, and working-memory compaction for long
    threads. Same bounded-loop shape as ToolCallingAgent, extended with a
    MemoryStore and a persistent per-ticket thread (so a second `run()` call
    on the same ticket_id continues the same conversation)."""

    def __init__(
        self,
        api: MockAPI | None = None,
        store: MemoryStore | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 6,
        token_budget: int = TOKEN_BUDGET,
    ):
        self.api = api or MockAPI()
        self.store = store or MemoryStore()
        self.order_callables, self.order_schemas = build_tools(self.api)
        self.model = model
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        self._threads: dict[str, list[dict]] = {}

    def run(self, ticket_id: str, customer_id: str, user_message: str) -> MemoryAgentRun:
        memory_callables, memory_schemas = build_memory_tools(self.store, customer_id)
        callables = {**self.order_callables, **memory_callables}
        tool_schemas = [*self.order_schemas, *memory_schemas]

        history = self._threads.get(ticket_id)
        compacted_this_run = False
        recalled_texts: list[str] = []

        if history is None:
            recalled = self.store.search(customer_id, user_message, k=3)
            recalled_texts = [f"[{r.kind}] {r.text}" for r, _score in recalled]
            context = (
                "Relevant memory of this customer:\n" + "\n".join(f"- {t}" for t in recalled_texts)
                if recalled_texts else "No prior memory of this customer."
            )
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": context},
            ]
        else:
            history, compacted_this_run = compact_if_needed(history, token_budget=self.token_budget, model=self.model)

        history.append({"role": "user", "content": user_message})
        tool_calls_made: list[str] = []

        for i in range(self.max_iterations):
            resp = llm_completion(messages=history, model=self.model, tools=tool_schemas)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                history.append({"role": "assistant", "content": msg.content or ""})
                self._threads[ticket_id] = history
                return MemoryAgentRun(
                    ticket_id=ticket_id,
                    customer_id=customer_id,
                    final_answer=msg.content or "",
                    transcript=list(history),
                    tool_calls_made=tool_calls_made,
                    recalled_memories=recalled_texts,
                    iterations=i + 1,
                    compacted=compacted_this_run,
                )

            history.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls_made.append(name)
                fn = callables.get(name)
                try:
                    result = fn(**args) if fn else {"error": f"unknown tool {name}"}
                except Exception as e:  # noqa: BLE001 -- surface any tool failure to the model as data
                    result = {"error": str(e)}
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                })

        self._threads[ticket_id] = history
        return MemoryAgentRun(
            ticket_id=ticket_id,
            customer_id=customer_id,
            final_answer="I wasn't able to resolve this within my step budget -- escalating to a human agent.",
            transcript=list(history),
            tool_calls_made=tool_calls_made,
            recalled_memories=recalled_texts,
            iterations=self.max_iterations,
            hit_iteration_cap=True,
            compacted=compacted_this_run,
        )
