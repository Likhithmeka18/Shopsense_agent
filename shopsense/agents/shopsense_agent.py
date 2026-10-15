"""
ShopSense Agents: the combined production agent (Milestones 2 + 3 + 4).

Everything built so far, in one bounded tool-calling loop: Milestone 2's
order/shipping/refund tools, Milestone 3's agent-managed memory
(remember/recall, recalled-context injection, working-memory compaction),
and Milestone 4's policy_lookup tool (hybrid search + rerank + grounded,
cited RAG answer over the ShopSense policy documents). The model alone
decides which tool(s) a given message needs -- an order lookup, a policy
citation, a memory write, any combination, or none at all.

This file is deliberately self-contained rather than inheriting from
ToolCallingAgent or MemoryAwareAgent -- the same choice already made
between tool_calling_agent.py and react_agent.py: each milestone's agent
is its own readable artifact, not a layer of subclassing someone has to
trace through to see the whole loop. See MILESTONES.md for how this file
relates to the narrower per-milestone agents it builds on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from ..memory.memory_store import MemoryStore
from ..memory.summarizer import TOKEN_BUDGET, compact_if_needed
from ..tools.memory_tools import build_memory_tools
from ..tools.order_tools import MockAPI, build_tools
from ..tools.policy_tools import PolicyKnowledgeBase, build_policy_tools

SYSTEM_PROMPT = (
    "You are the ShopSense customer support agent. You have three kinds of "
    "tools: order/shipping/refund lookups (use them before answering any "
    "question about a specific order -- never guess), a policy_lookup tool "
    "for return/refund/shipping/escalation POLICY questions (use it instead "
    "of answering policy questions from memory -- policy details change and "
    "must be cited), and remember/recall for this customer's long-term "
    "memory (save only what the customer actually said; recalled facts "
    "about this customer are given to you as context below -- trust what "
    "the customer says in THIS conversation over an old memory if they "
    "conflict). If a ticket needs human review per policy (legal, safety, "
    "an abuse flag, or a high refund approval tier), say so plainly instead "
    "of resolving it yourself. Keep your final reply short and "
    "customer-facing."
)


@dataclass
class ShopSenseAgentRun:
    ticket_id: str
    customer_id: str
    final_answer: str
    transcript: list[dict] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    recalled_memories: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_cap: bool = False
    compacted: bool = False


class ShopSenseAgent:
    """The full-stack agent: order tools (M2) + memory (M3) + policy RAG
    (M4) in one bounded tool-calling loop over one per-ticket thread."""

    def __init__(
        self,
        api: MockAPI | None = None,
        store: MemoryStore | None = None,
        kb: PolicyKnowledgeBase | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 8,
        token_budget: int = TOKEN_BUDGET,
    ):
        self.api = api or MockAPI()
        self.store = store or MemoryStore()
        self.kb = kb or PolicyKnowledgeBase()
        self.order_callables, self.order_schemas = build_tools(self.api)
        self.policy_callables, self.policy_schemas = build_policy_tools(self.kb, model=model)
        self.model = model
        self.max_iterations = max_iterations
        self.token_budget = token_budget
        self._threads: dict[str, list[dict]] = {}

    def run(self, ticket_id: str, customer_id: str, user_message: str) -> ShopSenseAgentRun:
        memory_callables, memory_schemas = build_memory_tools(self.store, customer_id)
        callables = {**self.order_callables, **self.policy_callables, **memory_callables}
        tool_schemas = [*self.order_schemas, *self.policy_schemas, *memory_schemas]

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
                return ShopSenseAgentRun(
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
        return ShopSenseAgentRun(
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
