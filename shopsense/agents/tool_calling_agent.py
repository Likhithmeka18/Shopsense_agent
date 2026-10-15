"""
ShopSense Agents: tool-calling single agent (Milestone 2, structured tool calling).

The MODEL decides which tool(s) to call and with what arguments -- this file
never hardcodes that decision. Loop: send messages + tool schemas to the
model; if it requests tool calls, run them and append results; ask again;
repeat until the model replies in plain text, capped at max_iterations so a
confused model can't loop forever.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from ..tools.order_tools import MockAPI, build_tools

SYSTEM_PROMPT = (
    "You are a ShopSense customer support agent. Use the available tools to look up "
    "order, shipping, and refund information before answering -- never guess an "
    "order's status. Resolve the customer's request directly when you can. If a "
    "refund needs manager or executive approval, or the order can't be found, say "
    "so clearly so it can be escalated. If you don't have an order ID, ask the "
    "customer for one instead of guessing. Keep your final reply short and "
    "customer-facing."
)


@dataclass
class AgentRun:
    ticket_id: str
    final_answer: str
    transcript: list[dict] = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_cap: bool = False


class ToolCallingAgent:
    """Bounded tool-calling loop over the ShopSense order tools."""

    def __init__(self, api: MockAPI | None = None, model: str = DEFAULT_MODEL, max_iterations: int = 6):
        self.api = api or MockAPI()
        self.callables, self.tool_schemas = build_tools(self.api)
        self.model = model
        self.max_iterations = max_iterations

    def run(self, ticket_id: str, user_message: str) -> AgentRun:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_calls_made: list[str] = []

        for i in range(self.max_iterations):
            resp = llm_completion(messages=messages, model=self.model, tools=self.tool_schemas)
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                return AgentRun(
                    ticket_id=ticket_id,
                    final_answer=msg.content or "",
                    transcript=messages,
                    tool_calls_made=tool_calls_made,
                    iterations=i + 1,
                )

            messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls_made.append(name)
                fn = self.callables.get(name)
                try:
                    result = fn(**args) if fn else {"error": f"unknown tool {name}"}
                except Exception as e:  # noqa: BLE001 -- surface any tool failure to the model as data
                    result = {"error": str(e)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                })

        return AgentRun(
            ticket_id=ticket_id,
            final_answer="I wasn't able to resolve this within my step budget -- escalating to a human agent.",
            transcript=messages,
            tool_calls_made=tool_calls_made,
            iterations=self.max_iterations,
            hit_iteration_cap=True,
        )
