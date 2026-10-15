"""
ShopSense Agents: ReAct loop + Reflection (Milestone 2, second pattern).

Thought / Action / Observation interleaved reasoning, parsed from the
model's own text, run against the same ShopSense tools as
tool_calling_agent.py. Useful for open-ended investigation questions
("why does this account have an abuse flag?") where the reasoning trace
should be visible as text rather than structured tool_calls. A capped
iteration count guards against runaway loops; a Reflection pass makes one
more model call to critique the draft answer against the evidence actually
gathered, and revises it if the critique isn't a clean APPROVED.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from ..tools.order_tools import MockAPI, build_tools

REACT_SYSTEM_PROMPT = """You are a ShopSense support investigation agent. Answer using
the available tools. Respond using EXACTLY this format, one step at a time:

Thought: <your reasoning about what to do next>
Action: <tool_name>[<argument>]

Available tools: order_lookup[order_id], refund_status[order_id],
shipping_tracker[order_id], calculate_refund_amount[order_id]

Wait -- you will be given an Observation. Do not write your own Observation.
When you have enough evidence, respond with:
Thought: <final reasoning>
Final Answer: <your answer>
"""

REFLECT_PROMPT = """You are reviewing a support agent's answer for accuracy.

Question: {question}

Evidence gathered:
{evidence}

Draft answer:
{draft}

Check: (1) the answer is actually supported by the evidence above, (2) it
actually addresses the question. If the draft is good, respond with exactly
APPROVED. Otherwise respond with "REVISE: <what's wrong>"."""


def _parse_action(text: str) -> tuple[str, str] | None:
    m = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
    if not m:
        return None
    return m.group(1), m.group(2)


def _parse_final_answer(text: str) -> str | None:
    m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else None


@dataclass
class ReActRun:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    transcript: str = ""
    critique: str = ""
    revised: bool = False
    hit_iteration_cap: bool = False


class ReActAgent:
    def __init__(self, api: MockAPI | None = None, model: str = DEFAULT_MODEL, max_iterations: int = 6):
        self.api = api or MockAPI()
        self.callables, _ = build_tools(self.api)
        self.model = model
        self.max_iterations = max_iterations

    def _run_tool(self, name: str, arg: str) -> str:
        fn = self.callables.get(name)
        if not fn:
            return f"Error: unknown tool {name}"
        try:
            return str(fn(arg))
        except Exception as e:  # noqa: BLE001 -- surface as an Observation, not a crash
            return f"Error: {e}"

    def run(self, question: str) -> ReActRun:
        transcript = f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n"
        evidence: list[str] = []

        for _ in range(self.max_iterations):
            resp = llm_completion(
                messages=[{"role": "user", "content": transcript}],
                model=self.model,
                stop=["Observation:"],
            )
            step = resp.choices[0].message.content
            transcript += step

            final = _parse_final_answer(step)
            if final:
                return self._reflect(question, final, evidence, transcript)

            action = _parse_action(step)
            if not action:
                transcript += "\nObservation: (no valid Action found; use the exact Thought/Action format)\n"
                continue

            name, arg = action
            observation = self._run_tool(name, arg)
            evidence.append(f"{name}[{arg}] -> {observation}")
            transcript += f"\nObservation: {observation}\n"

        return ReActRun(
            question=question,
            answer="Stopped: exceeded reasoning step budget without a final answer.",
            evidence=evidence,
            transcript=transcript,
            hit_iteration_cap=True,
        )

    def _reflect(self, question: str, draft: str, evidence: list[str], transcript: str) -> ReActRun:
        evidence_text = "\n".join(evidence) if evidence else "(no tool calls made)"
        critique_resp = llm_completion(
            messages=[{
                "role": "user",
                "content": REFLECT_PROMPT.format(question=question, evidence=evidence_text, draft=draft),
            }],
            model=self.model,
        )
        critique = critique_resp.choices[0].message.content.strip()

        if critique.upper().startswith("APPROVED"):
            return ReActRun(question=question, answer=draft, evidence=evidence, transcript=transcript, critique=critique)

        revise_resp = llm_completion(
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\nEvidence:\n{evidence_text}\n"
                    f"Previous draft: {draft}\nCritique: {critique}\n"
                    "Write a corrected final answer only."
                ),
            }],
            model=self.model,
        )
        revised_answer = revise_resp.choices[0].message.content.strip()
        return ReActRun(
            question=question, answer=revised_answer, evidence=evidence,
            transcript=transcript, critique=critique, revised=True,
        )
