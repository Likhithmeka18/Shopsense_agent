"""
Milestone 2 tests: the two agent patterns (structured tool-calling loop,
and ReAct + Reflection), each exercised against the scripted fake LLM so
the control flow -- the model decides, the loop just executes what it asks
for -- is verified without any real network call.
"""
import json

import pytest

from shopsense.agents.react_agent import ReActAgent
from shopsense.agents.tool_calling_agent import ToolCallingAgent
from shopsense.tools.order_tools import MockAPI


@pytest.fixture(scope="module")
def api():
    return MockAPI()


@pytest.fixture(scope="module")
def sample_order_id(api):
    return next(iter(api.orders.keys()))


# --------------------------------------------------------------------------
# ToolCallingAgent: the model decides which tool to call, loop just executes
# --------------------------------------------------------------------------

def test_tool_calling_agent_calls_tool_then_answers(scripted_litellm, api, sample_order_id):
    scripted_litellm.queue_tool_call("call_1", "order_lookup", {"order_id": sample_order_id})
    scripted_litellm.queue_text(f"Your order {sample_order_id} is on its way.")

    agent = ToolCallingAgent(api=api, max_iterations=6)
    run = agent.run("TCK-0001", f"Where is my order {sample_order_id}?")

    assert run.tool_calls_made == ["order_lookup"]
    assert run.final_answer == f"Your order {sample_order_id} is on its way."
    assert not run.hit_iteration_cap
    # the tool result must have been fed back to the model as a "tool" message
    tool_messages = [m for m in run.transcript if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["order_id"] == sample_order_id


def test_tool_calling_agent_answers_without_tools_when_not_needed(scripted_litellm):
    scripted_litellm.queue_text("Could you share your order ID so I can help?")
    agent = ToolCallingAgent(max_iterations=6)
    run = agent.run("TCK-0002", "I have a question but didn't mention an order.")
    assert run.tool_calls_made == []
    assert "order ID" in run.final_answer


def test_tool_calling_agent_hits_iteration_cap(scripted_litellm, api, sample_order_id):
    # Model keeps requesting the same tool forever -- loop must still terminate.
    for _ in range(3):
        scripted_litellm.queue_tool_call("call_x", "order_lookup", {"order_id": sample_order_id})

    agent = ToolCallingAgent(api=api, max_iterations=3)
    run = agent.run("TCK-0003", f"Status of {sample_order_id}?")

    assert run.hit_iteration_cap
    assert run.iterations == 3
    assert "escalat" in run.final_answer.lower()


# --------------------------------------------------------------------------
# ReActAgent: Thought/Action/Observation parsed from model text, + Reflection
# --------------------------------------------------------------------------

def test_react_agent_runs_action_then_final_answer_then_approved(scripted_litellm, api, sample_order_id):
    scripted_litellm.queue_text(
        f"Thought: I need order details.\nAction: order_lookup[{sample_order_id}]"
    )
    scripted_litellm.queue_text(
        f"Thought: I have what I need.\nFinal Answer: Order {sample_order_id} is confirmed in our system."
    )
    scripted_litellm.queue_text("APPROVED")

    agent = ReActAgent(api=api, max_iterations=6)
    run = agent.run(f"What's the status of {sample_order_id}?")

    assert not run.revised
    assert run.critique.startswith("APPROVED")
    assert len(run.evidence) == 1
    assert sample_order_id in run.answer


def test_react_agent_reflection_revises_unsupported_answer(scripted_litellm, api, sample_order_id):
    scripted_litellm.queue_text(
        f"Thought: checking.\nAction: order_lookup[{sample_order_id}]"
    )
    scripted_litellm.queue_text("Thought: done.\nFinal Answer: Your refund has already been processed.")
    scripted_litellm.queue_text("REVISE: the evidence says nothing about a refund")
    scripted_litellm.queue_text("Your order exists but no refund has been processed yet.")

    agent = ReActAgent(api=api, max_iterations=6)
    run = agent.run(f"Was my refund processed for {sample_order_id}?")

    assert run.revised
    assert "REVISE" in run.critique
    assert run.answer == "Your order exists but no refund has been processed yet."


def test_react_agent_hits_iteration_cap(scripted_litellm, api, sample_order_id):
    for _ in range(2):
        scripted_litellm.queue_text(f"Thought: still checking.\nAction: order_lookup[{sample_order_id}]")

    agent = ReActAgent(api=api, max_iterations=2)
    run = agent.run(f"Status of {sample_order_id}?")

    assert run.hit_iteration_cap
    assert "Stopped" in run.answer
