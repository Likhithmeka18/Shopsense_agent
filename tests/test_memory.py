"""
Milestone 3 tests: persistent memory (episodic/semantic/procedural),
rolling summarization of working memory, consolidation, and the
memory-aware agent -- exercised against the scripted fake LLM (chat
completions) and the dependency-free local fallback embedder. Neither
needs network or an API key, the same offline-verification approach
tests/test_core.py and tests/test_agents.py already use for Milestones 1-2.
"""
import time

import pytest

from shopsense.agents.memory_agent import MemoryAwareAgent
from shopsense.memory.consolidation import consolidate
from shopsense.memory.embeddings import cosine_similarity, embed_query
from shopsense.memory.memory_store import MemoryStore
from shopsense.memory.summarizer import compact_if_needed, summary_preserves
from shopsense.tools.memory_tools import build_memory_tools
from shopsense.tools.order_tools import MockAPI


@pytest.fixture(scope="module")
def api():
    return MockAPI()


@pytest.fixture(scope="module")
def sample_order_id(api):
    return next(iter(api.orders.keys()))


# --------------------------------------------------------------------------
# embeddings: deterministic local fallback (no network / API key required)
# --------------------------------------------------------------------------

def test_fallback_embedding_is_deterministic():
    v1 = embed_query("customer prefers email updates")
    v2 = embed_query("customer prefers email updates")
    assert v1 == v2


def test_fallback_embedding_ranks_similar_text_higher():
    base = embed_query("the customer is allergic to peanuts")
    same_topic = embed_query("customer has a peanut allergy")
    unrelated = embed_query("order shipped via BlueDart yesterday")
    assert cosine_similarity(base, same_topic) > cosine_similarity(base, unrelated)


# --------------------------------------------------------------------------
# MemoryStore: the three CoALA long-term kinds, persisted and searchable
# --------------------------------------------------------------------------

def test_memory_store_write_and_search_ranks_relevant_fact_first(tmp_path):
    store = MemoryStore(base_dir=tmp_path)
    store.write("CUST-1", "semantic", "Customer prefers email over SMS for delivery updates")
    store.write("CUST-1", "semantic", "Customer's favorite color is blue")
    hits = store.search("CUST-1", "how should we contact this customer?", k=1)
    assert hits[0][0].text.startswith("Customer prefers email")


def test_memory_store_kind_filter(tmp_path):
    store = MemoryStore(base_dir=tmp_path)
    store.write("CUST-2", "episodic", "On 2026-10-01, ticket TCK-0099: refund approved")
    store.write("CUST-2", "procedural", "Auto-approve refunds under 3000 for this customer")
    episodic_only = store.all("CUST-2", kind="episodic")
    assert len(episodic_only) == 1 and episodic_only[0].kind == "episodic"


def test_memory_store_persists_across_instances(tmp_path):
    store1 = MemoryStore(base_dir=tmp_path)
    store1.write("CUST-3", "semantic", "Customer is a ShopSense Plus member")
    store2 = MemoryStore(base_dir=tmp_path)  # fresh instance, same directory
    records = store2.all("CUST-3")
    assert len(records) == 1 and "Plus member" in records[0].text


# --------------------------------------------------------------------------
# summarizer: lossy by construction -- test recall, don't assume it
# --------------------------------------------------------------------------

def test_compaction_does_not_fire_under_budget():
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    compacted, did_compact = compact_if_needed(history, token_budget=10_000)
    assert not did_compact and compacted == history


def test_compaction_preserves_planted_fact(scripted_litellm):
    # Plant a fact early, bury it under enough turns to exceed the budget,
    # force compaction, and verify -- don't assume -- that the fact
    # survived the lossy summary. 
    planted_fact = "the customer's pet parrot is named Kiwi"
    scripted_litellm.queue_text(f"Summary: customer mentioned {planted_fact} and asked about shipping delays.")

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"By the way, {planted_fact}."},
        {"role": "assistant", "content": "Noted!"},
    ]
    for i in range(6):
        history.append({"role": "user", "content": "Tell me more about shipping policy " * 10 + str(i)})
        history.append({"role": "assistant", "content": "Here is a long explanation. " * 10})

    compacted, did_compact = compact_if_needed(history, token_budget=500, recent_keep=4)
    assert did_compact
    summary_content = compacted[1]["content"]
    assert summary_preserves(summary_content, "Kiwi")


def test_compaction_merges_with_previous_summary(scripted_litellm):
    scripted_litellm.queue_text("merged summary mentioning Kiwi and the new shipping question")
    history = [
        {"role": "system", "content": "sys"},
        {"role": "system", "content": "[Earlier conversation summary]\nCustomer's parrot is named Kiwi."},
    ]
    for i in range(6):
        history.append({"role": "user", "content": "more shipping talk " * 20 + str(i)})
        history.append({"role": "assistant", "content": "ok " * 20})

    compacted, did_compact = compact_if_needed(history, token_budget=500, recent_keep=4)
    assert did_compact
    # prev_summary must have been folded into the prompt sent to the model,
    # not dropped and re-derived from scratch.
    last_call = scripted_litellm.calls[-1]
    assert "Kiwi" in last_call["messages"][-1]["content"]


# --------------------------------------------------------------------------
# consolidation: merge near-duplicate semantic memories, prune stale ones
# --------------------------------------------------------------------------

def test_consolidate_prunes_stale_and_merges_duplicates(tmp_path, scripted_litellm):
    store = MemoryStore(base_dir=tmp_path)
    store.write("CUST-4", "semantic", "User prefers email updates.")
    store.write("CUST-4", "semantic", "User prefers email updates for orders.")
    store.write("CUST-4", "semantic", "Trial expires soon", expires_at=time.time() - 1)  # already stale
    store.write("CUST-4", "semantic", "Customer lives in Mumbai")

    scripted_litellm.queue_text("User prefers email over SMS for order updates.")
    report = consolidate(store, "CUST-4", merge_threshold=0.3)

    assert report.before == 4
    assert report.pruned_stale == 1
    assert report.merged_groups == 1
    assert report.after == 2
    remaining = store.all("CUST-4", kind="semantic")
    assert all("Trial expires" not in r.text for r in remaining)


# --------------------------------------------------------------------------
# memory_tools: agent-callable remember/recall, scoped to one customer
# --------------------------------------------------------------------------

def test_memory_tools_remember_then_recall(tmp_path):
    store = MemoryStore(base_dir=tmp_path)
    callables, schemas = build_memory_tools(store, "CUST-5")
    assert {"remember", "recall"} == set(callables.keys())
    assert {"remember", "recall"} == {s["function"]["name"] for s in schemas}

    saved = callables["remember"](text="Customer prefers email updates", kind="semantic")
    assert saved["saved"] is True

    result = callables["recall"](query="how to contact this customer")
    assert result["results"][0]["text"] == "Customer prefers email updates"


def test_memory_tools_are_scoped_to_one_customer(tmp_path):
    store = MemoryStore(base_dir=tmp_path)
    callables_a, _ = build_memory_tools(store, "CUST-A")
    callables_a["remember"](text="secret A fact", kind="semantic")

    callables_b, _ = build_memory_tools(store, "CUST-B")
    result = callables_b["recall"](query="secret A fact")
    assert result["results"] == []


# --------------------------------------------------------------------------
# MemoryAwareAgent: recall-before-respond, agent-managed remember, compaction
# --------------------------------------------------------------------------

def test_memory_agent_recalls_before_first_turn(tmp_path, scripted_litellm):
    store = MemoryStore(base_dir=tmp_path)
    store.write("CUST-6", "semantic", "Customer prefers email over SMS for updates")

    scripted_litellm.queue_text("Sure, I'll use email as noted.")
    agent = MemoryAwareAgent(store=store, max_iterations=6)
    run = agent.run("TCK-1001", "CUST-6", "Please keep me posted on my order.")

    assert run.recalled_memories and "email" in run.recalled_memories[0]
    sent_messages = scripted_litellm.calls[0]["messages"]
    assert any("email" in (m.get("content") or "") for m in sent_messages)


def test_memory_agent_model_can_call_remember(tmp_path, scripted_litellm):
    store = MemoryStore(base_dir=tmp_path)
    scripted_litellm.queue_tool_call("call_1", "remember", {"text": "Customer is allergic to peanuts", "kind": "semantic"})
    scripted_litellm.queue_text("Got it, I've noted that.")

    agent = MemoryAwareAgent(store=store, max_iterations=6)
    run = agent.run("TCK-1002", "CUST-7", "FYI I'm allergic to peanuts.")

    assert "remember" in run.tool_calls_made
    saved = store.all("CUST-7", kind="semantic")
    assert len(saved) == 1 and "peanuts" in saved[0].text


def test_memory_agent_can_still_use_order_tools(tmp_path, scripted_litellm, api, sample_order_id):
    store = MemoryStore(base_dir=tmp_path)
    scripted_litellm.queue_tool_call("call_1", "order_lookup", {"order_id": sample_order_id})
    scripted_litellm.queue_text(f"Your order {sample_order_id} is on its way.")

    agent = MemoryAwareAgent(api=api, store=store, max_iterations=6)
    run = agent.run("TCK-1004", "CUST-9", f"Where is my order {sample_order_id}?")

    assert "order_lookup" in run.tool_calls_made
    assert run.final_answer == f"Your order {sample_order_id} is on its way."


def test_memory_agent_compacts_long_thread(tmp_path, scripted_litellm):
    store = MemoryStore(base_dir=tmp_path)
    agent = MemoryAwareAgent(store=store, max_iterations=6, token_budget=50)

    scripted_litellm.queue_text("First reply covering plenty of detail about the order status.")
    agent.run("TCK-1003", "CUST-8", "Hello, I have a long question about my order " * 5)

    scripted_litellm.queue_text("Second reply, also fairly long with details.")
    agent.run("TCK-1003", "CUST-8", "Here is a long follow-up message " * 5)

    scripted_litellm.queue_text("Summary of earlier turns.")
    scripted_litellm.queue_text("Third reply, now compacted.")
    run = agent.run("TCK-1003", "CUST-8", "One more follow-up " * 5)

    assert run.compacted


# ---------------------------------------------------------------------------
# real-embedding path: batching + rate-limit patience (fake litellm.embedding)
# ---------------------------------------------------------------------------

def test_real_embedding_path_batches_and_waits_out_a_rate_limit(monkeypatch):
    import sys
    import types

    from shopsense.memory import embeddings

    calls = []

    def fake_embedding(*, model, input, **kwargs):
        calls.append(len(input))
        if len(calls) == 2:                       # second batch is rate limited once
            calls.append("limited")
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, try again in 2s")
        data = [{"embedding": [float(len(t)), 1.0]} for t in input]
        return types.SimpleNamespace(data=data)

    monkeypatch.setattr(sys.modules["litellm"], "embedding", fake_embedding, raising=False)
    monkeypatch.setattr(embeddings, "EMBED_BATCH_SIZE", 50)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

    texts = ["x" * (i + 1) for i in range(121)]
    vecs = embeddings.embed_texts(texts, model="fake/embed")
    assert len(vecs) == 121 and [v[0] for v in vecs] == [float(i + 1) for i in range(121)]   # order preserved
    assert [c for c in calls if c != "limited"] == [50, 50, 50, 21] and slept == [3.0]        # 3 batches + 1 retry, waited 2s+1


def test_a_non_rate_limit_embedding_error_is_not_swallowed(monkeypatch):
    import sys

    import pytest
    from shopsense.memory import embeddings

    def boom(*, model, input, **kwargs):
        raise RuntimeError("401 invalid api key")

    monkeypatch.setattr(sys.modules["litellm"], "embedding", boom, raising=False)
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["a"], model="fake/embed")
