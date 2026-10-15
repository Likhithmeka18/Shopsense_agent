"""
ShopSense Tools: agent-callable remember/recall tools (Milestone 3).

This is the "agent-managed memory" pattern: the model
decides, mid-conversation, what's worth writing down and when it's worth
looking something up -- these two functions are just plumbing around
memory.memory_store.MemoryStore. Nothing here pre-decides what counts as
memorable; that judgment is the LLM's, made through an ordinary tool call,
the same mechanism it already uses for order_lookup et al. in
tools/order_tools.py (Milestone 2).
"""
from __future__ import annotations

from typing import Any

from ..memory.memory_store import MemoryKind, MemoryStore


def build_memory_tools(store: MemoryStore, customer_id: str) -> tuple[dict[str, Any], list[dict]]:
    """Build (callables_by_name, tool_schemas) bound to one MemoryStore and
    one customer -- a fresh pair per ticket, so the model can never read or
    write another customer's memory through these tools."""

    def remember(text: str, kind: MemoryKind) -> dict:
        """Persist a durable fact, preference, past event, or standing rule
        about this customer so future conversations can recall it."""
        record = store.write(customer_id, kind, text)
        return {"saved": True, "id": record.id, "kind": kind}

    def recall(query: str, kind: str = "") -> dict:
        """Search this customer's long-term memory for facts relevant to
        `query`. Pass kind=episodic/semantic/procedural to narrow the
        search, or leave it blank to search every kind."""
        hits = store.search(customer_id, query, kind=kind or None, k=5)
        return {"results": [{"text": r.text, "kind": r.kind, "score": round(score, 3)} for r, score in hits]}

    callables = {"remember": remember, "recall": recall}
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save a durable fact, preference, past event, or standing rule about "
                    "this customer for future conversations. Use kind='episodic' for a "
                    "specific thing that happened, 'semantic' for a stable fact or "
                    "preference, 'procedural' for a standing rule on how to handle this "
                    "customer. Only save something the customer actually stated -- never "
                    "something you inferred or guessed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The fact to remember, in plain words."},
                        "kind": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    },
                    "required": ["text", "kind"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search this customer's long-term memory for facts relevant to a query, before answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural", ""],
                            "description": "Narrow to one kind, or omit/blank to search all kinds.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    return callables, schemas
