"""
ShopSense Tools: policy_lookup tool (Milestone 4).

Wraps the Milestone 4 RAG pipeline -- ingest -> index -> hybrid search ->
rerank -> grounded, cited answer -- as one agent-callable tool, the same
way Milestone 2 wrapped MockAPI as order_lookup/refund_status/etc. The
agent decides when a question needs a policy lookup instead of an order
lookup; this file only provides the capability, bound once per process so
ingestion and embedding never repeat per ticket.
"""
from __future__ import annotations

from typing import Any, Optional

from ..core.llm_client import DEFAULT_MODEL
from ..observability.tracing import add_event, get_tracer
from ..rag.answer import answer_from_ids
from ..rag.bm25 import BM25Index
from ..rag.hybrid import hybrid_search
from ..rag.index import VectorIndex
from ..rag.ingest import Chunk, load_policy_corpus
from ..rag.rerank import rerank


class PolicyKnowledgeBase:
    """Loads and indexes the ShopSense policy corpus once; reused across
    every ticket so chunking/embedding never repeats per request."""

    def __init__(self, chunks: Optional[list[Chunk]] = None):
        self.chunks = chunks if chunks is not None else load_policy_corpus()
        self.index = VectorIndex(self.chunks)
        self.bm25 = BM25Index(self.chunks)

    def retrieve(
        self, query: str, *, k: int = 3, pool: int = 10, use_rerank: bool = True, model: str = DEFAULT_MODEL
    ) -> list[int]:
        tracer = get_tracer()
        with tracer.span("rag.retrieve", kind="retriever", input={"query": query}) as sp:
            fused = hybrid_search(self.index, self.bm25, query, k=pool, pool=pool)
            ids = fused[:k]
            if use_rerank:
                with tracer.span("rag.rerank", kind="retriever", input={"candidates": len(fused)}) as rs:
                    try:
                        ids = rerank(query, fused, self.chunks, k=k, model=model)
                    except Exception as e:  # noqa: BLE001 -- the model is down or throttled: keep the fused (BM25 + dense) order
                        add_event("rag.rerank_failed", error=f"{type(e).__name__}: {str(e)[:100]}")
                        rs.fail(f"rerank failed, using fused order: {type(e).__name__}")
            sp.update(output={"ids": ids, "sources": [self.chunks[i].source for i in ids]})
            return ids

    def answer(self, query: str, *, k: int = 3, model: str = DEFAULT_MODEL, use_rerank: bool = True,
               question: str | None = None) -> dict:
        """`query` is what the index searches for; `question` (default: the same text) is what the answer model is asked."""
        ids = self.retrieve(query, k=k, use_rerank=use_rerank, model=model)
        if not ids:
            return {
                "answer": "I don't have a policy document covering this -- escalating to a human agent.",
                "sources": [],
            }
        with get_tracer().span("rag.answer", kind="retriever", input={"query": query, "chunks": ids}):
            text = answer_from_ids(question or query, ids, self.chunks, model=model)
        sources = [
            {"id": cid, "source": self.chunks[cid].source, "section": self.chunks[cid].heading}
            for cid in ids
        ]
        return {"answer": text, "sources": sources}


def build_policy_tools(kb: PolicyKnowledgeBase, *, model: str = DEFAULT_MODEL) -> tuple[dict[str, Any], list[dict]]:
    """Build (callables_by_name, tool_schemas) bound to one PolicyKnowledgeBase."""

    def policy_lookup(question: str) -> dict:
        """Answer a question about ShopSense return/refund/shipping/escalation policy, grounded in and citing the actual policy documents."""
        return kb.answer(question, model=model)

    callables = {"policy_lookup": policy_lookup}
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "policy_lookup",
                "description": (
                    "Answer a question about ShopSense policy -- returns, refunds, "
                    "shipping timelines, or escalation rules -- grounded in and citing "
                    "the actual policy documents. Use this instead of answering policy "
                    "questions from memory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        },
    ]
    return callables, schemas
