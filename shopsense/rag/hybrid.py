"""
ShopSense RAG: hybrid search via Reciprocal Rank Fusion (Milestone 4).

Dense semantic search and BM25 keyword search run independently, then get
fused with RRF: each chunk's fused score is sum(1 / (c + rank)) across the
ranked lists it appears in, so a chunk ranked highly by EITHER retriever
floats to the top. This is what gives hybrid search its guarantee: BM25's
exact-token match never depends on embedding quality, so hybrid is never
worse than dense alone on an exact-identifier query (a policy code, an
escalation code), and is sometimes much better.
"""
from __future__ import annotations

from .bm25 import BM25Index
from .index import VectorIndex


def rrf_fuse(rankings: list[list[int]], *, k: int = 10, c: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over any number of ranked id lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def hybrid_search(index: VectorIndex, bm25: BM25Index, query: str, *, k: int = 10, pool: int = 10) -> list[int]:
    """Fused dense + BM25 candidate list, no rerank."""
    dense_ids = index.dense_search(query, k=pool)
    bm25_ids = bm25.search(query, k=pool)
    return rrf_fuse([dense_ids, bm25_ids], k=k)
