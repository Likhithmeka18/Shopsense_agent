"""
ShopSense RAG: reranking (Milestone 4).

Fusion gives a good candidate pool, but ranks on coarse similarity signals.
A reranker reads the query and each candidate TOGETHER and scores true
relevance more precisely -- too expensive to run over the whole corpus, so
it only re-scores the fused top-N, the usual division of labor: a cheap broad net,
then an expensive precise sort.

The common alternative is a local cross-encoder (sentence-transformers,
cross-encoder/ms-marco-MiniLM-L-6-v2). That needs the sentence-transformers
+ torch stack -- a heavy, version-sensitive dependency this project avoids,
consistent with the rest of the codebase staying installable with a small,
pinned dependency set (see README). The swap made here is an
LLM-as-reranker: one prompt asks the model to score each candidate's
relevance 0-10 against the query, through the SAME provider-agnostic LLM
client used by every other model call in this codebase. This is a real
production alternative to a cross-encoder (no extra model to host), at the
cost of an LLM call instead of local CPU inference -- a genuine trade-off,
stated here rather than hidden.
"""
from __future__ import annotations

import json
import re

from ..core.llm_client import DEFAULT_MODEL, llm_completion
from .ingest import Chunk

RERANK_PROMPT = """Score how relevant each passage is to the question, from 0 (irrelevant) to 10 (directly answers it).

Question: {query}

Passages:
{passages}

Return ONLY a JSON array of integers, one score per passage, in the same order. Example: [7, 2, 9]"""


def rerank(query: str, cand_ids: list[int], chunks: list[Chunk], *, k: int = 5, model: str = DEFAULT_MODEL) -> list[int]:
    """Re-score `cand_ids` for true relevance to `query` and return the
    top-k, most relevant first. Falls back to the incoming (fused) order,
    truncated to k, if the model's response can't be parsed -- a malformed
    rerank reply should degrade gracefully, not break the pipeline."""
    if not cand_ids:
        return []
    passages = "\n".join(f"[{i}] {chunks[cid].text[:400]}" for i, cid in enumerate(cand_ids))
    resp = llm_completion(
        messages=[{"role": "user", "content": RERANK_PROMPT.format(query=query, passages=passages)}],
        model=model,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    match = re.search(r"\[[\d,\s]+\]", raw)
    if not match:
        return cand_ids[:k]
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return cand_ids[:k]
    if len(scores) != len(cand_ids):
        return cand_ids[:k]
    order = sorted(range(len(cand_ids)), key=lambda i: scores[i], reverse=True)
    return [cand_ids[i] for i in order[:k]]
