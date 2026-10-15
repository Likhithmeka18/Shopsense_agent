"""
ShopSense RAG: retrieval evaluation harness (Milestone 4).

"It seems better" is not an engineering claim. GOLDEN is a small set of
real ShopSense policy questions, each tied to the exact chunk(s) that
answer it via `answer_contains` (a policy code, an escalation code, or a
distinctive phrase) -- the same "resolve gold chunks from a string, no
manual chunk ids" approach, so
the gold labels can never drift out of sync with however the corpus
happens to be chunked.

precision@k / recall@k / MRR are computed purely from ranked id lists --
no LLM call, fast and deterministic -- so evaluate() can score any
retrieval variant (naive-dense, hybrid, hybrid+rerank) cheaply and compare
them head to head, exactly the proof-it-works step Milestone 4 asks for.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .ingest import Chunk

# The retrieval questions live in data/shopsense/eval/retrieval_qa.json (an asset, not code). The inline list
# is only a fallback so the module still imports if that file is missing.
_FALLBACK_GOLDEN: list[dict] = [
    {"q": "What escalation code is used when a shipment is declared lost?", "answer_contains": "ESC-LOST"},
    {"q": "What is the SLA code for warehouse quality-check turnaround?", "answer_contains": "SLA-48"},
    {"q": "Which escalation code routes a refund of INR 12,000 for executive review?", "answer_contains": "ESC-L2"},
]
RETRIEVAL_QA_PATH = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "eval" / "retrieval_qa.json"


def load_retrieval_qa(path: Path = RETRIEVAL_QA_PATH) -> list[dict]:
    try:
        return json.load(open(path))["questions"]
    except (OSError, KeyError, ValueError):
        return list(_FALLBACK_GOLDEN)


GOLDEN: list[dict] = load_retrieval_qa()


def _norm(s: str) -> str:
    """Lower-case and collapse all whitespace, so a phrase that a PDF
    extractor happened to wrap across a line break still matches."""
    return " ".join(s.lower().split())


def gold_chunks(chunks: list[Chunk], answer_contains: str) -> set[int]:
    key = _norm(answer_contains)
    return {c.cid for c in chunks if key in _norm(c.text)}


def precision_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return sum(1 for cid in topk if cid in gold) / len(topk)


def recall_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr(retrieved: list[int], gold: set[int]) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def evaluate(
    retrieve_fn: Callable[[str, int], list[int]],
    chunks: list[Chunk],
    golden: list[dict] = GOLDEN,
    *,
    k: int = 5,
) -> dict:
    """Score a retrieval function `(query, k) -> ranked_chunk_ids` against
    `golden`, averaged across questions that resolve to at least one gold
    chunk in this corpus."""
    precisions, recalls, mrrs = [], [], []
    for item in golden:
        gold = gold_chunks(chunks, item["answer_contains"])
        if not gold:
            continue  # golden item doesn't resolve against this corpus -- skip rather than silently fail
        got = retrieve_fn(item["q"], k)
        precisions.append(precision_at_k(got, gold, k))
        recalls.append(recall_at_k(got, gold, k))
        mrrs.append(mrr(got, gold))
    n = len(precisions) or 1
    return {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "n_questions": len(precisions),
    }
