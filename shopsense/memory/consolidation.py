"""
ShopSense Memory: consolidation (Milestone 3).

Long-term memory that nobody tidies accumulates duplicate and stale facts
across sessions -- three sessions can leave one cluttered file: a preference restated three different ways, and one fact (a trial
expiry date) that's gone stale weeks ago but is still sitting there,
quietly misleading. consolidate() is the fix, run as a batch job BETWEEN
sessions -- never mid-conversation -- the "dreaming" idea (offline consolidation, like sleep), implemented here as a
plain function rather than a background scheduler.

One pass over a customer's SEMANTIC memories:
  1. prune  -- drop any record whose metadata marks it expired (an explicit
     `expires_at` set at write time; consolidation never guesses what's
     stale on its own -- that would be the same "psychological profiling"
     mistake as inferring facts nobody stated)
  2. merge  -- cluster near-duplicate facts (cosine similarity above
     MERGE_THRESHOLD) and ask the LLM to merge each cluster into one record

Episodic and procedural memories are left alone: episodic entries are
meant to accumulate (they're a log of what happened, not a single current
state), and procedural rules are few and edited deliberately rather than
auto-merged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .embeddings import cosine_similarity, embed_query
from .memory_store import MemoryRecord, MemoryStore
from ..core.llm_client import DEFAULT_MODEL, llm_completion

MERGE_THRESHOLD = 0.92

MERGE_PROMPT = (
    "These are near-duplicate facts remembered about the same customer, "
    "written at different times:\n{facts}\n\n"
    "Merge them into ONE concise fact that preserves everything true and "
    "current. Return only the merged fact, one sentence, no commentary."
)


@dataclass
class ConsolidationReport:
    customer_id: str
    before: int
    after: int
    merged_groups: int
    pruned_stale: int


def _cluster(records: list[MemoryRecord], threshold: float) -> list[list[MemoryRecord]]:
    """Greedy single-pass clustering: each record joins the first existing
    cluster whose representative (first member) it's similar enough to,
    else starts a new cluster. Simple, deterministic, good enough at the
    scale one customer's memory file ever reaches."""
    clusters: list[list[MemoryRecord]] = []
    for r in records:
        placed = False
        for cluster in clusters:
            if cosine_similarity(r.embedding, cluster[0].embedding) >= threshold:
                cluster.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])
    return clusters


def consolidate(
    store: MemoryStore,
    customer_id: str,
    *,
    model: str = DEFAULT_MODEL,
    merge_threshold: float = MERGE_THRESHOLD,
) -> ConsolidationReport:
    """Run one consolidation pass over `customer_id`'s semantic memories:
    prune anything explicitly marked stale, then merge near-duplicate
    clusters via one LLM call per cluster. Rewrites the store in place."""
    all_records = store.all(customer_id)
    semantic = [r for r in all_records if r.kind == "semantic"]
    other = [r for r in all_records if r.kind != "semantic"]
    before = len(semantic)

    now = time.time()
    fresh = [r for r in semantic if not (r.metadata.get("expires_at") and r.metadata["expires_at"] < now)]
    pruned_stale = len(semantic) - len(fresh)

    clusters = _cluster(fresh, merge_threshold)
    merged: list[MemoryRecord] = []
    merged_groups = 0
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        merged_groups += 1
        facts = "\n".join(f"- {r.text}" for r in cluster)
        resp = llm_completion(
            messages=[{"role": "user", "content": MERGE_PROMPT.format(facts=facts)}],
            model=model,
            temperature=0,
        )
        merged_text = resp.choices[0].message.content.strip()
        kept = cluster[0]
        merged.append(MemoryRecord(
            id=kept.id,
            customer_id=customer_id,
            kind="semantic",
            text=merged_text,
            created_at=max(r.created_at for r in cluster),
            embedding=embed_query(merged_text, model=store.embed_model),
            metadata={**kept.metadata, "merged_from": [r.id for r in cluster]},
        ))

    store.replace_all(customer_id, other + merged)
    return ConsolidationReport(
        customer_id=customer_id,
        before=before,
        after=len(merged),
        merged_groups=merged_groups,
        pruned_stale=pruned_stale,
    )
