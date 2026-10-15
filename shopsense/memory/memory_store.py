"""
ShopSense Memory: persistent long-term memory store (Milestone 3).

Working memory already exists in this codebase as core.llm_client.SessionStore
-- the live per-ticket message buffer, gone once the session ends unless
something here writes a piece of it out. This module is the other three
kinds from the CoALA taxonomy (Sumers, Yao, Narasimhan & Griffiths), each a
conscious write, never an automatic save:

  episodic   -- a specific, timestamped thing that happened for one customer
                ("On 2026-10-09, ticket TCK-0031: refund for ORD-4021 approved")
  semantic   -- a durable fact/preference about a customer, no story attached
                ("Customer prefers email over SMS for delivery updates")
  procedural -- a standing rule for how to handle a recurring situation
                ("This customer's refunds under INR 3000 can be auto-approved")

Memory lives in one JSONL file per customer under
data/shopsense/memory/<customer_id>.jsonl -- plain files, no vector database
required to start, the same reasoning as file-based memory tools. Each record is embedded on write so search() can do
meaning-based recall within a kind, via the provider-agnostic
memory.embeddings.embed_query used everywhere in this package.

Nothing in this module decides on its own what's worth remembering -- that
judgment belongs to whatever calls write() (see tools/memory_tools.py and
agents/memory_agent.py). This module is storage and retrieval, not policy.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .embeddings import DEFAULT_EMBED_MODEL, cosine_similarity, embed_query

MemoryKind = Literal["episodic", "semantic", "procedural"]

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "memory"


@dataclass
class MemoryRecord:
    id: str
    customer_id: str
    kind: MemoryKind
    text: str
    created_at: float
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "MemoryRecord":
        return cls(**d)


class MemoryStore:
    """Per-customer JSONL-backed long-term memory, embedded for recall."""

    def __init__(self, base_dir: Optional[Path] = None, *, embed_model: str = DEFAULT_EMBED_MODEL):
        # SHOPSENSE_MEMORY_DIR moves the store (the test-suite points it at a temp folder so tests never write into data/).
        self.base_dir = Path(base_dir or os.environ.get("SHOPSENSE_MEMORY_DIR") or DEFAULT_MEMORY_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self._cache: dict[str, list[MemoryRecord]] = {}

    def _path(self, customer_id: str) -> Path:
        safe = customer_id.replace("/", "_")
        return self.base_dir / f"{safe}.jsonl"

    def _load(self, customer_id: str) -> list[MemoryRecord]:
        if customer_id in self._cache:
            return self._cache[customer_id]
        records: list[MemoryRecord] = []
        path = self._path(customer_id)
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(MemoryRecord.from_json(json.loads(line)))
        self._cache[customer_id] = records
        return records

    def _save(self, customer_id: str) -> None:
        records = self._cache.get(customer_id, [])
        with open(self._path(customer_id), "w") as f:
            for r in records:
                f.write(json.dumps(r.to_json()) + "\n")

    def write(self, customer_id: str, kind: MemoryKind, text: str, **metadata) -> MemoryRecord:
        """Persist one memory. This is the only way a fact graduates from
        working memory to long-term storage -- called explicitly by the
        agent (or a tool the agent decides to invoke), never automatically."""
        records = self._load(customer_id)
        record = MemoryRecord(
            id=str(uuid.uuid4())[:8],
            customer_id=customer_id,
            kind=kind,
            text=text,
            created_at=time.time(),
            embedding=embed_query(text, model=self.embed_model),
            metadata=metadata,
        )
        records.append(record)
        self._save(customer_id)
        return record

    def search(
        self, customer_id: str, query: str, *, kind: Optional[MemoryKind] = None, k: int = 3
    ) -> list[tuple[MemoryRecord, float]]:
        """Meaning-based recall: embed the query, rank stored memories (of
        the given kind, if any) by cosine similarity. Returns (record, score)
        pairs, highest similarity first."""
        records = [r for r in self._load(customer_id) if kind is None or r.kind == kind]
        if not records:
            return []
        q_vec = embed_query(query, model=self.embed_model)
        scored = [(r, cosine_similarity(q_vec, r.embedding)) for r in records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def all(self, customer_id: str, *, kind: Optional[MemoryKind] = None) -> list[MemoryRecord]:
        records = self._load(customer_id)
        return [r for r in records if kind is None or r.kind == kind]

    def replace_all(self, customer_id: str, records: list[MemoryRecord]) -> None:
        """Used by memory.consolidation to write back a deduped/pruned
        record set after a batch consolidation pass."""
        self._cache[customer_id] = records
        self._save(customer_id)
