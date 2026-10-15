"""
ShopSense RAG: local vector index (Milestone 4).

Stands in for a hosted vector collection (e.g. Qdrant) without requiring a cloud
account: every chunk's embedding is computed once, via the same
provider-agnostic memory.embeddings.embed_texts used by Milestone 3's
memory store, and cached to disk under data/shopsense/rag_cache/, so
re-running ingestion re-embeds nothing -- the "throttle once, cache
forever" approach for free-tier embedding calls. The interface
(VectorIndex.dense_search) is what the rest of the codebase depends on;
swapping this for a real Qdrant/pgvector/etc. client later means
reimplementing this one class and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..memory.embeddings import DEFAULT_EMBED_MODEL, cosine_similarity, embed_texts
from .ingest import Chunk

DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "rag_cache" / "embeddings.json"


class VectorIndex:
    """In-memory vector index over a fixed corpus of Chunks, with a disk
    cache keyed by (model, chunk text) so re-embedding is never repeated."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cache_path: Path = DEFAULT_CACHE_PATH,
    ):
        self.chunks = chunks
        self.embed_model = embed_model
        self.cache_path = cache_path
        self._vectors: list[list[float]] = self._embed_all(chunks)

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(cache, f)

    def _cache_key(self, text: str) -> str:
        return f"{self.embed_model or 'local-fallback'}::{text}"

    def _embed_all(self, chunks: list[Chunk]) -> list[list[float]]:
        cache = self._load_cache()
        missing_idx = [i for i, c in enumerate(chunks) if self._cache_key(c.text) not in cache]
        if missing_idx:
            fresh = embed_texts([chunks[i].text for i in missing_idx], model=self.embed_model)
            for i, vec in zip(missing_idx, fresh):
                cache[self._cache_key(chunks[i].text)] = vec
            self._save_cache(cache)
        return [cache[self._cache_key(c.text)] for c in chunks]

    def dense_search(self, query: str, *, k: int = 10, source: Optional[str] = None) -> list[int]:
        """Return the top-k chunk ids (cid) by cosine similarity to `query`,
        optionally restricted to chunks from one source document."""
        q_vec = embed_texts([query], model=self.embed_model)[0]
        candidates = [
            (c.cid, cosine_similarity(q_vec, self._vectors[i]))
            for i, c in enumerate(self.chunks)
            if source is None or c.source == source
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [cid for cid, _score in candidates[:k]]

    def chunk_by_id(self, cid: int) -> Chunk:
        return self.chunks[cid]
