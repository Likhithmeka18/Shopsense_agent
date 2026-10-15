"""
ShopSense RAG: minimal BM25 keyword search (Milestone 4).

Dense (embedding) search captures meaning, but
that match is probabilistic -- it depends on the embedding model being
good at representing an exact token like a policy code (RP-1.3) or an
escalation code (ESC-LOST). BM25 is the deterministic complement: pure
token overlap, scored with the standard Okapi BM25 formula, with no
dependency on any embedding model's quality. This is a small, from-scratch
implementation (no rank_bm25 package needed) so the RAG stack stays
installable with nothing beyond the Python standard library.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Standard Okapi BM25 over a fixed corpus of Chunks."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._docs = [_tokenize(c.text) for c in chunks]
        self._doc_len = [len(d) for d in self._docs]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._docs else 0.0
        self._term_freqs = [Counter(d) for d in self._docs]
        self._df: Counter = Counter()
        for d in self._docs:
            for term in set(d):
                self._df[term] += 1
        n_docs = len(self._docs) or 1
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        score = 0.0
        tf = self._term_freqs[doc_idx]
        dl = self._doc_len[doc_idx]
        for term in query_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        return score

    def search(self, query: str, *, k: int = 10) -> list[int]:
        """Return the top-k chunk ids (cid) by BM25 score, highest first."""
        query_terms = _tokenize(query)
        scores = [(c.cid, self._score(query_terms, i)) for i, c in enumerate(self.chunks)]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return [cid for cid, _score in scores[:k]]
