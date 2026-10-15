"""
ShopSense Memory: provider-agnostic embedding client (Milestone 3, part 1).

Wraps LiteLLM's embedding() the same way core/llm_client.py wraps
completion() -- swapping embedding providers/models is a `model=` string
change (e.g. "gemini/gemini-embedding-001", "text-embedding-3-small"),
nothing downstream changes. Milestone 3's semantic memory search and
Milestone 4's RAG retrieval both go through this one function.

A real embedding call needs network + an API key, same as every other LLM
call in this codebase. For offline development and the test suite, a
deterministic local fallback embedder is used (a hashed bag-of-words
projection), good enough to exercise similarity-ranking LOGIC -- the thing
this codebase's tests actually need to verify -- without ever calling a
network embedding model. It captures exact-token overlap only, not real
semantics; production ShopSense sets SHOPSENSE_EMBED_MODEL (or passes
model=) and never touches this fallback. This mirrors the mocked-litellm
approach used for chat completions in tests/conftest.py: same idea, applied
to embeddings instead of chat.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

# Empty string -> use the local fallback embedder. Set this to a real
# LiteLLM-supported embedding model string to use a real provider everywhere
# in the codebase at once.
DEFAULT_EMBED_MODEL = os.getenv("SHOPSENSE_EMBED_MODEL", "")
FALLBACK_DIM = 256


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _local_fallback_embed(text: str, dim: int = FALLBACK_DIM) -> list[float]:
    """Deterministic, dependency-free stand-in for a real embedding: hash
    each token into a fixed-size vector (a tiny random-projection bag-of-
    words). Stable and reproducible across runs/processes, which is what
    the test suite needs -- it is NOT a substitute for real semantic
    similarity and is never used once a real embed model is configured."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


EMBED_BATCH_SIZE = int(os.getenv("SHOPSENSE_EMBED_BATCH", "50"))  # providers cap items per call (Gemini: 100)
EMBED_MAX_RETRIES = 6


def _is_rate_limit(err: Exception) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return any(t in text for t in ("ratelimit", "rate limit", "429", "quota", "resource_exhausted", "too many requests"))


def embed_texts(texts: list[str], *, model: str = DEFAULT_EMBED_MODEL, **kwargs: Any) -> list[list[float]]:
    """Embed many texts. Uses the real provider via LiteLLM if `model` is
    set; otherwise falls back to the local deterministic embedder so
    memory/RAG code is fully testable offline, with no API key.

    The real path sends the texts in batches (a 121-chunk corpus in one call
    is rejected by providers that cap a batch at 100) and waits out a
    free-tier rate limit instead of crashing; any other error raises."""
    if not model:
        return [_local_fallback_embed(t) for t in texts]
    import time

    from litellm import embedding  # imported lazily -- only needed on the real-model path

    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                resp = embedding(model=model, input=batch, **kwargs)
                break
            except Exception as e:  # noqa: BLE001 -- only rate limits are retried
                if not _is_rate_limit(e) or attempt == EMBED_MAX_RETRIES:
                    raise
                from ..core.llm_client import parse_retry_wait

                wait = parse_retry_wait(str(e)) or 20.0
                time.sleep(min(wait + 1.0, 70.0))
        out.extend(item["embedding"] for item in resp.data)
    return out


def embed_query(text: str, *, model: str = DEFAULT_EMBED_MODEL, **kwargs: Any) -> list[float]:
    """Embed a single text (e.g. a search query)."""
    return embed_texts([text], model=model, **kwargs)[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        # Empty vector or mismatched embedding spaces (e.g. a vector cached
        # from a different model) -- treat as unrelated rather than crash.
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
