"""
ShopSense Memory: rolling summarization for working memory (Milestone 3).

core.llm_client.SessionStore keeps the full, growing message list for a
ticket thread -- "just resend everything". That is fine
for a short exchange, but a long-running thread (a customer going back and
forth many times) will eventually blow the context-window budget, exactly
like a support chat that slows to a crawl: re-reading
the entire afternoon before every reply, then dropping the oldest messages
-- including the order number -- once there's no more room.

compact_if_needed() is the fix: once a thread's history exceeds
TOKEN_BUDGET, the OLD turns (everything except the most recent
RECENT_KEEP) are compressed into a rolling summary via a real LLM call,
merged with any previous summary so an earlier round of compaction never
gets silently re-dropped. The compacted history replaces the old turns
with one summary message instead of deleting them outright.

Summarization is lossy by construction -- by nature --
so this module intentionally does NOT trust the summary on faith.
summary_preserves() below is a self-check: plant a
fact, force compaction, assert the fact survived. See
tests/test_memory.py::test_compaction_preserves_planted_fact for the
executable version of that check.
"""
from __future__ import annotations

from ..core.llm_client import DEFAULT_MODEL, llm_completion

RECENT_KEEP = 4       # turns always kept verbatim, never summarized
TOKEN_BUDGET = 1200   # character count (not true tokens) -- cheap, deterministic trigger for compaction

SUMMARY_PREFIX = "[Earlier conversation summary]\n"

SUMMARY_SYSTEM_PROMPT = (
    "You maintain a running summary of a customer support conversation. "
    "Update the existing summary with the new turns, preserving concrete "
    "facts (order IDs, amounts, dates, stated preferences, complaints). "
    "Return only the updated summary, under 120 words."
)


def _approx_size(history: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in history)


def summarize_turns(turns: list[dict], prev_summary: str = "", *, model: str = DEFAULT_MODEL) -> str:
    """Compress `turns` into an updated rolling summary, merging with
    `prev_summary` so nothing from an earlier compaction round is dropped.
    A real LLM call -- this is the lossy step, not a string truncation."""
    convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in turns)
    user = f"Existing summary:\n{prev_summary or '(none)'}\n\nNew turns:\n{convo}\n\nUpdated summary:"
    resp = llm_completion(
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def compact_if_needed(
    history: list[dict],
    *,
    token_budget: int = TOKEN_BUDGET,
    recent_keep: int = RECENT_KEEP,
    model: str = DEFAULT_MODEL,
) -> tuple[list[dict], bool]:
    """Return (possibly-compacted history, did_compact). `history[0]` is
    assumed to be the thread's primary system prompt and is always kept
    verbatim. Compaction only fires once the budget is exceeded -- a
    deliberate keep/discard policy, not blind truncation of the oldest
    messages (a common failure mode)."""
    if _approx_size(history) <= token_budget or len(history) <= recent_keep + 1:
        return history, False

    system_msg = history[0]
    rest = history[1:]
    old, recent = rest[: len(rest) - recent_keep], rest[len(rest) - recent_keep :]

    # If an earlier compaction already left a summary message at the front
    # of `rest`, fold it in as prev_summary instead of re-summarizing it
    # from scratch (which could silently drop what it already preserved).
    prev_summary = ""
    if old and old[0].get("role") == "system" and str(old[0].get("content", "")).startswith(SUMMARY_PREFIX):
        prev_summary = old[0]["content"][len(SUMMARY_PREFIX):]
        old = old[1:]

    if not old:
        # Nothing new to fold in beyond what's already summarized.
        return history, False

    summary_text = summarize_turns(old, prev_summary, model=model)
    summary_msg = {"role": "system", "content": f"{SUMMARY_PREFIX}{summary_text}"}
    return [system_msg, summary_msg, *recent], True


def summary_preserves(summary: str, must_contain: str) -> bool:
    """Self-check: does the compacted summary still mention a fact that
    mattered? Call this after compact_if_needed() instead of assuming a
    summary that 'sounds complete' actually kept what you needed."""
    return must_contain.lower() in summary.lower()
