"""
ShopSense MCP: a FastMCP server exposing the Milestone 4 policy knowledge
base, safely, over stdio (Milestone 6).

Run standalone with:  python -m shopsense.mcp.policy_server
(it then sits waiting on stdin for JSON-RPC -- that silence means it is
working. Stdout must never receive a stray print() in a stdio server.)

Needs the `mcp` package (in requirements.txt). The sandbox rule this
follows: every path this server touches is resolved and must land inside
`data/shopsense/policies/`; symlinks are refused; reads are size-capped;
and -- the design rule worth taking away -- there is NO write tool and NO
delete tool. Not guarded: absent. A model cannot misuse a capability that
was never exposed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve the sandbox root from __file__, not from os.getcwd() -- the
# client launches this file as a SUBPROCESS and that process's working
# directory is not guaranteed to be the caller's.
_REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = _REPO_ROOT / "data" / "shopsense" / "policies"
MAX_READ_BYTES = 20_000

sys.path.insert(0, str(_REPO_ROOT))
from shopsense.mcp.compat import load_server_class  # noqa: E402

FastMCP, _SERVER_LIB = load_server_class()
mcp = FastMCP("shopsense-policy")

_kb_singleton = None  # built lazily -- only on the first tool call that needs it


def _kb():
    """Lazy PolicyKnowledgeBase singleton. Importing shopsense.* requires
    the repo root on sys.path, which a subprocess doesn't get for free."""
    global _kb_singleton
    if _kb_singleton is None:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from shopsense.tools.policy_tools import PolicyKnowledgeBase

        _kb_singleton = PolicyKnowledgeBase()
    return _kb_singleton


def _safe_path(relative: str, root: Path = POLICY_DIR) -> Path:
    """Resolve `relative` inside `root`, or raise. THE security boundary
    of this server -- see the module docstring for why each of these three
    checks exists; dropping any one of them reopens the hole the others
    close."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the sandbox: {relative!r} is outside {root.name}/")
    if candidate.is_symlink():
        raise ValueError(f"symlinks are not allowed: {relative!r}")
    return candidate


@mcp.tool()
def list_policy_files() -> list[str]:
    """List the ShopSense policy documents available to search or read."""
    return sorted(p.name for p in POLICY_DIR.iterdir() if p.is_file() and p.suffix == ".md")


@mcp.tool()
def read_policy_file(name: str) -> str:
    """Read one ShopSense policy document by filename.

    Args:
        name: a bare filename inside the policy corpus, e.g.
            "escalation_sop.md". Directory traversal and absolute paths
            are rejected.

    Returns:
        The document's text, truncated to 20000 bytes.
    """
    target = _safe_path(name)
    if not target.is_file():
        raise ValueError(f"not a file: {name!r}")
    return target.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]


@mcp.tool()
def search_policy(query: str, k: int = 3) -> list[dict]:
    """Search ShopSense's return/refund/shipping/escalation policy corpus
    for the chunks most relevant to a question (hybrid dense + BM25
    retrieval, no LLM rerank -- deterministic, so this tool needs no model
    or API key of its own).

    Args:
        query: a plain-language question about ShopSense policy.
        k: how many chunks to return (default 3).

    Returns:
        Matching chunks, each with id, source document, section heading,
        and text.
    """
    kb = _kb()
    ids = kb.retrieve(query, k=k, use_rerank=False)
    return [
        {"id": cid, "source": kb.chunks[cid].source, "section": kb.chunks[cid].heading, "text": kb.chunks[cid].text}
        for cid in ids
    ]


# THERE IS NO write TOOL AND NO delete TOOL. Adding one would mean
# reasoning about whether a prompt can talk a model out of using it --
# a losing argument to have with a probability distribution. This server
# only ever needs to be read from.


@mcp.resource("policy://escalation/citation-rules")
def citation_rules() -> str:
    """House rules every generated escalation memo must satisfy."""
    # A RESOURCE, not a tool: the application decides to load this; no
    # model is involved in that decision.
    return (
        "ESCALATION MEMO CITATION POLICY\n"
        "1. Every factual claim carries a tag of the form [F<id>], matching a "
        "retrieved finding.\n"
        "2. A tag is valid only if that finding was actually retrieved for this memo.\n"
        "3. Unsupported tags must be removed, not reworded.\n"
    )


if __name__ == "__main__":
    # NEVER print to stdout in a stdio server -- stdout IS the JSON-RPC
    # channel. Log to stderr instead.
    print("[shopsense.mcp.policy_server] starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio")
