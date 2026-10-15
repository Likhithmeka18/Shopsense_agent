"""
ShopSense RAG: document ingestion + structure-aware chunking (Milestone 4).

Loads ShopSense's policy documents (data/shopsense/policies/*.md) and splits
each into overlapping chunks a retriever can search and a generator can
cite. Chunking is structure-aware: split on
markdown headers first, so a chunk never straddles two policy sections,
then on paragraph/sentence boundaries within a section, falling back to a
hard character cut only if a single paragraph is still too long. This is a
small, dependency-free reimplementation of the idea behind LangChain's
RecursiveCharacterTextSplitter / MarkdownHeaderTextSplitter -- the codebase
doesn't require langchain to be installed, the same "hand-roll it, stay
provider-agnostic" approach used for the LLM/embedding clients.

Every chunk keeps a `source` (which policy document), a `heading` (nearest
markdown heading, carried into chunk metadata the way
MarkdownHeaderTextSplitter does), and a `cid` (its index in the returned
list -- the id used everywhere downstream: the vector index, BM25, and the
evaluation harness all key off this same integer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "policies"

CHUNK_SIZE = 600     # characters
CHUNK_OVERLAP = 100  # characters


@dataclass
class Chunk:
    cid: int
    text: str
    source: str
    heading: str = ""
    metadata: dict = field(default_factory=dict)


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) sections on '#'-style
    headers. Text before the first header (if any) is kept under an empty
    heading, attributed to the document's implicit intro."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split on the largest separator that still fits the
    pieces within chunk_size: paragraph breaks, then single newlines, then
    sentences, then a hard character cut as the last resort. Adjacent
    chunks carry `overlap` characters of context from the previous chunk,
    so a fact split across the boundary is still findable in at least one
    chunk."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = None
    parts: list[str] = []
    for candidate_sep in ["\n\n", "\n", ". "]:
        if candidate_sep in text:
            sep = candidate_sep
            parts = [p for p in text.split(candidate_sep) if p.strip()]
            break
    if sep is None:
        sep = ""
        parts = list(text)  # character-by-character fallback

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:] if (current and overlap) else ""
        current = (tail + sep + part) if tail else part
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap:]
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_document(
    text: str, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, str]]:
    """Return a list of (heading, chunk_text) for one document's full text.

    The heading is prepended to the body before splitting, not just carried
    as separate metadata -- a policy code that appears only in a section's
    heading (e.g. "### RP-3.2 ...") must still be part of the searchable
    chunk text, the same way MarkdownHeaderTextSplitter keeps the
    header line inside chunk 1 rather than discarding it."""
    out: list[tuple[str, str]] = []
    for heading, body in _split_by_headers(text):
        full_body = f"{heading}\n{body}" if heading else body
        for piece in _split_text(full_body, chunk_size, overlap):
            out.append((heading, piece))
    return out


def load_policy_corpus(
    policy_dir: Path = DEFAULT_POLICY_DIR, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Load every *.md file in `policy_dir`, chunk it, and return one flat
    list of Chunk with cid assigned in load order. This is the corpus every
    other rag/ module (index, bm25, evaluate) operates over."""
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text()
        for heading, piece in chunk_document(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(cid=len(chunks), text=piece, source=path.stem, heading=heading))
    return chunks


# --------------------------------------------------------------------------
# PDF variant of the corpus (Data Sourcing Guidelines 3.3: "chunk this too").
# generate_corpus_pdfs.py renders every policy to a LOSSY PDF: heading markers
# gone, tables flattened, running header + page number on every page. A
# chunker that only works on clean markdown fails here, so this path does the
# two recoveries real policy PDFs need -- strip page furniture, then rebuild
# section boundaries from the policy codes ("RF-3.1 ...", "CA-BT-1.2 ...")
# that every section title starts with -- and then reuses the SAME
# chunk_document() as the markdown path.
# --------------------------------------------------------------------------
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "shopsense" / "policies_pdf"

_PAGE_NUMBER = re.compile(r"^Page \d+( of \d+)?$")
_RUNNING_HEADER = re.compile(r"^ShopSense Support Operations\b")
_SECTION_CODE = re.compile(r"^(?:CA-[A-Z]{2}|RP|RF|SP|ESC|RA|WP|FP|CN)-\d[\d.]*\s+\S")


def strip_page_furniture(page_texts: list[str]) -> str:
    """Join per-page text, dropping page numbers and any line that repeats
    at the top of every page (the running header)."""
    per_page = [[ln.strip() for ln in p.splitlines() if ln.strip()] for p in page_texts]
    repeated: set[str] = set()
    if len(per_page) > 1:
        repeated = set.intersection(*(set(lines[:2]) for lines in per_page if lines))
    out: list[str] = []
    for lines in per_page:
        for ln in lines:
            if _PAGE_NUMBER.match(ln) or _RUNNING_HEADER.match(ln) or ln in repeated:
                continue
            out.append(ln)
    return "\n".join(out)


def recover_headings(text: str) -> str:
    """Turn lines that look like section titles back into '## ' headings:
    a leading policy code (RF-3.1, CA-EL-2, ESC-2.1 ...), or a short FAQ
    question. Table rows start with codes like 'ESC-L1' (letters after the
    dash), so the digit requirement keeps them as body text."""
    lines = text.splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        is_code = bool(_SECTION_CODE.match(ln)) and "   " not in ln and len(ln) < 110
        is_question = ln.endswith("?") and len(ln) < 110 and i > 0
        if i == 0 and ln.startswith("ShopSense"):
            out.append(f"# {ln}")
        elif is_code or is_question:
            out.append(f"## {ln}")
        else:
            out.append(ln)
    return "\n".join(out)


def load_policy_corpus_pdf(
    pdf_dir: Path = DEFAULT_PDF_DIR, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """PDF counterpart of load_policy_corpus(): same Chunk shape, same
    chunk_document(), `source` is the PDF's stem (== the markdown stem, so
    gold chunks and citations line up across both variants)."""
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pypdf is required to ingest the PDF corpus: pip install pypdf") from e
    chunks: list[Chunk] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
        text = recover_headings(strip_page_furniture(pages))
        for heading, piece in chunk_document(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(cid=len(chunks), text=piece, source=path.stem, heading=heading,
                                metadata={"format": "pdf"}))
    return chunks
