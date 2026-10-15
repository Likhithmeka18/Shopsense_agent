"""
ShopSense dev script: render the policy corpus (data/shopsense/policies/*.md)
to PDF, deliberately LOSSY, per the Data Sourcing Guidelines section 3.3
("emits every document as both markdown and PDF ... flattens heading
structure the way real policy PDFs do").

What "lossy" means here -- the three things real policy PDFs do to a parser:
  1. Heading structure is flattened: '#'/'##'/'###' markers are dropped and
     every heading is set in the same body font as the text around it.
  2. Tables are flattened into plain rows of text (no cell structure).
  3. A running header and a "Page N of M" footer are printed on every page,
     so page furniture lands inside the extracted text.

Needs `reportlab` (dev-only; not a runtime dependency of the agent).
Deterministic: same markdown in -> same PDF text out.

    pip install reportlab
    python generate_corpus_pdfs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "shopsense" / "policies"
OUT = ROOT / "data" / "shopsense" / "policies_pdf"
RUNNING_HEADER = "ShopSense Support Operations -- Internal Policy Document"


def flatten_markdown(md: str) -> list[str]:
    """Markdown -> plain lines. Headings lose their '#', bold/backticks lose
    their markers, table rows become 'a  b  c' and the |---| rule is
    dropped. Blank lines are kept as paragraph breaks."""
    lines: list[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if re.match(r"^\|\s*-{2,}", line):
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            line = "   ".join(cells)
        line = line.replace("**", "").replace("`", "")
        line = re.sub(r"^\s*[-*]\s+", "- ", line)
        lines.append(line)
    return lines


def render_pdf(md_path: Path, out_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer
    from xml.sax.saxutils import escape

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=0)
    state = {"pages": 0}

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, RUNNING_HEADER)
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(out_path), pagesize=A4, title=md_path.stem, author="ShopSense",
                          leftMargin=20 * mm, rightMargin=20 * mm, topMargin=22 * mm, bottomMargin=20 * mm,
                          invariant=1)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=furniture)])

    story = []
    for line in flatten_markdown(md_path.read_text()):
        story.append(Spacer(1, 5) if not line.strip() else Paragraph(escape(line), body))
    doc.build(story)


def main() -> None:
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("reportlab is not installed -- run: pip install reportlab")
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for md_path in sorted(SRC.glob("*.md")):
        out_path = OUT / f"{md_path.stem}.pdf"
        render_pdf(md_path, out_path)
        print(f"  {md_path.name} -> {out_path.relative_to(ROOT)}")
    print(f"wrote {len(list(OUT.glob('*.pdf')))} PDFs")


if __name__ == "__main__":
    main()
